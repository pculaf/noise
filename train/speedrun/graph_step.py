"""Two graph launches per update, retaining the finite-credit check between them.

Schedule scalars are copied to a persistent GPU buffer each update. Optimizer state
and parameters are restored in place after warmup/capture. This is execution work,
not a new optimization algorithm; graph replay itself is established prior work.
"""
import copy
import torch
from adam_finish import addcdiv


class GraphStep:
    def __init__(self,module,estimator,optimizer,x,y):
        self.m,self.est,self.optimizer=module,estimator,optimizer
        self.x,self.y=x.clone(),y.clone()
        self.anvil,self.adam=optimizer.optimizers
        self.params=list(estimator.model.named_parameters())
        if bool((y<0).any()):raise ValueError('Graph step currently supports the unpadded benchmark batches only')
        self.generators=[estimator.generator,estimator.capture.generator]
        rng=[g.get_state() for g in self.generators];global_rng=torch.cuda.get_rng_state()
        lrs=[float(g['lr']) for g in optimizer.param_groups]
        self.host=torch.empty(4+2*len(self.anvil.param_groups)+2*len(self.adam.param_groups),pin_memory=True)
        self.control=torch.empty_like(self.host,device=x.device)
        step_values={int(self.adam.state[p]['step'].item()) for g in self.adam.param_groups for p in g['params']}
        if len(step_values)!=1:raise ValueError('Adam steps must agree for this fixed-model estimator')
        self.adam_step=step_values.pop()
        for g in self.adam.param_groups:
            g['capturable']=True
            for p in g['params']:
                state=self.adam.state[p]
                if not state:raise ValueError('Warm optimizer state before creating GraphStep')
                state['step']=state['step'].to(x.device)
        weights={k:v.clone() for k,v in estimator.model.state_dict().items()}
        states=[{p:{k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in s.items()}
                 for p,s in o.state.items()} for o in optimizer.optimizers]
        self.pack_controls()
        def restore():
            estimator.model.load_state_dict(weights)
            for o,snapshot in zip(optimizer.optimizers,states):
                for p,state in snapshot.items():
                    for k,v in state.items():
                        if torch.is_tensor(v):o.state[p][k].copy_(v)
                        else:o.state[p][k]=copy.deepcopy(v)
            for g,v in zip(optimizer.param_groups,lrs):g['lr']=v
            for g,v in zip(self.generators,rng):g.set_state(v)
            torch.cuda.set_rng_state(global_rng)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):self.estimate_body();self.optimizer_body()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();restore()
        self.estimate_graph=torch.cuda.CUDAGraph()
        for g in self.generators:self.estimate_graph.register_generator_state(g)
        with torch.cuda.graph(self.estimate_graph):self.estimate_body()
        self.optimizer_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.optimizer_graph):self.optimizer_body()
        torch.cuda.synchronize();restore()

    def pack_controls(self):
        a=self.anvil
        beta=a.rail_beta();engaged=a.t>=a.engage
        fast_beta=a.fast_beta if engaged else beta;weight=a.fast_weight if engaged else 1.
        values=[beta,1-fast_beta,weight,1-weight]
        for g in a.param_groups:
            lr=float(g['lr']);values += [lr,g['weight_decay']*lr]
        for g in self.adam.param_groups:
            b1,b2=g['betas'];step=self.adam_step+1
            values += [-float(g['lr'])/(1-b1**step),(1-b2**step)**.5]
        self.host.copy_(torch.tensor(values));self.control.copy_(self.host,non_blocking=True)

    def estimate_body(self):
        gradients=self.est.estimate(self.x,self.y)
        valid=(self.y!=-1).sum()
        self.flat=torch.cat([(gradients[name]/valid).float().reshape(-1) for name,_ in self.params])
        if self.est.world>1:torch.distributed.all_reduce(self.flat)
        self.flat/=self.est.world//self.est.split
        self.finite=torch.isfinite(self.flat).all()
        offset=0
        for name,p in self.params:
            p.grad=self.flat[offset:offset+p.numel()].view_as(p).to(p.dtype);offset+=p.numel()
        self.loss=self.est.capture.clean_loss.mean().reshape(1)
        if self.est.world>1:torch.distributed.all_reduce(self.loss);self.loss/=self.est.world

    def optimizer_body(self):
        a=self.anvil;beta,fast_weight,weight,other_weight=self.control[:4]
        for gi,group in enumerate(a.param_groups):
            lr,decay_rate=self.control[4+2*gi],self.control[5+2*gi]
            for p in group['params']:
                g=p.grad.float();lanes=1 if p.size(0)>=p.size(1) else 0
                state=a.state[p];fast,slow,energy=state['fast'],state['slow'],state['energy']
                fast.lerp_(g,fast_weight);slow.lerp_(g,1-a.slow_beta)
                update=self.m.polar(g.lerp(weight*fast+other_weight*slow,beta))
                energy.lerp_(update.square().mean(lanes,keepdim=True),1-a.lane_beta)
                gained=update*energy.clamp_min(1e-10).rsqrt()
                update=gained*(update.norm()/gained.norm().clamp_min(1e-10))
                w=p.float();decay=(slow*w>=0)*w*decay_rate
                p.copy_(w-lr*max(1,p.size(0)/p.size(1))**.5*(update+decay))
        for j,g in enumerate(self.adam.param_groups):
            if g['weight_decay']!=0 or g['amsgrad'] or g['maximize']:raise ValueError('Unsupported Adam variant')
            ps=g['params'];ms=[self.adam.state[p]['exp_avg'] for p in ps]
            vs=[self.adam.state[p]['exp_avg_sq'] for p in ps];gs=[p.grad for p in ps]
            torch._foreach_add_([self.adam.state[p]['step'] for p in ps],1)
            b1,b2=g['betas']
            torch._foreach_lerp_(ms,gs,1-b1)
            torch._foreach_mul_(vs,b2)
            torch._foreach_addcmul_(vs,gs,gs,1-b2)
            denom=torch._foreach_sqrt(vs)
            offset=4+2*len(a.param_groups)+2*j
            torch._foreach_div_(denom,[self.control[offset+1]]*len(ps))
            torch._foreach_add_(denom,g['eps'])
            for p,m,d in zip(ps,ms,denom):addcdiv(p,m,d,self.control,offset)

    def step(self,x,y):
        if x.shape!=self.x.shape or y.shape!=self.y.shape:raise ValueError('Graph shape changed')
        self.pack_controls();self.x.copy_(x);self.y.copy_(y)
        self.estimate_graph.replay()
        if not bool(self.finite):raise FloatingPointError('Non-finite estimate')
        self.optimizer_graph.replay();self.anvil.t+=1;self.adam_step+=1
        return self.loss.item()

    def close(self):
        # Captured NCCL operations hold communicator resources until graph reset.
        # Release them before torch.distributed.destroy_process_group().
        torch.cuda.synchronize()
        self.estimate_graph.reset()
        self.optimizer_graph.reset()
