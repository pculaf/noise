"""Research PyTorch implementation of PR29's recipe and work distribution.

See PROVENANCE.md. Torch seed-0 initialization uses the benchmark initializer;
it is not bitwise JAX initialization. No record claim before full validation.
"""
import math
import torch
import torch.distributed as dist
from torch.nn import functional as F
from moments import cycle_derivatives,rope32
from attention_moments import local_credit
from batched_optimizer import batch_update

compile_fn=lambda f:torch.compile(f,fullgraph=True,dynamic=False)


def norm(x):
    x=x.float()
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)


def linear(x,w):
    return F.linear(x.bfloat16(),w.bfloat16())


@compile_fn
def before_attention(x,x0,lr,l0,wq,wk,wv,wgate,ve,cos,sin):
    b,t,d=x.shape;h=d//64
    mixed=lr*x.float()+l0*x0.float();n=norm(mixed)
    qr,kr,vr=[linear(n,w).view(b,t,h,64) for w in (wq,wk,wv)]
    values=vr
    gate=linear(n[...,:32],wgate) if wgate is not None else None
    if ve is not None:
        values=values+(2*gate.float().sigmoid()).bfloat16()[...,None]*ve.view(b,t,h,64)
    q=rope32(qr,cos,sin).bfloat16();k=rope32(kr,cos,sin).bfloat16()
    return mixed,n,qr,kr,vr,gate,values,q,k


@compile_fn
def after_attention(mixed,o,wproj,wfc,wout):
    x=mixed.float()+linear(o,wproj).float();n=norm(x)
    pre=linear(n,wfc);act=F.relu(pre).square()
    out=x+linear(act,wout).float()
    return out,n,pre,act


@compile_fn
def head_pass(x,weight,y):
    final=norm(x);raw=linear(final,weight).float()
    loss=F.cross_entropy(15*torch.tanh(raw.flatten(0,1)/15),y.flatten(),reduction='none').view_as(y)
    return final,raw,loss


@compile_fn
def head_credit(raw,y):
    z=15*torch.tanh(raw/15);up=15*torch.tanh((raw+.1)/15);down=15*torch.tanh((raw-.1)/15)
    p=z.softmax(-1)
    change=torch.log1p(p*torch.expm1(up-z))-torch.log1p(p*torch.expm1(down-z))
    change=change.scatter_add(-1,y[:,None],-(up-down).gather(1,y[:,None]))
    return change/.2


@compile_fn
def hidden_credit(pre,credit,wout):
    step=.05*pre.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    slope=(F.relu(pre.float()+step).square()-F.relu(pre.float()-step).square())/(2*step)
    read=torch.mm(credit.bfloat16(),wout.bfloat16(),out_dtype=torch.float32)
    return slope*read


@compile_fn
def schedule(step,total):
    warm=round(240*total/1194);cool=round(50*total/1194);engage=round(514*total/1194)
    beta=torch.where(step<warm,.85+.08*step/max(1,warm),.93)
    beta=torch.where(step>total-cool,.93-.08*(step-(total-cool))/max(1,cool),beta)
    fast=torch.where(step>=engage,.85,beta);weight=torch.where(step>=engage,.4385,1.)
    scale=((total-step)/(total*.4)).clamp(max=1.)
    lr=.04*scale
    return torch.stack((beta,1-fast,weight,1-weight,lr,2.25*lr)),scale


@compile_fn
def adam(params,grads,ms,vs,rates,scale,step):
    c1=1-.7**(step+1);c2=(1-.98**(step+1)).sqrt()
    for p,g,m,v,lr in zip(params,grads,ms,vs,rates):
        # PR29 casts the estimate to parameter dtype before updating fp32 moments.
        g=g.to(p.dtype).float()
        m.mul_(.7).add_(g,alpha=.3);v.mul_(.98).addcmul_(g,g,value=.02)
        p.copy_(p.float()-lr*scale*(m/c1)/(v.sqrt()/c2+1e-8))


class Trainer:
    def __init__(self,model,total=975,paired=False,stable=False,batch_heads=False):
        from utils.model import flex,window_mask
        self.model,self.total,self.paired,self.stable=model,total,paired,stable
        self.batch_heads=batch_heads
        self.flex,self.window_mask=flex,window_mask
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        if self.world not in (1,8):raise ValueError('Use one diagnostic GPU or eight record GPUs')
        self.device=next(model.parameters()).device
        self.gen=torch.Generator(self.device).manual_seed(17)
        self.step_index=torch.zeros((),device=self.device)
        self.debug={}
        h=torch.ones(1,1,device=self.device)
        while h.shape[0]<512:h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
        self.h=h
        self.cycle=compile_fn(cycle_derivatives)
        self.attention=compile_fn(local_credit)
        self.params=dict(model.named_parameters())
        self.matrix_groups={};self.other=[]
        for name,p in self.params.items():
            if name.startswith('transformer.h.') and p.ndim==2 and 've_gate' not in name:
                self.matrix_groups.setdefault(tuple(sorted(p.shape)),[]).append(name)
            else:self.other.append(name)
        self.states={};self.matrix_rates={}
        for shape,names in self.matrix_groups.items():
            if len(names)%self.world:raise ValueError('ANVIL groups must divide across GPUs')
            count=len(names)//self.world;owned=names[self.rank*count:(self.rank+1)*count]
            states=[]
            for name in owned:
                p=self.params[name];w=p.T if p.shape[0]>p.shape[1] else p
                axis=-1 if w.shape[0]==w.shape[1] else -2
                states.append((torch.zeros_like(w),torch.zeros_like(w),torch.zeros_like(w.mean(axis,keepdim=True))))
            self.states[shape]=states
            self.matrix_rates[shape]=torch.tensor([max(1,self.params[n].shape[0]/self.params[n].shape[1])**.5 for n in owned],device=self.device)[:,None,None]
        self.ms=[torch.zeros_like(self.params[n],dtype=torch.float32) for n in self.other]
        self.vs=[torch.zeros_like(m) for m in self.ms]
        self.rates=[.9 if ('wte' in n or 'value_embeds' in n) else .003 for n in self.other]
        self.rope={};self.masks={}
        for t in (1024,2048):
            angle=torch.arange(t,device=self.device)[:,None]/10000**(torch.arange(0,64,2,device=self.device)/64)
            self.rope[t]=(angle.cos()[None,:,None,:],angle.sin()[None,:,None,:])
            for window in (1024,2048):self.masks[t,window]=window_mask(window,t,self.device)

    def state_tensors(self):
        return [*self.params.values(),self.step_index,*self.ms,*self.vs,
                *(x for states in self.states.values() for triple in states for x in triple)]

    def clean(self,x,y):
        b,t=x.shape;model=self.model;cos,sin=self.rope[t]
        x0=norm(model.transformer.wte.weight[x]);stream=x0;cache=[]
        for i,block in enumerate(model.transformer.h):
            ve=model.value_embeds[str(i)].weight[x] if str(i) in model.value_embeds else None
            gate=block.attn.ve_gate.weight if block.attn.ve_gate is not None else None
            mixed,n,qr,kr,vr,z,values,q,k=before_attention(stream,x0,model.resid_lambdas[i],model.x0_lambdas[i],
                block.attn.c_q.weight,block.attn.c_k.weight,block.attn.c_v.weight,gate,ve,cos,sin)
            window=model.window_sizes[i][0]
            o,lse=self.flex(q.transpose(1,2),k.transpose(1,2),values.transpose(1,2),block_mask=self.masks[t,window],return_lse=True)
            o=o.transpose(1,2).contiguous().view(b,t,512)
            prev=stream
            stream,n2,pre,act=after_attention(mixed,o,block.attn.c_proj.weight,block.mlp.c_fc.weight,block.mlp.c_proj.weight)
            cache.append(dict(n=n,qr=qr,kr=kr,vr=vr,z=z,ve=ve,values=values,o=o,lse=lse.transpose(1,2),
                              n2=n2,pre=pre,act=act,prev=prev,out=stream))
        final,raw,loss=head_pass(stream,model.lm_head.weight,y)
        return cache,x0,final,raw,loss

    def estimate(self,x,y):
        b,t=x.shape;world=self.world;rank=self.rank;model=self.model
        if b>=world:
            rows=b//world;xr=x[rank*rows:(rank+1)*rows];yr=y[rank*rows:(rank+1)*rows]
            mine=lambda a:a.reshape(-1,*a.shape[2:])
        else:
            per=world//b;part=t//per;xr=x[rank//per:rank//per+1];yr=y[rank//per:rank//per+1]
            mine=lambda a:a.reshape(t,*a.shape[2:])[rank%per*part:(rank%per+1)*part]
        cache,x0,final,raw,loss=self.clean(xr,yr)
        tokens=mine(xr).flatten();targets=mine(yr).flatten();n=t*b
        grads={};deferred={}
        def put(name,inp,error):
            if name.startswith('value_embeds') or name=='transformer.wte.weight':
                grads[name]=torch.zeros_like(self.params[name],dtype=torch.float32).index_add_(0,tokens,error)
            else:deferred.setdefault((error.shape[-1],inp.shape[-1]),[]).append((name,inp.float(),error.float()))
        grads['lm_head.weight']=head_credit(mine(raw),targets).T@mine(final)
        top=mine(cache[-1]['out']).float();h=self.h*(torch.randint(0,2,(512,),device=self.device,generator=self.gen).float()*2-1)
        head=model.lm_head.weight.float();c=h@head.T
        steps=.005*top.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        deriv=self.cycle(top@head.T,c,targets,steps,top.square().sum(-1,keepdim=True),top@h.T,h.square().sum(-1),512,self.stable)
        credit=deriv@h/512
        self.debug={'top_credit':credit}
        credits=[None]*len(cache)
        for i in reversed(range(len(cache))):
            credits[i]=credit
            if i:credit=credit*model.resid_lambdas[i]
        targets_o=[];packed=[]
        for i,(block,ca,credit) in enumerate(zip(model.transformer.h,cache,credits)):
            prefix=f'transformer.h.{i}.'
            put(prefix+'mlp.c_proj.weight',mine(ca['act']),credit)
            put(prefix+'mlp.c_fc.weight',mine(ca['n2']),hidden_credit(mine(ca['pre']),credit,block.mlp.c_proj.weight))
            put(prefix+'attn.c_proj.weight',mine(ca['o']),credit)
            hub=(credit@block.attn.c_proj.weight.float()).bfloat16().view(-1,8,64);targets_o.append(hub)
            # Reconstruct in fp32 then cast, as PR29's gathered_heads does.
            values=ca['vr'].float()
            if ca['ve'] is not None:values=values+2*ca['z'].float().sigmoid()[...,None]*ca['ve'].float().view_as(values)
            parts=[mine(ca[k]).reshape(-1,8,64).bfloat16() for k in ('qr','kr')]
            parts += [mine(values).reshape(-1,8,64).bfloat16(),mine(ca['o']).reshape(-1,8,64).bfloat16(),hub]
            parts += [mine(ca['lse']).contiguous().float().view(torch.bfloat16).reshape(-1,8,2)]
            packed.append(torch.cat(parts,-1))
        packed=torch.stack(packed) # [L,m,H,322]
        if world>1:
            send=packed.permute(2,0,1,3).contiguous();recv=torch.empty_like(send)
            dist.all_to_all_single(recv,send)
            whole=recv.permute(1,0,2,3).reshape(len(cache),b,t,322)
        else:
            whole=packed.reshape(len(cache),b,t,8,322).permute(0,1,3,2,4).reshape(len(cache),b*8,t,322)
        head_credits=[None]*len(cache);cos,sin=[z.reshape(t,32) for z in self.rope[t]]
        grouped={}
        for i in range(len(cache)):
            window=min(t,model.window_sizes[i][0])
            grouped.setdefault(window if self.batch_heads else (window,i),[]).append(i)
        for group in grouped.values():
            combined=torch.stack([whole[i] for i in group]).flatten(0,1)
            parts=[combined[...,j*64:(j+1)*64].float().contiguous() for j in range(5)]
            lse=combined[...,320:].contiguous().view(torch.float32).squeeze(-1)
            qr,kr,v,o,hub=parts
            self.debug.update({f'head_{group[0]}_lse':lse,f'head_{group[0]}_q':qr,f'head_{group[0]}_hub':hub})
            q,k,vcredit=self.attention(qr,kr,v,hub,cos,sin,min(t,model.window_sizes[group[0]][0]),16,64,lse,o,self.paired)
            parts=torch.stack((q,k,vcredit)).reshape(3,len(group),-1,t,64)
            for j,i in enumerate(group):head_credits[i]=parts[:,j]
        by_head=torch.stack(head_credits)
        if world>1:
            send=by_head.reshape(len(cache),3,world,-1,64).permute(2,0,1,3,4).contiguous();recv=torch.empty_like(send)
            dist.all_to_all_single(recv,send)
            by_token=recv.permute(1,2,3,0,4).reshape(len(cache),3,-1,512)
        else:
            by_token=by_head.reshape(len(cache),3,b,8,t,64).permute(0,1,2,4,3,5).reshape(len(cache),3,-1,512)
        scalars=[]
        for i,(ca,credit) in enumerate(zip(cache,credits)):
            prefix=f'transformer.h.{i}.'
            for j,kind in enumerate('qkv'):put(prefix+f'attn.c_{kind}.weight',mine(ca['n']),by_token[i,j])
            if ca['ve'] is not None:
                z=mine(ca['z']).float();ve=mine(ca['ve']).float().reshape(-1,8,64);vc=by_token[i,2].reshape(-1,8,64)
                slope=((z+.1).sigmoid()-(z-.1).sigmoid())/.1
                grads[prefix+'attn.ve_gate.weight']=(slope*(vc*ve).sum(-1)).T@mine(ca['n'])[...,:32]
                put(f'value_embeds.{i}.weight',None,(vc*2*z.sigmoid()[...,None]).flatten(1))
            scalars.append(torch.stack(((credit*mine(ca['prev'])).sum(),(credit*mine(x0)).sum())))
        # PR29's identity-path embedding update, without an additional norm step.
        emb=model.resid_lambdas[0]*credits[0]+sum(model.x0_lambdas[i]*cr for i,cr in enumerate(credits))
        put('transformer.wte.weight',None,emb)
        for group in deferred.values():
            xs=torch.stack([row[1] for row in group]);es=torch.stack([row[2] for row in group])
            gs=es.transpose(-1,-2)@xs
            for (name,_,_),g in zip(group,gs):grads[name]=g
        grads['resid_lambdas'],grads['x0_lambdas']=torch.stack(scalars,1)
        return {name:g/n for name,g in grads.items()},mine(loss).mean()

    def update(self,grads):
        control,scale=schedule(self.step_index,self.total)
        for shape,names in self.matrix_groups.items():
            orient=lambda x:x.T if x.shape[0]>x.shape[1] else x
            count=len(names)//self.world;start=self.rank*count
            g=torch.stack([orient(grads[n]) for n in names]).contiguous()
            if self.world>1:
                own=torch.empty((count,*g.shape[1:]),device=self.device);dist.reduce_scatter_tensor(own,g)
            else:own=g
            ps=[orient(self.params[n]) for n in names[start:start+count]]
            states=self.states[shape]
            rate=self.matrix_rates[shape]
            batch_update(ps,list(own.unbind()),*[list(a) for a in zip(*states)],control,rate)
            if self.world>1:
                gathered=torch.empty_like(g);dist.all_gather_into_tensor(gathered,torch.stack(ps).contiguous())
                for j,name in enumerate(names):orient(self.params[name]).copy_(gathered[j])
        flat=torch.cat([grads[n].flatten() for n in self.other])
        if self.world>1:dist.all_reduce(flat)
        gs=[];offset=0
        for name in self.other:
            p=self.params[name];gs.append(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
        adam([self.params[n] for n in self.other],gs,self.ms,self.vs,self.rates,scale,self.step_index)
        self.step_index.add_(1)

    def step(self,x,y):
        grads,loss=self.estimate(x,y);self.update(grads)
        if self.world>1:dist.all_reduce(loss);loss/=self.world
        return loss
