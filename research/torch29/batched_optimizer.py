"""Batched ANVIL execution, following PR29's grouping by wide matrix shape.

Same #26 update algebra, state and rates; caller may shard groups across ranks.
Weight and state storage remain the optimizer's ordinary tensors, so graph
warmup restoration still restores all state. No persistent shadow weights.
"""
import torch

MAPS=((3.923798038567,-6.095026865488,3.905234618423),
      (3.278126713798,-3.328923386476,.989127286973),
      (3.505298394150,-5.137358782410,1.968325560615),
      (2.815058591845,-3.685181239622,1.417196497642),
      (2.245503932403,-2.443826979899,.963091710461),
      (2.256537145403,-2.166840097229,.929501253245))


@torch.compile(fullgraph=True,dynamic=False)
def batch_update(params,grads,fasts,slows,energies,control,rate_factors):
    w,g=torch.stack(params).float(),torch.stack(grads).float()
    fast,slow,energy=torch.stack(fasts),torch.stack(slows),torch.stack(energies)
    beta,fast_weight,weight,other_weight=control[:4]
    lr,decay_rate=control[4:6]
    fast=fast.lerp(g,fast_weight);slow=slow.lerp(g,.02)
    pulled=g.lerp(weight*fast+other_weight*slow,beta)
    norm=lambda x:x.square().sum((-2,-1),keepdim=True).sqrt()
    x=(pulled/(1.1*norm(pulled)+1e-6)).bfloat16()
    for a,b,c in MAPS:
        s=x@x.transpose(-1,-2)
        x=a*x+(b*s+c*(s@s))@x
    update=x.float()
    axis=-1 if w.shape[-1]==w.shape[-2] else -2
    energy=energy.lerp(update.square().mean(axis,keepdim=True),.1)
    gained=update*energy.clamp_min(1e-10).rsqrt()
    update=gained*(norm(update)/norm(gained).clamp_min(1e-10))
    new=w-lr*rate_factors*(update+(slow*w>=0)*w*decay_rate)
    for i in range(len(params)):
        params[i].copy_(new[i]);fasts[i].copy_(fast[i]);slows[i].copy_(slow[i]);energies[i].copy_(energy[i])


def groups(optimizer):
    if len(optimizer.param_groups)!=1:raise ValueError('Single ANVIL rate group required')
    out={}
    for p in optimizer.param_groups[0]['params']:
        tall=p.shape[0]>p.shape[1]
        shape=tuple(sorted(p.shape))
        out.setdefault(shape,[]).append((p,tall))
    return list(out.values())


def run_groups(optimizer,control):
    for group in groups(optimizer):
        orient=lambda x,tall:x.T if tall else x
        ps=[orient(p,t) for p,t in group]
        gs=[orient(p.grad,t) for p,t in group]
        states=[[orient(optimizer.state[p][kind],t) for p,t in group] for kind in ('fast','slow','energy')]
        # Graph capture records this creation; values depend only on shapes.
        rates=torch.tensor([max(1,p.shape[0]/p.shape[1])**.5 for p,_ in group],device=ps[0].device)[:,None,None]
        batch_update(ps,gs,*states,control,rates)
