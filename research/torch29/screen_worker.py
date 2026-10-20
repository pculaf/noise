"""Compare proposed execution improvements to the PR29 formulas, one GPU."""
import argparse,json
from pathlib import Path
import torch
from components_worker import error,timed,exact_cycle
from moments import cycle_derivatives,rope32
from attention_moments import local_credit
from readout_moments import cycle


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(1)
    torch._dynamo.config.recompile_limit=32
    result={'attention':[],'readout':[]};save=lambda:(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for t in [1024,2048]:
        # Four sequences, one head/GPU: PR29's late-stage ownership.
        q,k,v,h=[torch.randn(4,t,64,device='cuda') for _ in range(4)]
        angle=torch.arange(t,device='cuda')[:,None]/10000**(torch.arange(0,64,2,device='cuda')/64);cos,sin=angle.cos(),angle.sin()
        qp,kp=rope32(q,cos,sin).bfloat16().float(),rope32(k,cos,sin).bfloat16().float()
        lag=torch.arange(t,device='cuda')[:,None]-torch.arange(t,device='cuda')[None,:]
        logits=(qp@kp.transpose(-1,-2)/8).masked_fill(lag<0,-torch.inf)
        lse=logits.logsumexp(-1);o=logits.softmax(-1)@v.bfloat16().float()
        reference=torch.compile(local_credit,fullgraph=True,dynamic=False)(q,k,v,h,cos,sin,t,16,64,lse,o)
        for paired,chunk,block,early in [(False,16,64,False),(True,16,64,False),(True,16,64,True),(True,32,32,True)]:
            fn=torch.compile(lambda:local_credit(q,k,v,h,cos,sin,t,chunk,block,lse,o,paired,early),fullgraph=True,dynamic=False)
            got=fn();err=[error(a,b) for a,b in zip(reference,got)]
            if max(e['relative'] for e in err)>.002:raise RuntimeError(f'Paired moments mismatch {err}')
            row=dict(t=t,paired=paired,chunk=chunk,block=block,early=early,errors=err,ms=timed(fn))
            result['attention'].append(row);save();print(json.dumps(row),flush=True)
    w,vocab,n=512,4096,16
    had=torch.ones(1,1,device='cuda')
    while had.shape[0]<w:had=torch.cat((torch.cat((had,had),1),torch.cat((had,-had),1)),0)
    for scale in [.001,.03]:
        x=torch.randn(n,w,device='cuda');y=torch.randint(vocab,(n,),device='cuda');weight=torch.randn(w,vocab,device='cuda')*scale
        args=(x@weight,had@weight,y,.005*x.square().mean(-1,keepdim=True).sqrt(),x.square().sum(-1,keepdim=True),x@had.T,had.square().sum(-1),w)
        ref=exact_cycle(*args)
        for label,fn in [('ported',lambda:cycle_derivatives(*args)),('stable',lambda:cycle_derivatives(*args,stable=True)),('fused',lambda:cycle(*args))]:
            got=fn();err=error(ref,got);row=dict(scale=scale,variant=label,error=err)
            result['readout'].append(row);save();print(json.dumps(row),flush=True)
            if err['relative']>.02:raise RuntimeError('Readout mismatch')
    for n in [512,1024]:
        x=torch.randn(n,w,device='cuda');y=torch.randint(vocab,(n,),device='cuda')
        args=(x@weight,had@weight,y,.005*x.square().mean(-1,keepdim=True).sqrt(),x.square().sum(-1,keepdim=True),x@had.T,had.square().sum(-1),w)
        ref=torch.compile(cycle_derivatives,fullgraph=True,dynamic=False)
        row=dict(tokens=n,variant='PR29_products',ms=timed(lambda:ref(*args)));result['readout'].append(row);save();print(json.dumps(row),flush=True)
        for bm,bp,bv in [(16,32,128),(32,32,64),(16,64,128)]:
            row=dict(tokens=n,variant='fused',bm=bm,bp=bp,bv=bv,ms=timed(lambda:cycle(*args,bm=bm,bp=bp,bv=bv)))
            result['readout'].append(row);save();print(json.dumps(row),flush=True)
    result['status']='passed';save()


if __name__=='__main__':main()
