"""Remote-only correctness and latency checks for the attributed PR29 port."""
import argparse,json,sys,time
from pathlib import Path
import torch
from moments import cycle_derivatives,attention_reference,rope32
from attention_moments import local_credit


def error(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    return dict(relative=float((a-b).norm()/a.norm().clamp_min(1e-20)),
                cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                max_abs=float((a-b).abs().max()))


def timed(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):graph.replay()
    b.record();b.synchronize()
    ms=a.elapsed_time(b)/20
    graph.reset()
    return ms


def exact_cycle(base,c,y,steps,sq,dots,ds,w):
    # Float64 explicit moved forward losses, chunked over directions.
    base,c,steps,sq,dots,ds=[v.double() for v in (base,c,steps,sq,dots,ds)]
    inv0=torch.rsqrt(sq/w+torch.finfo(torch.float32).eps)
    clean=(15*torch.tanh(base*inv0/15)).logsumexp(-1,keepdim=True)-15*torch.tanh(base.gather(1,y[:,None])*inv0/15)
    values=[]
    for j in range(0,c.shape[0],16):
        inv=torch.rsqrt((sq+2*steps*dots[:,j:j+16]+steps.square()*ds[j:j+16])/w+torch.finfo(torch.float32).eps)
        z=(base[:,None,:]+steps[:,:,None]*c[j:j+16])*inv[:,:,None]
        moved=(15*torch.tanh(z/15)).logsumexp(-1)-15*torch.tanh((base.gather(1,y[:,None])+steps*c[j:j+16].T[y])*inv/15)
        values.append((moved-clean)/steps)
    return torch.cat(values,-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(0)
    results={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'attention':[],'readout':[]}
    save=lambda:(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for t,window in [(64,32),(128,128)]:
        q,k,v,h=[torch.randn(1,t,64,device='cuda') for _ in range(4)]
        angle=torch.arange(t,device='cuda')[:,None]/10000**(torch.arange(0,64,2,device='cuda')/64)
        cos,sin=angle.cos(),angle.sin()
        exact=attention_reference(q,k,v,h,cos,sin,window)
        got=local_credit(q,k,v,h,cos,sin,window)
        err=[error(a,b) for a,b in zip(exact,got)]
        results['attention'].append(dict(t=t,window=window,errors=err));save()
        print(json.dumps(results['attention'][-1]),flush=True)
        if max(e['relative'] for e in err)>.04:raise RuntimeError('Moment attention mismatch')
    # Scale readout weights to cover both initialization and larger trained logits.
    w,vocab,n=512,4096,16
    had=torch.ones(1,1,device='cuda')
    while had.shape[0]<w:had=torch.cat((torch.cat((had,had),1),torch.cat((had,-had),1)),0)
    x=torch.randn(n,w,device='cuda');y=torch.randint(vocab,(n,),device='cuda')
    steps=.005*x.square().mean(-1,keepdim=True).sqrt();sq=x.square().sum(-1,keepdim=True);dots=x@had.T;ds=had.square().sum(-1)
    for scale in [.001,.03]:
        weight=torch.randn(w,vocab,device='cuda')*scale;base=x@weight;c=had@weight
        ref=exact_cycle(base,c,y,steps,sq,dots,ds,w)
        got=cycle_derivatives(base,c,y,steps,sq,dots,ds,w)
        row=dict(scale=scale,error=error(ref,got));results['readout'].append(row);save();print(json.dumps(row),flush=True)
        if row['error']['relative']>.02:raise RuntimeError('Readout expansion mismatch')
    compiled=torch.compile(cycle_derivatives,fullgraph=True,dynamic=False)
    for n in [256,1024]:
        x=torch.randn(n,w,device='cuda');y=torch.randint(vocab,(n,),device='cuda')
        args=(x@weight,c,y,.005*x.square().mean(-1,keepdim=True).sqrt(),x.square().sum(-1,keepdim=True),x@had.T,ds,w)
        ms=timed(lambda:compiled(*args));row=dict(tokens=n,compiled_readout_ms=ms)
        results.setdefault('timings',[]).append(row);save();print(json.dumps(row),flush=True)
    for t in [1024,2048]:
        q,k,v,h=[torch.randn(1,t,64,device='cuda') for _ in range(4)]
        angle=torch.arange(t,device='cuda')[:,None]/10000**(torch.arange(0,64,2,device='cuda')/64);cos,sin=angle.cos(),angle.sin()
        qp,kp=rope32(q,cos,sin).bfloat16().float(),rope32(k,cos,sin).bfloat16().float()
        lag=torch.arange(t,device='cuda')[:,None]-torch.arange(t,device='cuda')[None,:]
        logits=(qp@kp.transpose(-1,-2)/8).masked_fill(lag<0,-torch.inf)
        lse=logits.logsumexp(-1);o=logits.softmax(-1)@v.bfloat16().float()
        for chunk,block in [(16,32),(16,64),(32,32)]:
            call=torch.compile(lambda:local_credit(q,k,v,h,cos,sin,t,chunk,block,lse,o),fullgraph=True,dynamic=False)
            ms=timed(call);row=dict(tokens=t,chunk=chunk,block=block,attention_ms=ms)
            results['timings'].append(row);save();print(json.dumps(row),flush=True)
    results['status']='passed';save()


if __name__=='__main__':main()
