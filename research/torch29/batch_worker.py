"""Same moments, batching independent layers with the same causal window."""
import argparse,json
from pathlib import Path
import torch
from components_worker import timed,error
from attention_moments import local_credit
from moments import rope32


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch._dynamo.config.recompile_limit=32;torch.manual_seed(7)
    result=[]
    for t,b in [(1024,8),(2048,4)]:
        q,k,v,h=[torch.randn(8,b,t,64,device='cuda') for _ in range(4)]
        angle=torch.arange(t,device='cuda')[:,None]/10000**(torch.arange(0,64,2,device='cuda')/64)
        cos,sin=angle.cos(),angle.sin();lses=[];outs=[]
        for i in range(8):
            qp,kp=rope32(q[i],cos,sin).bfloat16().float(),rope32(k[i],cos,sin).bfloat16().float()
            lag=torch.arange(t,device='cuda')[:,None]-torch.arange(t,device='cuda')[None,:]
            window=2048 if i%4==3 else 1024
            logits=(qp@kp.transpose(-1,-2)/8).masked_fill((lag<0)|(lag>window),-torch.inf)
            lses.append(logits.logsumexp(-1));outs.append(logits.softmax(-1)@v[i].bfloat16().float())
        lse,o=torch.stack(lses),torch.stack(outs)
        def sequence(paired):
            return torch.stack([torch.stack(local_credit(q[i],k[i],v[i],h[i],cos,sin,min(t,2048 if i%4==3 else 1024),16,64,lse[i],o[i],paired)) for i in range(8)])
        def grouped():
            parts=[None]*8
            groups=[list(range(8))] if t==1024 else [[0,1,2,4,5,6],[3,7]]
            for indices in groups:
                a=[x[indices].flatten(0,1) for x in (q,k,v,h)]
                output=local_credit(*a,cos,sin,min(t,2048 if indices[0]%4==3 else 1024),16,64,lse[indices].flatten(0,1),o[indices].flatten(0,1),True)
                output=torch.stack(output).reshape(3,len(indices),b,t,64)
                for j,i in enumerate(indices):parts[i]=output[:,j]
            return torch.stack(parts)
        ref=torch.compile(lambda:sequence(False),fullgraph=True,dynamic=False)
        pair=torch.compile(lambda:sequence(True),fullgraph=True,dynamic=False)
        batch=torch.compile(grouped,fullgraph=True,dynamic=False)
        reference=ref()
        for name,fn in [('ported_PR29',ref),('paired',pair),('paired_batched_layers',batch)]:
            err=error(reference,fn())
            # Batching changes the compiler's reduction layout. Require an error
            # far below BF16 quantization, rather than accidental bit identity.
            if err['relative']>1e-4:raise RuntimeError(err)
            row=dict(t=t,sequences=b,variant=name,error=err,ms=timed(fn));result.append(row)
            (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
