"""One-GPU isolation of warmup, finite gradients, and multi-update capture."""
import argparse,json,math,time
from pathlib import Path
import torch
from train_smoke_worker import load_model
from torch_train import Trainer
from precise_mm import mm_transposed


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch._dynamo.config.recompile_limit=64
    m,model=load_model();trainer=Trainer(model,paired=True,stable=True)
    states=trainer.state_tensors();snapshot=[v.clone() for v in states];rng=trainer.gen.get_state()
    rows=m.load_rows(Path('/results/data/train.pt'),m.DATA_ROWS)
    result=[]
    def log(row):
        result.append(row);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
    def restore():
        for v,s in zip(states,snapshot):v.copy_(s)
        trainer.gen.set_state(rng)
    a=torch.randn(128,4096,device='cuda');b=torch.randn(512,4096,device='cuda')
    got=torch.compile(mm_transposed,fullgraph=True)(a,b)
    torch.set_float32_matmul_precision('highest');ref=a@b.T;torch.set_float32_matmul_precision('high')
    log(dict(test='precise_mm',relative=float((got-ref).norm()/ref.norm()),finite=bool(torch.isfinite(got).all())))
    x=rows[:1,:1024].contiguous().cuda();y=rows[:1,1:1025].contiguous().cuda()
    for i in range(3):
        grads,loss=trainer.estimate(x,y)
        bad=[n for n,g in grads.items() if not torch.isfinite(g).all().item()]
        log(dict(test='eager',step=i,loss=loss.item(),bad=bad,max_norm=max(g.norm().item() for g in grads.values())))
        if bad:raise FloatingPointError(bad)
        trainer.update(grads)
    restore()
    graph=torch.cuda.CUDAGraph();graph.register_generator_state(trainer.gen)
    with torch.cuda.graph(graph):loss=torch.stack([trainer.step(x,y) for _ in range(15)])
    torch.cuda.synchronize();restore();graph.replay();torch.cuda.synchronize()
    values=loss.tolist();log(dict(test='graph15',losses=values,finite=all(math.isfinite(v) for v in values)))
    graph.reset()
    if not all(math.isfinite(v) for v in values):raise FloatingPointError('Graph replay failed')
    restore()
    values=[]
    for _ in range(15):values.append(trainer.step(x,y).item())
    log(dict(test='eager15',losses=values,max_gap=max(abs(a-b) for a,b in zip(values,loss.tolist()))))


if __name__=='__main__':main()
