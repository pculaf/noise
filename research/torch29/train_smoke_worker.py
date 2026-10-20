"""Remote single-GPU smoke and graph test for the standalone PyTorch trainer."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import torch
from torch_train import Trainer


def load_model():
    sys.path[:0]=['/opt/baseline','/workspace/noise']
    spec=importlib.util.spec_from_file_location('baseline','/opt/baseline/run.py')
    m=importlib.util.module_from_spec(spec);sys.modules['baseline']=m;spec.loader.exec_module(m)
    cfg=m.GPTConfig(sequence_len=2048,n_layer=8,n_head=8,n_kv_head=8,n_embd=512)
    return m,m.make_model(torch.device('cuda'),0,cfg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    m,model=load_model();trainer=Trainer(model,paired=True,stable=True)
    rows=m.load_rows(Path('/results/data/train.pt'),m.DATA_ROWS)
    x,y=rows[:1,:1024].contiguous().cuda(),rows[:1,1:1025].contiguous().cuda()
    states=trainer.state_tensors();snapshot=[p.clone() for p in states];rng=trainer.gen.get_state()
    result={'warmup':[]};save=lambda:(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for i in range(2):
        start=time.monotonic();loss=trainer.step(x,y);torch.cuda.synchronize()
        row=dict(step=i,loss=loss.item(),seconds=time.monotonic()-start);result['warmup'].append(row);save();print(json.dumps(row),flush=True)
    graph=torch.cuda.CUDAGraph();graph.register_generator_state(trainer.gen)
    with torch.cuda.graph(graph):loss=trainer.step(x,y)
    torch.cuda.synchronize()
    for p,s in zip(states,snapshot):p.copy_(s)
    trainer.gen.set_state(rng)
    losses=[];start=time.monotonic()
    for i in range(10):
        batch=rows[i:i+1];x.copy_(batch[:,:1024]);y.copy_(batch[:,1:1025]);graph.replay();losses.append(loss.item())
    torch.cuda.synchronize();result.update(losses=losses,mean_step_ms=(time.monotonic()-start)*100,status='completed')
    if not all(math.isfinite(v) for v in losses):raise RuntimeError('Nonfinite loss')
    save();print(json.dumps(result),flush=True);graph.reset()


if __name__=='__main__':
    import math
    main()
