"""Bounded eight-GPU PR29-port verification and one fixed seed-0 run."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from train_smoke_worker import load_model
from torch_train import Trainer


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=975)
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch._dynamo.config.recompile_limit=64
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    m,model=load_model();trainer=Trainer(model,total=975,paired=True,stable=True)
    states=trainer.state_tensors();snapshot=[v.clone() for v in states];rng=trainer.gen.get_state()
    def restore():
        for v,s in zip(states,snapshot):v.copy_(s)
        trainer.gen.set_state(rng)
    def log(row):
        if rank==0:print(json.dumps(row),flush=True)
    # Dummy-only warmup/capture; always restore the seed-0 model and all moments.
    graphs={};controls={};losses={};compile_start=time.monotonic()
    for t in (1024,2048):
        b=4*2048//t
        x=torch.randint(0,4096,(15,b,t),device='cuda');y=torch.randint(0,4096,(15,b,t),device='cuda')
        for warm in range(2):
            grads,loss=trainer.estimate(x[0],y[0])
            bad=[name for name,g in grads.items() if not bool(torch.isfinite(g).all())]
            debug={name:dict(finite=bool(torch.isfinite(v).all()),min=float(v.min()),max=float(v.max())) for name,v in trainer.debug.items()}
            (out/f'diagnostic-rank{rank}-t{t}-warm{warm}.json').write_text(json.dumps(dict(loss=loss.item(),bad=bad,debug=debug),indent=2)+'\n')
            if bad:raise FloatingPointError(f'Nonfinite warmup gradients on rank {rank}: {bad}')
            trainer.update(grads)
            bad_state=[name for name,p in trainer.params.items() if not bool(torch.isfinite(p).all())]
            if bad_state:raise FloatingPointError(f'Nonfinite updated weights on rank {rank}: {bad_state}')
            log(dict(event='warmup',length=t,step=warm,loss=loss.item()))
        torch.cuda.synchronize();restore();dist.barrier()
        graph=torch.cuda.CUDAGraph();graph.register_generator_state(trainer.gen)
        with torch.cuda.graph(graph):
            loss=torch.stack([trainer.step(x[j],y[j]) for j in range(15)])
        torch.cuda.synchronize();restore();graphs[t]=graph;controls[t]=(x,y);losses[t]=loss
        log(dict(event='captured',length=t,seconds=time.monotonic()-compile_start))
    # Matched numerical check: optimized paired moments vs unpaired moments on
    # a fixed real batch. No optimizer update; restore the shared draw RNG.
    rows=m.load_rows(Path('/results/data/train.pt'),m.DATA_ROWS)
    order=np.random.default_rng((1,0)).permutation(len(rows))
    idx=np.stack([np.sort(order[j:j+4]) for j in range(0,4*975,4)])
    batches=rows[torch.from_numpy(idx)].cuda()
    bx=batches[0,:,:-1].contiguous().reshape(8,1024);by=batches[0,:,1:].contiguous().reshape(8,1024)
    restore();optimized,_=trainer.estimate(bx,by)
    trainer.paired=False;trainer.gen.set_state(rng);reference,_=trainer.estimate(bx,by);trainer.paired=True
    diffs={name:float((optimized[name]-reference[name]).norm()/reference[name].norm().clamp_min(1e-12)) for name in optimized}
    bad=[name for name,g in optimized.items() if not bool(torch.isfinite(g).all())]
    if bad:raise FloatingPointError(f'Nonfinite pre-training estimates on rank {rank}: {bad}')
    maximum=max(diffs.values());check=torch.tensor(maximum,device='cuda');dist.all_reduce(check,op=dist.ReduceOp.MAX)
    if check.item()>.002:raise RuntimeError(f'Paired-moment estimator mismatch {check.item()}')
    log(dict(event='paired_estimator_check',max_relative=check.item()))
    del optimized,reference
    restore();dist.barrier();torch.cuda.synchronize()
    if rank==0:
        (out/'config.json').write_text(json.dumps(dict(source_pr=29,source_revision='866615d3a0332219b3d62ef8eaadc9c9d905d2fe',
            seed=0,initialization='benchmark PyTorch seed 0 (not bitwise JAX)',batch=4,steps=args.steps,schedule=975,
            short_context_until=780,paired_moments=True,stable_readout=True,steps_per_call=15),indent=2)+'\n')
    records=[];start=time.perf_counter()
    for first in range(0,args.steps,15):
        if first+15>args.steps:raise ValueError('Steps must divide 15')
        t=1024 if first<780 else 2048
        x,y=controls[t];batch=batches[first:first+15]
        x.copy_(batch[:,:,:-1].reshape_as(x));y.copy_(batch[:,:,1:].reshape_as(y))
        graphs[t].replay()
        values=losses[t].tolist()
        if not all(math.isfinite(v) for v in values):
            log(dict(event='nonfinite',first=first,losses=values))
            raise FloatingPointError('Nonfinite training loss')
        elapsed=time.perf_counter()-start
        records.extend(dict(step=first+j+1,train_loss=v,train_seconds=elapsed) for j,v in enumerate(values))
        if first%150==0:log(dict(event='progress',step=first+15,loss=values[-1],seconds=elapsed))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    for graph in graphs.values():graph.reset()
    torch.cuda.synchronize()
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode());digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    hashes=[None]*dist.get_world_size();dist.all_gather_object(hashes,digest.hexdigest())
    if len(set(hashes))!=1:raise RuntimeError('Ranks disagree on final weights')
    result=dict(train_seconds=elapsed,steps=args.steps,tokens=args.steps*4*2048,rank_hashes=hashes,paired_estimator_relative=maximum)
    if rank==0:
        torch.save(state,out/'final.pt');(out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        log(dict(event='training_complete',**result))
    # Benchmark evaluator uses its own model, unchanged.
    val=m.evaluate(state,Path('/results/data'))
    result.update(val_loss=float(val),reached_target=val<=4.2)
    if rank==0:(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');log(result)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
