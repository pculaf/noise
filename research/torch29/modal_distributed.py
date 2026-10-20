"""Budgeted remote profiling and implementation comparisons. No unattended retries."""
import json
from pathlib import Path
import sys
import modal

HERE=Path(__file__).resolve().parent
if modal.is_local():sys.path.insert(0,'/home/culaf/dev/zero-order')
from modal_orthogonal import image as base_image,volume
REVISION='98dbf6339c7768b2a6c2b9022aa25c80eb22a259'
ROOT='/workspace/noise'
app=modal.App('noise-torch29-distributed')
image=(base_image.run_commands(f'git -C {ROOT} fetch origin {REVISION}',f'git -C {ROOT} checkout --detach {REVISION}')
       .env({'TORCHINDUCTOR_COMPILE_THREADS':'2','OMP_NUM_THREADS':'1'}))
if modal.is_local():
    image=image.add_local_python_source('modal_orthogonal').add_local_dir(
        HERE,'/opt/torch29',ignore=['results','reference','__pycache__','budget.json'])

    image=image.add_local_dir(HERE.parents[1]/'train/speedrun','/opt/baseline',ignore=['log','__pycache__'])

@app.function(image=image,gpu='H100!:8',cpu=(16,16),memory=(65536,65536),timeout=360,startup_timeout=90,
              retries=0,max_containers=1,scaledown_window=2,volumes={'/results':volume})
def execute(attempt:str,worker:str,timeout:int):
    import datetime,hashlib,os,shutil,signal,subprocess,threading,time
    if '/' in attempt or not attempt.startswith('torch29-'):raise ValueError('Invalid attempt')
    if worker not in ('distributed',):raise ValueError('Unknown worker')
    if not 60<=timeout<=360:raise ValueError('Invalid timeout')
    started=time.monotonic();volume.reload()
    out=Path('/results/runs')/attempt;out.mkdir();shutil.copytree('/opt/torch29',out/'source')
    result=dict(attempt=attempt,worker=worker,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    process=None
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True,timeout=10)
        (out/'gpu.csv').write_text(gpu)
        if len(gpu.strip().splitlines())!=8 or gpu.count('H100')!=8:raise RuntimeError('Expected eight H100')
        (out/'run.json').write_text(json.dumps(result));volume.commit()
        print(json.dumps(dict(event='gpu_started',attempt=attempt,gpu=gpu.strip())),flush=True)
        with (out/'console.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',f'/opt/torch29/{worker}_worker.py','--output',str(out)],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True,
                env={**os.environ,'PYTHONUNBUFFERED':'1'})
            def relay():
                for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            thread=threading.Thread(target=relay,daemon=True);thread.start()
            code=process.wait(timeout=max(1,timeout-35-(time.monotonic()-started)));thread.join(timeout=5)
            if code:raise RuntimeError(f'Worker exited {code}')
        result['status']='completed'
    except Exception as e:result.update(status='failed',error=repr(e))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        result['function_seconds']=time.monotonic()-started
        result['estimated_function_cost_usd']=result['function_seconds']*RATE
        (out/'run.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    # Weights stay on the volume; do not download a 150 MB checkpoint for a diagnostic.
    return {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file() and p.suffix!='.pt'}


RATE=8*.001097+16*.0000131+64*.00000222


@app.local_entrypoint()
def main(attempt:str,worker:str='distributed',timeout:int=360):
    import time
    budget_path=HERE/'budget.json'
    budget=json.loads(budget_path.read_text()) if budget_path.exists() else {'cap_usd':9.,'attempts':[]}
    if any(r['attempt']==attempt for r in budget['attempts']):raise RuntimeError('Duplicate reservation')
    reserve=RATE*(timeout+90+5)+.03
    used=sum(r.get('conservative_charge_usd',r['reserved_usd']) for r in budget['attempts'])
    if used+reserve>budget['cap_usd']:raise RuntimeError('Campaign budget would be exceeded')
    entry={'attempt':attempt,'worker':worker,'reserved_usd':reserve,'status':'reserved'}
    budget['attempts'].append(entry);budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    out=HERE/'results'/attempt;out.mkdir(parents=True)
    call=execute.with_options(timeout=timeout).spawn(attempt,worker,timeout)
    (out/'receipt.json').write_text(json.dumps({'call_id':call.object_id,'reservation':entry},indent=2)+'\n')
    try:artifacts=call.get(timeout=timeout+300)
    except BaseException:
        call.cancel(terminate_containers=True);raise
    for name,data in artifacts.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    result=json.loads(artifacts['run.json'])
    budget=json.loads(budget_path.read_text())
    entry=next(row for row in budget['attempts'] if row['attempt']==attempt)
    entry.update(status=result['status'],function_seconds=result['function_seconds'],
        conservative_charge_usd=RATE*(result['function_seconds']+90+5)+.03)
    budget_path.write_text(json.dumps(budget,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='completed':raise RuntimeError(result.get('error'))
