#!/usr/bin/env python3
"""Run the twelve remaining declared fits; seed157 has its own bounded worker."""
from pathlib import Path
import json,os,subprocess,sys,time
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
KINDS=['A_original','B_amount_skip','C_volume_skip']

def main():
    queue=[(s,k) for k in KINDS for s in [887,2357,6101,12203]];running={};logs={}
    (OUT/'worker_logs').mkdir(exist_ok=True)
    while queue or running:
        for key,p in list(running.items()):
            rc=p.poll()
            if rc is not None:
                logs[key].close();del running[key]
                if rc:raise RuntimeError(f'Fit worker failed without retry: {key}, exit={rc}')
        seed157_complete=all((OUT/f'fits/seed_157/{k}/fit_audit.json').exists() for k in KINDS)
        capacity=3 if seed157_complete else 2
        while queue and len(running)<capacity and psutil.virtual_memory().available>2.8e9:
            s,k=queue.pop(0);dest=OUT/f'fits/seed_{s}/{k}'
            if (dest/'fit_audit.json').exists():continue
            if (dest/'started.json').exists():raise RuntimeError(f'Refusing unresolved fit restart: {s} {k}')
            f=(OUT/f'worker_logs/seed_{s}_{k}.log').open('w');logs[(s,k)]=f
            p=subprocess.Popen([sys.executable,'-B',str(OUT/'train_controls.py'),'--seed',str(s),'--kind',k],cwd=ROOT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
            running[(s,k)]=p;print('LAUNCHED',s,k,'pid',p.pid,flush=True)
            time.sleep(3)
        status=dict(running=[dict(seed=s,kind=k,pid=p.pid) for (s,k),p in running.items()],queued=len(queue),
            completed_fits=len(list((OUT/'fits').glob('seed_*/*/fit_audit.json'))),started_fits=len(list((OUT/'fits').glob('seed_*/*/started.json'))),
            maximum_declared_fits=15,available_memory_gb=psutil.virtual_memory().available/1e9)
        (OUT/'training_status.json').write_text(json.dumps(status,indent=2))
        if queue or running:time.sleep(5)
    print('REMAINING_TWELVE_COMPLETE',flush=True)

if __name__=='__main__':main()
