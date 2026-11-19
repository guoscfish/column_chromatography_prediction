"""Read-only prerequisites and measured historical compute for the v2 study."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ORIGINAL=HERE.with_name('epig_prediction_oriented_20261117')
sys.path.insert(0,str(ORIGINAL))
import study_support as support

SEEDS=(157,887,2357,6101,12203)


def preflight():
    freeze=support.verify_freeze()
    checkpoints=[]
    costs=[]
    for seed in SEEDS:
        base=support.BASE/f'runtime/seed_{seed}'
        model=base/'lcmd/round_00/model'
        audit=support.read(model/'fit_audit.json')
        files={name:support.digest(model/name) for name in ('best.pt','predictions.csv.gz','history.csv','fit_audit.json')}
        if files['best.pt'] != audit['checkpoint_sha256'] or files['predictions.csv.gz'] != audit['prediction_sha256']:
            raise RuntimeError(f'seed {seed}: checkpoint/prediction hash mismatch')
        if audit['parameter_count']!=458952 or audit['model_variant']!='qgeognn_v2' or audit['train_rows']!=333 or audit['validation_rows']!=416:
            raise RuntimeError(f'seed {seed}: architecture or split drift')
        checkpoints.append({'seed':seed,'model_directory':str(model.relative_to(ROOT)),
                            'hashes':files,'parameter_count':audit['parameter_count'],
                            'initialization_hash':audit['initialization_hash']})
        for r in range(1,22):
            p=base/f'lcmd/round_{r:02d}/model/fit_audit.json'
            if p.exists():
                a=support.read(p)
                costs.append({'seed':seed,'round':r,'training_seconds':a['training_seconds'],
                              'epochs_run':a['epochs_run'],'budget':a['train_rows'],
                              'audit_path':str(p.relative_to(ROOT)),'sha256':support.digest(p)})
    locks=[]
    for p in [ORIGINAL/'runtime/formal.lock',HERE/'runtime/exploratory.lock']:
        rec={'path':str(p.relative_to(ROOT)),'exists':p.exists(),'held':False}
        if p.exists():
            with p.open() as handle:
                try:
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(handle,fcntl.LOCK_UN)
                except BlockingIOError:
                    rec['held']=True
        locks.append(rec)
    disk=shutil.disk_usage(ROOT)
    def command(args):
        try:
            return {'exit_code':0,'stdout':subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()}
        except (subprocess.CalledProcessError,OSError) as exc:
            return {'exit_code':getattr(exc,'returncode',None),'error':str(exc),
                    'stdout':getattr(exc,'output','')}
    hardware=command(['sysctl','-n','hw.memsize','hw.ncpu','hw.machine'])
    processes=command(['ps','-axo','pid,ppid,%cpu,%mem,etime,command'])
    if processes.get('exit_code')==0:
        processes['stdout']='\n'.join(line for line in processes['stdout'].splitlines() if any(s in line.lower() for s in ('python','qgeognn','epig','train','caffeinate','ssh','rsync')))
    vals=[r['training_seconds'] for r in costs]
    report={'schema':'epig-v2-resource-preflight','created_unix':time.time(),
            'original_freeze_verified':True,'original_freeze_commit':freeze['git_commit'],
            'current_commit':command(['git','rev-parse','HEAD']),
            'python':sys.executable,'platform':platform.platform(),
            'hardware':hardware,'disk_free_bytes':disk.free,'disk_total_bytes':disk.total,
            'initial_checkpoint_records':checkpoints,'initial_checkpoints_present':len(checkpoints),
            'historical_cost_reference':'same original training protocol; LCMD trajectories are descriptive cost evidence',
            'historical_downstream_fits':len(costs),'historical_training_seconds':sum(vals),
            'estimated_two_arm_seconds_at_historical_cost':sum(vals)*2,
            'historical_median_fit_seconds':statistics.median(vals),
            'historical_max_fit_seconds':max(vals),'historical_fit_costs':costs,
            'locks':locks,'process_inventory':processes,
            'labels_read':{'l0':0,'future_U':0,'validation':0,'test':0},
            'note':'No training or target CSV is read by this preflight. Process/hardware sandbox failures are explicit, not evidence of absent resources.'}
    report['dependency_ready']=len(checkpoints)==5 and not any(r['held'] for r in locks)
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'evidence/runner/resource_preflight.json')
    args=parser.parse_args()
    result=preflight()
    support.dump(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('historical_fit_costs','initial_checkpoint_records','process_inventory')},ensure_ascii=False,indent=2))
