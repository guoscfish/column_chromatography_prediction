"""Restore both Free trajectories to813: one Free API request, independent training."""
import sys,os,time,subprocess,json
from pathlib import Path
import free_scoped_request_gate as gate
runner=gate.runner
base=runner.base
s=runner.s

def work(seed):
    gate.activate(seed)
    return runner.worker(seed)

def main():
    if '--seed' in sys.argv:return work(int(sys.argv[sys.argv.index('--seed')+1]))
    with s.exclusive_lock(base.OLD/'execution.lock'):
        gate.activate(157)
        record={'authorization':'User2026-10-13 resumed both Free seeds through813, including seed157training; prior pause superseded',
            'target_labels':813,'seeds':[157,6101],'concurrent_Free_requests':1,'training_processes':2,
            'request_lock':str(base.NEW/'free_only_request.execution.lock'),
            'model':'gpt-6.1-sol','reasoning_effort':'xhigh','scientific_settings_changed':False,
            'transition_manifest_sha256':s.sha256_file(base.NEW/'transition_manifest.json'),
            'entry_sha256':s.sha256_file(Path(__file__)),'request_gate_sha256':s.sha256_file(Path(gate.__file__)),
            'worker_sha256':s.sha256_file(Path(runner.__file__))}
        if '--check' in sys.argv:print('SCOPED_BOTH813_PREFLIGHT_PASS',flush=True);return 0
        base.original.once(base.NEW/'scoped_resume_schedule_20261013.json',record)
        children=[]
        try:
            for seed in (157,6101):
                log=base.NEW/f'scoped_resume_seed{seed}_{time.strftime("%Y%m%d_%H%M%S")}.log'
                with log.open('ab') as out:
                    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--seed',str(seed)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                children.append((seed,p));s._progress({'action':'scoped_resume_worker_started','seed':seed,'pid':p.pid,'log':str(log)})
            while True:
                errors=[(seed,p.poll()) for seed,p in children if p.poll() not in (None,0)]
                if any(code==78 for _,code in errors):raise RuntimeError(f'QUOTA_EXHAUSTED {errors}')
                if all(p.poll() is not None for _,p in children):
                    if errors:raise RuntimeError(f'Incomplete seed: {errors}')
                    break
                time.sleep(1)
            for seed in (157,6101):
                result=base.original.read(base.NEW/f'{seed}_result.json')
                assert result['final_labels']==813 and len(set(result['selected_ids']))==480
            s.verify_files(base.NEW,base.original.read(base.NEW/'transition_manifest.json')['inherited_files'])
            base.original.once(base.NEW/'result.json',{'status':'BOTH_FREE_COMPLETE813','test_evaluation_performed':False})
            s._progress({'action':'sol_free813_completed'});return 0
        finally:
            for _,p in children:
                if p.poll() is None:p.terminate()
            for _,p in children:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':sys.exit(main())
