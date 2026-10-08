"""Hybrid-first scheduling only; frozen protocol and original runner unchanged."""
import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
import resume_813_free6101_authorized as authorized
from extend_authorized_to813 import *
s.run_selector = authorized.extra_selector

# Finish the already-running hybrid fit before changing the scheduler.
if '--handoff-pid' in sys.argv:
    oldpid=int(sys.argv[sys.argv.index('--handoff-pid')+1])
    target=s.STUDY/'runtime/seed_6101/cw16_llm16_scientist_v2/round_05/prediction_freeze.json'
    print(f'Waiting for PID {oldpid} to freeze hybrid6101 L493 before scheduler handoff',flush=True)
    expected='resume_813_free6101_authorized.py'
    while True:
        result=subprocess.run(['ps','-p',str(oldpid),'-o','command='],capture_output=True,text=True)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError('Previous worker exited before handoff; monitor must diagnose cause before restart')
        if expected not in result.stdout or 'python' not in result.stdout.lower():
            raise RuntimeError('Old PID identity changed; refusing to signal')
        if target.exists():
            f=read(target)
            for kind in ('checkpoint','prediction'):
                assert s.sha256_file(Path(f[kind+'_path']))==f[kind+'_sha256']
            os.kill(oldpid,signal.SIGTERM)
            for _ in range(100):
                if subprocess.run(['ps','-p',str(oldpid)],stdout=subprocess.DEVNULL).returncode:break
                time.sleep(.1)
            else:raise RuntimeError('Previous worker did not exit; refusing concurrent run')
            print('Verified L493 freeze; old worker exited; starting hybrid-first schedule',flush=True)
            break
        time.sleep(.1)

if __name__=='__main__':
    with s.exclusive_lock(s.STUDY/'execution.lock'):
        manifest,random=preflight()
        if '--check' in sys.argv:
            print(json.dumps({'status':'PREFLIGHT_PASS','targets':manifest['targets'],'budgets':manifest['budgets'],'random':random['status']}),flush=True)
            sys.exit(0)
        EXT.mkdir(exist_ok=True);once(EXT/'manifest.json',manifest);once(EXT/'random157_reuse.json',random)
        once(EXT/'hybrid_first_schedule.json',{'authorization':'User requested both hybrid trajectories finish before Free on 2026-10-07','order':[list(t[:2]) for t in TARGETS],'runner_sha256':s.sha256_file(Path(__file__)),'scientific_settings_changed':False})
        s._progress({'action':'extension813_started','pid':os.getpid(),'schedule':'hybrid_first'})
        try:
            # User-authorized order: finish both hybrid trajectories before Free.
            for seed,method,start in TARGETS:
                for r in range(start,15):
                    d=selection_directory(seed,method,r)
                    pred=s.STUDY/f'runtime/seed_{seed}/{method}/round_{r+1:02d}/prediction_freeze.json'
                    details={'seed':seed,'method':method,'round':r,'extension813':True}
                    if pred.exists() and (d/'feedback.json').exists():
                        s.state(s.Context(seed),method,r+1);continue
                    s._progress({'action':'extension813_cycle_started',**details})
                    with s.status_phase(f'813 extension seed={seed} {method} {s.BUDGETS[r]}->{s.BUDGETS[r+1]} select',30,**details):
                        s.select(seed,method,r)
                    with s.status_phase(f'813 extension seed={seed} {method} train {s.BUDGETS[r+1]}',30,**details):
                        s.advance(seed,method,r)
                    s._progress({'action':'extension813_cycle_completed',**details})
            for seed,method,_ in TARGETS: finish(seed,method)
            for name,h in manifest['prior_freezes'].items():
                assert s.sha256_file(s.STUDY/name)==h
                s.verify_files(s.STUDY,read(s.STUDY/name)['files'])
            once(EXT/'result.json',{'status':'ALL_AUTHORIZED_TRAJECTORIES_COMPLETE','final_active_labels':813,'test_evaluation_performed':False})
            s._progress({'action':'extension813_completed'})
        except Exception as e:
            s._progress({'action':'extension813_stopped','error_type':type(e).__name__,'message':str(e)[:1000]})
            raise
