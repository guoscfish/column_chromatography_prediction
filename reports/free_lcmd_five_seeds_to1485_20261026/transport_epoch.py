"""User-authorized two-launch transport recovery epoch; never reset history."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import control as c
import free_continue as free

ENTRY=Path(__file__).resolve()
OUT=ENTRY.parent/'transport_epoch_20261026_2338'
MANIFEST=OUT/'authorization.json'
CUTS={887:33,2357:35,12203:24}
SEEDS=tuple(CUTS)

def source_out(seed):return free.OLD if seed==2357 else free.OUT

def complete(seed):
    study=free.OLD_STUDY if seed==2357 else free.STUDY
    return study/f'{seed}_{free.m.FREE}_result.json'

def validate():
    record=c.read(MANIFEST)
    assert record['authorization_answer']=='授权各再恢复最多2次'
    for path,digest in record['protected_files'].items():
        if c.sha(c.ROOT/path)!=digest:raise RuntimeError('Frozen recovery epoch source drift: '+path)
    return record

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():validate();print('ALREADY_PREPARED');return
    free.old_validate()
    rows={}
    for seed,r in CUTS.items():
        sup=c.read(source_out(seed)/f'supervisor_seed_{seed}.json')
        state=c.read(source_out(seed)/f'status_seed_{seed}.json')
        assert sup['status']=='NEEDS_ATTENTION' and state['phase']=='STOPPED' and state['completed_rounds']==r
        assert sup['session_recovery_count']<12
        rows[str(seed)]={'baseline_supervisor':sup,'baseline_status':state,'source_out':str(source_out(seed)),
            'current_round':r,'maximum_additional_worker_launches_at_current_round':2}
    for name in ('connectivity_latest.json','transport_recovery_proposal.json'):
        c.once(OUT/name,c.read(ENTRY.parent/name))
    files=[ENTRY,Path(c.__file__),Path(free.__file__),free.MANIFEST,
           free.OLD/'continue_to1485.py',OUT/'connectivity_latest.json',
           OUT/'transport_recovery_proposal.json']
    record={'time':c.now(),'authorization_date':'2026-10-26',
        'authorization_answer':'授权各再恢复最多2次',
        'authorization_question':'Allow at most two additional transport recoveries per seed at the listed current round, without changing logical budgets/model/prompt/training?',
        'seed_limits':rows,'logical_budget_changed':False,'training_changed':False,'model_changed':False,
        'session_failure_limit_unchanged':12,'all_historical_failure_counts_retained':True,
        'protected_files':{str(p.relative_to(c.ROOT)):c.sha(p) for p in files}}
    c.once(MANIFEST,record)
    c.atomic(ENTRY.parent/'transport_epoch_latest.json',{'epoch':str(OUT),'entry':str(ENTRY),
        'authorization':str(MANIFEST),'time':c.now(),'seeds':list(SEEDS)})
    print({'status':'PREPARED','epoch':str(OUT),'maximum_additional_launches_at_current_round':2},flush=True)

def work(seed):
    validate()
    if seed in free.CUTS:free.activate(seed)
    # In a fresh 2357 process the imported m still has its original frozen globals.
    free.m.work(seed)

def supervise(seed):
    authorization=validate()['seed_limits'][str(seed)]
    baseline=authorization['baseline_supervisor']
    source=source_out(seed)
    state_path=OUT/f'supervisor_seed_{seed}.json'
    with c.lock(OUT/f'supervisor_seed_{seed}.lock'),c.lock(source/f'supervisor_seed_{seed}.lock'):
        with c.lock(source/f'worker_seed_{seed}.lock'):pass
        previous=c.read(state_path) if state_path.exists() else baseline
        failures=previous['session_recovery_count'];counts=previous['failures_per_round']
        historical=previous['recovery_count']-failures
        launches=previous.get('authorized_current_round_launches',0)
        while True:
            saved={'seed':seed,'time':c.now(),'recovery_count':historical+failures,
                'session_recovery_count':failures,'failures_per_round':counts,
                'authorized_current_round_launches':launches,'authorization':str(MANIFEST),
                'baseline_recovery_count':baseline['recovery_count'],'source_status':str(source/f'status_seed_{seed}.json')}
            if complete(seed).exists():c.atomic(state_path,{'status':'COMPLETE',**saved});return
            state=c.read(source/f'status_seed_{seed}.json');r=state['completed_rounds']
            if r==CUTS[seed]:
                if launches>=2:
                    c.atomic(state_path,{'status':'NEEDS_ATTENTION','reason':'Authorized current-round transport recovery limit exhausted',**saved});return
                launches+=1;saved['authorized_current_round_launches']=launches
            log=OUT/f'worker_seed_{seed}.log';offset=log.stat().st_size if log.exists() else 0
            with log.open('ab') as handle:
                child=subprocess.Popen([sys.executable,'-u',str(ENTRY),'--worker',str(seed)],cwd=c.ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            saved.update(pid=child.pid,supervisor_pid=os.getpid(),log=str(log))
            c.atomic(state_path,{'status':'RUNNING',**saved})
            c.event(OUT,seed,'authorized_transport_worker_started',pid=child.pid,round=r,current_round_launches=launches)
            phase=None;started=time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                state=c.read(source/f'status_seed_{seed}.json');key=(state['completed_rounds'],state['phase'])
                c.atomic(OUT/f'status_seed_{seed}.json',state)
                if key!=phase:phase=key;started=time.monotonic()
                if state['phase']=='SELECTING' and time.monotonic()-started>2400:
                    c.event(OUT,seed,'selection_watchdog_timeout',round=key[0])
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
            code=child.wait()
            if code==0 and complete(seed).exists():continue
            state=c.read(source/f'status_seed_{seed}.json');r=str(state['completed_rounds'])
            failures+=1;counts[r]=counts.get(r,0)+1
            with log.open('rb') as handle:handle.seek(offset);body=handle.read().decode(errors='replace')
            fatal=code==78 or any(word in body for word in ('QUOTA_EXHAUSTED','DAILY_LIMIT_EXCEEDED',
                'USAGE_LIMIT_EXCEEDED','AuthenticationError','401 Unauthorized','403 Forbidden',
                'full-pool selection remained invalid','Frozen ','drift','integrity failure',
                'immutable dialog artifact changed','Provider model mismatch','AssertionError'))
            transient=code<0 or any(word in body for word in ('ConnectError','RemoteProtocolError','ReadError',
                'WriteError','ReadTimeout','WriteTimeout','APIConnectionError','APITimeoutError','TimeoutException',
                '429 Too Many Requests','500 Internal Server Error','502 Bad Gateway','503 Service Unavailable','504 Gateway Timeout'))
            limit=(launches>=2) if int(r)==CUTS[seed] else counts[r]>2
            stop=fatal or not transient or limit or failures>12
            saved.update(time=c.now(),returncode=code,recovery_count=historical+failures,
                session_recovery_count=failures,failures_per_round=counts,authorized_current_round_launches=launches)
            c.atomic(state_path,{'status':'NEEDS_ATTENTION' if stop else 'RETRYING',**saved})
            c.event(OUT,seed,'supervisor_stopped' if stop else 'authorized_transport_retry',round=int(r),returncode=code)
            if stop:return
            time.sleep(5)

def launch():
    validate()
    with c.lock(OUT/'launch.lock'):
        ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
        with ExitStack() as locks:
            for seed in SEEDS:
                if any(any(name in line for name in ('free_continue.py','continue_to1485.py','transport_epoch.py')) and
                    any(f'{flag} {seed}' in line for flag in ('--worker','--supervise')) for line in ps.splitlines()):
                    raise RuntimeError('Existing Free process; refusing duplicate')
                for folder in (OUT,source_out(seed)):
                    for role in ('supervisor','worker'):locks.enter_context(c.lock(folder/f'{role}_seed_{seed}.lock'))
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        rows=[]
        for seed in SEEDS:
            with (OUT/f'supervisor_seed_{seed}.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ENTRY),'--supervise',str(seed)],cwd=c.ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            subprocess.Popen(['caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            rows.append({'seed':seed,'pid':child.pid})
        c.atomic(OUT/'launch_receipt.json',{'time':c.now(),'processes':rows,'authorization':str(MANIFEST)})
        print({'status':'LAUNCHED','processes':rows},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--worker',type=int,choices=SEEDS);p.add_argument('--supervise',type=int,choices=SEEDS)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.launch:launch()
    elif a.worker is not None:work(a.worker)
    elif a.supervise is not None:supervise(a.supervise)
