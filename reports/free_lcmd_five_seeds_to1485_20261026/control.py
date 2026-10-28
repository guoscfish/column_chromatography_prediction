"""Operational locks and bounded supervision for the authorized continuation.

No scientific computation or selector policy is defined here.
"""
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT/'reports/free_llm_five_seeds_to1485_20261024'

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(tmp,path)

def once(path,value):
    if Path(path).exists():
        if read(path)!=value:raise RuntimeError('Frozen artifact drift: '+str(path))
    else:atomic(path,value)

@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

@contextmanager
def training_slot():
    chosen=None
    while chosen is None:
        try:
            with lock(POOL/'training_allocation.lock'):
                for i in range(3):
                    handle=(POOL/f'training_slot_{i}.lock').open('a+')
                    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:handle.close();continue
                    chosen=handle;break
        except BlockingIOError:pass
        if chosen is None:time.sleep(1)
    try:yield
    finally:fcntl.flock(chosen.fileno(),fcntl.LOCK_UN);chosen.close()

def event(out,seed,name,**fields):
    with (out/f'events_seed_{seed}.jsonl').open('a') as handle:
        handle.write(json.dumps({'time':now(),'seed':seed,'event':name,**fields})+'\n')

def supervise(entry,out,seed,complete):
    """Retain all previous failure counts; stop on scientific/credential errors."""
    with lock(out/f'supervisor_seed_{seed}.lock'):
        with lock(out/f'worker_seed_{seed}.lock'):pass
        path=out/f'supervisor_seed_{seed}.json'
        previous=read(path) if path.exists() else {}
        failures=previous.get('session_recovery_count',0)
        counts=previous.get('failures_per_round',{})
        historical=previous.get('recovery_count',failures)-failures
        while True:
            saved={'seed':seed,'time':now(),'recovery_count':historical+failures,
                   'session_recovery_count':failures,'failures_per_round':counts}
            if complete.exists():atomic(path,{'status':'COMPLETE',**saved});return
            log=out/f'worker_seed_{seed}.log'
            offset=log.stat().st_size if log.exists() else 0
            with log.open('ab') as handle:
                child=subprocess.Popen([sys.executable,'-u',str(entry),'--worker',str(seed)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(path,{'status':'RUNNING',**saved,'pid':child.pid,'supervisor_pid':os.getpid(),'log':str(log)})
            event(out,seed,'supervised_worker_started',pid=child.pid)
            phase=None;started=time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                state_path=out/f'status_seed_{seed}.json'
                if not state_path.exists():continue
                state=read(state_path);key=(state['completed_rounds'],state['phase'])
                if key!=phase:phase=key;started=time.monotonic()
                # monotonic time on this host excludes system sleep.
                if state['phase']=='SELECTING' and time.monotonic()-started>2400:
                    event(out,seed,'selection_watchdog_timeout',round=key[0])
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
            code=child.wait()
            if code==0 and complete.exists():continue
            state=read(out/f'status_seed_{seed}.json');r=str(state['completed_rounds'])
            failures+=1;counts[r]=counts.get(r,0)+1
            with log.open('rb') as handle:handle.seek(offset);body=handle.read().decode(errors='replace')
            fatal=code==78 or any(w in body for w in ('QUOTA_EXHAUSTED','DAILY_LIMIT_EXCEEDED',
                'USAGE_LIMIT_EXCEEDED','AuthenticationError','401 Unauthorized','403 Forbidden',
                'full-pool selection remained invalid','Frozen ','drift','integrity failure',
                'immutable dialog artifact changed','Provider model mismatch','AssertionError'))
            transient=code<0 or any(w in body for w in ('ConnectError','RemoteProtocolError','ReadError',
                'WriteError','ReadTimeout','WriteTimeout','APIConnectionError','APITimeoutError',
                'TimeoutException','429 Too Many Requests','500 Internal Server Error',
                '502 Bad Gateway','503 Service Unavailable','504 Gateway Timeout'))
            saved.update(time=now(),returncode=code,recovery_count=historical+failures,
                         session_recovery_count=failures,failures_per_round=counts,log=str(log))
            stop=fatal or not transient or counts[r]>2 or failures>12
            atomic(path,{'status':'NEEDS_ATTENTION' if stop else 'RETRYING',**saved})
            event(out,seed,'supervisor_stopped' if stop else 'automatic_transport_retry',round=int(r),returncode=code)
            if stop:return
            if counts[r]>1:time.sleep(30)

def launch(entry,out,seeds,complete,previous_out=None):
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'launch.lock'):
        # Process inventory plus kernel locks: never trust stale PID files alone.
        ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
        for seed in seeds:
            if any(str(entry) in line and any(f'{flag} {seed}' in line for flag in ('--worker','--supervise')) for line in ps.splitlines()):
                raise RuntimeError('Continuation process already exists')
            folders=[out]+([previous_out] if previous_out else [])
            for folder in folders:
                for name in ('supervisor','worker'):
                    with lock(folder/f'{name}_seed_{seed}.lock'):pass
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        rows=[]
        for seed in seeds:
            if complete(seed).exists():continue
            with (out/f'supervisor_seed_{seed}.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-u',str(entry),'--supervise',str(seed)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            subprocess.Popen(['caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            rows.append({'seed':seed,'supervisor_pid':child.pid})
        record={'time':now(),'entry':str(entry),'runner_sha256':sha(entry),'processes':rows}
        atomic(out/'launch_receipt.json',record)
        print(json.dumps(record),flush=True)
