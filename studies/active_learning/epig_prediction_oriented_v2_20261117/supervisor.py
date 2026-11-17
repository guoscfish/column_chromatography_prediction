"""Durable local process supervisor for the frozen full study, no efficacy stops."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)


def refresh_report():
    with (HERE/'runtime/report.log').open('a') as output:
        done=subprocess.run([sys.executable,str(HERE/'report_study.py')],cwd=ROOT,
            stdout=output,stderr=subprocess.STDOUT,timeout=55)
    return done.returncode


def worker_lock_held():
    """Inspect the OS lock, not the stale PID/text inside its file."""
    path=HERE/'runtime/exploratory.lock'
    if not path.exists():return False
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(handle,fcntl.LOCK_UN)
    return False


def main():
    runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'supervisor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another v2 supervisor is active')
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        path=runtime/'supervisor_status.json'
        state=read(path) if path.exists() else {'schema':'epig-v2-supervisor','launches':[]}
        # The ledger independently enforces 231 total neural attempts, 3 per
        # key, finite auxiliary work and active-time limits across all resumes.
        while len(state['launches']) < 22:
            # If a supervisor died while its training child survived, adopt
            # monitoring without launching a competing worker or editing state.
            while worker_lock_held():
                state.update(status='MONITORING_EXISTING_WORKER',supervisor_pid=os.getpid(),
                             last_observed_unix=time.time());dump(path,state)
                try:refresh_report()
                except Exception as exc:state['last_report_error']=str(exc)
                time.sleep(45)
            previous=HERE/'exploratory_status.json'
            if previous.exists() and read(previous).get('status')=='COMPLETED_L1005':
                refresh_report();state.update(status='COMPLETED',ended_unix=time.time());dump(path,state);return
            log=runtime/f'worker_{len(state["launches"]):02d}.log'
            command=[sys.executable,'-u',str(HERE/'run_exploratory.py'),'run']
            if Path('/usr/bin/caffeinate').exists():command=['/usr/bin/caffeinate','-i',*command]
            with log.open('ab') as output:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            rec={'pid':proc.pid,'started_unix':time.time(),'log':str(log.relative_to(HERE))}
            state['launches'].append(rec);state.update(status='RUNNING',supervisor_pid=os.getpid());dump(path,state)
            while proc.poll() is None:
                try:rec['last_report_exit_code']=refresh_report()
                except Exception as exc:rec['last_report_error']=f'{type(exc).__name__}: {exc}'
                rec['last_observed_unix']=time.time();dump(path,state)
                try:proc.wait(timeout=45)
                except subprocess.TimeoutExpired:pass
            rec.update(returncode=proc.returncode,ended_unix=time.time());dump(path,state)
            try:refresh_report()
            except Exception as exc:rec['last_report_error']=str(exc)
            result=read(previous) if previous.exists() else {}
            if proc.returncode==0 and result.get('status')=='COMPLETED_L1005':
                state.update(status='COMPLETED',ended_unix=time.time());dump(path,state);return
            # Protocol/covariance/label errors and resource limits require an
            # audit. Only explicit transient IO/interruption errors may restart.
            transient=result.get('error_type') in {'OSError','IOError','TimeoutError','InterruptedError'}
            if not transient:
                state.update(status='PAUSED_REQUIRES_AUDIT',worker_result=result,ended_unix=time.time());dump(path,state);return
        state.update(status='PROCESS_RESTART_CAP_REACHED',ended_unix=time.time());dump(path,state)


if __name__=='__main__':main()
