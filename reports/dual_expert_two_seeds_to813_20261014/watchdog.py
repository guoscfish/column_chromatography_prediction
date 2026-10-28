"""Monitor selection replies without changing frozen scientific code or inputs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

IDLE_SECONDS = 15 * 60
POLL_SECONDS = 30
MAX_RECOVERIES = 2

def reply_progress(directory, started_at):
    """Only role receipt files count; study/log heartbeats are deliberately ignored."""
    paths = (Path(directory) / 'agents').glob('**/*.json')
    return max([started_at, *(p.stat().st_mtime for p in paths if p.is_file())])

def overdue_selection(directory, started_at, now, idle_seconds=IDLE_SECONDS):
    directory = Path(directory)
    if (directory / 'batch_freeze.json').exists():
        return False
    return now - reply_progress(directory, started_at) >= idle_seconds

def recent_failures(directory, started_at):
    result=[]
    for p in (Path(directory) / 'agents').glob('**/retries/**/*.json'):
        if p.stat().st_mtime >= started_at:
            result.append(json.loads(p.read_text()))
    return result

def probe_provider(key_file, base_url):
    import httpx
    result={}
    try:
        with httpx.Client(timeout=httpx.Timeout(10, connect=5)) as client:
            response=client.get(base_url.rstrip('/')+'/models',
                headers={'Authorization':'Bearer '+Path(key_file).read_text().strip()})
        result['http_status']=response.status_code
        if response.status_code==200:
            result['requested_model_listed']=any(x.get('id')=='gpt-6.1-sol' for x in response.json().get('data',[]))
    except Exception as error:
        result['error_type']=type(error).__name__
    return result

def stop_worker(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()

def run_monitored(command, *, cwd, log, directory, probe, emit,
                  idle_seconds=IDLE_SECONDS, poll_seconds=POLL_SECONDS):
    started_at=time.time()
    with Path(log).open('a') as handle:
        process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            try:
                return {'returncode':process.wait(timeout=poll_seconds),
                        'watchdog_timeout':False,'started_at':started_at,'pid':process.pid}
            except subprocess.TimeoutExpired:
                now=time.time()
                if not overdue_selection(directory,started_at,now,idle_seconds):continue
                health=probe()
                # The response may have completed during the health check.
                if process.poll() is not None:
                    return {'returncode':process.returncode,'watchdog_timeout':False,
                            'started_at':started_at,'pid':process.pid}
                if not overdue_selection(directory,started_at,time.time(),idle_seconds):continue
                emit(event='selection_reply_timeout',idle_seconds=round(time.time()-reply_progress(directory,started_at),1),
                     worker_pid=process.pid,provider_probe=health)
                stop_worker(process)
                return {'returncode':124,'watchdog_timeout':True,'provider_probe':health,
                        'started_at':started_at,'pid':process.pid}
