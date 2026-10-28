"""Explicitly authorized new resume session; preserve frozen science and history."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import parallel_resume as adapter

base = adapter.base
OUT = base.OUT
HERE = Path(__file__).resolve()
HISTORY = OUT/'resume_sessions'
PROBE = OUT/'resume_probe_latest.json'
SESSION = OUT/'resume_session_latest.json'


def lane(seed):
    key = adapter.KEYS[seed].read_text().strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def check_seed(seed):
    """A real minimal generation checks quota; GET /models alone cannot do so."""
    key = adapter.KEYS[seed].read_text().strip()
    checks = []
    for proxy in (None, 'http://127.0.0.1:3066'):
        started = time.monotonic()
        result = {'seed': seed, 'proxy': proxy, 'experiment_packet_sent': False}
        try:
            with httpx.Client(proxy=proxy, trust_env=False, timeout=httpx.Timeout(60, connect=10)) as client:
                with client.stream('POST', base.CONFIG['base_url']+'/responses',
                    headers={'Authorization': 'Bearer '+key},
                    json={'model': base.CONFIG['model'], 'input': [{'role': 'user', 'content': 'Reply with OK.'}],
                          'reasoning': {'effort': base.CONFIG['reasoning_effort']},
                          'tools': [], 'store': False, 'stream': True}) as response:
                    result['http_status'] = response.status_code
                    if response.status_code != 200:
                        text = response.read().decode(errors='replace')
                        try:
                            error = json.loads(text).get('error', {})
                        except ValueError:
                            error = {}
                        if isinstance(error, dict):
                            result['provider_error_code'] = str(error.get('code', ''))[:100]
                        quota = any(word in text for word in ('DAILY_LIMIT_EXCEEDED', 'USAGE_LIMIT_EXCEEDED', 'insufficient_quota'))
                        result['status'] = 'QUOTA_EXHAUSTED' if quota else ('AUTHENTICATION_FAILED' if response.status_code in (401,403) else 'PROVIDER_ERROR')
                    else:
                        completed = None
                        for line in response.iter_lines():
                            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                                continue
                            data = json.loads(line[5:].strip())
                            if data.get('type') == 'response.completed':
                                completed = data['response']
                                break
                            if data.get('type') in ('error','response.failed','response.incomplete'):
                                error = data.get('error') or data.get('response', {}).get('error') or {}
                                code = str(error.get('code','')) if isinstance(error,dict) else ''
                                result.update(status='QUOTA_EXHAUSTED' if any(w in str(error) for w in ('DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','insufficient_quota')) else 'PROVIDER_ERROR', provider_error_code=code)
                                break
                        if completed:
                            model = completed['model']
                            assert model == base.CONFIG['model'] or model.startswith(base.CONFIG['model']+'-')
                            assert completed['status'] == 'completed'
                            assert all(item['type'] in ('message','reasoning') for item in completed['output'])
                            result.update(status='PASS', served_model=model, response_id=completed['id'])
                        elif 'status' not in result:
                            result['status'] = 'INCOMPLETE_STREAM'
        except Exception as error:
            result.update(status='CONNECTION_FAILED', error_type=type(error).__name__)
        result['elapsed_seconds'] = round(time.monotonic()-started, 2)
        checks.append(result)
        if result['status'] in ('PASS','QUOTA_EXHAUSTED','AUTHENTICATION_FAILED'):
            break
    return {'seed': seed, 'status': checks[-1]['status'], 'proxy': checks[-1]['proxy'], 'checks': checks}


def probe():
    base.activate()
    for seed in base.SEEDS:
        adapter.configure(seed)
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(check_seed, base.SEEDS))
    record = {'time': base.now(), 'authorization_date': '2026-10-24', 'seeds': rows,
              'model': base.CONFIG['model'], 'reasoning_effort': base.CONFIG['reasoning_effort']}
    HISTORY.mkdir(exist_ok=True)
    base.atomic_json(HISTORY/f'probe_{time.time_ns()}.json', record)
    base.atomic_json(PROBE, record)
    print(json.dumps(record), flush=True)


def supervisor(seed, session_id):
    session = base.s.read(SESSION)
    assert session['session_id'] == session_id
    assert session['supervisor_sha256'] == base.s.sha256_file(HERE)
    adapter.configure(seed)
    base.activate()
    prefix = session['previous_supervisors'][str(seed)]['recovery_count']
    session_failures = 0
    failures_per_round = {}
    with base.s.exclusive_lock(OUT/f'supervisor_seed_{seed}.lock'):
        with base.s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'):
            pass
        while True:
            if (base.STUDY/f'{seed}_{base.FREE}_result.json').exists():
                base.atomic_json(OUT/f'supervisor_seed_{seed}.json',
                    {'status':'COMPLETE','seed':seed,'time':base.now(),'session_id':session_id,
                     'recovery_count':prefix+session_failures,'session_recovery_count':session_failures})
                return
            log = OUT/f'continued_worker_seed_{seed}_{session_id}.log'
            offset = log.stat().st_size if log.exists() else 0
            with log.open('ab') as handle:
                child = subprocess.Popen([sys.executable,'-u',str(Path(adapter.__file__)),'--worker',str(seed)],
                    cwd=base.ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            base.atomic_json(OUT/f'supervisor_seed_{seed}.json',
                {'status':'RUNNING','seed':seed,'pid':child.pid,'supervisor_pid':os.getpid(),
                 'time':base.now(),'session_id':session_id,'recovery_count':prefix+session_failures,
                 'session_recovery_count':session_failures,'log':str(log)})
            base.event(seed,'renewed_session_worker_started',child_pid=child.pid,session_id=session_id)
            current_round = None
            phase_started = time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                state_path = OUT/f'status_seed_{seed}.json'
                if not state_path.exists():
                    continue
                state = base.s.read(state_path)
                r = state['completed_rounds']
                if r != current_round:
                    current_round, phase_started = r, time.monotonic()
                if state['phase'] == 'SELECTING' and time.monotonic()-phase_started > 2400:
                    base.event(seed,'selection_watchdog_timeout',round=r,session_id=session_id)
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                    break
            code = child.wait()
            if code == 0:
                continue
            session_failures += 1
            state = base.s.read(OUT/f'status_seed_{seed}.json')
            r = str(state['completed_rounds'])
            failures_per_round[r] = failures_per_round.get(r,0)+1
            with log.open('rb') as handle:
                handle.seek(offset)
                body = handle.read().decode(errors='replace')
            fatal = code == 78 or any(word in body for word in (
                'QUOTA_EXHAUSTED','DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','AuthenticationError',
                '401 Unauthorized','403 Forbidden','full-pool selection remained invalid',
                'Frozen source drift','Frozen runner drift','immutable dialog artifact changed','Provider model mismatch'))
            transient = code < 0 or any(word in body for word in (
                'ConnectError','RemoteProtocolError','ReadTimeout','WriteTimeout','APIConnectionError',
                'APITimeoutError','TimeoutException','429 Too Many Requests','500 Internal Server Error',
                '502 Bad Gateway','503 Service Unavailable','504 Gateway Timeout'))
            saved = {'seed':seed,'time':base.now(),'returncode':code,'session_id':session_id,
                     'recovery_count':prefix+session_failures,'session_recovery_count':session_failures,
                     'failures_per_round':failures_per_round,'log':str(log)}
            if fatal or not transient or failures_per_round[r] > 2 or session_failures > 12:
                base.atomic_json(OUT/f'supervisor_seed_{seed}.json',{'status':'NEEDS_ATTENTION',**saved})
                base.event(seed,'renewed_session_stopped',returncode=code,session_id=session_id)
                return
            base.atomic_json(OUT/f'supervisor_seed_{seed}.json',{'status':'RETRYING',**saved})
            base.event(seed,'automatic_transport_retry',session_id=session_id,round=int(r),
                       recovery_count=prefix+session_failures)
            if failures_per_round[r] > 1:
                time.sleep(30)


def launch():
    base.activate()
    checks = base.s.read(PROBE)
    assert time.time()-PROBE.stat().st_mtime < 600, 'Quota probes must be fresh'
    healthy = [x for x in checks['seeds'] if x['status']=='PASS']
    if not healthy:
        print(json.dumps({'status':'NO_HEALTHY_CREDENTIALS','probe':checks}),flush=True)
        return
    session_id = str(time.time_ns())
    destination = HISTORY/session_id
    destination.mkdir(parents=True)
    with base.s.exclusive_lock(OUT/'resume_20261024_launch.lock'):
        previous = {}
        with ExitStack() as guards:
            for row in healthy:
                seed=row['seed']
                guards.enter_context(base.s.exclusive_lock(OUT/f'supervisor_seed_{seed}.lock'))
                guards.enter_context(base.s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'))
                previous[str(seed)]=base.s.read(OUT/f'supervisor_seed_{seed}.json')
                base.s.once(destination/f'supervisor_seed_{seed}.json',previous[str(seed)])
                base.s.once(destination/f'status_seed_{seed}.json',base.s.read(OUT/f'status_seed_{seed}.json'))
                marker=OUT/f'quota_{lane(seed)}.json'
                if marker.exists():
                    marker.rename(destination/marker.name)
            manifest=base.s.read(adapter.MANIFEST)
            base.s.once(destination/'previous_parallel_resume_manifest.json',manifest)
            for row in healthy:
                manifest['routes'][str(row['seed'])]={'http_status':200,'requested_model_listed':True,'proxy':row['proxy']}
            base.atomic_json(adapter.MANIFEST,manifest)
            record={'time':base.now(),'authorization':'User explicitly requested continuation on 2026-10-24',
                    'session_id':session_id,'supervisor_sha256':base.s.sha256_file(HERE),
                    'probe':checks,'previous_supervisors':previous,'max_transport_retries_per_round':2,
                    'max_session_transport_retries':12,'cumulative_failures_preserved':True,
                    'logical_correction_budget_changed':False,'model_or_training_changed':False,
                    'training_concurrency':3,'LLM_request_concurrency':3,'supervisors':[]}
            base.atomic_json(SESSION,record)
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for row in healthy:
            seed=row['seed'];log=OUT/f'supervisor_seed_{seed}_{session_id}.log'
            with log.open('ab') as handle:
                child=subprocess.Popen([sys.executable,'-u',str(HERE),'--supervise',str(seed),'--session',session_id],
                    cwd=base.ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            subprocess.Popen(['caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            record['supervisors'].append({'seed':seed,'pid':child.pid,'log':str(log)})
            base.atomic_json(SESSION,record)
        base.atomic_json(destination/'launch.json',record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--supervise',type=int,choices=base.SEEDS)
    parser.add_argument('--session')
    args=parser.parse_args()
    if args.probe:
        probe()
    elif args.launch:
        launch()
    elif args.supervise:
        supervisor(args.supervise,args.session)
