"""Three independent credentials and three fit slots; preserve frozen science."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import run as base

OUT = base.OUT
KEYS = dict(base.KEYS)
KEYS[12203] = base.KEY_ROOT/'free-12203-20261023.api-key'
MANIFEST = OUT/'parallel_resume_manifest.json'


@contextmanager
def training_slot():
    # Use the same allocator and original two slot files, adding a third slot.
    chosen = None
    while chosen is None:
        with base.lock(OUT/'training_allocation.lock'):
            for i in range(3):
                handle = (OUT/f'training_slot_{i}.lock').open('a+')
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                chosen = handle
                break
        if chosen is None:
            time.sleep(1)
    try:
        yield
    finally:
        fcntl.flock(chosen.fileno(), fcntl.LOCK_UN)
        chosen.close()


def probe(key_file, route):
    try:
        with httpx.Client(proxy=route, trust_env=False, timeout=15) as client:
            response = client.get(base.CONFIG['base_url']+'/models',
                headers={'Authorization': 'Bearer '+key_file.read_text().strip()})
            result = {'http_status': response.status_code, 'proxy': route}
            if response.status_code == 200:
                result['requested_model_listed'] = any(
                    x.get('id') == base.CONFIG['model'] for x in response.json().get('data', []))
            return result
    except Exception as exc:
        return {'error_type': type(exc).__name__, 'proxy': route}


def configure(seed):
    record = base.s.read(MANIFEST)
    assert record['frozen_runner_sha256'] == base.s.sha256_file(Path(base.__file__))
    assert record['adapter_sha256'] == base.s.sha256_file(Path(__file__))
    route = record['routes'][str(seed)]['proxy']
    base.KEYS = KEYS
    base.training_slot = training_slot
    base.client_for = lambda _old_proxy=None, timeout=600: httpx.Client(
        proxy=route, trust_env=False, timeout=timeout)


def supervise(seed):
    import signal
    configure(seed)
    with base.s.exclusive_lock(OUT/f'supervisor_seed_{seed}.lock'):
        # A kernel lock check prevents starting alongside an original worker.
        with base.s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'):
            pass
        state_path = OUT/f'supervisor_seed_{seed}.json'
        saved = base.s.read(state_path) if state_path.exists() else {}
        recovery_count = saved.get('recovery_count', 0)
        while True:
            result = base.STUDY/f'{seed}_{base.FREE}_result.json'
            if result.exists():
                base.atomic_json(state_path, {'status': 'COMPLETE', 'seed': seed,
                    'time': base.now(), 'recovery_count': recovery_count})
                return
            log = OUT/f'parallel_worker_seed_{seed}.log'
            offset = log.stat().st_size if log.exists() else 0
            started = time.time()
            with log.open('ab') as handle:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', str(seed)],
                    cwd=base.ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True)
            base.atomic_json(state_path, {'status': 'RUNNING', 'seed': seed, 'pid': child.pid,
                'supervisor_pid': os.getpid(), 'time': base.now(), 'recovery_count': recovery_count})
            base.event(seed, 'supervised_worker_started', child_pid=child.pid, recovery_count=recovery_count)
            current_round = -1
            round_started = time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                state_file = OUT/f'status_seed_{seed}.json'
                if not state_file.exists():
                    continue
                state = base.s.read(state_file)
                r = state['completed_rounds']
                if r != current_round:
                    current_round, round_started = r, time.monotonic()
                # A healthy stream can reason for many minutes. Only a hard
                # 40 minute selection limit kills an otherwise live process.
                if state['phase'] == 'SELECTING' and time.monotonic()-round_started > 2400:
                    base.event(seed, 'selection_watchdog_timeout', round=r)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
            code = child.wait()
            if code == 0:
                continue
            with log.open('rb') as handle:
                handle.seek(offset)
                body = handle.read().decode(errors='replace')
            errors = []
            for path in (base.STUDY/f'selections/seed_{seed}/{base.FREE}').glob('round_*/retries/turn_*/*.json'):
                if path.stat().st_mtime >= started:
                    errors.append(base.s.read(path))
            fatal = any(word in body for word in (
                'DAILY_LIMIT_EXCEEDED', 'USAGE_LIMIT_EXCEEDED', 'QUOTA_EXHAUSTED',
                'AuthenticationError', '401 Unauthorized', '403 Forbidden',
                'full-pool selection remained invalid', 'Frozen source drift',
                'Frozen runner drift', 'immutable dialog artifact changed', 'Provider model mismatch'))
            transient = code < 0 or any(word in body for word in (
                'ConnectError', 'RemoteProtocolError', 'ReadTimeout', 'WriteTimeout',
                'APIConnectionError', 'APITimeoutError', 'TimeoutException',
                '429 Too Many Requests', '500 Internal Server Error', '502 Bad Gateway',
                '503 Service Unavailable', '504 Gateway Timeout'))
            recovery_count += 1
            if fatal or not transient or recovery_count > 12:
                base.atomic_json(state_path, {'status': 'NEEDS_ATTENTION', 'seed': seed,
                    'time': base.now(), 'returncode': code, 'recovery_count': recovery_count,
                    'error_types': sorted({e.get('error_type', '') for e in errors}), 'log': str(log)})
                base.event(seed, 'supervisor_stopped', returncode=code, recovery_count=recovery_count)
                return
            # First recovery is immediate. Persistent failures back off to avoid
            # hammering a disconnected provider; completed turns stay on disk.
            base.event(seed, 'automatic_transport_retry', recovery_count=recovery_count)
            base.atomic_json(state_path, {'status': 'RETRYING', 'seed': seed,
                'time': base.now(), 'recovery_count': recovery_count})
            if recovery_count > 1:
                time.sleep(min(60, 10*(recovery_count-1)))


def launch():
    base.activate()
    with base.s.exclusive_lock(OUT/'parallel_launch.lock'):
        routes = {}
        for seed in base.SEEDS:
            direct = probe(KEYS[seed], None)
            if direct.get('http_status') == 200 and direct.get('requested_model_listed'):
                routes[str(seed)] = direct
            else:
                routed = probe(KEYS[seed], 'http://127.0.0.1:3066')
                if routed.get('http_status') != 200 or not routed.get('requested_model_listed'):
                    print(json.dumps({'seed': seed, 'direct': direct, 'proxy': routed}), flush=True)
                    raise RuntimeError('No healthy authenticated route for seed '+str(seed))
                routes[str(seed)] = routed
        key_values = [KEYS[seed].read_text().strip() for seed in base.SEEDS]
        assert len(set(key_values)) == 3, 'Three independent credentials required'
        record = {'time': base.now(), 'authorization': 'User requested independent seed12203 credential, three parallel calls and fits, immediate monitored retries',
                  'model': base.CONFIG['model'], 'reasoning_effort': base.CONFIG['reasoning_effort'],
                  'target_labels': 1005, 'training_concurrency': 3, 'LLM_request_concurrency': 3,
                  'frozen_runner_sha256': base.s.sha256_file(Path(base.__file__)),
                  'adapter_sha256': base.s.sha256_file(Path(__file__)), 'routes': routes,
                  'credential_files': {str(s): str(KEYS[s]) for s in base.SEEDS}, 'supervisors': []}
        if MANIFEST.exists():
            raise RuntimeError('Parallel launch already recorded; inspect existing supervisors before resume')
        base.atomic_json(MANIFEST, record)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for seed in base.SEEDS:
            log = OUT/f'supervisor_seed_{seed}.log'
            with log.open('ab') as handle:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--supervise', str(seed)],
                    cwd=base.ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
            subprocess.Popen(['caffeinate', '-i', '-w', str(child.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record['supervisors'].append({'seed': seed, 'pid': child.pid, 'log': str(log)})
            base.atomic_json(MANIFEST, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', type=int, choices=base.SEEDS)
    parser.add_argument('--supervise', type=int, choices=base.SEEDS)
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.worker:
        configure(args.worker)
        base.work(args.worker)
    elif args.supervise:
        supervise(args.supervise)
