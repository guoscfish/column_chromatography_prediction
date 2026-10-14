"""Authorized two-credential scheduling; preserves the original scientific runner."""
import os, sys, time, json, hashlib, fcntl, signal, subprocess
from pathlib import Path
from contextlib import contextmanager
import extend_four_to1005 as base

HERE = Path(__file__).resolve().parent
SCHEDULE = HERE / 'seed_key_parallel_schedule.json'
SEED = None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guard():
    record = json.loads(SCHEDULE.read_text())
    assert sha(__file__) == record['runner_sha256']
    assert sha(HERE / 'extend_four_to1005.py') == record['original_runner_sha256']
    assert sha(HERE / 'extension_manifest.json') == record['original_extension_manifest_sha256']
    base.activate()
    return record

def quota_path(seed):
    return HERE / f'quota_seed_{seed}_parallel_keys.json'

@contextmanager
def request_lock():
    with (HERE / f'llm_request_seed_{SEED}.execution.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def api_call(messages, config):
    with request_lock():
        if quota_path(SEED).exists() or (SEED == 157 and (HERE / 'quota_exhausted.json').exists()):
            raise SystemExit(78)
        base.s._progress({'action': 'LLM_request_started', 'seed': SEED, 'pid': os.getpid(),
            'model': config['model'], 'effort': config['reasoning_effort'], 'credential_scope': f'seed{SEED}'})
        try:
            import httpx
            from openai import APIConnectionError
            try:
                return base.raw_call(messages, config)
            except httpx.RemoteProtocolError as exc:
                raise APIConnectionError(message='RemoteProtocolError: ' + str(exc),
                    request=httpx.Request('POST', config['base_url'] + '/responses')) from exc
            except Exception as exc:
                if any(code in str(exc) for code in ('DAILY_LIMIT_EXCEEDED', 'USAGE_LIMIT_EXCEEDED', 'insufficient_quota')):
                    base.once(quota_path(SEED), {'status': 'QUOTA_EXHAUSTED', 'seed': SEED, 'time': time.time()})
                    base.s._progress({'action': 'quota_exhausted', 'seed': SEED, 'pid': os.getpid()})
                    raise SystemExit(78) from exc
                headers = getattr(getattr(exc, 'response', None), 'headers', {})
                retry = headers.get('Retry-After') if headers else None
                if retry:
                    try:
                        delay = max(0, float(retry))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        delay = max(0, parsedate_to_datetime(retry).timestamp() - time.time())
                    base.s._progress({'action': 'rate_limit_retry_after', 'seed': SEED, 'seconds': delay})
                    time.sleep(delay)
                raise
        finally:
            base.s._progress({'action': 'LLM_request_finished', 'seed': SEED, 'pid': os.getpid()})

def alive(pid):
    state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith('Z')

def worker(seed, method):
    global SEED
    SEED = seed
    record = guard()
    os.environ.pop('SCIENTIST_API_KEY', None)
    os.environ['SCIENTIST_API_KEY_FILE'] = record['credential_files'][str(seed)]
    path = Path(os.environ['SCIENTIST_API_KEY_FILE'])
    assert path.is_file() and path.stat().st_mode & 0o077 == 0
    base.api_call = api_call
    return base.work(seed, method)

def launch(seed, method, record):
    env = os.environ.copy()
    env.pop('SCIENTIST_API_KEY', None)
    env['SCIENTIST_API_KEY_FILE'] = record['credential_files'][str(seed)]
    log = HERE / f'parallel_keys_worker_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    with log.open('ab') as out:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(seed), method],
            env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    base.s._progress({'action': 'worker_started', 'seed': seed, 'method': method, 'pid': child.pid, 'log': str(log),
        'schedule': 'independent_seed_credentials'})
    return child

def main():
    if '--worker' in sys.argv:
        i = sys.argv.index('--worker')
        return worker(int(sys.argv[i + 1]), sys.argv[i + 2])
    record = guard()
    if '--check' in sys.argv:
        from src.qgeognn_al.active_learning_v2.scientist_transport import provider_key
        keys = []
        for seed in (157, 6101):
            os.environ.pop('SCIENTIST_API_KEY', None)
            os.environ['SCIENTIST_API_KEY_FILE'] = record['credential_files'][str(seed)]
            keys.append(provider_key(base.NEW_CONFIG))
        assert keys[0] != keys[1]
        print('SEED_KEYS_PARALLEL_PREFLIGHT_PASS', flush=True)
        return 0
    adopted = {}
    old_parent = None
    if '--adopt-seed157' in sys.argv:
        i = sys.argv.index('--adopt-seed157')
        old_parent = int(sys.argv[i + 1])
        adopted = {(157, base.HYBRID): int(sys.argv[i + 2]), (157, base.FREE): int(sys.argv[i + 3])}
        for pid in adopted.values():
            assert alive(pid)
    children = {}
    with base.s.exclusive_lock(HERE / 'seed_key_parallel_coordinator.execution.lock'):
        try:
            for seed, method in base.TARGETS:
                if (seed, method) not in adopted:
                    children[seed, method] = launch(seed, method, record)
            stopped_seeds = set()
            while True:
                if (HERE / 'quota_exhausted.json').exists() or quota_path(157).exists():
                    stopped_seeds.add(157)
                for seed in (157, 6101):
                    if quota_path(seed).exists():
                        stopped_seeds.add(seed)
                for (seed, method), child in children.items():
                    if seed in stopped_seeds and child.poll() is None:
                        child.terminate()
                for (seed, method), pid in adopted.items():
                    if seed in stopped_seeds and alive(pid):
                        os.kill(pid, signal.SIGTERM)
                if old_parent and not any(alive(pid) for pid in adopted.values()):
                    if alive(old_parent):
                        os.kill(old_parent, signal.SIGCONT)
                    old_parent = None
                if all(child.poll() is not None for child in children.values()) and not any(alive(pid) for pid in adopted.values()):
                    break
                time.sleep(1)
            guard()
            for seed, method in base.TARGETS:
                result = base.read(HERE / f'{seed}_{method}_result.json')
                assert result['final_labels'] == 1005 and len(set(result['selected_ids'])) == 672
                base.s.verify_files(HERE, result['files'])
            base.once(HERE / 'result.json', {'status': 'ALL_FOUR_COMPLETE1005', 'test_evaluation_performed': False})
            base.s._progress({'action': 'all_four1005_completed', 'schedule': 'independent_seed_credentials'})
            return 0
        finally:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
            for child in children.values():
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            if old_parent and alive(old_parent):
                # Keep adopted scientific workers running if this coordinator alone fails.
                os.kill(old_parent, signal.SIGCONT)

if __name__ == '__main__':
    sys.exit(main())
