"""Diagnosed half-open recovery; preserve cumulative failure counts and science."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import parallel_resume as resume


if __name__ == '__main__':
    base = resume.base
    base.activate()
    probe_path = max(base.OUT.glob('inference_probe_*.json'), key=lambda p: p.stat().st_mtime)
    probe = base.s.read(probe_path)
    assert probe['status'] == 'PASS' and time.time()-probe_path.stat().st_mtime < 300
    assert probe['provenance']['served_model'] == base.CONFIG['model']
    stamp = str(time.time_ns())
    with base.s.exclusive_lock(base.OUT/'diagnosed_recovery_launch.lock'):
        guards = []
        try:
            for seed in base.SEEDS:
                sup = base.s.read(base.OUT/f'supervisor_seed_{seed}.json')
                assert sup['status'] == 'NEEDS_ATTENTION'
                assert set(sup['error_types']) <= {'RemoteProtocolError', 'ConnectError', 'ReadTimeout', 'APIConnectionError', 'APITimeoutError'}
                for name in (f'supervisor_seed_{seed}.lock', f'worker_seed_{seed}.lock'):
                    guard = base.s.exclusive_lock(base.OUT/name)
                    guard.__enter__()
                    guards.append(guard)
            history = base.OUT/'diagnosed_recovery_history'
            history.mkdir(exist_ok=True)
            previous = {}
            for seed in base.SEEDS:
                path = base.OUT/f'supervisor_seed_{seed}.json'
                previous[str(seed)] = base.s.read(path)
                base.s.once(history/f'{stamp}_supervisor_seed_{seed}.json', previous[str(seed)])
        finally:
            for guard in reversed(guards):
                guard.__exit__(None, None, None)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        record = {'time': base.now(), 'diagnostic_probe': str(probe_path),
                  'reason': 'New completed diagnostic inference confirms generation service responds after earlier persistent connection failures',
                  'preserved_cumulative_failure_counts': {s: v['recovery_count'] for s, v in previous.items()},
                  'transport_policy': 'Original supervisors retain their counters; one diagnosed half-open worker per seed; another failure stops again',
                  'logical_format_budget_changed': False, 'model_or_prompt_changed': False, 'supervisors': []}
        for seed in base.SEEDS:
            log = base.OUT/f'supervisor_seed_{seed}.log'
            with log.open('ab') as handle:
                child = subprocess.Popen([sys.executable, '-u', str(Path(resume.__file__)), '--supervise', str(seed)],
                    cwd=base.ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True)
            subprocess.Popen(['caffeinate', '-i', '-w', str(child.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record['supervisors'].append({'seed': seed, 'pid': child.pid})
        base.atomic_json(history/f'{stamp}_launch.json', record)
        base.atomic_json(base.OUT/'diagnosed_recovery_latest.json', record)
        print(json.dumps(record), flush=True)
