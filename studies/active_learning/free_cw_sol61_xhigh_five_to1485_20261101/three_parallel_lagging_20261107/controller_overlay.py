#!/usr/bin/env python3
"""User-authorized three-worker/three-compute scheduler, lagging seeds first."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ENTRY = Path(__file__).resolve()
PRIOR = STUDY / 'upstream_error_recovery_20261104/controller_overlay.py'
SEEDS = (157, 887, 2357, 6101, 12203)
LIMIT = 3


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def priority(counts):
    return sorted((s for s in SEEDS if counts[s] < 36), key=lambda s: (counts[s], s))


def round_counts():
    result = {}
    for seed in SEEDS:
        rounds = sorted(int(p.stem.rsplit('_', 1)[1]) for p in
                        (STUDY / 'progress').glob(f'completed_seed_{seed}_round_*.json'))
        if rounds != list(range(len(rounds))) or len(rounds) > 36:
            raise RuntimeError('non-contiguous completed rounds')
        result[seed] = len(rounds)
    return result


def build():
    upstream = module('retained_upstream_three', PRIOR)
    prior = module('retained_immediate_three', upstream.PRIOR)
    adapter = module('retained_persistent_three', prior.BASE)
    h2 = module('retained_h2_three', STUDY / 'transport_recovery_20261102/controller_overlay.py')
    shim = module('retained_shim_three', STUDY / 'transport_recovery_20261102/upstream_h2_worker.py')
    recovery = module('retained_403_three', prior.PRIOR)
    h2.bind(adapter, shim)
    recovery.bind(adapter)
    prior.bind(adapter)
    upstream.bind(adapter)
    bind(adapter)
    return adapter


def bind(adapter):
    previous_load, previous_validate = adapter.load_base, adapter.validate

    @contextmanager
    def compute_slot(seed):
        # Slot zero is the existing lease, so adopted old compute is counted.
        paths = [STUDY / 'locks/new_free_resource.lock'] + [
            STUDY / f'locks/new_free_resource_{i}.lock' for i in (1, 2)]
        while True:
            adapter.m.training_priority_wait()
            for path in paths:
                handle = path.open('a+')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    adapter.m.training_priority_wait()
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                return
            adapter.m.base.status_record(seed, 'WAITING_COMPUTE_SLOT',
                completed_rounds=round_counts()[seed], maximum_expensive_compute=LIMIT)
            time.sleep(5)

    def load():
        previous_load()
        adapter.a.compute_slot = compute_slot

    def validate():
        result = previous_validate()
        policy = json.loads((HERE / 'authorization.json').read_text())
        required = {'maximum_seed_workers': 3, 'maximum_expensive_compute': 3,
                    'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh',
                    'target_budget': 1485, 'transport_backoff_seconds': 0,
                    'local_restart_cooldown_seconds': 60, 'watchdog_seconds': 660,
                    'priority': 'completed_rounds_ascending_then_seed_ascending'}
        if any(policy.get(k) != v for k, v in required.items()):
            raise RuntimeError('three-parallel authorization drift')
        for name, digest in policy['code_hashes'].items():
            if hashlib.sha256((STUDY / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('three-parallel source binding drift: ' + name)
        return {**result, 'three_parallel_authorization_sha256': adapter.m.sha(HERE / 'authorization.json')}

    def retire_old(seed, pid):
        """Only retire verified terminal failure with no pending request or compute."""
        info = adapter.process_command(pid)
        if str(PRIOR) not in info or f'worker --seed {seed}' not in info:
            return False
        state = adapter.m.read(STUDY / f'progress/seed_{seed}.json')
        stream = adapter.m.read(STUDY / f'progress/stream_seed_{seed}.json')
        if (state.get('pid') != pid or state.get('phase') not in ('SELECTING', 'TRANSPORT_BACKOFF')
                or stream.get('pid') != pid or stream.get('phase') != 'FAILED'):
            return False
        os.kill(pid, signal.SIGSTOP)
        try:
            for _ in range(30):
                if adapter.process_command(pid).startswith('T'):
                    break
                time.sleep(.03)
            else:
                return False
            state = adapter.m.read(STUDY / f'progress/seed_{seed}.json')
            if state.get('pid') != pid or state.get('phase') not in ('SELECTING', 'TRANSPORT_BACKOFF'):
                return False
            if adapter.unresolved(seed):
                return False
            stream = adapter.m.read(STUDY / f'progress/stream_seed_{seed}.json')
            if stream.get('pid') != pid or stream.get('phase') != 'FAILED':
                return False
            adapter.a.once(HERE / f'handover/seed_{seed}_pid_{pid}.json', {
                'seed': seed, 'pid': pid, 'time': adapter.m.now(), 'command': info,
                'reason': 'three-parallel resource migration at terminal failure boundary',
                'unresolved_requests': 0, 'NoCW_signaled': False, 'state': state, 'stream': stream})
            os.kill(pid, signal.SIGTERM)
            return True
        finally:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

    def supervisor():
        adapter.activate_worker()
        validate()
        adapter.m.s.require_authorization()
        children, cooldowns = {}, {}
        with adapter.m.base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/supervisor.lock'):
            while True:
                validate()
                for seed, child in list(children.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    adapter.a.once(HERE / f'worker_exits/seed_{seed}_pid_{child.pid}.json',
                        {'seed': seed, 'pid': child.pid, 'exit_code': code, 'time': adapter.m.now()})
                    if code:
                        cooldowns[seed] = time.time() + 60
                        adapter.m.base.atomic_json(HERE / f'restart_after_seed_{seed}.json',
                            {'epoch': cooldowns[seed], 'reason': 'persistent local recovery cooldown'})
                    del children[seed]
                counts = round_counts()
                rank = priority(counts)
                states = {s: adapter.m.read(STUDY / f'progress/seed_{s}.json') for s in SEEDS}
                live = {s: states[s].get('pid') for s in SEEDS if adapter.a.lock_busy(s)}
                # Drain old selectors at safe failure boundaries. Never interrupt old compute.
                for seed, pid in live.items():
                    if str(PRIOR) in adapter.process_command(pid):
                        if not retire_old(seed, pid):
                            old_entry = adapter.ENTRY
                            try:
                                adapter.ENTRY = PRIOR
                                adapter.retire_worker(seed, pid, 'wall-clock stream inactivity', idle_limit=660)
                            finally:
                                adapter.ENTRY = old_entry
                    elif str(ENTRY) in adapter.process_command(pid):
                        adapter.retire_worker(seed, pid, 'wall-clock stream inactivity', idle_limit=660)
                # Recheck locks: any old in-flight/compute worker consumes one of the three.
                live = {s: states[s].get('pid') for s in SEEDS if adapter.a.lock_busy(s)}
                occupied = set(live) | set(children)
                for seed in rank:
                    if len(occupied) >= LIMIT:
                        break
                    cooldown = max(cooldowns.get(seed, 0), *[
                        adapter.m.read(directory / f'restart_after_seed_{seed}.json').get('epoch', 0)
                        if (directory / f'restart_after_seed_{seed}.json').exists() else 0
                        for directory in (HERE, adapter.EXT)])
                    if seed in occupied or cooldown > time.time():
                        continue
                    if not adapter.transient_stop(states[seed]):
                        continue
                    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='2',
                               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    env.pop('SCIENTIST_API_KEY', None)
                    with (STUDY / f'progress/seed_{seed}_persistent.log').open('ab') as out:
                        child = subprocess.Popen([sys.executable, str(ENTRY), 'worker', '--seed', str(seed),
                            '--allow-paid-llm', '--allow-training', '--allow-label-reveal'],
                            cwd=adapter.ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                    children[seed] = child
                    occupied.add(seed)
                    adapter.a.once(HERE / f'dispatch/seed_{seed}_pid_{child.pid}.json',
                        {'seed': seed, 'pid': child.pid, 'time': adapter.m.now(), 'completed_rounds': counts[seed]})
                rows = [{'seed': s, 'pid': children[s].pid if s in children else live.get(s),
                         'supervision': 'RUNNING' if s in occupied else 'COMPLETE' if counts[s] == 36 else
                         'BLOCKED_CONFIGURATION_OR_PROTOCOL' if not adapter.transient_stop(states[s]) else 'QUEUED_LAGGING_PRIORITY',
                         'completed_rounds': counts[s], 'current_budget': 333 + 32 * counts[s],
                         'progress': states[s]} for s in SEEDS]
                complete = not rank
                record = {'pid': os.getpid(), 'phase': 'ALL_FIVE_COMPLETE' if complete else 'RUNNING',
                          'updated_utc': adapter.m.now(), 'target_budget': 1485,
                          'maximum_seed_workers': 3, 'maximum_expensive_compute': 3,
                          'priority_order': rank, 'active_seeds': sorted(occupied),
                          'migration_draining': len(occupied) > 3,
                          'transport_attempt_cap': None, 'answer_attempt_cap': None,
                          'three_parallel_authorization_sha256': adapter.m.sha(HERE / 'authorization.json'), 'seeds': rows}
                for name in ('persistent_supervisor.json', 'parallel_supervisor.json'):
                    adapter.m.base.atomic_json(STUDY / 'progress' / name, record)
                if complete:
                    adapter.m.update_reports()
                    return
                time.sleep(3)

    adapter.load_base, adapter.validate = load, validate
    adapter.supervisor, adapter.ENTRY = supervisor, ENTRY


if __name__ == '__main__':
    build().main()
