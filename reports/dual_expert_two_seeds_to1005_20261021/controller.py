"""One seed per process and credential; bounded, persistent transport recovery."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reports/dual_expert_two_seeds_to813_20261014'))
from watchdog import run_monitored, recent_failures, probe_provider

BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/revisions'
METHOD = 'free_llm32_dual_expert_v1'


def latest(revision, seed):
    paths = (BASE / revision / f'runtime/seed_{seed}/{METHOD}').glob('round_*/prediction_freeze.json')
    return max((int(p.parent.name.split('_')[1]) for p in paths), default=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(157, 6101), required=True)
    seed = parser.parse_args().seed
    # Kernel lock prevents duplicate controllers even after an interrupted launch.
    import fcntl
    lock = (OUT / f'controller_seed_{seed}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key_file = Path.home() / '.config/qgeognn-scientist' / (
        'dual-expert-813.api-key' if seed == 157 else 'dual-expert-813-seed6101.api-key')
    credential = key_file.read_text().strip()
    if not credential:
        raise RuntimeError('Task credential is empty')
    os.environ['SCIENTIST_API_KEY_FILE'] = str(key_file)
    os.environ['SCIENTIST_API_KEY'] = credential
    parent = ('sol61_xhigh_seed157_from621_to813_20261014' if seed == 157
              else 'sol61_xhigh_seed6101_to813_20261014')
    child = f'sol61_xhigh_seed{seed}_from813_to1005_20261021'
    recovery_file = OUT / f'recoveries_seed_{seed}.json'
    recoveries = json.loads(recovery_file.read_text()) if recovery_file.exists() else {}

    def event(**data):
        data = {'time': datetime.now().astimezone().isoformat(), 'seed': seed, **data}
        with (OUT / f'events_seed_{seed}.jsonl').open('a') as handle:
            handle.write(json.dumps(data) + '\n')
        print(json.dumps(data), flush=True)

    def status(state, revision, r, **extra):
        data = {'seed': seed, 'status': state, 'revision': revision,
                'completed_rounds': r, 'completed_labels': 333 + 32*r,
                'target_labels': 1005, 'updated_at': datetime.now().astimezone().isoformat(), **extra}
        (OUT / f'status_seed_{seed}.json').write_text(json.dumps(data, indent=2) + '\n')

    while True:
        extended = latest(parent, seed) >= 15
        revision = child if extended else parent
        r = max(15, latest(child, seed)) if extended else latest(parent, seed)
        if r == 21:
            status('COMPLETE', revision, r)
            event(event='complete', labels=1005)
            return
        directory = BASE / revision / f'selections/seed_{seed}/{METHOD}/round_{r:02d}'
        command = [sys.executable, '-u', str(OUT/'worker.py'), '--seed', str(seed), '--rounds', str(r+1)]
        if extended:
            command.append('--extended')
        log = OUT / f'seed_{seed}_to_{333+32*(r+1)}.log'
        status('RUNNING', revision, r)
        event(event='batch_started', from_labels=333+32*r, to_labels=333+32*(r+1), revision=revision)
        offset = log.stat().st_size if log.exists() else 0
        result = run_monitored(command, cwd=ROOT, log=log, directory=directory,
                               probe=lambda: probe_provider(key_file, 'https://token4research.cn/v1'),
                               emit=lambda **data: event(**data))
        if result['returncode'] == 0:
            assert latest(revision, seed) == r+1
            event(event='batch_complete', labels=333+32*(r+1))
            continue
        errors = recent_failures(directory, result['started_at'])
        with log.open('rb') as handle:
            handle.seek(offset)
            body = handle.read().decode('utf-8', errors='replace')
        if any('DAILY_LIMIT_EXCEEDED' in e.get('message', '') for e in errors) or 'DAILY_LIMIT_EXCEEDED' in body:
            state = 'BLOCKED_PROVIDER_DAILY_LIMIT'
        elif any(e.get('error_type') == 'AuthenticationError' for e in errors):
            state = 'BLOCKED_AUTHENTICATION'
        elif result['watchdog_timeout'] or ('failed after 3 attempts' in body and any(
                e.get('error_type') in ('RemoteProtocolError','APIConnectionError','APITimeoutError',
                                      'ReadTimeout','RuntimeError','InternalServerError') for e in errors)):
            # A new explicit user resume can authorize another bounded session;
            # retain all previous counts under their original keys.
            epoch = os.environ.get('DUAL_RECOVERY_EPOCH', 'initial')
            key = f'{epoch}:{revision}:{r}' if epoch != 'initial' else f'{revision}:{r}'
            if recoveries.get(key, 0) < 2:
                recoveries[key] = recoveries.get(key, 0)+1
                recovery_file.write_text(json.dumps(recoveries, indent=2)+'\n')
                event(event='bounded_transport_recovery', round=r, attempt=recoveries[key])
                time.sleep(30)
                continue
            state = 'BLOCKED_PROVIDER_CONNECTION'
        else:
            state = 'BLOCKED_PROTOCOL_OR_FORMAT'
        status(state, revision, r, log=str(log))
        event(event='stopped', status=state, round=r)
        return


if __name__ == '__main__':
    main()
