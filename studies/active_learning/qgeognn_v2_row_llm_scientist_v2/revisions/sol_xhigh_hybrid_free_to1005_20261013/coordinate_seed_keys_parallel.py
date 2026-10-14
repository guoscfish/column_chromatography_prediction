"""Adopt the running seed6101 workers and resume seed157 with its independent key."""
import os, sys, json, time, signal, subprocess, hashlib
from pathlib import Path
import run_seed_keys_parallel as runner
base = runner.base
HERE = runner.HERE

def guard():
    record = json.loads((HERE / 'seed_key_parallel_coordinator_manifest.json').read_text())
    assert runner.sha(__file__) == record['coordinator_sha256']
    runner.guard()
    return record

def main():
    record = guard()
    with base.s.exclusive_lock(HERE / 'seed_key_parallel_live_coordinator.execution.lock'):
        if '--check' in sys.argv:
            print('PARALLEL_COORDINATOR_PREFLIGHT_PASS', flush=True)
            return 0
        adopted = {}
        for method, pid in record['adopted_seed6101_workers'].items():
            pid = int(pid)
            args = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True).stdout
            assert 'run_seed_keys_parallel.py' in args and f'--worker 6101 {method}' in args and runner.alive(pid)
            adopted[6101, method] = pid
        # The new seed6101 requests stay alive during this administrative handoff.
        for pid in record['replaced_coordinator_pids']:
            if runner.alive(pid):
                args = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True).stdout
                assert 'run_seed_keys_parallel.py' in args and '--worker' not in args
                os.kill(pid, signal.SIGSTOP)
        events = [json.loads(line) for line in (HERE / 'execution_progress.jsonl').read_text().splitlines()]
        for pid in record['retired_seed157_workers']:
            if not runner.alive(pid):
                continue
            starts = [e for e in events if e['action'] == 'LLM_request_started' and e['pid'] == pid]
            ends = [e for e in events if e['action'] == 'LLM_request_finished' and e['pid'] == pid]
            assert not starts or (ends and starts[-1]['time'] <= ends[-1]['time']), 'Do not cancel an active request'
            worker = next(e for e in reversed(events) if e['action'] == 'worker_started' and e['pid'] == pid)
            latest = next(e for e in reversed(events) if e.get('seed') == 157 and e.get('method') == worker['method'] and e['action'] in ('training_started', 'training_completed', 'transport_backoff'))
            assert latest['action'] != 'training_started', 'Do not cancel an active training fit'
            os.kill(pid, signal.SIGTERM)
        for pid in record['replaced_coordinator_pids']:
            if runner.alive(pid):
                os.kill(pid, signal.SIGTERM); os.kill(pid, signal.SIGCONT)
        old_parent = record['old_serial_parent_pid']
        if runner.alive(old_parent):
            os.kill(old_parent, signal.SIGCONT)
        for _ in range(100):
            if not any(runner.alive(pid) for pid in record['retired_seed157_workers']):
                break
            time.sleep(.1)
        assert not any(runner.alive(pid) for pid in record['retired_seed157_workers'])
        children = {}
        try:
            for seed, method in base.TARGETS:
                if seed == 157:
                    children[seed, method] = runner.launch(seed, method, runner.guard())
            pids = {f'{seed}:{method}': child.pid for (seed, method), child in children.items()}
            pids.update({f'{seed}:{method}': pid for (seed, method), pid in adopted.items()})
            (HERE / 'seed_key_parallel_live_pids.json').write_text(json.dumps(pids, indent=2) + '\n')
            base.s._progress({'action': 'both_seed_keys_parallel_ready', 'worker_pids': pids})
            stopped = set()
            while True:
                for seed in (157, 6101):
                    if runner.quota_path(seed).exists() or (seed == 157 and (HERE / 'quota_exhausted.json').exists()):
                        stopped.add(seed)
                for (seed, method), child in children.items():
                    if seed in stopped and child.poll() is None:
                        child.terminate()
                for (seed, method), pid in adopted.items():
                    if seed in stopped and runner.alive(pid):
                        os.kill(pid, signal.SIGTERM)
                if all(child.poll() is not None for child in children.values()) and not any(runner.alive(pid) for pid in adopted.values()):
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

if __name__ == '__main__':
    sys.exit(main())
