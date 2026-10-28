"""Read-only experiment audit plus a compact heartbeat receipt; no API calls."""
from datetime import datetime
from pathlib import Path
import json
import subprocess
from zoneinfo import ZoneInfo

import run


def main():
    run.activate()
    current = datetime.now(ZoneInfo('Asia/Shanghai'))
    supervisors = {seed: run.s.read(run.OUT/f'supervisor_seed_{seed}.json') for seed in run.SEEDS}
    pids = sorted({pid for sup in supervisors.values()
                   for pid in (sup.get('pid'), sup.get('supervisor_pid')) if pid})
    processes = {}
    if pids:
        result = subprocess.run(['ps', '-p', ','.join(map(str, pids)), '-o', 'pid=,stat='],
                                text=True, errors='replace', capture_output=True, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError('Cannot inspect experiment processes: '+result.stderr.strip())
        processes = {int(row.split()[0]): row.split()[1] for row in result.stdout.splitlines()}
    rows, issues = [], []
    for seed in run.SEEDS:
        state = run.s.read(run.OUT/f'status_seed_{seed}.json')
        sup = supervisors[seed]
        initial = run.s.read(run.s.historical_round0_paths(seed)['member0_audit'])
        verified = []
        for path in sorted((run.STUDY/f'runtime/seed_{seed}/{run.FREE}').glob('round_*/prediction_freeze.json')):
            r = int(path.parent.name.split('_')[1])
            record = run.s.read(path)
            for kind in ('checkpoint', 'prediction'):
                assert run.s.sha256_file(Path(record[kind+'_path'])) == record[kind+'_sha256'], str(path)
            fit = run.s.read(Path(record['checkpoint_path']).parent/'fit_audit.json')
            assert fit['train_rows'] == run.BUDGETS[r], str(path)
            assert fit['initialization_hash'] == initial['initialization_hash'], str(path)
            assert fit['validation_ids_hash'] == initial['validation_ids_hash'], str(path)
            assert record['test_truth_access_count'] == 0, str(path)
            verified.append(run.BUDGETS[r])
        assert verified == list(run.BUDGETS[:len(verified)]), f'seed {seed}: missing frozen prediction'
        context = run.s.Context(seed)
        labeled, _, _, _ = run.s.state(context, run.FREE, len(verified)-1)
        assert len(labeled) == verified[-1], f'seed {seed}: state/fit budget mismatch'
        batch_count = 0
        for path in (run.STUDY/f'selections/seed_{seed}/{run.FREE}').glob('round_*/batch_freeze.json'):
            batch = run.s.audit_batch(path.parent)
            assert len(batch['batch_ids']) == 32 and not batch['pending_ids'], str(path)
            batch_count += 1
        events = [json.loads(line) for line in (run.OUT/f'events_seed_{seed}.jsonl').read_text().splitlines()]
        latest = events[-1]
        event_age = max(0, (datetime.now(ZoneInfo('Asia/Shanghai'))-datetime.fromisoformat(latest['time'])).total_seconds())
        active = {str(pid): processes.get(pid, 'MISSING')
                  for pid in (sup.get('pid'), sup.get('supervisor_pid')) if pid}
        if sup['status'] not in ('COMPLETE', 'RETRYING'):
            if sup['status'] != 'RUNNING' or any(s == 'MISSING' or s.startswith('Z') for s in active.values()):
                issues.append({'seed': seed, 'reason': 'Supervisor or worker needs attention', 'supervisor': sup})
        if state['phase'] == 'SELECTING' and event_age > 900:
            issues.append({'seed': seed, 'reason': 'No recorded stream/selection event for 15 minutes',
                           'last_event_age_seconds': round(event_age)})
        rows.append({'seed': seed, 'phase': state['phase'], 'completed_labels': verified[-1],
                     'verified_frozen_budgets': verified, 'verified_batch_count': batch_count,
                     'supervisor_status': sup['status'], 'recovery_count': sup['recovery_count'],
                     'processes': active, 'latest_event': latest,
                     'latest_event_age_seconds': round(event_age)})
    receipt = {'time': current.isoformat(), 'status': 'NEEDS_ATTENTION' if issues else 'HEALTHY',
               'issues': issues, 'rows': rows, 'test_truth_revealed': False}
    run.atomic_json(run.OUT/'heartbeat_latest.json', receipt)
    with (run.OUT/'monitor_audit.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'issues': issues,
        'rows': [{k: row[k] for k in ('seed', 'phase', 'completed_labels', 'supervisor_status',
                                      'recovery_count', 'latest_event_age_seconds')} for row in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
