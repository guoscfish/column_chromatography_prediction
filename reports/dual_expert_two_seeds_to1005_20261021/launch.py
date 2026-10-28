"""Preflight and detach two isolated controllers without exposing credentials."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'reports/dual_expert_two_seeds_to813_20261014'))
from watchdog import probe_provider
from src.qgeognn_al.active_learning_v2 import dual_expert_study as study

processes = subprocess.check_output(['ps', '-axo', 'pid,args'], text=True)
for line in processes.splitlines():
    if ('python' in line and ('/controller.py --seed' in line or
            'run_qgeognn_v2_row_llm_dual_expert.py run' in line or
            'run_alternating.py' in line or '/worker.py --seed' in line)):
        raise RuntimeError('An existing Dual Expert process needs inspection; no duplicate launch')

records = []
execution_epoch = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
for seed, revision, filename in (
    (157, 'sol61_xhigh_seed157_from621_to813_20261014', 'dual-expert-813.api-key'),
    (6101, 'sol61_xhigh_seed6101_to813_20261014', 'dual-expert-813-seed6101.api-key'),
):
    study.configure_revision(revision)
    validation = study.validate()
    freezes = sorted((study.STUDY/f'runtime/seed_{seed}/{study.METHOD}').glob('round_*/prediction_freeze.json'))
    frozen = json.loads(freezes[-1].read_text())
    for kind in ('checkpoint', 'prediction'):
        assert study.sha256_file(Path(frozen[f'{kind}_path'])) == frozen[f'{kind}_sha256']
    audit = json.loads((Path(frozen['checkpoint_path']).parent/'fit_audit.json').read_text())
    health = probe_provider(Path.home()/'.config/qgeognn-scientist'/filename,
                            'https://token4research.cn/v1')
    record = {'seed': seed, 'revision': revision, 'completed_labels': audit['train_rows'],
              'validation_nrmse': audit['best_validation_combined_normalized_rmse'],
              'frozen_validation': validation, 'provider_probe': health}
    if health.get('http_status') == 200 and health.get('requested_model_listed'):
        log = OUT/f'controller_seed_{seed}.log'
        env = os.environ.copy()
        env['MPLCONFIGDIR'] = str(OUT/'matplotlib_cache')
        env['DUAL_RECOVERY_EPOCH'] = execution_epoch
        with log.open('a') as handle:
            child = subprocess.Popen([sys.executable, '-u', str(OUT/'controller.py'), '--seed', str(seed)],
                                     cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=child.pid, log=str(log), status='STARTED')
    else:
        record['status'] = 'PROVIDER_PREFLIGHT_FAILED'
    records.append(record)

result = {'time': datetime.now().astimezone().isoformat(), 'target_labels': 1005,
          'recovery_epoch': execution_epoch,
          'user_requested_threshold': 1000, 'batch_size': 32,
          'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh',
          'renewed_execution_authorized_on': '2026-10-21',
          'history_preserved': True, 'old_recovery_receipts_preserved': True,
          'new_execution_max_transport_recoveries_per_round': 2,
          'extension': 'finish frozen 813 revisions; inherit at round15 into new 1005 revisions',
          'budget_override': 'worker sets study.BUDGETS=333..1005 only for the new revision',
          'scientific_source_changed': False,
          'wrapper_hashes': {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                             for name in ('worker.py', 'controller.py', 'launch.py')},
          'controllers': records}
previous = OUT/'launch.json'
if previous.exists():
    (OUT/f'launch_before_{execution_epoch}.json').write_bytes(previous.read_bytes())
previous.write_text(json.dumps(result, indent=2)+'\n')
with (OUT/'launch_history.jsonl').open('a') as handle:
    handle.write(json.dumps(result)+'\n')
print(json.dumps(result, indent=2))
