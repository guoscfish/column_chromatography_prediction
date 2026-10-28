"""Read saved dual-expert receipts; no model calls or label access."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/revisions'
SPECS=[(157,'sol61_xhigh_seed157_from621_to813_20261014',9),(6101,'sol61_xhigh_seed6101_to813_20261014',0)]
rows=[]
for seed,revision,start in SPECS:
 p=BASE/revision;method='free_llm32_dual_expert_v1';latest=0;value=None
 for r in range(16):
  freeze=p/f'runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json'
  if freeze.exists():
   d=json.loads(freeze.read_text());a=Path(d['checkpoint_path']).parent/'fit_audit.json'
   if a.exists(): latest=r;value=json.loads(a.read_text())['best_validation_combined_normalized_rmse']
 directory=p/f'selections/seed_{seed}/{method}/round_{latest:02d}'
 accepted=[];errors=[]
 for role in ['chemistry_scientist','ml_scientist','scientific_planner']:
  q=directory/'agents'/role
  if (q/('selection.json' if role=='scientific_planner' else 'memo.json')).exists(): accepted.append(role)
  for f in sorted(q.glob('rejected_turn_*.json')): errors.append({'role':role,'file':f.name,'error':json.loads(f.read_text())['error']})
  for f in sorted(q.glob('retries/**/*.json')):
   d=json.loads(f.read_text());errors.append({'role':role,'file':str(f.relative_to(q)),'error':d.get('error_type'),'message':d.get('message','')[:300]})
 status_path=p/'execution_status.json'
 if status_path.exists():
  status=json.loads(status_path.read_text());status.update(latest_completed_labels=333+32*latest,latest_validation_NRMSE=value,completed_total_acquisition_rounds=latest,accepted_current_roles=accepted)
  status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 rows.append({'seed':seed,'revision':revision,'latest_completed_labels':333+32*latest,'round':latest,'validation_NRMSE':value,'accepted_current_roles':accepted,'current_errors':errors})
print(json.dumps(rows,ensure_ascii=False,indent=2))
(ROOT/'reports/dual_expert_two_seeds_to813_20261014/live_status.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
