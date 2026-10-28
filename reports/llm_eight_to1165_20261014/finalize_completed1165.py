import json,time,hashlib,sys,re
from pathlib import Path
ROOT=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');P=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1165_20261014';OUT=ROOT/'reports/llm_eight_to1165_20261014';OUT.mkdir(exist_ok=True)
while not (P/'result.json').exists():time.sleep(10)
sys.path.insert(0,str(P));import resume1165_compatible as runner
runner.guard_adapter()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
curves=json.loads((ROOT/'reports/llm_four_to1005_20261013/validation_curves.json').read_text());audits={};targets={}
extra=runner.read(P/'selections/seed_6101/cw16_llm16_scientist_v2/round_21/authorized_extra_correction.json')
assert extra['additional_logical_calls']==1
assert (P/'selections/seed_6101/cw16_llm16_scientist_v2/round_21/turn_03.json').exists()
for seed in (157,6101):
 for method in runner.METHODS:
  result=runner.read(P/f'{seed}_{method}_result.json');assert result['final_labels']==1165 and len(result['selected_ids'])==len(set(result['selected_ids']))==832 and len(result['additional_ids'])==len(set(result['additional_ids']))==160
  batches=[];newrows=[]
  for r in range(21,26):
   d=P/(f'selections/seed_{seed}/{method}/round_{r:02d}' if method in (runner.HYBRID,runner.FREE) else f'controls/seed_{seed}/{method}/round_{r:02d}')
   b=runner.read(d/'batch_freeze.json');assert len(b['batch_ids'])==len(set(b['batch_ids']))==32;batches.extend(b['batch_ids'])
   if method in (runner.HYBRID,runner.FREE):
    assert len(b['pending_ids'])==(16 if method==runner.HYBRID else 0)
    for name,h in b['artifact_hashes'].items():assert sha(d/name)==h
    t=runner.read(d/'model_transition.json');assert (t['model'],t['reasoning_effort'])==('gpt-6.1-sol','xhigh')
    for file in d.glob('turn_*.json'):
     if re.fullmatch(r'turn_\d\d.json',file.name):assert runner.read(file)['provenance']['served_model'].startswith('gpt-6.1-sol')
   f=P/f'runtime/seed_{seed}/{method}/round_{r+1:02d}/prediction_freeze.json';freeze=runner.read(f)
   for k in ('checkpoint','prediction'):assert sha(freeze[k+'_path'])==freeze[k+'_sha256']
   a=Path(freeze['checkpoint_path']).parent/'fit_audit.json';fit=runner.read(a);assert fit['train_rows']==333+32*(r+1) and fit['test_labels_used_for_fit_or_checkpoint_selection']==0
   newrows.append({'labels':fit['train_rows'],'nrmse':fit['best_validation_combined_normalized_rmse']});audits[f'{seed}:{method}:{r+1}']={'prediction_freeze':freeze,'prediction_freeze_sha256':sha(f),'fit_audit':fit,'fit_audit_sha256':sha(a)}
  assert batches==result['additional_ids']
  curves[str(seed)][method]=curves[str(seed)][method]+newrows
  targets[f'{seed}:{method}']={'labels':1165,'total_batches':26,'unique_new_ids_since333':832,'additional_batches':5,'additional_unique_ids_since1005':160,'validation_nrmse':newrows[-1]['nrmse']}
(OUT/'completion_audit.json').write_text(json.dumps({'status':'ALL_EIGHT_COMPLETE1165_VERIFIED','test_evaluation_performed':False,'classification':'exploratory two-seed validation; original mixed-model prefixes and authorized historical deviations retained','authorized_new_protocol_deviation':extra,'targets':targets,'frozen_fit_audits':audits},indent=2)+'\n')
(OUT/'validation_curves.json').write_text(json.dumps(curves,indent=2)+'\n')
source=(ROOT/'reports/llm_four_to1005_20261013/plot_completed1005.py').read_text();source=source.replace('range(333, 1006, 32)','range(333, 1166, 32)').replace('to 1005 |','to 1165 |').replace('325, 1020','325, 1180').replace('813, 909, 1005]','813, 909, 1005, 1101, 1165]');source=source.replace('Free6101 r0 and r19 (extra turn03).', 'Free6101 r0/r19; Hybrid6101 r21 (extra turn03).');(OUT/'plot_completed1165.py').write_text(source);exec(compile(source,str(OUT/'plot_completed1165.py'),'exec'),{'__file__':str(OUT/'plot_completed1165.py'),'__name__':'__main__'})
print('ALL_EIGHT1165_AUDIT_AND_FIGURE_READY',flush=True)
