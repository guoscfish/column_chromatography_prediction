import json,time,hashlib,sys,re
from pathlib import Path
ROOT=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');P=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1293_20261015';OUT=ROOT/'reports/llm_eight_to1293_20261015';OUT.mkdir(exist_ok=True)
while not (P/'result.json').exists():time.sleep(10)
sys.path.insert(0,str(P));import continue_eight_to1293 as runner
runner.guard()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
curves=json.loads((ROOT/'reports/llm_eight_to1165_20261014/validation_curves.json').read_text());audits={};targets={}
extra=runner.read(runner.PREVIOUS/'selections/seed_6101/cw16_llm16_scientist_v2/round_21/authorized_extra_correction.json')
assert extra['additional_logical_calls']==1
assert (runner.PREVIOUS/'selections/seed_6101/cw16_llm16_scientist_v2/round_21/turn_03.json').exists()
for seed in (157,6101):
 for method in runner.METHODS:
  result=runner.read(P/f'{seed}_{method}_result.json');assert result['final_labels']==1293 and len(result['selected_ids'])==len(set(result['selected_ids']))==960 and len(result['additional_ids'])==len(set(result['additional_ids']))==128
  batches=[];newrows=[]
  for r in range(26,30):
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
  targets[f'{seed}:{method}']={'labels':1293,'total_batches':30,'unique_new_ids_since333':960,'additional_batches':4,'additional_unique_ids_since1165':128,'validation_nrmse':newrows[-1]['nrmse']}
(OUT/'completion_audit.json').write_text(json.dumps({'status':'ALL_EIGHT_COMPLETE1293_VERIFIED','test_evaluation_performed':False,'classification':'exploratory two-seed validation; original mixed-model prefixes and authorized historical deviations retained','inherited_authorized_protocol_deviation':extra,'new_extra_corrections_authorized':False,'targets':targets,'frozen_fit_audits':audits},indent=2)+'\n')
(OUT/'validation_curves.json').write_text(json.dumps(curves,indent=2)+'\n')
source=(ROOT/'reports/llm_eight_to1165_20261014/plot_completed1165.py').read_text()
source=source.replace('range(333, 1166, 32)','range(333, 1294, 32)').replace('to 1165 |','to 1293 |').replace('325, 1180','325, 1310').replace('1101, 1165]','1101, 1197, 1293]')
(OUT/'plot_completed1293.py').write_text(source)
exec(compile(source,str(OUT/'plot_completed1293.py'),'exec'),{'__file__':str(OUT/'plot_completed1293.py'),'__name__':'__main__'})
print('ALL_EIGHT1293_AUDIT_AND_FIGURE_READY',flush=True)
