from pathlib import Path
import json,hashlib
root=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');p=root/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018';out=root/'reports/llm_eight_to1549_20261018/current';out.mkdir(exist_ok=True)
curves=json.loads((root/'reports/llm_eight_to1421_20261017/validation_curves.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
for seed,methods in curves.items():
 for method,rows in methods.items():
  for r in range(35,39):
   freeze=p/f'runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json'
   if not freeze.exists():break
   f=json.loads(freeze.read_text())
   for k in ('checkpoint','prediction'):assert sha(f[k+'_path'])==f[k+'_sha256']
   a=json.loads((Path(f['checkpoint_path']).parent/'fit_audit.json').read_text());assert a['train_rows']==333+32*r and a['test_labels_used_for_fit_or_checkpoint_selection']==0
   rows.append({'labels':a['train_rows'],'nrmse':a['best_validation_combined_normalized_rmse']})
(out/'validation_curves.json').write_text(json.dumps(curves,indent=2)+'\n')
s=(root/'reports/llm_eight_to1421_20261017/plot_completed1421.py').read_text()
s=s.replace('list(range(333, 1422, 32))',"list(range(333, rows[-1]['labels'] + 1, 32))").replace('to 1421 | verified completed fits','toward 1549 | completed fits; in progress').replace('325, 1440','325, 1568').replace('1293, 1421]','1293, 1421, 1549]')
(out/'plot_current.py').write_text(s);exec(compile(s,str(out/'plot_current.py'),'exec'),{'__file__':str(out/'plot_current.py'),'__name__':'__main__'})
