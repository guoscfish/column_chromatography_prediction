from pathlib import Path
import subprocess,sys,time,shutil,json
ROOT=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');P=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018';OUT=ROOT/'reports/llm_eight_to1549_20261018'
last=None
while True:
 if (OUT/'completion_audit.json').exists() and (OUT/'validation_progress.png').exists():
  try:
   assert json.loads((OUT/'completion_audit.json').read_text())['status']=='ALL_EIGHT_COMPLETE1549_VERIFIED'
   for ext in ('png','svg','pdf'):
    if (OUT/f'validation_progress.{ext}').exists():shutil.copyfile(OUT/f'validation_progress.{ext}',OUT/f'current/validation_progress.{ext}')
   print('Final verified1549 figure ready',flush=True);break
  except (ValueError,AssertionError):pass
 files=sorted((P/'runtime').glob('seed_*/*/round_*/prediction_freeze.json')) if (P/'runtime').exists() else []
 signature=tuple((str(f),f.stat().st_mtime_ns) for f in files)
 if signature!=last:
  result=subprocess.run([sys.executable,str(OUT/'plot_progress1549.py')],capture_output=True,text=True)
  if result.returncode==0:last=signature;print('Updated verified partial figure; frozen_new_fits='+str(len(files)),flush=True)
  else:print('Progress plot awaiting consistent frozen artifacts; exit='+str(result.returncode),flush=True)
 time.sleep(10)
