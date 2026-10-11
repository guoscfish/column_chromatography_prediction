"""Stop the identified worker after the current prediction freeze is written."""
import datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
pid=int(sys.argv[1])
target=p/'runtime/seed_6101/cw16_llm16_scientist_v2/round_09/prediction_freeze.json'
identity='resume_6101_hybrid_compatible.py'
def alive():
 r=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
 return r.returncode==0 and identity in r.stdout and 'python' in r.stdout.lower()
print(f'Waiting for PID {pid} to freeze 621 labels',flush=True)
while not target.exists():
 if not alive():
  print('Worker exited before target freeze; no signal sent',flush=True);sys.exit(2)
 time.sleep(.02)
if not alive():
 print('Worker already exited',flush=True);sys.exit(2)
os.kill(pid,signal.SIGSTOP)
# File visibility may precede the end of its write. Suspend only briefly until parse succeeds.
try:
 for attempt in range(100):
  try:
   freeze=json.loads(target.read_text());break
  except json.JSONDecodeError:
   os.kill(pid,signal.SIGCONT);time.sleep(.01);os.kill(pid,signal.SIGSTOP)
 else:raise RuntimeError('Freeze JSON remained incomplete')
 for kind in ('checkpoint','prediction'):
  q=Path(freeze[kind+'_path'])
  if hashlib.sha256(q.read_bytes()).hexdigest()!=freeze[kind+'_sha256']:raise RuntimeError('Freeze hash mismatch')
 a=json.loads((Path(freeze['checkpoint_path']).parent/'fit_audit.json').read_text())
 if a['train_rows']!=621:raise RuntimeError('Wrong final budget')
 result={'time':datetime.datetime.now().astimezone().isoformat(),'status':'STOPPED_AFTER_USER_REQUESTED_ROUND','seed':6101,'method':'cw16_llm16_scientist_v2','labels':621,'validation_nrmse':a['best_validation_combined_normalized_rmse'],'pid':pid,'prediction_freeze_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 for _ in range(100):
  if not alive():break
  time.sleep(.1)
 else:raise RuntimeError('Worker did not exit')
 (p/'extension_authorized_to813/stop_after_621_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
except BaseException:
 # Fail closed: a failed hash check must not let the worker enter another round.
 if alive():
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 raise
