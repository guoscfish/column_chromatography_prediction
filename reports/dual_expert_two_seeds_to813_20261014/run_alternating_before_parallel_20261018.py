"""Resume two frozen studies one batch at a time, with one recovery per network failure."""
import json,subprocess,sys,time,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from watchdog import run_monitored, recent_failures, probe_provider, MAX_RECOVERIES
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/revisions'
PYTHON='/Users/fish/miniforge3/envs/fish/bin/python'

# Task-local authentication only; never fall back to another experiment's key.
TASK_KEY_FILE=Path.home()/'.config/qgeognn-scientist/dual-expert-813.api-key'
if not TASK_KEY_FILE.is_file():raise RuntimeError('Task-specific API credential is missing')
os.environ['SCIENTIST_API_KEY_FILE']=str(TASK_KEY_FILE)
os.environ['SCIENTIST_API_KEY']=TASK_KEY_FILE.read_text().strip()
if not os.environ['SCIENTIST_API_KEY']:raise RuntimeError('Task-specific API credential is empty')

SPECS=[(6101,'sol61_xhigh_seed6101_to813_20261014',0),(157,'sol61_xhigh_seed157_from621_to813_20261014',9)]
METHOD='free_llm32_dual_expert_v1'
def event(**record):
 record={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**record}
 with (OUT/'execution_events.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
 print(json.dumps(record,ensure_ascii=False),flush=True)
def latest(seed,revision):
 p=BASE/revision/f'runtime/seed_{seed}/{METHOD}'
 return max([r for r in range(16) if (p/f'round_{r:02d}/prediction_freeze.json').exists()],default=0)
def status(revision,**changes):
 p=BASE/revision/'execution_status.json';d=json.loads(p.read_text());d.update(changes);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def resume_priority(spec):
 directory=BASE/spec[1]/f'selections/seed_{spec[0]}/{METHOD}/round_{latest(spec[0],spec[1]):02d}'
 return (not (directory/'batch_freeze.json').exists(),-len(list((directory/'agents').glob('*/memo.json'))))
SPECS.sort(key=resume_priority)
blocked={};recoveries={};quota=False
while any(latest(seed,revision)<15 and seed not in blocked for seed,revision,start in SPECS) and not quota:
 for seed,revision,start in SPECS:
  r=latest(seed,revision)
  if r>=15 or seed in blocked:continue
  key=(seed,r)
  command=[PYTHON,'-u','scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py','run','--seed',str(seed),'--rounds',str(r+1),'--start-round',str(start),'--revision',revision,'--model','gpt-6.1-sol','--reasoning-effort','xhigh']
  log=OUT/f'seed_{seed}_through_round_{r+1:02d}.log'
  event(event='batch_started',seed=seed,from_labels=333+32*r,to_labels=333+32*(r+1),revision=revision,log=str(log))
  status(revision,status='RUNNING',phase=f'round_{r:02d}_selection_or_fit')
  log_offset=log.stat().st_size if log.exists() else 0
  directory=BASE/revision/f'selections/seed_{seed}/{METHOD}/round_{r:02d}'
  base_url=json.loads((BASE/revision/'protocol.json').read_text())['transport']['base_url']
  monitored=run_monitored(command,cwd=ROOT,log=log,directory=directory,
      probe=lambda:probe_provider(TASK_KEY_FILE,base_url),
      emit=lambda **data:event(seed=seed,round=r,**data))
  code=monitored['returncode']
  if code==0:
   assert latest(seed,revision)==r+1
   status(revision,latest_completed_labels=333+32*(r+1),completed_total_acquisition_rounds=r+1,status='COMPLETE' if r+1==15 else 'RUNNING')
   event(event='batch_complete',seed=seed,labels=333+32*(r+1));continue
  errors=recent_failures(directory,monitored['started_at'])
  if any('DAILY_LIMIT_EXCEEDED' in e.get('message','') for e in errors):
   quota=True;blocked[seed]='BLOCKED_PROVIDER_DAILY_LIMIT';event(event='quota_exhausted',seed=seed);break
  with log.open('rb') as f:f.seek(log_offset);body=f.read().decode('utf-8',errors='replace')
  network=monitored['watchdog_timeout'] or ('failed after 3 attempts' in body and any(e.get('error_type') in ['RemoteProtocolError','APIConnectionError','APITimeoutError','ReadTimeout','RuntimeError','InternalServerError'] for e in errors))
  if network and recoveries.get(key,0)<MAX_RECOVERIES:
   recoveries[key]=recoveries.get(key,0)+1;status(revision,status='RECOVERING_PROVIDER_CONNECTION');event(event='bounded_network_recovery',seed=seed,round=r,attempt=recoveries[key],watchdog_timeout=monitored['watchdog_timeout']);time.sleep(30)
  else:
   blocked[seed]='BLOCKED_PROVIDER_CONNECTION' if network else 'BLOCKED_VALIDATION_OR_PROTOCOL';status(revision,status=blocked[seed],last_failed_round=r);event(event='seed_blocked',seed=seed,status=blocked[seed],round=r)
results=[]
for seed,revision,start in SPECS:
 r=latest(seed,revision);state='COMPLETE' if r==15 else ('BLOCKED_PROVIDER_DAILY_LIMIT' if quota else blocked.get(seed,'INCOMPLETE'))
 status(revision,status=state,latest_completed_labels=333+32*r,completed_total_acquisition_rounds=r,target_labels_completed=r==15,process_running=False)
 results.append({'seed':seed,'revision':revision,'status':state,'labels':333+32*r,'rounds':r})
(OUT/'run_result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');event(event='run_finished',results=results)
sys.exit(0 if all(r['rounds']==15 for r in results) else 1)
