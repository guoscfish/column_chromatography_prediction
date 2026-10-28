"""Bounded recovery of original pending requests; no new scientific calls/budgets."""
import sys,os,time,json,subprocess,fcntl
from pathlib import Path
import httpx
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));import resume_routed_transport_20261016 as r
m=r.m;r.guard()
assert r.sha(__file__)==m.read(P/'recovery_monitor_manifest.json')['source_sha256']
counts={157:0,6101:0};last_recovery={157:0,6101:0};seen_complete={157:0,6101:0};blocked={};children={}
def event(action,**data):
 row={'time':time.time(),'action':action,**data}
 with (P/'recovery_monitor_events.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def active():
 result=set()
 for line in subprocess.check_output(['ps','-axo','pid,stat,args'],text=True).splitlines():
  parts=line.strip().split(None,2)
  if len(parts)!=3 or parts[1].startswith('Z') or not parts[2].startswith(sys.executable) or str(P) not in parts[2] or '--worker' not in parts[2]:continue
  args=parts[2].split();i=args.index('--worker');result.add((int(args[i+1]),args[i+2]))
 return result
def pending(seed):
 result=[]
 order=(m.FREE,m.HYBRID) if seed==157 else (m.HYBRID,m.FREE)
 for method in order:
  key=(seed,method)
  if (P/f'{seed}_{method}_result.json').exists() or key in blocked:continue
  if any((m.directory(seed,method,n)/'rejected_turn_02.json').exists() and not (m.directory(seed,method,n)/'batch_freeze.json').exists() for n in range(30,34)):
   blocked[key]='logical_budget_exhausted';event('requires_decision',seed=seed,method=method);continue
  result.append(key)
 return result
def launch(key):
 if key in active():return
 seed,method=key;r.guard()
 entry='resume_priority_free157_1421.py' if key==(157,m.FREE) else 'resume_routed_diagnostic_20261017.py'
 log=P/f'monitored_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
 with log.open('ab') as out:c=subprocess.Popen([sys.executable,'-u',str(P/entry),'--worker',str(seed),method],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
 children[key]=c;event('worker_started',seed=seed,method=method,pid=c.pid,log=str(log))
class HealthGateError(Exception):
 def __init__(self,message,retry_after=0):super().__init__(message);self.retry_after=retry_after

def check_health_error(response,seed):
 if any(code in response.text for code in ('DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','insufficient_quota')):
  m.once(P/f'quota_seed_{seed}.json',{'seed':seed,'status':'QUOTA_EXHAUSTED','time':time.time(),'source':'original endpoint health gate'})
  event('quota_exhausted',seed=seed);raise HealthGateError('quota_stop')
 if response.status_code==429:
  value=response.headers.get('Retry-After','600')
  try:seconds=max(0,float(value))
  except ValueError:
   from email.utils import parsedate_to_datetime
   seconds=max(0,parsedate_to_datetime(value).timestamp()-time.time())
  raise HealthGateError('ordinary_rate_limit',seconds)

while not (P/'result.json').exists():
 for key,c in list(children.items()):
  rc=c.poll()
  if rc is None:continue
  event('worker_exited',seed=key[0],method=key[1],pid=c.pid,exit_code=rc);del children[key]
  if rc not in (0,75):blocked[key]='quota' if rc==78 else 'non_transport_requires_diagnosis'
 if (P/'routed_transport_events.jsonl').exists():
  for line in (P/'routed_transport_events.jsonl').read_text().splitlines():
   try:row=json.loads(line)
   except ValueError:continue
   if row['action']=='complete' and row['time']>seen_complete[row['seed']]:
    seed=row['seed'];seen_complete[seed]=row['time']
    if row['time']>last_recovery[seed]:counts[seed]=0
 live=active()
 for seed in (157,6101):
  targets=pending(seed)
  if not targets or (P/f'quota_seed_{seed}.json').exists():continue
  circuit=P/f'transport_circuit_seed_{seed}.json'
  if circuit.exists():
   if counts[seed]>=3:
    for key in targets:blocked[key]='three_failed_recovery_requests_require_diagnosis'
    event('bounded_recovery_stopped',seed=seed);continue
   if time.time()<max(m.read(circuit)['time'],last_recovery[seed])+600:continue
   try:
    with (P/f'llm_seed_{seed}.lock').open('a+') as lock:
     fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
     secret=Path(m.guard()['credential_files'][str(seed)]).read_text().strip()
     with httpx.Client(proxy=r.PROXY,trust_env=False,timeout=20) as client:
      response=client.get(m.CONFIG['base_url']+'/models',headers={'Authorization':'Bearer '+secret})
      check_health_error(response,seed)
      assert response.status_code==200
      assert any(x.get('id')==m.CONFIG['model'] for x in response.json().get('data',[]))
      post=client.post(m.CONFIG['base_url']+'/responses',headers={'Authorization':'Bearer '+secret},json={})
      check_health_error(post,seed)
      assert post.status_code==400 and 'model is required' in post.text
     r.guard();history=P/'transport_circuit_history';history.mkdir(exist_ok=True)
     for name in (f'transport_circuit_seed_{seed}.json',f'routed_failure_count_seed_{seed}.json'):
      f=P/name
      if f.exists():f.rename(history/f'{time.time_ns()}_{name}')
     (P/f'routed_failure_count_seed_{seed}.json').write_text(json.dumps({'time':time.time(),'consecutive_failures':1,'policy':'one bounded half-open original pending request after original endpoint auth and empty POST checks; no logical budget expansion'})+'\n')
     counts[seed]+=1;last_recovery[seed]=time.time();event('diagnosed_half_open',seed=seed,recovery_number=counts[seed],auth_status=200,empty_post_status=400)
    missing=[key for key in targets if key not in active()]
    if missing:launch(missing[0])
   except BlockingIOError:pass
   except Exception as error:
    last_recovery[seed]=time.time()+max(0,getattr(error,'retry_after',0)-600);event('health_gate_not_ready',seed=seed,error_type=type(error).__name__,retry_after=getattr(error,'retry_after',None))
  else:
   counter=P/f'routed_failure_count_seed_{seed}.json'
   if counter.exists() and m.read(counter).get('consecutive_failures')==0:
    for key in targets:
     if key not in active():launch(key)
 status={'time':time.time(),'failed_recoveries':counts,'blocked':{f'{s}:{method}':reason for (s,method),reason in blocked.items()},'last_recovery':last_recovery,'policy':'600seconds between failed half-open attempts; max3 consecutive failed full recoveries per key; quota/logical exhaustion stop immediately; max1 call/key and2 fits unchanged'}
 (P/'recovery_monitor_status.tmp').write_text(json.dumps(status,indent=2)+'\n');(P/'recovery_monitor_status.tmp').replace(P/'recovery_monitor_status.json')
 time.sleep(10)
