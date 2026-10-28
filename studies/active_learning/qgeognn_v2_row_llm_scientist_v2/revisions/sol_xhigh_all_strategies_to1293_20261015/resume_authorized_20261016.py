"""Resume remaining four LLM targets after explicit renewed user authorization."""
import resume_upstream_http2_compatible as adapter
import os,sys,time,json,subprocess,httpx
from pathlib import Path
m=adapter.runner;P=m.HERE
record=m.read(P/'resume_authorization_20261016.json')
assert m.sha(__file__)==record['controller_sha256']
assert m.sha(P/'resume_upstream_http2_compatible.py')==record['adapter_sha256']
adapter.guard()
status={'authorization_date':'2026-10-16','auth':{},'workers':{},'events':[]};children={};attempted=set();blocked=set();next_check={157:0,6101:0}
auth=m.read(P/'auth_preflight.json')
for row in auth['results']:
    if row['status']==200 and row.get('requested_model_available'):status['auth'][str(row['seed'])]='verified'

def event(action,**fields):
    item={'time':time.time(),'action':action,**fields};status['events']=status['events'][-49:]+[item]
    with (P/'resume_monitor_20261016.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')

def running():
    rows={}
    for line in subprocess.check_output(['ps','-axo','pid,stat,args']).decode().splitlines():
        fields=line.strip().split(None,2)
        if len(fields)!=3 or fields[1].startswith('Z'):continue
        command=fields[2]
        if not command.startswith(sys.executable) or str(P) not in command or '--worker' not in command:continue
        args=command.split();i=args.index('--worker');rows[int(fields[0])]={'seed':int(args[i+1]),'method':args[i+2]}
    return rows

def permit(seed):
    q=P/f'quota_seed_{seed}.json'
    if q.exists():
        history=P/'quota_history';history.mkdir(exist_ok=True)
        q.rename(history/f'{time.time_ns()}_quota_seed_{seed}.json')
    event('auth_restored_quota_history_preserved',seed=seed)

for seed in (157,6101):
    if status['auth'].get(str(seed))=='verified':permit(seed)
while True:
    for key,child in list(children.items()):
        code=child.poll()
        if code is None:continue
        seed,method=key;status['workers'][f'{seed}:{method}']['exit_code']=code;event('worker_exited',seed=seed,method=method,exit_code=code);del children[key]
        if code==78:
            blocked.add(seed);status['auth'][str(seed)]='quota_exhausted_requires_user_resume'
            for (other_seed,other_method),other in children.items():
                if other_seed==seed and other.poll() is None:other.terminate()
        elif code!=0:status['workers'][f'{seed}:{method}']['requires_diagnosis']=True
    active=running()
    for seed in (157,6101):
        if seed in blocked:continue
        if status['auth'].get(str(seed))!='verified' and time.time()>=next_check[seed]:
            secret=Path(record['credential_files'][str(seed)]).read_text().strip()
            try:
                response=httpx.get(m.CONFIG['base_url']+'/models',headers={'Authorization':'Bearer '+secret},timeout=httpx.Timeout(15,connect=10))
                try:data=response.json()
                except ValueError:data={}
                code=str(data.get('error',{}).get('code',''));message=str(data.get('error',{}).get('message',''))
                if response.status_code==200 and any(x.get('id')=='gpt-6.1-sol' for x in data.get('data',[])):
                    status['auth'][str(seed)]='verified';permit(seed)
                elif any(x in code+' '+message for x in ('DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','insufficient_quota')):
                    blocked.add(seed);status['auth'][str(seed)]='quota_exhausted_requires_user_resume';event('auth_quota',seed=seed,status=response.status_code,code=code)
                else:status['auth'][str(seed)]='auth_pending';event('auth_pending',seed=seed,status=response.status_code,code=code)
            except Exception as error:status['auth'][str(seed)]='connection_pending';event('auth_connection_pending',seed=seed,error_type=type(error).__name__)
            next_check[seed]=time.time()+60
        if status['auth'].get(str(seed))!='verified':continue
        for method in (m.HYBRID,m.FREE):
            key=seed,method
            if key in attempted or (P/f'{seed}_{method}_result.json').exists():continue
            if any(r['seed']==seed and r['method']==method for r in active.values()):continue
            bad=next((r for r in range(26,30) if (m.directory(seed,method,r)/'rejected_turn_02.json').exists() and not (m.directory(seed,method,r)/'batch_freeze.json').exists()),None)
            if bad is not None:
                attempted.add(key);status['workers'][f'{seed}:{method}']={'requires_decision_round':bad};event('correction_budget_exhausted',seed=seed,method=method,round=bad);continue
            adapter.guard()
            log=P/f'resumed_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
            with log.open('ab') as out:child=subprocess.Popen([sys.executable,'-u',str(P/'resume_upstream_http2_compatible.py'),'--worker',str(seed),method],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
            children[key]=child;attempted.add(key);status['workers'][f'{seed}:{method}']={'pid':child.pid,'log':str(log)};event('worker_started',seed=seed,method=method,pid=child.pid)
    status['updated_at']=time.time();(P/'resume_status_20261016.json').write_text(json.dumps(status,indent=2)+'\n')
    if (P/'result.json').exists():break
    if len(blocked)==2 and not children:break
    time.sleep(5)
