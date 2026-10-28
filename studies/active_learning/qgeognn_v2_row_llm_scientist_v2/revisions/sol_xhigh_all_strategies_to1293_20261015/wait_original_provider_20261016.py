"""Bounded per-key circuit recovery while the user waits for the original provider.
At most one half-open original request every600seconds across both keys. Scientific
selection/training functions and correction limits remain in the immutable runner.
"""
import resume_routed_transport_20261016 as routed
import os,sys,time,json,subprocess,fcntl,httpx,hashlib
from pathlib import Path
P=routed.P;m=routed.m
manifest=m.read(P/'provider_wait_manifest_20261016.json')
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==manifest['controller_sha256']
assert m.sha(P/'resume_routed_transport_20261016.py')==manifest['routed_runner_sha256']
routed.guard()
interval=600;children={};blocked={};healthy=set();probes={};last_probe=max([m.read(f)['time'] for f in P.glob('transport_circuit_seed_*.json')]+[0]);last_seed=157;next_health=0
status={'started':time.time(),'policy':'one half-open original request per600seconds globally; per-key circuit; immutable science','workers':{},'events':[]}

def event(action,**data):
    row={'time':time.time(),'action':action,**data};status['events']=status['events'][-39:]+[row]
    with (P/'provider_wait_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def active():
    out={}
    for line in subprocess.check_output(['ps','-axo','pid,stat,args'],text=True).splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3 or parts[1].startswith('Z') or not parts[2].startswith(sys.executable):continue
        if str(P) not in parts[2] or '--worker' not in parts[2]:continue
        args=parts[2].split();i=args.index('--worker');out[(int(args[i+1]),args[i+2])]=int(parts[0])
    return out

def pending(seed):
    return [(seed,method) for method in (m.HYBRID,m.FREE) if not (P/f'{seed}_{method}_result.json').exists() and (seed,method) not in blocked]

def launch(key,half_open=False):
    if key in active():return
    seed,method=key
    bad=next((r for r in range(26,30) if (m.directory(seed,method,r)/'rejected_turn_02.json').exists() and not (m.directory(seed,method,r)/'batch_freeze.json').exists()),None)
    if bad is not None:blocked[key]='correction_budget_exhausted';event('requires_decision',seed=seed,method=method,round=bad);return
    routed.guard();path=P/f'provider_wait_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    with path.open('ab') as out:
        c=subprocess.Popen([sys.executable,'-u',str(P/'resume_routed_transport_20261016.py'),'--worker',str(seed),method],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
    children[key]=c;status['workers'][f'{seed}:{method}']={'pid':c.pid,'log':str(path),'half_open':half_open};event('worker_started',seed=seed,method=method,pid=c.pid,half_open=half_open)
    if half_open:probes[c.pid]=seed

def open_once(seed):
    history=P/'transport_circuit_history';history.mkdir(exist_ok=True)
    with (P/f'llm_seed_{seed}.lock').open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in (f'transport_circuit_seed_{seed}.json',f'routed_failure_count_seed_{seed}.json'):
            q=P/name
            if q.exists():q.rename(history/f'{time.time_ns()}_{name}')
        (P/f'routed_failure_count_seed_{seed}.json').write_text(json.dumps({'consecutive_failures':1,'time':time.time(),'policy':'HALF_OPEN: carry one historical failure credit; exactly one original paid request is permitted before reopening on failure.'})+'\n')
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)

while not (P/'result.json').exists():
    for key,c in list(children.items()):
        rc=c.poll()
        if rc is None:continue
        event('worker_exited',seed=key[0],method=key[1],pid=c.pid,exit_code=rc);del children[key]
        if rc not in (0,75):blocked[key]='quota' if rc==78 else 'requires_diagnosis';event('blocked_target',seed=key[0],method=key[1],exit_code=rc)
    a=active()
    if (P/'routed_transport_events.jsonl').exists():
        for line in (P/'routed_transport_events.jsonl').read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if r['action']=='complete' and r['pid'] in probes:
                seed=probes.pop(r['pid']);healthy.add(seed);event('model_service_answered',seed=seed,response_id=r.get('response_id'))
    for seed in (157,6101):
        if (P/f'quota_seed_{seed}.json').exists():
            healthy.discard(seed)
            for key in pending(seed):blocked[key]='quota_requires_user_resume'
            continue
        if (P/f'transport_circuit_seed_{seed}.json').exists():healthy.discard(seed)
        if seed in healthy:
            for key in pending(seed):
                if key not in a:launch(key)
    candidates=[]
    for seed in (157,6101):
        circuit=P/f'transport_circuit_seed_{seed}.json'
        if not circuit.exists() or (P/f'quota_seed_{seed}.json').exists() or not pending(seed):continue
        if any(k[0]==seed for k in a):continue
        record=m.read(circuit)
        if time.time()<max(record['time'],last_probe)+interval:continue
        terminals=[json.loads(l) for l in (P/'stream_terminal_errors.jsonl').read_text().splitlines() if l.strip()]
        overload=next((r for r in reversed(terminals) if r.get('seed')==seed and r['time']>=record['time']-10 and 'overloaded' in r.get('message','').lower()),None)
        if not overload:
            for key in pending(seed):blocked[key]='non_overload_transport_requires_diagnosis'
            event('requires_network_diagnosis',seed=seed);continue
        candidates.append(seed)
    if candidates and time.time()>=next_health:
        seed=next((s for s in candidates if s!=last_seed),candidates[0]);secret=Path(m.guard()['credential_files'][str(seed)]).read_text().strip()
        try:
            with httpx.Client(proxy=routed.PROXY,trust_env=False,timeout=15) as c:r=c.get(m.CONFIG['base_url']+'/models',headers={'Authorization':'Bearer '+secret})
            assert r.status_code==200 and any(x.get('id')==m.CONFIG['model'] for x in r.json().get('data',[]))
            open_once(seed);last_probe=time.time();last_seed=seed;launch(pending(seed)[0],half_open=True);event('half_open_probe',seed=seed,next_possible_probe_after=last_probe+interval)
        except Exception as error:event('health_gate_not_ready',seed=seed,error_type=type(error).__name__);next_health=time.time()+60
    status['updated_at']=time.time();status['blocked_targets']={f'{s}:{method}':reason for (s,method),reason in blocked.items()};status['healthy_seeds']=sorted(healthy);status['next_half_open_not_before']=last_probe+interval
    (P/'provider_wait_status.tmp').write_text(json.dumps(status,indent=2)+'\n');(P/'provider_wait_status.tmp').replace(P/'provider_wait_status_20261016.json')
    if not any(pending(seed) for seed in (157,6101)) and not a:break
    time.sleep(10)
