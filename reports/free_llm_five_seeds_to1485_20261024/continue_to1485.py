"""Continue three frozen Free LLM trajectories to the old seeds' completed round 36."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve()
OUT = HERE.parent
ROOT = HERE.parents[2]
PREFIX_OUT = ROOT/'reports/free_llm_five_seeds_to1005_20261023'
sys.path.insert(0, str(PREFIX_OUT))
import parallel_resume as old_adapter
import continue_20261024 as old_control

base = old_adapter.base
s = base.s
base.activate()
PREFIX = base.STUDY
PREFIX_PROTOCOL = s.read(PREFIX/'protocol.json')
REVISION = 'sol61_xhigh_free_five_seeds_to1485_20261024'
STUDY = PREFIX.parent/REVISION
SEEDS = base.SEEDS
FREE = base.FREE
KEYS = old_adapter.KEYS
START, END = 21, 36
TARGET = 333+32*END
BUDGETS = tuple(range(333, TARGET+1, 32))
ORIGINAL_PREDICTION = s.prediction_freeze
PROTOCOL = STUDY/'protocol.json'
SESSION = OUT/'resume_session_latest.json'
PROBE = OUT/'resume_probe_latest.json'


def sha(path):
    return s.sha256_file(Path(path))


def runtime(seed, r):
    owner = PREFIX if r <= START else STUDY
    return owner/f'runtime/seed_{seed}/{FREE}/round_{r:02d}'


def directory(seed, method, r):
    if seed not in SEEDS or method != FREE or r not in range(END):
        raise ValueError('Unregistered continuation seed/method/round')
    owner = PREFIX if r < START else STUDY
    return owner/f'selections/seed_{seed}/{method}/round_{r:02d}'


def validate():
    record = s.read(PROTOCOL)
    if record['runner_sha256'] != sha(HERE) or record['code_hashes'] != s.code_hashes():
        raise RuntimeError('Frozen source or runner drift')
    if record['transport'] != base.CONFIG or record['training'] != s.TRAINING_CONFIG:
        raise RuntimeError('Frozen configuration drift')
    if record['prompt_sha256'] != s.stable_hash(s.SYSTEM_PROMPT) or (STUDY/'selector_prompt.txt').read_text() != s.SYSTEM_PROMPT:
        raise RuntimeError('Frozen prompt drift')
    for name, digest in record['protected_files'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Frozen prefix artifact drift: '+name)
    return {'status':'VALID','protocol_sha256':s.stable_hash(record)}


def prediction(context, method, r, labeled):
    if r > START:
        return ORIGINAL_PREDICTION(context, method, r, labeled)
    frozen = s.read(runtime(context.seed,r)/'prediction_freeze.json')
    for kind in ('checkpoint','prediction'):
        if sha(frozen[kind+'_path']) != frozen[kind+'_sha256']:
            raise RuntimeError('Frozen prefix predictor drift')
    if frozen['L_t_ids_hash'] != s.ids_hash(context.ids(labeled)):
        raise RuntimeError('Frozen prefix labeled identity drift')
    # New gradient caches belong to this continuation, leaving the prefix intact.
    scratch = context.runtime/method/f'round_{r:02d}'
    scratch.mkdir(parents=True, exist_ok=True)
    return Path(frozen['checkpoint_path']), scratch


@contextmanager
def training_slot():
    chosen = None
    while chosen is None:
        with base.lock(OUT/'training_allocation.lock'):
            for i in range(3):
                handle = (OUT/f'training_slot_{i}.lock').open('a+')
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                chosen = handle
                break
        if chosen is None:
            time.sleep(1)
    try:
        yield
    finally:
        fcntl.flock(chosen.fileno(),fcntl.LOCK_UN)
        chosen.close()


def status(seed, phase, r, **fields):
    base.atomic_json(OUT/f'status_seed_{seed}.json', {
        'updated_at':base.now(),'seed':seed,'phase':phase,'completed_rounds':r,
        'completed_labels':BUDGETS[r],'target_labels':TARGET,**fields})


def activate(seed=None):
    validate()
    s.STUDY = STUDY
    s.BUDGETS = BUDGETS
    s.SEEDS = SEEDS
    s.selection_directory = directory
    s.prediction_freeze = prediction
    s.validate = validate
    base.OUT = OUT
    base.STUDY = STUDY
    base.KEYS = KEYS
    base.BUDGETS = BUDGETS
    base.status = status
    if seed is not None:
        import httpx
        routes = s.read(OUT/'launch.json')['routes']
        route = routes[str(seed)]['proxy']
        base.client_for = lambda _old_proxy=None, timeout=600: httpx.Client(proxy=route,trust_env=False,timeout=timeout)
        base.transport.responses_call = lambda messages, config: base.responses_call(seed,messages,config)
        os.environ['SCIENTIST_API_KEY_FILE'] = str(KEYS[seed])
        os.environ.pop('SCIENTIST_API_KEY',None)


def audit_seed(seed, require_complete=False):
    activate()
    initial = s.read(s.historical_round0_paths(seed)['member0_audit'])
    rows, files = [], {}
    for r, budget in enumerate(BUDGETS):
        path = runtime(seed,r)/'prediction_freeze.json'
        if not path.exists():
            if require_complete or r <= START:
                raise RuntimeError('Missing frozen fit: '+str(path))
            break
        frozen = s.read(path)
        for kind in ('checkpoint','prediction'):
            if sha(frozen[kind+'_path']) != frozen[kind+'_sha256']:
                raise RuntimeError('Frozen predictor integrity failure')
        fit = s.read(Path(frozen['checkpoint_path']).parent/'fit_audit.json')
        assert fit['train_rows'] == budget
        assert fit['initialization_hash'] == initial['initialization_hash']
        assert fit['validation_ids_hash'] == initial['validation_ids_hash']
        assert fit['test_labels_used_for_fit_or_checkpoint_selection'] == frozen['test_truth_access_count'] == 0
        rows.append({'labels':budget,'validation_nrmse':fit['best_validation_combined_normalized_rmse']})
        files[str(path)] = sha(path)
    context = s.Context(seed)
    labeled,*_ = s.state(context,FREE,len(rows)-1)
    assert len(labeled) == rows[-1]['labels']
    ids = []
    for r in range(len(rows)-1):
        path = directory(seed,FREE,r)
        batch = s.audit_batch(path)
        assert len(batch['batch_ids']) == 32 and not batch['pending_ids']
        ids.extend(batch['batch_ids'])
        for receipt in path.glob('turn_*.json'):
            if '.started.' in receipt.name:
                continue
            model = s.read(receipt)['provenance']['served_model']
            assert model == 'gpt-6.1-sol' or model.startswith('gpt-6.1-sol-')
        for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):
            files[str(path/name)] = sha(path/name)
    assert len(ids) == len(set(ids)) == 32*(len(rows)-1)
    return {'seed':seed,'method':FREE,'final_labels':rows[-1]['labels'],
            'selected_ids':ids,'validation_learning_curve':rows,'files':files,'test_evaluation_performed':False}


def work(seed):
    activate(seed)
    with s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'):
        try:
            for r in range(START,END):
                if (runtime(seed,r+1)/'prediction_freeze.json').exists():
                    s.state(s.Context(seed),FREE,r+1)
                    continue
                status(seed,'SELECTING',r)
                base.event(seed,'batch_started',from_labels=BUDGETS[r],to_labels=BUDGETS[r+1])
                with s.status_phase(f'Free seed{seed} select {BUDGETS[r]}->{BUDGETS[r+1]}'):
                    s.select(seed,FREE,r)
                status(seed,'WAITING_FOR_TRAINING',r)
                with training_slot():
                    status(seed,'TRAINING',r,training_labels=BUDGETS[r+1])
                    with s.status_phase(f'Free seed{seed} train {BUDGETS[r+1]}'):
                        s.advance(seed,FREE,r)
                base.event(seed,'batch_complete',labels=BUDGETS[r+1])
            result = audit_seed(seed,True)
            assert result['final_labels'] == TARGET
            s.once(STUDY/f'{seed}_{FREE}_result.json',result)
            status(seed,'COMPLETE',END)
        except BaseException as exc:
            traceback.print_exc()
            completed = max([START]+[r for r in range(START+1,END+1) if (runtime(seed,r)/'prediction_freeze.json').exists()])
            status(seed,'STOPPED',completed,error_type=type(exc).__name__,reason='QUOTA_EXHAUSTED' if isinstance(exc,SystemExit) and exc.code==78 else 'See current worker log')
            raise


def prepare():
    protected = {str((PREFIX/'protocol.json').relative_to(ROOT)):sha(PREFIX/'protocol.json')}
    protected.update(PREFIX_PROTOCOL['source_hashes'])
    for path in (Path(base.__file__),Path(old_adapter.__file__),Path(old_control.__file__)):
        protected[str(path.relative_to(ROOT))] = sha(path)
    matched = []
    old = PREFIX.parent/'sol_xhigh_all_strategies_to1549_20261018'
    for seed in (157,6101):
        paths = sorted((old/f'runtime/seed_{seed}/{FREE}').glob('round_*/prediction_freeze.json'))
        assert int(paths[-1].parent.name.split('_')[1]) == END
        frozen = s.read(paths[-1])
        for kind in ('checkpoint','prediction'):
            assert sha(frozen[kind+'_path']) == frozen[kind+'_sha256']
        fit = s.read(Path(frozen['checkpoint_path']).parent/'fit_audit.json')
        assert fit['train_rows'] == TARGET and frozen['test_truth_access_count'] == 0
        protected[str(paths[-1].relative_to(ROOT))] = sha(paths[-1])
        matched.append({'seed':seed,'completed_round':END,'completed_labels':TARGET,'prediction_freeze':str(paths[-1])})
    for seed in SEEDS:
        path = PREFIX/f'{seed}_{FREE}_result.json'
        result = s.read(path)
        assert result['final_labels']==1005 and len(result['validation_learning_curve'])==22
        assert len(result['selected_ids'])==len(set(result['selected_ids']))==672
        protected[str(path.relative_to(ROOT))] = sha(path)
        for name,digest in result['files'].items():
            path = PREFIX/name
            assert sha(path)==digest
            protected[str(path.relative_to(ROOT))] = digest
    STUDY.mkdir(parents=True,exist_ok=True)
    record = {'authorization_date':'2026-10-24','user_request':'Continue to the same completed rounds as the previous two Free LLM seeds',
        'matched_reference_trajectories':matched,'seeds':list(SEEDS),'methods':[FREE],'budgets':list(BUDGETS),
        'prefix':str(PREFIX),'prefix_completed_round':START,'additional_rounds':END-START,'target_labels':TARGET,
        'transport':base.CONFIG,'training':s.TRAINING_CONFIG,'training_concurrency':3,'LLM_request_concurrency':3,
        'model_call_budget':base.transport.FULL_SELECTION_ATTEMPTS,'logical_budget_changed':False,
        'candidate_delivery':PREFIX_PROTOCOL['candidate_delivery'],'selector_boundary':PREFIX_PROTOCOL['selector_boundary'],
        'truth_barrier':PREFIX_PROTOCOL['truth_barrier'],'test_evaluation_performed':False,
        'prompt_sha256':s.stable_hash(s.SYSTEM_PROMPT),'code_hashes':s.code_hashes(),
        'runner_sha256':sha(HERE),'protected_files':protected}
    s.once(PROTOCOL,record)
    s.once(OUT/'extension_manifest.json',record)
    prompt = STUDY/'selector_prompt.txt'
    if prompt.exists():
        assert prompt.read_text()==s.SYSTEM_PROMPT
    else:
        prompt.write_text(s.SYSTEM_PROMPT)
    activate()
    for seed in SEEDS:
        result=audit_seed(seed)
        if not (OUT/f'status_seed_{seed}.json').exists():
            status(seed,'READY',len(result['validation_learning_curve'])-1)
    print(json.dumps({'status':'PREPARED','target_labels':TARGET,'target_round':END,'additional_rounds':END-START,'prefix_integrity':'PASS'}),flush=True)


def probe():
    validate()
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows=list(executor.map(old_control.check_seed,SEEDS))
    record={'time':base.now(),'model':base.CONFIG['model'],'reasoning_effort':base.CONFIG['reasoning_effort'],'seeds':rows}
    base.atomic_json(PROBE,record)
    print(json.dumps(record),flush=True)


def supervise(seed):
    activate(seed)
    session=s.read(SESSION)
    assert session['runner_sha256']==sha(HERE)
    state_path=OUT/f'supervisor_seed_{seed}.json'
    with s.exclusive_lock(OUT/f'supervisor_seed_{seed}.lock'):
        with s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'):
            pass
        previous=s.read(state_path) if state_path.exists() else {}
        failures=previous.get('session_recovery_count',0)
        counts=previous.get('failures_per_round',{})
        prefix=session['historical_recovery_counts'][str(seed)]
        while True:
            log=OUT/f'worker_seed_{seed}_{session["session_id"]}.log'
            saved={'seed':seed,'session_id':session['session_id'],'time':base.now(),'log':str(log),
                   'recovery_count':prefix+failures,'session_recovery_count':failures,'failures_per_round':counts}
            if (STUDY/f'{seed}_{FREE}_result.json').exists():
                base.atomic_json(state_path,{'status':'COMPLETE',**saved})
                return
            offset=log.stat().st_size if log.exists() else 0
            with log.open('ab') as handle:
                child=subprocess.Popen([sys.executable,'-u',str(HERE),'--worker',str(seed)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            base.atomic_json(state_path,{'status':'RUNNING',**saved,'pid':child.pid,'supervisor_pid':os.getpid()})
            base.event(seed,'supervised_worker_started',child_pid=child.pid)
            phase_key=None
            phase_started=time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                state=s.read(OUT/f'status_seed_{seed}.json')
                key=(state['completed_rounds'],state['phase'])
                if key!=phase_key:
                    phase_key,phase_started=key,time.monotonic()
                if state['phase']=='SELECTING' and time.monotonic()-phase_started>2400:
                    base.event(seed,'selection_watchdog_timeout',round=key[0])
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
            code=child.wait()
            if code==0 and (STUDY/f'{seed}_{FREE}_result.json').exists():
                continue
            state=s.read(OUT/f'status_seed_{seed}.json')
            r=str(state['completed_rounds'])
            failures+=1
            counts[r]=counts.get(r,0)+1
            with log.open('rb') as handle:
                handle.seek(offset)
                body=handle.read().decode(errors='replace')
            fatal=code==78 or any(w in body for w in ('QUOTA_EXHAUSTED','DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED',
                'AuthenticationError','401 Unauthorized','403 Forbidden','full-pool selection remained invalid',
                'Frozen ','drift','integrity failure','immutable dialog artifact changed','Provider model mismatch'))
            transient=code<0 or any(w in body for w in ('ConnectError','RemoteProtocolError','ReadTimeout','WriteTimeout',
                'APIConnectionError','APITimeoutError','TimeoutException','429 Too Many Requests','500 Internal Server Error',
                '502 Bad Gateway','503 Service Unavailable','504 Gateway Timeout'))
            saved.update(time=base.now(),returncode=code,recovery_count=prefix+failures,session_recovery_count=failures,failures_per_round=counts)
            if fatal or not transient or counts[r]>2 or failures>12:
                base.atomic_json(state_path,{'status':'NEEDS_ATTENTION',**saved})
                base.event(seed,'supervisor_stopped',returncode=code)
                return
            base.atomic_json(state_path,{'status':'RETRYING',**saved})
            base.event(seed,'automatic_transport_retry',round=int(r),session_recovery_count=failures)
            if counts[r]>1:
                time.sleep(30)


def launch():
    validate()
    checks=s.read(PROBE)
    assert time.time()-PROBE.stat().st_mtime<600
    assert all(row['status']=='PASS' for row in checks['seeds']), 'All three credentials must pass real inference checks'
    assert len({KEYS[seed].read_text().strip() for seed in SEEDS})==3
    with s.exclusive_lock(OUT/'launch.lock'):
        if SESSION.exists():
            raise RuntimeError('Continuation already launched; inspect supervisors before any recovery')
        with ExitStack() as locks:
            for seed in SEEDS:
                for name in ('supervisor','worker'):
                    locks.enter_context(s.exclusive_lock(PREFIX_OUT/f'{name}_seed_{seed}.lock'))
                locks.enter_context(s.exclusive_lock(OUT/f'supervisor_seed_{seed}.lock'))
                locks.enter_context(s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'))
            record={'session_id':str(time.time_ns()),'time':base.now(),'runner_sha256':sha(HERE),
                'target_labels':TARGET,'target_round':END,'training_concurrency':3,'LLM_request_concurrency':3,
                'max_transport_retries_per_round':2,'max_session_transport_retries':12,'logical_budget_changed':False,
                'historical_recovery_counts':{str(seed):s.read(PREFIX_OUT/f'supervisor_seed_{seed}.json')['recovery_count'] for seed in SEEDS},
                'credential_files':{str(seed):str(KEYS[seed]) for seed in SEEDS},'probe':checks,'supervisors':[]}
            base.atomic_json(SESSION,record)
            base.atomic_json(OUT/'launch.json',{'time':base.now(),'routes':{str(row['seed']):{'proxy':row['proxy']} for row in checks['seeds']}})
        env=dict(os.environ,KMP_DUPLICATE_LIB_OK='TRUE',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for seed in SEEDS:
            log=OUT/f'supervisor_seed_{seed}.log'
            with log.open('ab') as handle:
                child=subprocess.Popen([sys.executable,'-u',str(HERE),'--supervise',str(seed)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            subprocess.Popen(['caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            record['supervisors'].append({'seed':seed,'pid':child.pid,'log':str(log)})
            base.atomic_json(SESSION,record)
        print(json.dumps({'status':'LAUNCHED','target_labels':TARGET,'target_round':END,'supervisors':record['supervisors']}),flush=True)


def audit():
    rows=[]
    for seed in SEEDS:
        result=audit_seed(seed)
        state=s.read(OUT/f'status_seed_{seed}.json')
        sup=s.read(OUT/f'supervisor_seed_{seed}.json') if (OUT/f'supervisor_seed_{seed}.json').exists() else {'status':'NOT_LAUNCHED'}
        processes={}
        for pid in (sup.get('pid'),sup.get('supervisor_pid')):
            if pid:
                value=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True,errors='replace').stdout.strip()
                processes[str(pid)]=value or 'MISSING'
        events=OUT/f'events_seed_{seed}.jsonl'
        latest=json.loads(events.read_text().splitlines()[-1]) if events.exists() else None
        rows.append({'seed':seed,'phase':state['phase'],'completed_labels':result['final_labels'],
            'verified_fits':len(result['validation_learning_curve']),'verified_batches':len(result['selected_ids'])//32,
            'supervisor_status':sup['status'],'session_recovery_count':sup.get('session_recovery_count',0),
            'processes':processes,'latest_event':latest})
    issues=[row['seed'] for row in rows if row['supervisor_status'] not in ('RUNNING','RETRYING','COMPLETE','NOT_LAUNCHED') or any(v=='MISSING' or v.startswith('Z') for v in row['processes'].values())]
    record={'time':base.now(),'status':'NEEDS_ATTENTION' if issues else 'HEALTHY','issues':issues,'rows':rows,'test_evaluation_performed':False}
    base.atomic_json(OUT/'heartbeat_latest.json',record)
    with (OUT/'monitor_audit.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--worker',type=int,choices=SEEDS)
    parser.add_argument('--supervise',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.probe:probe()
    elif args.launch:launch()
    elif args.audit:audit()
    elif args.worker is not None:work(args.worker)
    elif args.supervise is not None:supervise(args.supervise)
