"""Parallel per-seed LLM processes; one cross-process training lock."""
import os,sys,time,signal,subprocess,json,re,fcntl
from contextlib import contextmanager
from pathlib import Path
saved_arguments=sys.argv[:]
sys.argv=[sys.argv[0]]
import resume_free_sol_xhigh_to813 as base
sys.argv=saved_arguments
s=base.s

@contextmanager
def training_lock():
    # Each seed has its own process and exclusive seed lock; at most two fits.
    yield

original_response_call=base.transport.responses_call

def serial_response_call(messages,config):
    with (base.NEW/'llm_request.execution.lock').open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:
            base.s._progress({'action':'serial_llm_request_started','pid':os.getpid(),'model':config['model']})
            return original_response_call(messages,config)
        finally:
            base.s._progress({'action':'serial_llm_request_finished','pid':os.getpid()})
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

base.transport.responses_call=serial_response_call

def activate():
    manifest=base.original.read(base.NEW/'transition_manifest.json')
    assert s.sha256_file(Path(base.__file__))==manifest['runner_sha256']
    assert s.sha256_file(base.OLD/'protocol.json')==manifest['original_protocol_sha256']
    assert s.sha256_file(base.original.EXT/'manifest.json')==manifest['original_manifest_sha256']
    s.verify_files(base.NEW,manifest['inherited_files'])
    for name,digest in manifest['independent_modules_sha256'].items():
        assert s.sha256_file(base.original.ROOT/name)==digest
    s.configure_revision(base.NEW.name)
    s.selection_directory=base.directory;s.run_selector=base.choose;s.prediction_freeze=base.prediction
    base.inherited.repair.DIRECTORY=base.directory(157,base.METHOD,5)
    base.original.authorized.DIRECTORY=base.directory(6101,base.METHOD,0)
    s.validate()
    return manifest

def finish(seed):
    assert len(s.state(s.Context(seed),base.METHOD,15)[0])==813
    ids=[];files={}
    for r in range(15):
        d=base.directory(seed,base.METHOD,r);b=s.audit_batch(d)
        assert len(b['batch_ids'])==32 and not b['pending_ids'];ids+=b['batch_ids']
        for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):
            files[str((d/name).relative_to(base.NEW))]=s.sha256_file(d/name)
        if r>=base.STARTS[seed]:
            for f in d.glob('turn_*.json'):
                if not re.fullmatch(r'turn_\d\d.json',f.name):continue
                receipt=base.original.read(f)
                model=receipt['provenance']['served_model']
                assert model=='gpt-6.1-sol' or model.startswith('gpt-6.1-sol-')
            assert base.original.read(d/'model_transition.json')['reasoning_effort']=='xhigh'
    assert len(ids)==len(set(ids))==480
    rows=base.original.curve(seed,base.METHOD);assert len(rows)==16
    base.original.once(base.NEW/f'{seed}_result.json',{'seed':seed,'final_labels':813,'selected_ids':ids,
        'validation_learning_curve':rows,'files':files,'transition_labels':s.BUDGETS[base.STARTS[seed]],
        'suffix_model':'gpt-6.1-sol','suffix_effort':'xhigh','test_evaluation_performed':False})

def worker(seed):
    activate()
    with s.exclusive_lock(base.NEW/f'seed_{seed}.execution.lock'):
        for r in range(base.STARTS[seed],15):
            pred=base.NEW/f'runtime/seed_{seed}/{base.METHOD}/round_{r+1:02d}/prediction_freeze.json'
            if pred.exists():s.state(s.Context(seed),base.METHOD,r+1);continue
            failures=0
            existing=sorted((base.directory(seed,base.METHOD,r)/'retries').glob('turn_*/*.json'),key=lambda f:f.stat().st_mtime)
            if existing:
                last=base.original.read(existing[-1]);message=last.get('message','')
                if 'DAILY_LIMIT_EXCEEDED' in message or 'USAGE_LIMIT_EXCEEDED' in message:return 78
                if last.get('error_type') in {'APIConnectionError','ReadTimeout','ConnectTimeout','APITimeoutError','ConnectError'} or 'response.completed' in message:
                    delay=3600 if len(existing)>=6 else 900
                    remaining=max(0,existing[-1].stat().st_mtime+delay-time.time())
                    if remaining:
                        s._progress({'action':'parallel_transport_backoff','seed':seed,'round':r,'seconds':remaining,'reason':'resume respects existing failure backoff'})
                        time.sleep(remaining)
            while True:
                call_started=time.time()
                try:
                    details={'seed':seed,'method':base.METHOD,'round':r,'parallel_selection':True}
                    with s.status_phase(f'Parallel Sol seed{seed} {s.BUDGETS[r]}->{s.BUDGETS[r+1]} select',30,**details):
                        s.select(seed,base.METHOD,r)
                    break
                except Exception:
                    import traceback
                    traceback.print_exc()
                    retries=base.directory(seed,base.METHOD,r)/'retries'
                    records=[base.original.read(f) for f in retries.glob('turn_*/*.json') if f.stat().st_mtime>=call_started]
                    if any('DAILY_LIMIT_EXCEEDED' in x.get('message','') or 'USAGE_LIMIT_EXCEEDED' in x.get('message','') for x in records):
                        s._progress({'action':'parallel_quota_exhausted','seed':seed,'round':r});return 78
                    latest=sorted((f for f in retries.glob('turn_*/*.json') if f.stat().st_mtime>=call_started),key=lambda f:f.stat().st_mtime)
                    transport_types={'APIConnectionError','ReadTimeout','ConnectTimeout','APITimeoutError','RateLimitError','ConnectError'}
                    if not latest or (base.original.read(latest[-1]).get('error_type') not in transport_types and 'response.completed' not in base.original.read(latest[-1]).get('message','')):
                        s._progress({'action':'parallel_requires_decision','seed':seed,'round':r});return 79
                    failures+=1;delay=900 if failures==1 else 3600
                    s._progress({'action':'parallel_transport_backoff','seed':seed,'round':r,'failures':failures,'seconds':delay})
                    time.sleep(delay)
            s._progress({'action':'training_queued','seed':seed,'round':r})
            with training_lock():
                s._progress({'action':'parallel_training_started','seed':seed,'round':r,'pid':os.getpid()})
                with s.status_phase(f'Parallel train seed{seed} {s.BUDGETS[r+1]}',30,seed=seed,round=r):
                    s.advance(seed,base.METHOD,r)
                s._progress({'action':'parallel_training_completed','seed':seed,'round':r})
        finish(seed)
        s._progress({'action':'parallel_seed_complete','seed':seed})
    return 0

def handoff_group(parent, workers):
    # Keep seed157 current request; stop seed6101 request per explicit user request.
    try:os.kill(parent,signal.SIGSTOP)
    except ProcessLookupError:pass
    try:
        try:os.kill(workers[6101],signal.SIGTERM)
        except ProcessLookupError:pass
        target=base.NEW/f'runtime/seed_157/{base.METHOD}/round_12/prediction_freeze.json'
        print('Waiting for seed157717 freeze; seed6101 removed from concurrent requests',flush=True)
        while not target.exists():
            state=subprocess.run(['ps','-p',str(workers[157]),'-o','stat='],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'):raise RuntimeError('Oldseed157 exited before717; inspect failure')
            time.sleep(.2)
        try:os.kill(workers[157],signal.SIGSTOP)
        except ProcessLookupError:pass
        f=base.original.read(target)
        for kind in ('checkpoint','prediction'):assert s.sha256_file(Path(f[kind+'_path']))==f[kind+'_sha256']
        for pid in (workers[157],parent):
            try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            live=[]
            for pid in [parent]+list(workers.values()):
                state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
                if state and not state.startswith('Z'):live.append(pid)
            if not live:print('Handoff complete; starting serial LLM request schedule',flush=True);return
            time.sleep(.2)
        raise RuntimeError(f'Old processes still active {live}')
    except Exception:
        for pid in (parent,workers[157]):
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        raise

def main():
    if '--seed' in sys.argv:return worker(int(sys.argv[sys.argv.index('--seed')+1]))
    if '--handoff-group' in sys.argv:
        i=sys.argv.index('--handoff-group');parent,a,b=map(int,sys.argv[i+1:i+4]);handoff_group(parent,{157:a,6101:b})
    with s.exclusive_lock(base.OLD/'execution.lock'):
        activate()
        record={'authorization':'User requested one LLM request at a time, training may remain parallel, on2026-10-12',
            'selection_processes':2,'concurrent_llm_requests':1,'llm_request_lock':str(base.NEW/'llm_request.execution.lock'),'training_processes':2,'cpu_threads_per_training':2,'isolation':'one independent process and exclusive lock per seed',
            'scientific_settings_changed':False,'runner_sha256':s.sha256_file(Path(__file__)),
            'transition_manifest_sha256':s.sha256_file(base.NEW/'transition_manifest.json')}
        if '--check' in sys.argv:print('FULL_PARALLEL_PREFLIGHT_PASS',flush=True);return 0
        base.original.once(base.NEW/'serial_request_schedule.json',record)
        workers=[]
        try:
            for seed in (157,6101):
                external=base.NEW/'independent_serial_seed6101.json'
                if seed==6101 and external.exists():
                    ext=base.original.read(external)
                    state=subprocess.run(['ps','-p',str(ext['pid']),'-o','stat='],capture_output=True,text=True).stdout.strip()
                    if state and not state.startswith('Z'):
                        print(f"Adopting external seed6101 PID{ext['pid']}",flush=True)
                        continue
                log=base.NEW/f'parallel_seed{seed}_{time.strftime("%Y%m%d_%H%M%S")}.log'
                with log.open('ab') as out:
                    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--seed',str(seed)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                workers.append((seed,p));s._progress({'action':'parallel_worker_started','seed':seed,'pid':p.pid,'log':str(log)})
            while True:
                errors=[(seed,p.poll()) for seed,p in workers if p.poll() not in (None,0)]
                if any(code==78 for _,code in errors):raise RuntimeError(f'Quota exhausted: {errors}')
                if all(p.poll() is not None for _,p in workers):
                    if errors:raise RuntimeError(f'Incomplete seeds require attention: {errors}')
                    break
                time.sleep(1)
            while not (base.NEW/'6101_result.json').exists():
                external=base.original.read(base.NEW/'independent_serial_seed6101.json')
                state=subprocess.run(['ps','-p',str(external['pid']),'-o','stat='],capture_output=True,text=True).stdout.strip()
                if not state or state.startswith('Z'):raise RuntimeError('Independent6101 exited before result; inspect log')
                time.sleep(1)
            for seed in (157,6101):
                result=base.original.read(base.NEW/f'{seed}_result.json')
                assert result['final_labels']==813 and len(set(result['selected_ids']))==480
            base.original.once(base.NEW/'result.json',{'status':'BOTH_FREE_COMPLETE813','test_evaluation_performed':False})
            s._progress({'action':'sol_free813_completed'});return 0
        finally:
            for _,p in workers:
                if p.poll() is None:p.terminate()
            for _,p in workers:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':sys.exit(main())
