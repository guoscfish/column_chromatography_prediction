"""Four preserved813 prefixes, Sol/xhigh continuation; one API request, at most two fits."""
import os,sys,json,time,shutil,fcntl,subprocess,hashlib,re,traceback
from pathlib import Path
from contextlib import contextmanager
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'astra_high_full_pool_20261001'
sys.path.insert(0,str(OLD))
import resume_free_sol_xhigh_to813 as inherited
from src.qgeognn_al.active_learning_v2 import scientist_transport as transport
s=inherited.s
read=inherited.original.read
once=inherited.original.once
ROOT=inherited.original.ROOT
FREE='free_llm32_scientist_v2';HYBRID='cw16_llm16_scientist_v2'
TARGETS=((157,HYBRID),(6101,HYBRID),(157,FREE),(6101,FREE))
FREE_SOURCE=inherited.NEW
SOURCES={HYBRID:OLD,FREE:FREE_SOURCE}
BUDGETS=tuple(range(333,1006,32))
OLD_CONFIG=inherited.OLD_CONFIG
NEW_CONFIG=inherited.NEW_CONFIG
old_prediction=inherited.old_predict
raw_call=transport.responses_call

@contextmanager
def request_lock():
    with (HERE/'llm_request.execution.lock').open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

@contextmanager
def training_slot():
    # A separate acquisition mutex serializes slot allocation; two independently locked fit slots.
    chosen=None
    while chosen is None:
        with (HERE/'training_allocation.lock').open('a+') as allocator:
            fcntl.flock(allocator.fileno(),fcntl.LOCK_EX)
            for i in range(2):
                f=(HERE/f'training_slot_{i}.lock').open('a+')
                try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:f.close();continue
                chosen=f;break
            fcntl.flock(allocator.fileno(),fcntl.LOCK_UN)
        if chosen is None:time.sleep(1)
    try:yield
    finally:fcntl.flock(chosen.fileno(),fcntl.LOCK_UN);chosen.close()

def directory(seed,method,r):
    if (seed,method) not in TARGETS or r not in range(21):raise ValueError('Outside authorized four1005 trajectories')
    return HERE/f'selections/seed_{seed}/{method}/round_{r:02d}'

def selector(packet,catalog,path,config,call=None):
    seed,method,r=packet['seed'],packet['method'],packet['round']
    if r<15:
        if method==FREE:
            if r<inherited.STARTS[seed]:return inherited.prefix_selector(packet,catalog,path,OLD_CONFIG,call=call)
            return transport.run_selector(packet,catalog,path,NEW_CONFIG,call=call)
        return transport.run_selector(packet,catalog,path,OLD_CONFIG,call=call)
    once(Path(path)/'model_transition.json',{'model':'gpt-6.1-sol','reasoning_effort':'xhigh','extension_start_labels':813,'extension_target_labels':1005})
    return transport.run_selector(packet,catalog,path,NEW_CONFIG,call=call)

def prediction(context,method,r,labeled):
    if r<=15:
        path=HERE/f'runtime/seed_{context.seed}/{method}/round_{r:02d}'
        f=read(path/'prediction_freeze.json')
        for k in ('checkpoint','prediction'):assert s.sha256_file(Path(f[k+'_path']))==f[k+'_sha256']
        assert f['L_t_ids_hash']==s.ids_hash(context.ids(labeled))
        return Path(f['checkpoint_path']),path
    return old_prediction(context,method,r,labeled)

def activate():
    manifest=read(HERE/'extension_manifest.json')
    assert s.sha256_file(Path(__file__))==manifest['runner_sha256']
    for name,h in manifest['original_source_hashes'].items():assert s.sha256_file(ROOT/name)==h,name
    for name,h in manifest['additional_modules_sha256'].items():assert s.sha256_file(ROOT/name)==h,name
    for name,h in manifest['source_manifests'].items():assert s.sha256_file(Path(name))==h,name
    s.verify_files(HERE,manifest['inherited_files'])
    s.configure_revision(HERE.name);s.BUDGETS=BUDGETS
    s.selection_directory=directory;s.run_selector=selector;s.prediction_freeze=prediction
    inherited.inherited.repair.DIRECTORY=directory(157,FREE,5)
    inherited.original.authorized.DIRECTORY=directory(6101,FREE,0)
    s.validate()
    return manifest

def prepare():
    s.validate()
    protocol=read(OLD/'protocol.json')
    transition=read(FREE_SOURCE/'transition_manifest.json')
    assert s.sha256_file(OLD/'protocol.json')==transition['original_protocol_sha256']
    assert s.sha256_file(OLD/'extension_authorized_to813/manifest.json')==transition['original_manifest_sha256']
    s.verify_files(FREE_SOURCE,transition['inherited_files'])
    extra={str(p.relative_to(ROOT)):s.sha256_file(p) for p in (ROOT/'src/qgeognn_al').rglob('*.py') if str(p.relative_to(ROOT)) not in protocol['code_hashes']}
    hashes={};source_manifests={str(OLD/'protocol.json'):s.sha256_file(OLD/'protocol.json'),str(OLD/'extension_authorized_to813/manifest.json'):s.sha256_file(OLD/'extension_authorized_to813/manifest.json'),str(FREE_SOURCE/'transition_manifest.json'):s.sha256_file(FREE_SOURCE/'transition_manifest.json')}
    for name in ('protocol.json','selector_prompt.txt','control_reuse.json','historical_artifacts.json'):
        if not (HERE/name).exists():shutil.copy2(OLD/name,HERE/name)
        assert s.sha256_file(HERE/name)==s.sha256_file(OLD/name)
        hashes[name]=s.sha256_file(HERE/name)
    for seed,method in TARGETS:
        source=SOURCES[method]
        ids=[]
        for r in range(15):
            d=source/f'selections/seed_{seed}/{method}/round_{r:02d}';f=read(d/'batch_freeze.json')
            s.verify_files(d,f['artifact_hashes']);assert len(f['batch_ids'])==32 and len(f['pending_ids'])==(16 if method==HYBRID else 0)
            ids+=f['batch_ids']
        assert len(ids)==len(set(ids))==480
        for r in range(16):
            f=read(source/f'runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json')
            for kind in ('checkpoint','prediction'):assert s.sha256_file(Path(f[kind+'_path']))==f[kind+'_sha256']
            assert read(Path(f['checkpoint_path']).parent/'fit_audit.json')['train_rows']==BUDGETS[r]
        for subtree,count in (('selections',15),('runtime',16)):
            a=source/f'{subtree}/seed_{seed}/{method}';b=HERE/f'{subtree}/seed_{seed}/{method}';b.mkdir(parents=True,exist_ok=True)
            for r in range(count):
                src=a/f'round_{r:02d}';dst=b/src.name
                if not dst.exists():shutil.copytree(src,dst)
                for p in src.rglob('*'):
                    if p.is_file():
                        q=dst/p.relative_to(src);h=s.sha256_file(p);assert s.sha256_file(q)==h
                        hashes[str(q.relative_to(HERE))]=h
        for name in ('context.json',):
            src=source/f'runtime/seed_{seed}/{name}';dst=HERE/f'runtime/seed_{seed}/{name}'
            if not dst.exists():shutil.copy2(src,dst)
            assert s.sha256_file(dst)==s.sha256_file(src)
            hashes[str(dst.relative_to(HERE))]=s.sha256_file(dst)
    cw={}
    for seed in (157,6101):
        d=ROOT/f'studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005/runtime/seed_{seed}/center_width_lcmd/round_21'
        contract=read(d/'round_freeze.json');assert contract['status']=='FROZEN_BEFORE_EXTENSION_TEST_TRUTH' and contract['test_truth_access_count']==0
        for kind in ('checkpoint','prediction'):assert s.sha256_file(Path(contract[kind+'_path']))==contract[kind+'_sha256']
        a=read(d/'model/fit_audit.json');assert a['train_rows']==1005
        cw[str(seed)]={'labels':1005,'validation_nrmse':a['best_validation_combined_normalized_rmse'],'round_freeze_sha256':s.sha256_file(d/'round_freeze.json')}
    record={'authorization':'User2026-10-13 extended both seeds hybrid andFree813->1005 withSol/xhigh',
        'targets':[{'seed':a,'method':b,'source':str(SOURCES[b]),'starting_labels':813,'final_labels':1005} for a,b in TARGETS],
        'budgets':list(BUDGETS),'additional_batches_each':6,'additional_labels_each':192,'new_transport':NEW_CONFIG,
        'original_source_hashes':protocol['code_hashes'],'additional_modules_sha256':extra,'source_manifests':source_manifests,
        'inherited_files':hashes,'runner_sha256':s.sha256_file(Path(__file__)),
        'LLM_request_concurrency':1,'training_concurrency':2,'training_threads_each':2,
        'hybrid_counts':{'CW':16,'LLM':16},'Free_count':32,'full_candidate_delivery':True,'correction_budget_changed':False,
        'training_prompt_changed':False,'CW32_1005_reuse':cw,'classification':'exploratory extension; inherited prefixes retain historical model transitions and authorized correction deviations',
        'test_evaluation_performed':False}
    once(HERE/'extension_manifest.json',record)
    activate()
    for seed,method in TARGETS:assert len(s.state(s.Context(seed),method,15)[0])==813
    print('FOUR1005_PREFLIGHT_PASS',flush=True)

def api_call(messages,config):
    with request_lock():
        if (HERE/'quota_exhausted.json').exists():raise SystemExit(78)
        s._progress({'action':'LLM_request_started','pid':os.getpid(),'model':config['model'],'effort':config['reasoning_effort']})
        try:
            import httpx
            from openai import APIConnectionError
            try:return raw_call(messages,config)
            except httpx.RemoteProtocolError as exc:
                raise APIConnectionError(message='RemoteProtocolError: '+str(exc),request=httpx.Request('POST',config['base_url']+'/responses')) from exc
            except Exception as exc:
                if 'DAILY_LIMIT_EXCEEDED' in str(exc) or 'USAGE_LIMIT_EXCEEDED' in str(exc):
                    once(HERE/'quota_exhausted.json',{'status':'QUOTA_EXHAUSTED','time':time.time(),'pid':os.getpid()})
                    s._progress({'action':'quota_exhausted','pid':os.getpid()})
                    raise SystemExit(78) from exc
                headers=getattr(getattr(exc,'response',None),'headers',{})
                retry=headers.get('Retry-After') if headers else None
                if retry:
                    try:delay=max(0,float(retry))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        delay=max(0,parsedate_to_datetime(retry).timestamp()-time.time())
                    s._progress({'action':'rate_limit_retry_after','pid':os.getpid(),'seconds':delay});time.sleep(delay)
                raise
        finally:s._progress({'action':'LLM_request_finished','pid':os.getpid()})

def finish(seed,method):
    assert len(s.state(s.Context(seed),method,21)[0])==1005
    ids=[];files={};rows=[]
    for r in range(21):
        d=directory(seed,method,r);b=s.audit_batch(d)
        assert len(b['batch_ids'])==32 and len(b['pending_ids'])==(16 if method==HYBRID else 0);ids+=b['batch_ids']
        for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):files[str((d/name).relative_to(HERE))]=s.sha256_file(d/name)
        if r>=15:
            assert read(d/'model_transition.json')['reasoning_effort']=='xhigh'
            for f in d.glob('turn_*.json'):
                if re.fullmatch(r'turn_\d\d.json',f.name):
                    model=read(f)['provenance']['served_model'];assert model=='gpt-6.1-sol' or model.startswith('gpt-6.1-sol-')
    assert len(ids)==len(set(ids))==672
    for r,b in enumerate(BUDGETS):
        p=HERE/f'runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json';f=read(p)
        for k in ('checkpoint','prediction'):assert s.sha256_file(Path(f[k+'_path']))==f[k+'_sha256']
        a=read(Path(f['checkpoint_path']).parent/'fit_audit.json');assert a['train_rows']==b
        rows.append({'labels':b,'validation_nrmse':a['best_validation_combined_normalized_rmse']});files[str(p.relative_to(HERE))]=s.sha256_file(p)
    once(HERE/f'{seed}_{method}_result.json',{'seed':seed,'method':method,'final_labels':1005,'selected_ids':ids,'additional_ids':ids[480:],'files':files,'validation_learning_curve':rows,'test_evaluation_performed':False})
    s._progress({'action':'trajectory1005_complete','seed':seed,'method':method})

def work(seed,method):
    activate();transport.responses_call=api_call
    with s.exclusive_lock(HERE/f'{seed}_{method}.execution.lock'):
        for r in range(15,21):
            p=HERE/f'runtime/seed_{seed}/{method}/round_{r+1:02d}/prediction_freeze.json'
            if p.exists():s.state(s.Context(seed),method,r+1);continue
            failures=0
            while True:
                started=time.time()
                try:
                    with s.status_phase(f'Sol1005 seed{seed} {method} {BUDGETS[r]}->{BUDGETS[r+1]} select',30,seed=seed,method=method,round=r):s.select(seed,method,r)
                    break
                except Exception as exc:
                    traceback.print_exc()
                    errors=[read(p) for p in directory(seed,method,r).glob('retries/turn_*/*.json') if p.stat().st_mtime>=started]
                    if any('DAILY_LIMIT_EXCEEDED' in x.get('message','') or 'USAGE_LIMIT_EXCEEDED' in x.get('message','') for x in errors):return 78
                    recoverable={'APIConnectionError','APITimeoutError','ConnectTimeout','ReadTimeout','ConnectError','RemoteProtocolError','InternalServerError','RateLimitError'}
                    if not errors or any(x.get('error_type') not in recoverable and 'response.completed' not in x.get('message','') for x in errors):
                        s._progress({'action':'requires_decision','seed':seed,'method':method,'round':r,'error':str(exc)[:500]});return 79
                    failures+=1;delay=900 if failures==1 else 3600
                    s._progress({'action':'transport_backoff','seed':seed,'method':method,'round':r,'seconds':delay});time.sleep(delay)
            s._progress({'action':'training_queued','seed':seed,'method':method,'round':r})
            with training_slot():
                s._progress({'action':'training_started','seed':seed,'method':method,'round':r,'pid':os.getpid()})
                with s.status_phase(f'Sol1005 seed{seed} {method} train{BUDGETS[r+1]}',30,seed=seed,method=method,round=r):s.advance(seed,method,r)
                s._progress({'action':'training_completed','seed':seed,'method':method,'round':r})
        finish(seed,method)
    return 0

def main():
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');return work(int(sys.argv[i+1]),sys.argv[i+2])
    with s.exclusive_lock(OLD/'execution.lock'):
        if '--check' in sys.argv:prepare();return 0
        manifest=activate();children=[]
        try:
            for seed,method in TARGETS:
                log=HERE/f'worker_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
                with log.open('ab') as out:
                    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(seed),method],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                children.append((seed,method,p));s._progress({'action':'worker_started','seed':seed,'method':method,'pid':p.pid,'log':str(log)})
            while True:
                errors=[(seed,method,p.poll()) for seed,method,p in children if p.poll() not in (None,0)]
                if any(code==78 for _,_,code in errors):raise RuntimeError('QUOTA_EXHAUSTED')
                if all(p.poll() is not None for _,_,p in children):
                    if errors:raise RuntimeError(f'Incomplete trajectories {errors}')
                    break
                time.sleep(1)
            for seed,method in TARGETS:
                f=read(HERE/f'{seed}_{method}_result.json');assert f['final_labels']==1005 and len(set(f['selected_ids']))==672
            s.verify_files(HERE,manifest['inherited_files'])
            for path,h in manifest['source_manifests'].items():assert s.sha256_file(Path(path))==h
            once(HERE/'result.json',{'status':'ALL_FOUR_COMPLETE1005','test_evaluation_performed':False});s._progress({'action':'all_four1005_completed'});return 0
        finally:
            for _,_,p in children:
                if p.poll() is None:p.terminate()
            for _,_,p in children:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':sys.exit(main())
