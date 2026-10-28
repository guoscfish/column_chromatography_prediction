"""Four authorized B32 continuation rounds for eight frozen trajectories; validation only."""
import os,sys,json,time,hashlib,fcntl,subprocess,traceback,re
from pathlib import Path
from contextlib import contextmanager
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'sol_xhigh_all_strategies_to1165_20261014'
sys.path.insert(0,str(PREVIOUS))
import resume_hybrid6101_round21_extra as previous_extra
old=previous_extra.original
s=old.s; base=old.base; transport=old.transport; ROOT=old.ROOT; SNAP=old.SNAP
SOURCE=old.SOURCE; read=old.read; once=old.once
HYBRID,FREE=old.HYBRID,old.FREE
METHODS=old.METHODS;CONFIG=dict(old.CONFIG);BUDGETS=tuple(range(333,1294,32));SEED=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guard():
    previous_extra.guard()
    record=read(HERE/'extension_manifest.json')
    assert sha(__file__)==record['runner_sha256']
    for name,h in record['protected_files'].items():assert sha(name)==h,name
    assert read(PREVIOUS/'result.json')['status']=='ALL_EIGHT_COMPLETE1165'
    return record

def directory(seed,method,r):
    assert 0<=r<30
    if r<26:return old.directory(seed,method,r)
    return HERE/f'selections/seed_{seed}/{method}/round_{r:02d}'

def selector(packet,catalog,path,config,call=None):
    if packet['round']<26:return previous_extra.extra_selector(packet,catalog,path,config,call=call)
    once(Path(path)/'model_transition.json',{'model':'gpt-6.1-sol','reasoning_effort':'xhigh','extension_start_labels':1165,'extension_target_labels':1293})
    return transport.run_selector(packet,catalog,path,CONFIG,call=call)

def prediction(context,method,r,labeled):
    if r<=26:
        d=(SOURCE if r<=21 else PREVIOUS)/f'runtime/seed_{context.seed}/{method}/round_{r:02d}'
        f=read(d/'prediction_freeze.json')
        for k in ('checkpoint','prediction'):assert sha(f[k+'_path'])==f[k+'_sha256']
        assert f['L_t_ids_hash']==s.ids_hash(context.ids(labeled))
        return Path(f['checkpoint_path']),d
    return old.original_prediction(context,method,r,labeled)

def extended_label_budget(active):
    if active<=1165:return previous_extra.stream.compatible.extended_label_budget(active)
    assert int(active) in (1197,1229,1261,1293)
    p=previous_extra.stream.compatible.sequential_protocol
    return {'active_label_count':int(active),'active_label_fraction_outer_train':int(active)/p.OUTER_TRAIN_ROWS,'shared_validation_label_count':p.VALIDATION_ROWS,'total_observed_non_test_labels':int(active)+p.VALIDATION_ROWS,'total_observed_fraction_full_dataset':(int(active)+p.VALIDATION_ROWS)/p.FULL_DATASET_ROWS}

def activate():
    guard();old.activate()
    s.STUDY=HERE;s.BUDGETS=BUDGETS
    s.selection_directory=directory;s.run_selector=selector;s.prediction_freeze=prediction
    previous_extra.stream.compatible.sequential_runner.label_budget=extended_label_budget
    s.validate()



@contextmanager
def blocking_lock(path):
    with Path(path).open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f.fileno(),fcntl.LOCK_UN)


@contextmanager
def training_slot():
    chosen=None
    while chosen is None:
        with blocking_lock(HERE/'fit_allocation.lock'):
            for i in range(2):
                f=(HERE/f'fit_slot_{i}.lock').open('a+')
                try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: f.close();continue
                chosen=f;break
        if chosen is None:time.sleep(1)
    try:yield
    finally:fcntl.flock(chosen.fileno(),fcntl.LOCK_UN);chosen.close()


def api_call(messages,config):
    with blocking_lock(HERE/f'llm_seed_{SEED}.lock'):
        if (HERE/f'quota_seed_{SEED}.json').exists():raise SystemExit(78)
        s._progress({'action':'LLM_request_started','seed':SEED,'pid':os.getpid(),'model':config['model'],'effort':config['reasoning_effort']})
        try:
            import httpx
            from openai import APIConnectionError
            try:return base.raw_call(messages,config)
            except httpx.RemoteProtocolError as e:
                raise APIConnectionError(message=str(e),request=httpx.Request('POST',config['base_url']+'/responses')) from e
            except Exception as e:
                if any(x in str(e) for x in ('DAILY_LIMIT_EXCEEDED','USAGE_LIMIT_EXCEEDED','insufficient_quota')):
                    once(HERE/f'quota_seed_{SEED}.json',{'seed':SEED,'status':'QUOTA_EXHAUSTED','time':time.time()})
                    s._progress({'action':'quota_exhausted','seed':SEED,'pid':os.getpid()});raise SystemExit(78) from e
                headers=getattr(getattr(e,'response',None),'headers',{})
                delay=headers.get('Retry-After') if headers else None
                if delay:
                    try:seconds=max(0,float(delay))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        seconds=max(0,parsedate_to_datetime(delay).timestamp()-time.time())
                    time.sleep(seconds)
                raise
        finally:s._progress({'action':'LLM_request_finished','seed':SEED,'pid':os.getpid()})


def finish_llm(seed,method):
    labeled,*_=s.state(s.Context(seed),method,30);assert len(labeled)==1293
    ids=[];new=[];files={}
    for r in range(30):
        d=directory(seed,method,r);b=s.audit_batch(d);assert len(b['batch_ids'])==32 and len(b['pending_ids'])==(16 if method==HYBRID else 0)
        ids.extend(b['batch_ids'])
        if r>=26:
            new.extend(b['batch_ids'])
            for f in d.glob('turn_*.json'):
                if re.fullmatch(r'turn_\d\d.json',f.name):assert read(f)['provenance']['served_model'].startswith('gpt-6.1-sol')
            for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):files[str(d/name)]=sha(d/name)
    assert len(ids)==len(set(ids))==960 and len(new)==len(set(new))==128
    rows=[]
    for r in range(31):
        d=(SOURCE if r<=21 else (PREVIOUS if r<=26 else HERE))/f'runtime/seed_{seed}/{method}/round_{r:02d}';f=read(d/'prediction_freeze.json')
        for k in ('checkpoint','prediction'):assert sha(f[k+'_path'])==f[k+'_sha256']
        a=read(Path(f['checkpoint_path']).parent/'fit_audit.json');assert a['train_rows']==BUDGETS[r]
        rows.append({'labels':BUDGETS[r],'validation_nrmse':a['best_validation_combined_normalized_rmse']})
        files[str(d/'prediction_freeze.json')]=sha(d/'prediction_freeze.json')
    once(HERE/f'{seed}_{method}_result.json',{'seed':seed,'method':method,'final_labels':1293,'selected_ids':ids,'additional_ids':new,'validation_learning_curve':rows,'files':files,'test_evaluation_performed':False})


def llm_worker(seed,method):
    activate();transport.responses_call=api_call
    with s.exclusive_lock(HERE/f'{seed}_{method}.lock'):
        for r in range(26,30):
            target=HERE/f'runtime/seed_{seed}/{method}/round_{r+1:02d}/prediction_freeze.json'
            if target.exists():s.state(s.Context(seed),method,r+1);continue
            failures=0
            while True:
                started=time.time()
                try:
                    s._progress({'action':'select_started','seed':seed,'method':method,'round':r})
                    with s.status_phase(f'seed{seed} {method} select {BUDGETS[r]}->{BUDGETS[r+1]}',30,seed=seed,method=method,round=r):s.select(seed,method,r)
                    break
                except Exception as e:
                    traceback.print_exc();d=directory(seed,method,r)
                    if (d/'rejected_turn_02.json').exists() or 'bounded corrections' in str(e):
                        s._progress({'action':'requires_decision','seed':seed,'method':method,'round':r,'error':str(e)[:500]});return 79
                    errors=[read(f) for f in d.glob('retries/turn_*/*.json') if f.stat().st_mtime>=started]
                    recoverable={'APIConnectionError','APITimeoutError','ConnectTimeout','ReadTimeout','ConnectError','RemoteProtocolError','InternalServerError','RateLimitError'}
                    if not errors or any(x.get('error_type') not in recoverable and 'response.completed' not in x.get('message','') for x in errors):raise
                    failures+=1;delay=min(60,10*failures)
                    s._progress({'action':'transport_retry','seed':seed,'method':method,'round':r,'seconds':delay});time.sleep(delay)
            s._progress({'action':'training_queued','seed':seed,'method':method,'round':r})
            with training_slot():
                s._progress({'action':'training_started','seed':seed,'method':method,'round':r,'pid':os.getpid()})
                with s.status_phase(f'seed{seed} {method} train {BUDGETS[r+1]}',30,seed=seed,method=method,round=r):s.advance(seed,method,r)
                s._progress({'action':'training_completed','seed':seed,'method':method,'round':r})
        finish_llm(seed,method)
    return 0


def control_worker(seed,method):
    activate()
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore,ids_hash
    from src.qgeognn_al.active_learning_v2.sequential_acquisition import deterministic_random_permutation
    if method=='center_width_lcmd':
        from src.qgeognn_al.active_learning_v2.cw_lcmd_to_1005_runner import CW1005Context
        from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import _acquire
        context=CW1005Context(seed,HERE/'control_cw');src=ROOT/f'studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005/runtime/seed_{seed}/center_width_lcmd/round_21'
        state=pd.read_csv(src/'state.csv');labeled=state.loc[state.role.eq('labeled'),'canonical_index'].to_numpy(int);unlabeled=state.loc[state.role.eq('unlabeled'),'canonical_index'].to_numpy(int)
        anchor=read(src/'round_freeze.json');checkpoint=Path(anchor['checkpoint_path']);anchor_audit=read(checkpoint.parent/'fit_audit.json')
    else:
        from src.qgeognn_al.active_learning_v2.sequential_runner import SequentialSeedContext
        src=ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/random/round_21'
        contract=read(src.parent.parent/'context.json')['contract'];partition=pd.read_csv(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv')
        context=SequentialSeedContext(seed,partition,HERE/f'control_random/runtime/seed_{seed}',contract['preregistration_commit'])
        labeled=np.load(src/'labeled_indices.npy');unlabeled=np.load(src/'unlabeled_indices.npy');checkpoint=src/'model/best.pt';anchor_audit=read(src/'model/fit_audit.json')
        permutation=deterministic_random_permutation(context.ids(context.roles['u0']),seed)
        assert context.ids(labeled)==context.ids(context.roles['l0'])+list(permutation[:672])
    assert len(labeled)==1005
    for previous_round in range(21,26):
        batch=read(PREVIOUS/f'controls/seed_{seed}/{method}/round_{previous_round:02d}/batch_freeze.json')['batch_ids']
        positions=dict(zip(context.ids(unlabeled),unlabeled));assert len(batch)==len(set(batch))==32
        labeled=np.r_[labeled,[positions[i] for i in batch]]
        unlabeled=np.array([i for i in unlabeled if i not in set(labeled)],dtype=int)
    assert len(labeled)==1165
    assert context.ids(labeled)[333:]==read(PREVIOUS/f'{seed}_{method}_result.json')['selected_ids']
    anchor=read(PREVIOUS/f'runtime/seed_{seed}/{method}/round_26/prediction_freeze.json')
    for k in ('checkpoint','prediction'):assert sha(anchor[k+'_path'])==anchor[k+'_sha256']
    checkpoint=Path(anchor['checkpoint_path']);anchor_audit=read(checkpoint.parent/'fit_audit.json')
    if method=='random32':assert context.ids(labeled)==context.ids(context.roles['l0'])+list(permutation[:832])
    selected=context.ids(labeled)[333:];rows=[];newids=[]
    store=RestrictedLabelStore(s.SOURCE_DATA,context.partition) if method=='center_width_lcmd' else context.new_method_store()
    if method=='center_width_lcmd':
        assert np.array_equal(store.reveal(context.ids(context.roles['l0']),'initial_fit'),context.l0_truth)
        assert np.array_equal(store.reveal(context.ids(context.roles['validation']),'initial_fit'),context.validation_truth)
    store.freeze_acquisitions(selected);truth=np.vstack([context.l0_truth,store.reveal(selected,'after_acquisition_fit')])
    with s.exclusive_lock(HERE/f'{seed}_{method}.lock'):
        for r in range(26,30):
            d=HERE/f'controls/seed_{seed}/{method}/round_{r:02d}';d.mkdir(parents=True,exist_ok=True)
            target=HERE/f'runtime/seed_{seed}/{method}/round_{r+1:02d}';freeze=target/'prediction_freeze.json'
            s._progress({'action':'control_select_started','seed':seed,'method':method,'round':r})
            if (d/'batch_freeze.json').exists():batch=read(d/'batch_freeze.json')['batch_ids']
            elif method=='center_width_lcmd':
                with training_slot():batch,receipt=_acquire(context,r,labeled,unlabeled,checkpoint,d)
            else:batch=list(permutation[r*32:(r+1)*32])
            assert len(batch)==len(set(batch))==32 and set(batch)<=set(context.ids(unlabeled))
            once(d/'batch_freeze.json',{'seed':seed,'method':method,'round':r,'labels_before':BUDGETS[r],'batch_ids':batch,'checkpoint_sha256':sha(checkpoint),'L_t_ids_hash':ids_hash(context.ids(labeled)),'U_t_ids_hash':ids_hash(context.ids(unlabeled)),'frozen_before_reveal':True})
            positions=dict(zip(context.ids(unlabeled),unlabeled));labeled=np.r_[labeled,[positions[i] for i in batch]];unlabeled=np.array([i for i in unlabeled if i not in set(labeled)],dtype=int)
            selected+=batch;newids+=batch;store.freeze_acquisitions(selected);values=store.reveal(batch,'after_acquisition_fit');truth=np.vstack([truth,values])
            once(d/'label_access_receipt.json',{'batch_freeze_sha256':sha(d/'batch_freeze.json'),'audit':store.audit[-1]})
            target.mkdir(parents=True,exist_ok=True)
            if not freeze.exists():
                s._progress({'action':'training_queued','seed':seed,'method':method,'round':r})
                with training_slot():
                    s._progress({'action':'training_started','seed':seed,'method':method,'round':r,'pid':os.getpid()})
                    with s.status_phase(f'seed{seed} {method} train {BUDGETS[r+1]}',30,seed=seed,method=method,round=r):
                        if method=='center_width_lcmd':audit=context.fit(r+1,labeled,truth,target/'model')
                        else:audit=context.fit(method='random',round_index=r+1,labeled_indices=labeled,labeled_truth=truth,runtime=target/'model')
                    assert audit['initialization_hash']==anchor_audit['initialization_hash']
                    once(freeze,{'checkpoint_path':str(target/'model/best.pt'),'checkpoint_sha256':sha(target/'model/best.pt'),'prediction_path':str(target/'model/predictions.csv.gz'),'prediction_sha256':sha(target/'model/predictions.csv.gz'),'L_t_ids_hash':ids_hash(context.ids(labeled)),'test_truth_access_count':0})
                s._progress({'action':'training_completed','seed':seed,'method':method,'round':r})
            f=read(freeze)
            for k in ('checkpoint','prediction'):assert sha(f[k+'_path'])==f[k+'_sha256']
            checkpoint=Path(f['checkpoint_path']);a=read(checkpoint.parent/'fit_audit.json');assert a['train_rows']==BUDGETS[r+1]
            rows.append({'labels':BUDGETS[r+1],'validation_nrmse':a['best_validation_combined_normalized_rmse']})
        assert len(labeled)==1293 and len(selected)==len(set(selected))==960 and len(newids)==len(set(newids))==128
        once(HERE/f'{seed}_{method}_result.json',{'seed':seed,'method':method,'final_labels':1293,'selected_ids':selected,'additional_ids':newids,'validation_learning_curve':rows,'test_evaluation_performed':False})
    return 0


def worker(seed,method):
    global SEED
    SEED=seed
    old.SEED=seed
    stream=previous_extra.stream
    stream.HERE=HERE
    stream.ResponseStreamState.handle_event=stream.handle_event
    activate();record=guard();os.environ.pop('SCIENTIST_API_KEY',None)
    os.environ['SCIENTIST_API_KEY_FILE']=record['credential_files'][str(seed)]
    assert Path(os.environ['SCIENTIST_API_KEY_FILE']).stat().st_mode&0o077==0
    import torch
    torch.set_num_threads(2)
    s._progress({'action':'worker_started','seed':seed,'method':method,'pid':os.getpid()})
    try:return llm_worker(seed,method) if method in (HYBRID,FREE) else control_worker(seed,method)
    finally:s._progress({'action':'worker_finished','seed':seed,'method':method,'pid':os.getpid()})


def main():
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');return worker(int(sys.argv[i+1]),sys.argv[i+2])
    activate()
    if '--check' in sys.argv:
        for seed in (157,6101):
            for method in (HYBRID,FREE):assert len(s.state(s.Context(seed),method,26)[0])==1165
        for seed in (157,6101):
            for method in METHODS:
                result=read(PREVIOUS/f'{seed}_{method}_result.json')
                assert result['final_labels']==1165
                assert len(result['selected_ids'])==len(set(result['selected_ids']))==832
                freeze=read(PREVIOUS/f'runtime/seed_{seed}/{method}/round_26/prediction_freeze.json')
                for k in ('checkpoint','prediction'):assert sha(freeze[k+'_path'])==freeze[k+'_sha256']
                audit=read(Path(freeze['checkpoint_path']).parent/'fit_audit.json')
                assert audit['train_rows']==1165 and audit['test_labels_used_for_fit_or_checkpoint_selection']==0
        assert extended_label_budget(1293)['active_label_count']==1293
        print('EIGHT1293_PREFLIGHT_PASS: frozen source, original protocol, eight1165 anchors; no API call.',flush=True);return 0
    children={}
    scheduled_methods=METHODS[2:] if '--controls-only' in sys.argv else METHODS
    with s.exclusive_lock(HERE/'coordinator.lock'):
        for seed in (157,6101):
            for method in scheduled_methods:
                if (HERE/f'{seed}_{method}_result.json').exists():continue
                log=HERE/f'worker_{seed}_{method}_{time.strftime("%Y%m%d_%H%M%S")}.log'
                with log.open('ab') as f:child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(seed),method],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=os.environ.copy())
                children[seed,method]=child
        once(HERE/f'launch_{time.strftime("%Y%m%d_%H%M%S")}.json',{'worker_pids':{f'{seed}:{method}':c.pid for (seed,method),c in children.items()},'coordinator_pid':os.getpid()})
        while any(c.poll() is None for c in children.values()):
            for (seed,method),child in children.items():
                if method in (HYBRID,FREE) and (HERE/f'quota_seed_{seed}.json').exists() and child.poll() is None:child.terminate()
            time.sleep(1)
        exits={f'{seed}:{method}':c.returncode for (seed,method),c in children.items()}
        if any(exits.values()):s._progress({'action':'incomplete_workers','exit_codes':exits});return 79
        guard()
        for seed in (157,6101):
            for method in scheduled_methods:assert read(HERE/f'{seed}_{method}_result.json')['final_labels']==1293
        output='controls_result.json' if '--controls-only' in sys.argv else 'result.json'
        status='FOUR_CONTROLS_COMPLETE1293' if '--controls-only' in sys.argv else 'ALL_EIGHT_COMPLETE1293'
        once(HERE/output,{'status':status,'test_evaluation_performed':False});return 0


if __name__=='__main__':sys.exit(main())
