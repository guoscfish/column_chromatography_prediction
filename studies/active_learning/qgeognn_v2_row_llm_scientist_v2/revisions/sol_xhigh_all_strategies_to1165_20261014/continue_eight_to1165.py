"""Five additional B32 rounds for two seeds and four frozen strategies; validation only."""
import os, sys, json, time, hashlib, fcntl, subprocess, traceback, re
from pathlib import Path
from contextlib import contextmanager
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_hybrid_free_to1005_20261013'
SNAPSHOT = json.loads((HERE/'source_snapshot.json').read_text())
SNAP = Path(SNAPSHOT['root'])
sys.path.insert(0, str(SNAP))
from src.qgeognn_al.active_learning_v2 import scientist_study as s
sys.path.insert(0, str(SOURCE))
import resume_free6101_round19_extra as previous_extra
base = previous_extra.base
transport = base.transport
read, once = base.read, base.once
HYBRID, FREE = base.HYBRID, base.FREE
METHODS = (HYBRID, FREE, 'center_width_lcmd', 'random32')
BUDGETS = tuple(range(333, 1166, 32))
CONFIG = dict(base.NEW_CONFIG)
SEED = None

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guard():
    manifest = read(HERE/'extension_manifest.json')
    assert Path(s.__file__).is_relative_to(SNAP), 'Scientific module must load from frozen snapshot'
    assert sha(__file__)==manifest['runner_sha256']
    assert sha(HERE/'source_snapshot.json')==manifest['source_snapshot_sha256']
    for name, expected in SNAPSHOT['expected_source_hashes'].items():
        assert sha(SNAP/name)==expected, name
    for name, expected in manifest['protected_files'].items():
        assert sha(Path(name))==expected, name
    return manifest

def directory(seed, method, r):
    assert seed in (157,6101) and method in (HYBRID,FREE) and 0<=r<26
    return (SOURCE if r<21 else HERE)/f'selections/seed_{seed}/{method}/round_{r:02d}'

def selector(packet,catalog,path,config,call=None):
    if packet['round']<21:
        if (packet['seed'],packet['method'],packet['round'])==(6101,FREE,19):
            return previous_extra.extra_selector(packet,catalog,path,base.NEW_CONFIG,call=call)
        return base.selector(packet,catalog,path,config,call=call)
    once(Path(path)/'model_transition.json',{'model':'gpt-6.1-sol','reasoning_effort':'xhigh','extension_start_labels':1005,'extension_target_labels':1165})
    return transport.run_selector(packet,catalog,path,CONFIG,call=call)

original_prediction = base.old_prediction

def prediction(context,method,r,labeled):
    if r<=21:
        d=SOURCE/f'runtime/seed_{context.seed}/{method}/round_{r:02d}'
        f=read(d/'prediction_freeze.json')
        for k in ('checkpoint','prediction'): assert sha(f[k+'_path'])==f[k+'_sha256']
        assert f['L_t_ids_hash']==s.ids_hash(context.ids(labeled))
        return Path(f['checkpoint_path']),d
    return original_prediction(context,method,r,labeled)

def activate():
    guard()
    s.STUDY=HERE; s.BUDGETS=BUDGETS
    # Strictly check every original protocol entry against its immutable source snapshot.
    protocol=read(HERE/'protocol.json')
    s.code_hashes=lambda:{name:sha(SNAP/name) for name in protocol['code_hashes']}
    s.selection_directory=directory;s.run_selector=selector;s.prediction_freeze=prediction
    base.inherited.inherited.repair.DIRECTORY=directory(157,FREE,5)
    base.inherited.original.authorized.DIRECTORY=directory(6101,FREE,0)
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
    labeled,*_=s.state(s.Context(seed),method,26);assert len(labeled)==1165
    ids=[];new=[];files={}
    for r in range(26):
        d=directory(seed,method,r);b=s.audit_batch(d);assert len(b['batch_ids'])==32 and len(b['pending_ids'])==(16 if method==HYBRID else 0)
        ids.extend(b['batch_ids'])
        if r>=21:
            new.extend(b['batch_ids'])
            for f in d.glob('turn_*.json'):
                if re.fullmatch(r'turn_\d\d.json',f.name):assert read(f)['provenance']['served_model'].startswith('gpt-6.1-sol')
            for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):files[str(d/name)]=sha(d/name)
    assert len(ids)==len(set(ids))==832 and len(new)==len(set(new))==160
    rows=[]
    for r in range(27):
        d=(SOURCE if r<=21 else HERE)/f'runtime/seed_{seed}/{method}/round_{r:02d}';f=read(d/'prediction_freeze.json')
        for k in ('checkpoint','prediction'):assert sha(f[k+'_path'])==f[k+'_sha256']
        a=read(Path(f['checkpoint_path']).parent/'fit_audit.json');assert a['train_rows']==BUDGETS[r]
        rows.append({'labels':BUDGETS[r],'validation_nrmse':a['best_validation_combined_normalized_rmse']})
        files[str(d/'prediction_freeze.json')]=sha(d/'prediction_freeze.json')
    once(HERE/f'{seed}_{method}_result.json',{'seed':seed,'method':method,'final_labels':1165,'selected_ids':ids,'additional_ids':new,'validation_learning_curve':rows,'files':files,'test_evaluation_performed':False})

def llm_worker(seed,method):
    activate();transport.responses_call=api_call
    with s.exclusive_lock(HERE/f'{seed}_{method}.lock'):
        for r in range(21,26):
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
    selected=context.ids(labeled)[333:];rows=[];newids=[]
    store=context.new_store() if method=='center_width_lcmd' else context.new_method_store()
    store.freeze_acquisitions(selected);truth=np.vstack([context.l0_truth,store.reveal(selected,'after_acquisition_fit')])
    with s.exclusive_lock(HERE/f'{seed}_{method}.lock'):
        for r in range(21,26):
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
        assert len(labeled)==1165 and len(selected)==len(set(selected))==832 and len(newids)==len(set(newids))==160
        once(HERE/f'{seed}_{method}_result.json',{'seed':seed,'method':method,'final_labels':1165,'selected_ids':selected,'additional_ids':newids,'validation_learning_curve':rows,'test_evaluation_performed':False})
    return 0

def worker(seed,method):
    global SEED
    SEED=seed
    record=guard();os.environ.pop('SCIENTIST_API_KEY',None)
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
            for method in (HYBRID,FREE):assert len(s.state(s.Context(seed),method,21)[0])==1005
        print('EIGHT1165_PREFLIGHT_PASS: frozen source, original protocol, eight1005 anchors; no API call.',flush=True);return 0
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
            for method in scheduled_methods:assert read(HERE/f'{seed}_{method}_result.json')['final_labels']==1165
        output='controls_result.json' if '--controls-only' in sys.argv else 'result.json'
        status='FOUR_CONTROLS_COMPLETE1165' if '--controls-only' in sys.argv else 'ALL_EIGHT_COMPLETE1165'
        once(HERE/output,{'status':status,'test_evaluation_performed':False});return 0

if __name__=='__main__':sys.exit(main())
