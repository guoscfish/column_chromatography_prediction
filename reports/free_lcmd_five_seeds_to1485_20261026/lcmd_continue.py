"""Extend the five frozen LCMD32 trajectories from r21/1005 to r36/1485.

Only the budget schedule is extended. The original 512D q50-gradient LCMD-TP
implementation is loaded from its hash-identical archived source. No test labels
are requested, and every batch is frozen before its labels are revealed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import control as c

ENTRY=Path(__file__).resolve();ROOT=c.ROOT;OUT=ENTRY.parent/'lcmd'
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from qgeognn_al.active_learning_v2 import sequential_runner as sr, sequential_protocol as sp
from qgeognn_al.active_learning_v2.protocol import stable_hash,ids_hash
BASE=ROOT/'studies/active_learning/qgeognn_v2_row_sequential_b32'
STUDY=BASE.parent/'qgeognn_v2_row_lcmd_to1485_20261026'
SEEDS=(157,887,2357,6101,12203)
BUDGETS=tuple(range(333,1486,32))
MANIFEST=OUT/'extension_manifest.json'
VENDOR=ENTRY.parent/'frozen_sources/gradient_features.py'
ORIGINAL=c.read(BASE/'protocol.json')

def configure():
    # Relative imports resolve within the original package; the source is exact.
    name='qgeognn_al.active_learning_v2._frozen_lcmd_gradient_20261026'
    spec=importlib.util.spec_from_file_location(name,VENDOR)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    sr.extract_q50_gradient_sketches=module.extract_q50_gradient_sketches
    sr.STUDY=STUDY
    sr.ACTIVE_LABEL_BUDGETS=sp.ACTIVE_LABEL_BUDGETS=BUDGETS
    sr.ACQUISITION_ROUNDS=sp.ACQUISITION_ROUNDS=36
    sr.FINAL_ACTIVE_LABELS=sp.FINAL_ACTIVE_LABELS=1485
    # Contexts identify the effective frozen scientific implementation.
    sr.code_hashes=lambda:dict(ORIGINAL['code_hashes'])

def validate():
    record=c.read(MANIFEST)
    for path,digest in record['protected_files'].items():
        if c.sha(ROOT/path)!=digest:raise RuntimeError('Frozen LCMD source/prefix drift: '+path)
    assert c.sha(VENDOR)==ORIGINAL['code_hashes']['src/qgeognn_al/active_learning_v2/gradient_features.py']
    configure()

def directory(seed,r):
    owner=BASE if r<=21 else STUDY
    return owner/f'runtime/seed_{seed}/lcmd/round_{r:02d}'

def acquisition_dir(seed,r):
    owner=BASE if r<21 else STUDY
    return owner/f'runtime/seed_{seed}/lcmd/round_{r:02d}'

def complete(seed):return STUDY/f'seed_{seed}_lcmd_result.json'

def status(seed,phase,r,**fields):
    c.atomic(OUT/f'status_seed_{seed}.json',{'updated_at':c.now(),'seed':seed,'method':'lcmd',
        'phase':phase,'completed_rounds':r,'completed_labels':BUDGETS[r],'target_labels':1485,**fields})

def check_fit(seed,r):
    d=directory(seed,r)
    record=c.read(d/('contract.json' if r<=21 else 'fit_freeze.json'))
    if r<=21:sr._verify_round_artifacts(d,record)
    else:sr._assert_protected(d,record['files'])
    fit=c.read(d/'model/fit_audit.json')
    initial=c.read(directory(seed,0)/'model/fit_audit.json')
    assert fit['train_rows']==BUDGETS[r]
    assert fit['initialization_hash']==initial['initialization_hash']
    assert fit['validation_ids_hash']==initial['validation_ids_hash']
    assert fit['test_labels_used_for_fit_or_checkpoint_selection']==record['test_truth_access_count']==0
    assert c.sha(d/'model/best.pt')==fit['checkpoint_sha256']
    assert c.sha(d/'model/predictions.csv.gz')==fit['prediction_sha256']
    assert ids_hash(np.load(d/'labeled_ids.npy').astype(str).tolist())==fit['train_ids_hash']
    return fit,record

def audit_seed(seed,require_complete=False):
    rows=[];selected=[];files={}
    for r in range(37):
        d=directory(seed,r);freeze=d/('contract.json' if r<=21 else 'fit_freeze.json')
        if not freeze.exists():
            if r<=21 or require_complete:raise RuntimeError('Missing frozen LCMD fit: '+str(freeze))
            break
        fit,record=check_fit(seed,r)
        rows.append({'labels':BUDGETS[r],'nrmse':fit['best_validation_combined_normalized_rmse']})
        files[str(freeze)]=c.sha(freeze)
        if r==0:continue
        prev=directory(seed,r-1);ad=acquisition_dir(seed,r-1)
        acq=c.read(ad/'acquisition_artifacts/contract.json')
        sr._assert_protected(ad,acq['files'])
        ids=pd.read_csv(ad/'acquisition_artifacts/selected_next_batch.csv').sample_id.astype(str).tolist()
        assert len(ids)==len(set(ids))==32
        assert stable_hash(ids)==acq['selected_ids_ordered_hash'] and ids_hash(ids)==acq['selected_ids_set_hash']
        assert acq['input']['test_truth_access_count']==0
        sr.validate_trajectory_transition(np.load(prev/'labeled_ids.npy').astype(str).tolist(),
            np.load(prev/'unlabeled_ids.npy').astype(str).tolist(),ids,
            np.load(d/'labeled_ids.npy').astype(str).tolist(),np.load(d/'unlabeled_ids.npy').astype(str).tolist())
        assert acq['input']['evaluation_checkpoint_sha256']==c.sha(prev/'model/best.pt')
        assert acq['input']['L_t_ids_hash']==ids_hash(np.load(prev/'labeled_ids.npy').astype(str).tolist())
        assert acq['input']['U_t_ids_hash']==ids_hash(np.load(prev/'unlabeled_ids.npy').astype(str).tolist())
        if r>21:
            receipt=c.read(ad/'label_access_receipt.json')
            assert receipt['acquisition_sha256']==c.sha(ad/'acquisition_artifacts/contract.json')
            assert receipt['selected_ids']==ids
            assert receipt['audit']['purpose']=='after_acquisition_fit' and receipt['audit']['acquisitions_frozen']
            assert receipt['audit']['requested_ids_hash']==ids_hash(ids)
            assert record['label_receipt_sha256']==c.sha(ad/'label_access_receipt.json')
            files[str(ad/'label_access_receipt.json')]=c.sha(ad/'label_access_receipt.json')
        files[str(ad/'acquisition_artifacts/contract.json')]=c.sha(ad/'acquisition_artifacts/contract.json')
        selected.extend(ids)
    assert len(selected)==len(set(selected))==32*(len(rows)-1)
    result={'time':c.now(),'seed':seed,'method':'lcmd','final_labels':rows[-1]['labels'],
        'verified_fits':len(rows),'verified_batches':len(selected)//32,'selected_ids':selected,
        'validation_learning_curve':rows,'files':files,'test_truth_access_count':0}
    return result

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():validate();print('ALREADY_PREPARED');return
    hashes=sp.code_hashes()
    changed=[k for k,v in hashes.items() if ORIGINAL['code_hashes'][k]!=v]
    assert changed==['src/qgeognn_al/active_learning_v2/gradient_features.py'],changed
    assert c.sha(VENDOR)==ORIGINAL['code_hashes'][changed[0]]
    protected={str(p.relative_to(ROOT)):c.sha(p) for p in (ENTRY,Path(c.__file__),VENDOR,BASE/'protocol.json')}
    # All other scientific sources must still match the original preregistration.
    for name,digest in ORIGINAL['code_hashes'].items():
        if name!=changed[0]:protected[name]=digest
    proofs={}
    for seed in SEEDS:
        proof=audit_seed(seed)
        assert proof['final_labels']==1005 and proof['verified_fits']==22 and proof['verified_batches']==21
        proofs[str(seed)]=proof
        for name,digest in proof['files'].items():protected[str(Path(name).relative_to(ROOT))]=digest
        for name in ('context.json','scaler.json'):
            p=BASE/f'runtime/seed_{seed}'/name;protected[str(p.relative_to(ROOT))]=c.sha(p)
        p=BASE/f'splits/row_seed_{seed}.csv';protected[str(p.relative_to(ROOT))]=c.sha(p)
        status(seed,'READY',21)
    record={'time':c.now(),'authorization_date':'2026-10-26','user_request':'继续跑完，另外lcmd32也跑到一样的',
        'method':'lcmd','seeds':list(SEEDS),'target_round':36,'target_labels':1485,
        'prefix_round':21,'prefix_labels':1005,'new_fits_per_seed':15,'batch_size':32,
        'active_label_budgets':list(BUDGETS),'training':ORIGINAL['training'],
        'algorithm':'Original current full-network q50 512D CountSketch corrected LCMD-TP',
        'effective_scientific_code_hashes':ORIGINAL['code_hashes'],
        'historical_gradient_git_revision':'8081e86','historical_gradient_sha256':c.sha(VENDOR),
        'historical_source_reason':'Working tree gradient module gained later functions; load the exact original frozen source without changing it.',
        'prefix_study':str(BASE),'study':str(STUDY),'training_pool':str(c.POOL),'training_concurrency':3,
        'initialization_changed':False,'split_changed':False,'training_changed':False,
        'test_truth_access_count':0,'test_evaluation_performed':False,
        'protected_files':protected,'prefix_audits':proofs}
    STUDY.mkdir(parents=True,exist_ok=True)
    c.once(MANIFEST,record);c.once(STUDY/'protocol.json',record)
    validate();print(json.dumps({'status':'PREPARED','method':'LCMD32','seeds':SEEDS,'target_labels':1485,'prefix_audits':'PASS'}),flush=True)

def work(seed):
    validate()
    with c.lock(OUT/f'worker_seed_{seed}.lock'):
        try:
            # Verify every original round before resuming from its ordered state.
            initial_audit=audit_seed(seed)
            runtime=STUDY/f'runtime/seed_{seed}'
            partition=pd.read_csv(BASE/f'splits/row_seed_{seed}.csv')
            old_context=c.read(BASE/f'runtime/seed_{seed}/context.json')['contract']
            ctx=sr.SequentialSeedContext(seed,partition,runtime,old_context['preregistration_commit'])
            for key in ('partition_hash','l0_ids_hash','u0_ids_hash','validation_ids_hash','test_ids_hash','normalization','preprocessing'):
                if ctx.context_contract[key]!=old_context[key]:raise RuntimeError('Frozen LCMD context drift: '+key)
            store=ctx.new_method_store()
            L=np.load(directory(seed,21)/'labeled_indices.npy');U=np.load(directory(seed,21)/'unlabeled_indices.npy')
            all_ids=ctx.ids(L[len(ctx.roles['l0']):])
            store.freeze_acquisitions(all_ids)
            truth=np.vstack([ctx.l0_truth,store.reveal(all_ids,'after_acquisition_fit')])
            for r in range(21,36):
                target=directory(seed,r+1)
                ad=acquisition_dir(seed,r);ad.mkdir(parents=True,exist_ok=True)
                fit,_=check_fit(seed,r)
                if r==21:
                    model=ad/'model'
                    if not model.exists():model.symlink_to(directory(seed,21)/'model',target_is_directory=True)
                    assert model.resolve()==(directory(seed,21)/'model').resolve()
                status(seed,'WAITING_FOR_TRAINING',r)
                with c.training_slot():
                    status(seed,'ACQUIRING',r)
                    ids,acq,extra=sr._acquire_next(ctx,'lcmd',r,L,U,truth,fit,ad)
                    assert not extra and len(ids)==len(set(ids))==32 and not set(ids)&set(all_ids)
                    oldL,oldU=ctx.ids(L),ctx.ids(U)
                    lookup=dict(zip(oldU,U));nextL=np.r_[L,[lookup[x] for x in ids]]
                    selected_set=set(ids);nextU=np.asarray([i for i,x in zip(U,oldU) if x not in selected_set],dtype=int)
                    sr.validate_trajectory_transition(oldL,oldU,ids,ctx.ids(nextL),ctx.ids(nextU))
                    all_ids+=ids;store.freeze_acquisitions(all_ids)
                    values=store.reveal(ids,'after_acquisition_fit')
                    receipt={'seed':seed,'source_round':r,'selected_ids':ids,
                        'acquisition_sha256':c.sha(ad/'acquisition_artifacts/contract.json'),
                        'revealed_values_hash':stable_hash(values.tolist()),'audit':store.audit[-1],
                        'test_truth_access_count':0}
                    c.once(ad/'label_access_receipt.json',receipt)
                    truth=np.vstack([truth,values]);L,U=nextL,nextU
                    contract=sr._round_input_contract(ctx,'lcmd',r+1,L,U,ids)
                    incoming=pd.DataFrame({'outer_seed':[seed]*32,'method':['lcmd']*32,'round':[r+1]*32,
                        'selection_order':np.arange(32),'sample_id':ids})
                    sr._save_round_state(target,contract,L,U,ctx.ids(L),ctx.ids(U),incoming)
                    if (target/'fit_freeze.json').exists():
                        check_fit(seed,r+1);continue
                    model=target/'model'
                    # A killed fit has no valid freeze. Preserve partial output before a deterministic restart.
                    required=[model/name for name in ('best.pt','predictions.csv.gz','fit_audit.json')]
                    if model.exists() and any(p.exists() for p in required) and not all(p.exists() for p in required):
                        archive=OUT/'partial_fits'/f'seed_{seed}_round_{r+1:02d}_{time.time_ns()}'
                        archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(model,archive)
                        c.event(OUT,seed,'partial_fit_archived',round=r+1,path=str(archive))
                    status(seed,'TRAINING',r,training_labels=BUDGETS[r+1])
                    ctx.fit(method='lcmd',round_index=r+1,labeled_indices=L,labeled_truth=truth,runtime=model)
                    f=c.read(model/'fit_audit.json');ref=c.read(directory(seed,0)/'model/fit_audit.json')
                    assert f['initialization_hash']==ref['initialization_hash'] and f['validation_ids_hash']==ref['validation_ids_hash']
                    assert f['train_rows']==BUDGETS[r+1] and f['train_ids_hash']==ids_hash(ctx.ids(L))
                    assert f['test_labels_used_for_fit_or_checkpoint_selection']==0
                    protected=[p for p in target.iterdir() if p.is_file()]
                    protected += [model/name for name in ('best.pt','predictions.csv.gz','fit_audit.json')]
                    freeze={'seed':seed,'method':'lcmd','round':r+1,'labels':BUDGETS[r+1],
                        'input_contract_sha256':c.sha(target/'input_contract.json'),
                        'label_receipt_sha256':c.sha(ad/'label_access_receipt.json'),
                        'checkpoint_sha256':f['checkpoint_sha256'],'prediction_sha256':f['prediction_sha256'],
                        'initialization_hash':f['initialization_hash'],'validation_ids_hash':f['validation_ids_hash'],
                        'test_truth_access_count':0,
                        'files':{str(p.relative_to(target)):c.sha(p) for p in protected}}
                    c.once(target/'fit_freeze.json',freeze)
                    check_fit(seed,r+1)
                status(seed,'FROZEN',r+1);c.event(OUT,seed,'round_frozen',round=r+1,labels=BUDGETS[r+1])
            result=audit_seed(seed,True);result.pop('time')
            c.once(complete(seed),result);status(seed,'COMPLETE',36)
        except BaseException as exc:
            traceback.print_exc()
            r=max([21]+[j for j in range(22,37) if (directory(seed,j)/'fit_freeze.json').exists()])
            status(seed,'STOPPED',r,error_type=type(exc).__name__,reason='See worker log')
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true')
    p.add_argument('--audit',action='store_true');p.add_argument('--require-complete',action='store_true')
    p.add_argument('--worker',type=int,choices=SEEDS);p.add_argument('--supervise',type=int,choices=SEEDS)
    a=p.parse_args()
    if a.prepare:prepare()
    else:
        validate()
        if a.launch:c.launch(ENTRY,OUT,SEEDS,complete)
        elif a.audit:
            rows=[audit_seed(seed,a.require_complete) for seed in SEEDS]
            c.atomic(OUT/'audit_latest.json',{'time':c.now(),'rows':rows,'test_truth_access_count':0})
            print(json.dumps([{'seed':x['seed'],'labels':x['final_labels'],'fits':x['verified_fits'],'batches':x['verified_batches']} for x in rows]),flush=True)
        elif a.worker is not None:work(a.worker)
        elif a.supervise is not None:c.supervise(ENTRY,OUT,a.supervise,complete(a.supervise))
