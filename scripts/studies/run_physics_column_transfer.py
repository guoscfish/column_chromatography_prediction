#!/usr/bin/env python3
"""Blind, resumable physical-context experiment; frozen historical ledgers."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.studies import run_source_anchored_transfer as old
from src.qgeognn_al.transfer.column_physics import TrainingNormalizer, context_matrix
from src.qgeognn_al.transfer.column_conditioned import ColumnConditionedQGeoGNN
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.training.predictor import atomic_json, seed_everything, loader_pair
from src.qgeognn_al.artifacts import sha256_file
STUDY=ROOT/'studies/transfer/physics_column_conditioned_transfer'
METHODS=('packing_mass_physical_scale','physics_scale_residual','raw_column_conditioned','mass_normalized_column_conditioned')
NEURAL=METHODS[2:]
BASELINES=(*old.BASELINES,'standard_shallow_finetune')
COLUMNS,SEEDS,BUDGETS=old.COLUMNS,old.SEEDS,old.BUDGETS
CONFIG=dict(learning_rate=1e-4,weight_decay=1e-5,maximum_epochs=500,patience=100,source_weight=.5,target_weight=.5,ridge_alpha=100.,batch_norm='frozen_running_statistics_trainable_affine',threads=1)
expected_contexts=old.expected_contexts
context_path=old.context_path
verify_manifest=old.verify_manifest
selected_truth=old.selected_truth
target_path=old.target_path

def prepare():
    audit=json.loads((STUDY/'audit_frozen.json').read_text())
    audit['files'].update(json.loads((STUDY/'audit_supplement_frozen.json').read_text())['files'])
    for name,digest in audit['files'].items():
        if sha256_file(ROOT/name)!=digest: raise RuntimeError('physical audit changed: '+name)
    if not audit['normalized_arm']: raise RuntimeError('normalized arm not supported by frozen gate')
    files=[Path(__file__),ROOT/'src/qgeognn_al/transfer/column_conditioned.py',ROOT/'src/qgeognn_al/transfer/column_physics.py',ROOT/'scripts/studies/evaluate_physics_column_transfer.py',STUDY/'MODEL_PREREGISTRATION.md',STUDY/'audit_frozen.json',STUDY/'audit_supplement_frozen.json',ROOT/'scripts/studies/run_source_anchored_transfer.py',ROOT/'scripts/studies/run_scaling_failure_audit.py',ROOT/'scripts/studies/evaluate_source_anchored_transfer.py',old.SOURCE,old.SOURCE_DATA,old.SOURCE_GRAPH_CACHE,old.SOURCE_SPLIT,old.SCHEDULE,ROOT/'src/qgeognn_al/models/qgeognn_v2.py',ROOT/'src/qgeognn_al/training/predictor.py',ROOT/'src/qgeognn_al/data.py']
    for c in COLUMNS: files += [target_path(c),old.OLD/'data_audit'/f'graph_cache_{c}_only.pt',old.CONDITIONAL/f'features_{c}.csv']
    record=dict(config=CONFIG,methods=METHODS,baselines=BASELINES,contexts=120,neural_fits=240,residual_fits=120,hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in files},developmental_evidence=True)
    record=json.loads(json.dumps(record))
    path=STUDY/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=record: raise RuntimeError('frozen protocol changed')
    else: atomic_json(path,record)
    return record


def metadata(path,ids):
    return pd.read_csv(path,usecols=[*old.FEATURES,'column_specs']).set_index('sample_id').loc[ids].reset_index()


def attach(graphs,frame,normalizer):
    x=normalizer.transform(frame)
    mass=context_matrix(frame)[:,0]
    for a,values,m in zip(graphs[0],x,mass):
        a.column_context=torch.tensor(values).reshape(1,4)
        a.packing_mass=torch.tensor([[m]],dtype=torch.float32)
    return graphs


def balanced_indices(seed,epoch,source_count,target_count):
    target=np.random.default_rng(seed*10000+epoch).permutation(target_count)
    source=np.random.default_rng(np.random.SeedSequence([seed,epoch,41004])).choice(source_count,target_count,replace=False)
    return source,target


def batch(graphs,ids=None): return old.batch(graphs,ids)


def predict(model,graphs):
    model.eval()
    out=[]
    with torch.no_grad():
        for a,b in zip(*loader_pair(*graphs,range(len(graphs[0])),128)):
            p=model(a,b)
            if model.mass_normalized: p=p*a.packing_mass
            out.append(p.numpy())
    result=np.vstack(out)
    if not np.isfinite(result).all(): raise RuntimeError('nonfinite prediction')
    return result


def fit(method,seed,target,source,contract,scales):
    out=STUDY/'runtime'/context_path(contract['column'],contract['protocol'],seed,contract['budget'])/method
    out.mkdir(parents=True,exist_ok=True)
    summary_path=out/'fit_summary.json'
    if summary_path.exists():
        record=json.loads(summary_path.read_text())
        if record['contract']!=contract or sha256_file(out/'best.pt')!=record['checkpoint_sha256']: raise RuntimeError('fit drift')
        return record
    seed_everything(seed)
    model=ColumnConditionedQGeoGNN(load_predictor_checkpoint(old.SOURCE),method.startswith('mass_normalized'))
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    valid=batch(target['validation']); count=len(target['gradient_train'][0])
    best,stale,best_epoch,start,elapsed=float('inf'),0,0,1,0.
    history=[]; used=set()
    resume=out/'last.pt'
    if resume.exists():
        r=torch.load(resume,weights_only=False)
        if r['contract']!=contract: raise RuntimeError('resume drift')
        model.load_state_dict(r['model']); optimizer.load_state_dict(r['optimizer'])
        best,stale,best_epoch,start,elapsed=r['best'],r['stale'],r['best_epoch'],r['epoch']+1,r['elapsed']
        history=r['history'];used=set(r['used'])
    started=time.monotonic()
    for epoch in range(start,CONFIG['maximum_epochs']+1):
        if stale>=CONFIG['patience']: break
        model.training_joint()
        si,ti=balanced_indices(seed,epoch,len(source[0]),count)
        sa,sb=batch(source,si);ta,tb=batch(target['gradient_train'],ti)
        sy=sa.y/sa.packing_mass if model.mass_normalized else sa.y
        ty=ta.y/ta.packing_mass if model.mass_normalized else ta.y
        sl=old.loss_pair(sy,model(sa,sb));tl=old.loss_pair(ty,model(ta,tb))
        loss=.5*sl+.5*tl
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise RuntimeError('nonfinite gradient')
        optimizer.step();used.update(map(int,si));model.eval()
        with torch.no_grad():
            pv=model(*valid)
            if model.mass_normalized: pv=pv*valid[0].packing_mass
            score=old.point_metrics(valid[0].y.numpy(),pv.numpy(),scales)['combined_normalized_rmse']
        if not np.isfinite(score): raise RuntimeError('nonfinite validation')
        history.append(dict(epoch=epoch,source_loss=float(sl.detach()),target_loss=float(tl.detach()),validation_score=score,source_rows=len(si),target_rows=len(ti)))
        if score<best:
            best,stale,best_epoch=score,0,epoch
            old.save_torch(out/'best.pt',dict(model=model.state_dict(),contract=contract))
        else: stale+=1
        if epoch%25==0 or stale>=CONFIG['patience'] or epoch==CONFIG['maximum_epochs']:
            old.save_torch(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),contract=contract,best=best,stale=stale,best_epoch=best_epoch,epoch=epoch,elapsed=elapsed+time.monotonic()-started,history=history,used=sorted(used)))
            atomic_json(out/'progress.json',dict(epoch=epoch,best_epoch=best_epoch))
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    result=dict(contract=contract,method=method,epochs_run=len(history),best_epoch=best_epoch,validation_score=best,source_draws=len(history)*count,source_unique_rows=len(used),source_positions=sorted(used),target_rows=count,balanced_batches=all(r['source_rows']==r['target_rows'] for r in history),shared_head=True,parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha256_file(out/'best.pt'),wall_seconds=elapsed+time.monotonic()-started,fit_success=True)
    atomic_json(summary_path,result)
    return result


def representations(model,graphs):
    model.eval(); result=[]
    with torch.no_grad():
        for a,b in zip(*loader_pair(*graphs,range(len(graphs[0])),128)): result.append(model.extract_representation(a,b).numpy())
    return np.vstack(result)


def source_predict(model,graphs):
    model.eval()
    with torch.no_grad():
        return np.vstack([model(a,b).numpy() for a,b in zip(*loader_pair(*graphs,range(len(graphs[0])),128))])


def residual_features(source_model,graphs,frame):
    ea=frame['PE/EA'].map(lambda s: float(s.split('/')[1])/sum(map(float,s.split('/')))).to_numpy()
    return np.column_stack([representations(source_model,graphs),ea,context_matrix(frame)])


def run_context(c,p,s,b):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);prepare()
    relative=context_path(c,p,s,b);out=STUDY/'contexts'/relative
    if (out/'frozen.json').exists(): verify_manifest(out/'frozen.json');return
    usage=old.ledger(c,p,s,b)
    preprocessing=torch.load(old.SOURCE,weights_only=False)['preprocessing']
    target,source,source_ids,cache,_=old.load_fitting_inputs(usage,preprocessing)
    source_frame=metadata(old.SOURCE_DATA,source_ids)
    frames={role:metadata(target_path(c),usage[role]) for role in ['gradient_train','validation']}
    train_frame=pd.concat([source_frame,frames['gradient_train']],ignore_index=True)
    norm=TrainingNormalizer().fit(train_frame,[*source_ids,*usage['gradient_train']])
    attach(source,source_frame,norm)
    for role in frames: attach(target[role],frames[role],norm)
    contract=dict(protocol_sha256=sha256_file(STUDY/'protocol.json'),column=c,protocol=p,seed=s,budget=b,target_train_ids=usage['gradient_train'],target_validation_ids=usage['validation'])
    fits={}
    for method in NEURAL:
        fits[method]=fit(method,s,target,source,contract,preprocessing['target_scales'])
        print(f'FIT {c}/{p}/{s}/{b} {method} epochs={fits[method]["epochs_run"]}',flush=True)
    source_model=load_predictor_checkpoint(old.SOURCE)
    x=residual_features(source_model,target['gradient_train'],frames['gradient_train'])
    source_q=source_predict(source_model,target['gradient_train'])[:,[1,4]]
    mass=context_matrix(frames['gradient_train'])[:,0]/4
    y=selected_truth(target_path(c),usage['gradient_train'])
    scaler=StandardScaler().fit(x)
    residual=Ridge(alpha=CONFIG['ridge_alpha']).fit(scaler.transform(x),y-source_q*mass[:,None])
    # Only now construct blind test graphs; no test outcomes are parsed.
    test_frame=metadata(target_path(c),usage['test'])
    test=attach(old.read_graphs(target_path(c),usage['test'],cache,preprocessing),test_frame,norm)
    refpath=old.STUDY/'contexts'/relative
    refmanifest=verify_manifest(refpath/'frozen.json')
    columns=[f'{m}_{t}' for m in BASELINES for t in ['V1','V2']]
    table=pd.read_csv(refpath/'predictions_blind.csv.gz',usecols=['sample_id',*columns]).set_index('sample_id').loc[usage['test']]
    q=source_predict(source_model,test)[:,[1,4]]
    physical=q*(context_matrix(test_frame)[:,0]/4)[:,None]
    table[[f'{METHODS[0]}_{t}' for t in ['V1','V2']]]=physical
    table[[f'{METHODS[1]}_{t}' for t in ['V1','V2']]]=physical+residual.predict(scaler.transform(residual_features(source_model,test,test_frame)))
    for method in NEURAL:
        model=ColumnConditionedQGeoGNN(source_model,method.startswith('mass_normalized'))
        model.load_state_dict(torch.load(STUDY/'runtime'/relative/method/'best.pt',weights_only=False)['model'])
        table[[f'{method}_{t}' for t in ['V1','V2']]]=predict(model,test)[:,[1,4]]
    if not np.isfinite(table.to_numpy()).all(): raise RuntimeError('nonfinite blind predictions')
    out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'predictions_blind.csv.gz',compression={'method':'gzip','mtime':0})
    atomic_json(out/'label_usage.json',{**usage,'source_train_ids':source_ids,'normalization_ids':norm.ids,'normalization_mean':norm.mean.tolist(),'normalization_scale':norm.scale.tolist(),'residual_normalization_ids':usage['gradient_train'],'residual_alpha':100.,'residual_coef':residual.coef_.tolist(),'residual_intercept':residual.intercept_.tolist(),'residual_feature_mean':scaler.mean_.tolist(),'residual_feature_scale':scaler.scale_.tolist(),'reference_freeze_sha256':sha256_file(refpath/'frozen.json')})
    atomic_json(out/'fit_audit.json',fits)
    files=['predictions_blind.csv.gz','label_usage.json','fit_audit.json']
    atomic_json(out/'frozen.json',dict(protocol_sha256=contract['protocol_sha256'],frozen_at_unix=time.time(),files={n:sha256_file(out/n) for n in files}))


def freeze_all():
    prepare();files={}
    for c in expected_contexts():
        path=STUDY/'contexts'/context_path(*c)/'frozen.json';r=verify_manifest(path)
        if r['protocol_sha256']!=sha256_file(STUDY/'protocol.json'): raise RuntimeError('context protocol drift')
        files[str(path.relative_to(STUDY))]=sha256_file(path)
    payload=dict(contexts=120,files=files,protocol_sha256=sha256_file(STUDY/'protocol.json'),target_test_evaluation_started=False)
    p=STUDY/'all_predictions_frozen.json'
    if p.exists() and json.loads(p.read_text())!=payload: raise RuntimeError('global freeze drift')
    atomic_json(p,payload)


def execute(workers):
    prepare();logs=STUDY/'runtime/logs';logs.mkdir(parents=True,exist_ok=True)
    def worker(c):
        log=logs/('_'.join(map(str,c))+'.log')
        with log.open('a') as f: r=subprocess.run([sys.executable,str(Path(__file__)),'--context',*map(str,c)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        if r.returncode: raise RuntimeError(f'context failed: {c}; {log}')
        return c
    pending=[c for c in expected_contexts() if not (STUDY/'contexts'/context_path(*c)/'frozen.json').exists()]
    print(f'Pending {len(pending)} contexts; workers={workers}',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,c) for c in pending]
        for i,f in enumerate(as_completed(futures),1): print(f'FROZEN {120-len(pending)+i}/120 {f.result()}',flush=True)
    freeze_all()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--context',nargs=4);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--freeze',action='store_true');a=parser.parse_args()
    if a.context: c,p,s,b=a.context;run_context(c,p,int(s),int(b))
    elif a.execute: execute(a.workers)
    elif a.freeze: freeze_all()
    else: prepare()
