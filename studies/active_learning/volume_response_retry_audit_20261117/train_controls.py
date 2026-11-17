#!/usr/bin/env python3
"""Exactly fifteen declared fits, with immutable per-fit start receipts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/volume_controls_mpl')
from pathlib import Path
import argparse,hashlib,json,time,sys,gc
import numpy as np
import pandas as pd
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT/'src'))
from qgeognn_al.training.predictor import seed_everything,loader_pair,target_loss
from qgeognn_al.evaluation.point import point_metrics
from qgeognn_al.schemas.conditions import ConditionNormalization
from isolated_models import build_control,checkpoint,load_control,VARIANTS
from prepare_data import load_seed,assert_identity,dump,sha

def predict(model,atom,angle,indices,identity):
    model.eval();out=[];order=[]
    with torch.no_grad():
        for a,b in zip(*loader_pair(atom,angle,indices,identity['training_config']['batch_size'])):
            assert_identity(a,identity['ordered_ids'],identity['raw_volume_values'])
            out.append(model(a,b).cpu().numpy());order.extend(a.canonical_position.cpu().numpy().reshape(-1).tolist())
    assert order==list(indices)
    return np.vstack(out)

def fit(seed,kind):
    torch.set_num_threads(2)
    gate=json.loads((OUT/'DATA_FEASIBILITY.json').read_text());assert gate['status']=='PASS' and gate['maximum_new_fits']==15
    assert all(sha(OUT/n)==h for n,h in gate['training_code_sha256'].items()),'Training code changed after data gate'
    assert dict(seed=seed,kind=kind,initialization_seed=seed+2000003,training_seed=seed+3000017) in gate['planned_fits']
    dest=OUT/f'fits/seed_{seed}/{kind}';dest.mkdir(parents=True,exist_ok=True)
    done=dest/'fit_audit.json'
    if done.exists():
        old=json.loads(done.read_text());assert sha(dest/'best.pt')==old['checkpoint_sha256'];print('REUSE_COMPLETED',seed,kind,flush=True);return
    start=dest/'started.json'
    fd=os.open(start,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,json.dumps(dict(seed=seed,kind=kind,pid=os.getpid(),started_unix=time.time(),data_gate_sha256=sha(OUT/'DATA_FEASIBILITY.json'))).encode());os.close(fd)
    assert len(list((OUT/'fits').glob('seed_*/*/started.json')))<=15
    identity,atom,angle,values=load_seed(seed)
    expected=next(x for x in gate['pretraining_checks'] if x['seed']==seed)
    assert sha(OUT/f'data/seed_{seed}/identity.json')==expected['data_identity_sha256']
    metadata=pd.read_csv(OUT/f'data/seed_{seed}/metadata.csv')
    identity['raw_volume_values']=metadata['V/ul'].to_numpy(np.float32)
    config=identity['training_config'];train=identity['train_indices'];valid=identity['validation_indices'];check=identity['check_indices']
    assert not set(train)&set(valid) and not set(train)&set(check) and not set(valid)&set(check)
    # Only training and validation graphs receive labels; held-out checks remain zero.
    for indices,ys in [(train,values['train_y']),(valid,values['validation_y'])]:
        for idx,y in zip(indices,ys):
            a=atom[idx].clone();a.y=torch.tensor([y],dtype=torch.float32);atom[idx]=a
    assert all(torch.count_nonzero(atom[i].y)==0 for i in check)
    seed_everything(identity['initialization_seed'])
    v=identity['volume_scale'];norm=ConditionNormalization(**identity['normalization'])
    model=build_control(kind,norm,v['min'],v['max'])
    rng_hash=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert rng_hash==expected['post_init_torch_rng_hash'][kind]
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best=float('inf');best_epoch=None;stale=0;history=[];began=time.time()
    try:
        for epoch in range(1,config['maximum_epochs']+1):
            model.train();order=np.random.default_rng(config['seed']*10000+epoch).permutation(train);losses=[]
            for a,b in zip(*loader_pair(atom,angle,order,config['batch_size'])):
                assert_identity(a,identity['ordered_ids'],identity['raw_volume_values'])
                pred=model(a,b);loss=target_loss(a.y[:,0],pred[:,:3])+target_loss(a.y[:,1],pred[:,3:])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
            pred=predict(model,atom,angle,valid,identity);metrics=point_metrics(values['validation_y'],pred,identity['preprocessing']['target_scales']);score=metrics['combined_normalized_rmse']
            if not np.isfinite(score):raise RuntimeError('Nonfinite validation score')
            history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_selection_score=score,**{'validation_'+k:v for k,v in metrics.items()}))
            if score<best:
                best=score;best_epoch=epoch;stale=0
                saved=checkpoint(model,kind,norm,identity['preprocessing'],config,v,dict(epoch=epoch,validation_selection_score=score,
                    train_ids_hash=identity['train_ids_hash'],check_ids_hash=identity['check_ids_hash'],validation_ids_hash=identity['validation_ids_hash'],check_used_for_selection=False))
                temp=dest/'best.pt.tmp';torch.save(saved,temp);temp.replace(dest/'best.pt')
            else:stale+=1
            if epoch==1 or epoch%10==0:
                progress=dict(seed=seed,kind=kind,epoch=epoch,best_epoch=best_epoch,current_validation=score,best_validation=best,stale=stale,seconds=time.time()-began)
                try:
                    import psutil
                    progress['rss_gb']=psutil.Process().memory_info().rss/1e9
                except ImportError:pass
                dump(dest/'progress.json',progress);pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
                print(json.dumps(progress),flush=True)
            if stale>=config['patience']:break
        pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
        del optimizer,model;gc.collect()
        best_model,ck=load_control(dest/'best.pt')
        for name,indices in [('validation',valid),('check',check)]:
            p=predict(best_model,atom,angle,indices,identity)
            frame=pd.DataFrame(p,columns=ck['output_order']);frame.insert(0,'sample_id',[identity['ordered_ids'][i] for i in indices]);frame.to_csv(dest/f'{name}_predictions.csv',index=False)
        audit=dict(status='COMPLETE',seed=seed,kind=kind,model_variant=VARIANTS[kind],train_rows=len(train),check_rows=len(check),validation_rows=len(valid),
            best_epoch=best_epoch,epochs_run=len(history),best_validation_nrmse=best,training_seconds=time.time()-began,
            checkpoint_sha256=sha(dest/'best.pt'),training_config=config,initialization_seed=identity['initialization_seed'],post_init_torch_rng_hash=rng_hash,
            common_initial_hash=expected['common_initial_hash'][kind],train_ids_hash=identity['train_ids_hash'],check_ids_hash=identity['check_ids_hash'],validation_ids_hash=identity['validation_ids_hash'],
            holdout_used_for_early_stopping=False,test_truth_used=0,unobserved_targets_used=0,parameter_count=ck['parameter_count'])
        dump(done,audit);print('FIT_COMPLETE',seed,kind,best,time.time()-began,flush=True)
    except BaseException as error:
        dump(dest/'FAILED.json',dict(error=repr(error),seed=seed,kind=kind,automatic_retry_allowed=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--kind',choices=list(VARIANTS));args=p.parse_args()
    for kind in ([args.kind] if args.kind else list(VARIANTS)):fit(args.seed,kind)
