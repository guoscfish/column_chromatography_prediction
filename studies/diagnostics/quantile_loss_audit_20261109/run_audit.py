"""Read-only L333 training/validation diagnosis on independently pinned snapshots.

No optimizer, model fitting, provider, test truth or U truth reader exists here.
Only aggregate diagnostics leave the private snapshot directory.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from quantile_loss_audit_20261109.quantiles import quantile_summary,sorting_diagnostic,ranking_metrics,score_overlap
from quantile_loss_audit_20261109.incremental import incremental_width_diagnostic,seed_bootstrap_interval
from quantile_loss_audit_20261109.loss_diagnostics import measure_gradients,state_hash


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def readonly_file(path,digest):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink!=1 or path.stat().st_mode&0o222 or sha(path)!=digest:
        raise ValueError('independent readonly snapshot/hash required: '+str(path))


def source_namespace_isolated(source):
    source=Path(source).resolve()
    for name,module in list(sys.modules.items()):
        if name in {'QGeoGNN','utils'} or name=='src.qgeognn_al' or name.startswith('src.qgeognn_al.'):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(source):
                raise ValueError('mutable/mixed scientific module namespace: '+name)


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def nearest_distance(query,train):
    query,train=np.asarray(query,float),np.asarray(train,float)
    squared=np.maximum(np.square(query).sum(1)[:,None]+np.square(train).sum(1)[None,:]-2*query@train.T,0)
    return np.sqrt(squared.min(1))


def _per_row_gradient_norms(model,atom,angle,positions,scales,loader_pair):
    import torch
    parameters=[p for p in model.parameters() if p.requires_grad];rows=[]
    for a,b in zip(*loader_pair(atom,angle,positions,batch_size=1)):
        output=model(a,b);norms=[]
        for endpoint,column in enumerate((1,4)):
            values=torch.autograd.grad(output[0,column]/float(scales[endpoint]),parameters,retain_graph=endpoint==0,allow_unused=True)
            norm=sum(float(torch.square(value.detach().to(torch.float64)).sum()) for value in values if value is not None)
            norms.append(np.sqrt(norm))
        rows.append(norms)
    return np.asarray(rows)


def run(manifest_path,legacy_source,output):
    import torch
    import pandas as pd
    torch.set_num_threads(1)
    manifest_path,legacy_source,output=Path(manifest_path).resolve(),Path(legacy_source).resolve(),Path(output).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest['allowed_roles']!=['l0','validation'] or any(manifest['label_access'].get(k)!=0 for k in ['u0_values_converted','test_values_converted']):
        raise ValueError('this diagnosis admits only authorized L0 and validation target rows')
    identity=json.loads((legacy_source/'SOURCE_IDENTITY.json').read_text())
    if identity['commit']!=manifest['source_code_commit']:raise ValueError('frozen code commit mismatch')
    for relative,digest in identity['files'].items():readonly_file(legacy_source/relative,digest)
    # The module namespace is pinned to a separate read-only code snapshot, not
    # the active main source tree or a mutable refactor API.
    source_namespace_isolated(legacy_source)
    sys.path.insert(0,str(legacy_source))
    from src.qgeognn_al.models import load_predictor_checkpoint
    from src.qgeognn_al.data import build_model_data,condition_matrix
    from src.qgeognn_al.training.predictor import loader_pair
    from torch_geometric.loader import DataLoader
    source_namespace_isolated(legacy_source)
    cache_record=next(r for r in manifest['source_files'] if r.get('snapshot','').endswith('graph_cache_4g.pt'))
    readonly_file(cache_record['snapshot'],cache_record['sha256'])
    cache=torch.load(cache_record['snapshot'],map_location='cpu',weights_only=False)
    output.mkdir(parents=True,exist_ok=True)
    component_rows=[];competition_rows=[];quantile_rows=[];group_rows=[];signal_rows=[];incremental_rows=[];regression=[];source_checks=[]
    for record in manifest['seeds']:
        started=time.monotonic();seed=record['seed'];folder=Path(record['authorized_rows']).parent
        readonly_file(record['authorized_rows'],record['authorized_rows_sha256']);readonly_file(record['checkpoint']['snapshot'],record['checkpoint']['sha256'])
        frame=pd.read_csv(record['authorized_rows'])
        if set(frame.role)!={'l0','validation'} or frame.sample_id.duplicated().any():raise ValueError('snapshot identity/role mismatch')
        train=np.flatnonzero(frame.role.eq('l0'));valid=np.flatnonzero(frame.role.eq('validation'))
        if len(train)!=333 or len(valid)!=416:raise ValueError('declared L333 validation partition mismatch')
        payload=torch.load(record['checkpoint']['snapshot'],map_location='cpu',weights_only=False)
        model=load_predictor_checkpoint(record['checkpoint']['snapshot']);before=state_hash(model)
        scales=np.asarray([payload['preprocessing']['target_scales'][t] for t in ['V1','V2']])
        truth=frame[['V1_ml','V2_ml']].to_numpy(np.float32)
        np.testing.assert_allclose(truth[train].std(0),scales,rtol=1e-6,atol=1e-6)
        atom,angle=build_model_data(frame,cache,pd.DataFrame(),payload['preprocessing']['scaler'])
        predictions=[];hidden=[]
        with torch.no_grad():
            for a,b in zip(*loader_pair(atom,angle,range(len(frame)),batch_size=64)):
                predictions.append(model(a,b).numpy());hidden.append(model.extract_representation(a,b).numpy())
        prediction=np.vstack(predictions);latent=np.vstack(hidden)
        # A compact real-input checkpoint/graph regression fixture for task B.
        fixed=np.concatenate([train[:8],valid[:8]])
        ga=[atom[i] for i in fixed];gb=[angle[i] for i in fixed]
        with torch.no_grad():
            aa=next(iter(DataLoader(ga,batch_size=16)));bb=next(iter(DataLoader(gb,batch_size=16)))
            fixed_output=model(aa,bb).numpy();fixed_hidden=model.extract_representation(aa,bb).numpy()
        graph_path=folder/'regression_graphs.pt';expected_path=folder/'regression_expected.npz'
        torch.save({'atom':ga,'angle':gb,'sample_ids':frame.sample_id.iloc[fixed].astype(str).tolist(),'roles':frame.role.iloc[fixed].tolist(),'source_positions':frame.canonical_index.iloc[fixed].astype(int).tolist()},graph_path)
        np.savez(expected_path,outputs=fixed_output,hidden=fixed_hidden);graph_path.chmod(0o444);expected_path.chmod(0o444)
        regression.append({'seed':seed,'checkpoint':record['checkpoint']['snapshot'],'checkpoint_sha256':record['checkpoint']['sha256'],'graphs':str(graph_path),'graphs_sha256':sha(graph_path),'expected':str(expected_path),'expected_sha256':sha(expected_path),'rows':16,'allowed_roles':['l0','validation'],'old_code_commit':identity['commit'],'tolerance':{'rtol':0,'atol':1e-6}})
        batches=((a,b,a.y) for a,b in zip(*loader_pair(atom,angle,train,batch_size=16)))
        gradient=measure_gradients(model,batches,row_count=len(train))
        common={'status':'NEW_OFFLINE_MEASURED','seed':seed,'budget':333,'data_role':'row'}
        for row in gradient['components']:component_rows.append({**common,'split':'train',**row})
        for row in gradient['competition']:competition_rows.append({**common,'split':'train',**row})
        for split,positions in [('train',train),('validation',valid)]:
            for endpoint,target in enumerate(['V1','V2']):
                raw=prediction[positions,3*endpoint:3*endpoint+3];y=truth[positions,endpoint]
                metrics=quantile_summary(y,raw);sorting=sorting_diagnostic(y,raw)
                metrics.update({f'sorting_{k}':v for k,v in sorting.items()})
                metrics.update(training_sigma=float(scales[endpoint]),normalized_q50_mse=metrics['q50_mse']/float(scales[endpoint]**2),normalized_q50_rmse=metrics['q50_rmse']/float(scales[endpoint]))
                quantile_rows.append({**common,'split':split,'target':target,**metrics})
                if split=='validation':
                    training_cuts=np.quantile(truth[train,endpoint],[.25,.5,.75])
                    response_groups=np.searchsorted(training_cuts,y,side='right')
                    error=np.abs(y-raw[:,1]);difficulty_groups=np.searchsorted(np.quantile(error,[.25,.5,.75]),error,side='right')
                    for grouping,groups in [('training_response_quartiles',response_groups),('retrospective_validation_error_quartiles',difficulty_groups)]:
                        for group in range(4):
                            take=groups==group
                            if take.any():group_rows.append({**common,'split':split,'target':target,'grouping':grouping,'group':group,'rows':int(take.sum()),'coverage':float(np.mean((y[take]>=raw[take,0])&(y[take]<=raw[take,2]))),'mean_raw_width':float(np.mean(raw[take,2]-raw[take,0]))})
        # The fixed validation diagnostic controls feasible label-free signals.
        # No different outer-seed models are pooled into a fake ensemble.
        grad_norms=_per_row_gradient_norms(model,atom,angle,valid,scales,loader_pair)
        distance=nearest_distance(latent[valid],latent[train])
        conditions=condition_matrix(frame).astype(float);mean=conditions[train].mean(0);std=conditions[train].std(0);std[std<1e-8]=1.
        standardized=(conditions-mean)/std;condition_distance=nearest_distance(standardized[valid],standardized[train])
        width=np.column_stack([prediction[valid,2]-prediction[valid,0],prediction[valid,5]-prediction[valid,3]])/scales
        crossing=np.column_stack([(prediction[valid,0]>prediction[valid,1])|(prediction[valid,1]>prediction[valid,2]),(prediction[valid,3]>prediction[valid,4])|(prediction[valid,4]>prediction[valid,5])]).astype(float)
        base=np.column_stack([prediction[valid][:,[1,4]]/scales,np.log1p(distance),np.log1p(grad_norms),condition_distance,standardized[valid]])
        widths=np.column_stack([width,np.maximum(width,0),crossing])
        for endpoint,target in enumerate(['V1','V2']):
            error=np.abs(prediction[valid,3*endpoint+1]-truth[valid,endpoint])/scales[endpoint]
            signals={'signed_width':width[:,endpoint],'clipped_width':np.maximum(width[:,endpoint],0),'latent_distance':distance,'q50_gradient_norm':grad_norms[:,endpoint],'condition_distance':condition_distance,'q50_value':prediction[valid,3*endpoint+1]/scales[endpoint]}
            for name,values in signals.items():signal_rows.append({**common,'split':'validation','target':target,'record_type':'error_ranking','signal':name,**ranking_metrics(error,values)})
            for name,values in signals.items():
                if name not in ['signed_width','clipped_width']:signal_rows.append({**common,'split':'validation','target':target,'record_type':'score_overlap_with_signed_width','signal':name,**score_overlap(width[:,endpoint],values)})
            inc=incremental_width_diagnostic(base,widths,error,frame.canonical_smiles.iloc[valid].astype(str).tolist())
            incremental_rows.append({**common,'split':'validation','target':target,'base_controls':'both q50/sigma; latent distance to L0; both normalized q50 full-network gradient norms; 9D condition coverage/distance','missing_control':'same-seed K3 ensemble disagreement NOT_MEASURED',**inc})
        cross=prediction[valid,5]-prediction[valid,0];error=np.linalg.norm((prediction[valid][:,[1,4]]-truth[valid])/scales,axis=1)
        signal_rows.append({**common,'split':'validation','target':'V2_q90_minus_V1_q10','record_type':'confounded_cross_target_width','signal':'elution_span_plus_margins','not_a_quantile_of_difference':True,**ranking_metrics(error,cross)})
        if state_hash(model)!=before or any(p.grad is not None for p in model.parameters()):raise RuntimeError('checkpoint state or parameter gradient storage changed')
        source_checks.append({'seed':seed,'state_before':before,'state_after':state_hash(model),'checkpoint_sha256':record['checkpoint']['sha256'],'authorized_rows_sha256':record['authorized_rows_sha256'],'training_rows':len(train),'validation_rows':len(valid),'optimizer_steps':0,'qgeognn_training_runs':0,'U_truth_access_count':0,'test_truth_access_count':0})
        print(json.dumps({'seed':seed,'budget':333,'rows':len(frame),'state_unchanged':True,'seconds':round(time.monotonic()-started,2)}),flush=True)
    summary=[]
    for target in ['V1','V2']:
        for metric in ['rmse_delta_with_minus_base','baseline_oof_rmse','with_width_oof_rmse']:
            values=[r[metric] for r in incremental_rows if r['target']==target]
            summary.append({'status':'NEW_OFFLINE_MEASURED','budget':333,'target':target,'metric':metric,**seed_bootstrap_interval(values)})
    for group in ['backbone','condition_branch']:
        for metric in ['V2_to_V1_gradient_norm','cosine']:
            values=[r[metric] for r in competition_rows if r['parameter_group']==group and r[metric] is not None]
            summary.append({'status':'NEW_OFFLINE_MEASURED','budget':333,'target':group,'metric':metric,**seed_bootstrap_interval(values)})
    for name,rows in [('loss_components.csv',component_rows),('gradient_competition.csv',competition_rows),('quantile_metrics.csv',quantile_rows),('coverage_groups.csv',group_rows),('signal_diagnostics.csv',signal_rows),('incremental.csv',incremental_rows),('seed_summary.csv',summary)]:write_csv(output/name,rows)
    regression_manifest={'schema_version':1,'status':'NEW_OFFLINE_MEASURED','source_code_commit':identity['commit'],'snapshot_manifest':str(manifest_path),'snapshot_manifest_sha256':sha(manifest_path),'fixtures':regression,'labels_policy':'Only previously authorized L0/train and validation; no U/test truth.'}
    regression_path=manifest_path.parent/'REGRESSION_MANIFEST.json';regression_path.write_text(json.dumps(regression_manifest,indent=2)+'\n');regression_path.chmod(0o444)
    provenance={'status':'NEW_OFFLINE_MEASURED','source_code_commit':identity['commit'],'source_identity_sha256':sha(legacy_source/'SOURCE_IDENTITY.json'),'input_manifest_sha256':sha(manifest_path),'inputs':manifest,'code_hashes':{p.name:sha(p) for p in HERE.glob('*.py')},'seeds':source_checks,'forbidden_operations':{'qgeognn_training_runs':0,'optimizer_steps':0,'paid_llm_requests':0,'new_acquisitions':0,'new_label_reveals':0,'U_truth_access_count':0,'test_truth_access_count':0},'limitations':['Frozen checkpoint eval-mode gradients are not optimizer trajectory or causal target-weight effects.','Budgets L525/L1005/L1485 and compound AL snapshots are not measured.','No same-seed K3 ensemble control is available.','Validation was already used for checkpoint selection; these are retrospective diagnostics, not blind scientific confirmation.']}
    (output/'INPUT_AND_CODE_HASHES.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for relative,digest in identity['files'].items():readonly_file(legacy_source/relative,digest)
    return {'status':'PASS','seeds':len(source_checks),'regression_manifest':str(regression_path),'results':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-manifest',type=Path,required=True)
    parser.add_argument('--legacy-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.snapshot_manifest,args.legacy_source,args.output),indent=2))
