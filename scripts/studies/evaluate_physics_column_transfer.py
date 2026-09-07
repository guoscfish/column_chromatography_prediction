#!/usr/bin/env python3
"""Evaluation is unreachable until every preregistered prediction is frozen."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.studies import run_physics_column_transfer as run
from scripts.studies.evaluate_source_anchored_transfer import metric,sensitivity,error_summary
OUT=run.STUDY


def validate_freeze():
    run.prepare()
    f=run.verify_manifest(OUT/'all_predictions_frozen.json')
    expected={str(Path('contexts')/run.context_path(*c)/'frozen.json') for c in run.expected_contexts()}
    if f['contexts']!=120 or set(f['files'])!=expected: raise RuntimeError('incomplete 120-context freeze')
    if f['protocol_sha256']!=run.sha256_file(OUT/'protocol.json'): raise RuntimeError('protocol mismatch')
    for name in f['files']:
        r=run.verify_manifest(OUT/name)
        if r['protocol_sha256']!=f['protocol_sha256']: raise RuntimeError('context mismatch')
    return f


def paired(frame,value,endpoint):
    rows=[]
    for (c,p),g in frame.groupby(['column','protocol']):
        v=g.pivot(index='seed',columns='method',values=value)
        if len(v)!=5 or v.isna().any().any(): raise RuntimeError('missing paired seeds')
        for method in run.METHODS:
            for reference in run.BASELINES:
                delta=v[method]-v[reference];gain=1-v[method].mean()/v[reference].mean()
                no_regression=True
                if endpoint=='budget100':
                    means=g.groupby('method')[['V1_rmse','V2_rmse']].mean()
                    no_regression=bool((means.loc[method]<=means.loc[reference]).all())
                wins=int((delta < -1e-7).sum())
                rows.append(dict(column=c,protocol=p,method=method,reference=reference,endpoint=endpoint,relative_gain=gain,mean_delta=delta.mean(),median_delta=delta.median(),std_delta=delta.std(),wins=wins,ties=int((abs(delta)<=1e-7).sum()),seeds=5,no_output_regression=no_regression,material=bool(gain>=.05 and delta.median()<0 and wins>=4 and no_regression),stronger_10pct=bool(gain>=.1 and delta.median()<0 and wins>=4 and no_regression)))
    return pd.DataFrame(rows)


def decision(pairs):
    gates={}
    for method in run.METHODS:
        gates[method]={}
        for endpoint in ['aulc','budget100']:
            q=pairs.loc[pairs.method.eq(method)&pairs.endpoint.eq(endpoint)]
            contexts=[(c,p) for (c,p),g in q.groupby(['column','protocol']) if len(g)==len(run.BASELINES) and g.material.all()]
            replicated=len({c for c,p in contexts if p=='compound'})>=2 or any((c,'row') in contexts and (c,'compound') in contexts for c in run.COLUMNS)
            gates[method][endpoint]=dict(contexts=contexts,replicated=replicated)
    supported=any(gates[m]['aulc']['replicated'] or gates[m]['budget100']['replicated'] for m in run.NEURAL)
    joint=[m for m in run.METHODS if all(gates[m][e]['replicated'] for e in ['aulc','budget100'])]
    return dict(decision='EXPLICIT_COLUMN_CONTEXT_SUPPORTED_FOR_LOW_LABEL_TRANSFER' if supported else 'CURRENT_DATA_DO_NOT_IDENTIFY_COLUMN_PHYSICS_SUFFICIENTLY',gates=gates,joint_endpoint_candidates=joint,transfer_solved=False,developmental_evidence=True,new_models_after_test=0)


def evaluate():
    freeze=validate_freeze()
    event=OUT/'test_evaluation_started.json'
    if not event.exists(): run.atomic_json(event,dict(unix_time=time.time(),prediction_freeze_sha256=run.sha256_file(OUT/'all_predictions_frozen.json')))
    scales=run.torch.load(run.old.SOURCE,weights_only=False)['preprocessing']['target_scales']
    features={c:pd.read_csv(run.old.CONDITIONAL/f'features_{c}.csv',usecols=['sample_id','canonical_smiles','EA_fraction','source_V1','source_V2','Flow mL/min']).set_index('sample_id') for c in run.COLUMNS}
    rows,strata,center_width,training=[],[],[],[]
    for name in freeze['files']:
        out=(OUT/name).parent;usage=json.loads((out/'label_usage.json').read_text())
        keys={k:usage[k] for k in ['column','protocol','seed','budget','actual_budget']}
        table=pd.read_csv(out/'predictions_blind.csv.gz').set_index('sample_id')
        if list(table.index)!=usage['test']: raise RuntimeError('identity mismatch')
        if not np.isfinite(table.to_numpy()).all(): raise RuntimeError('nonfinite predictions')
        if set(usage['normalization_ids'])!=set(usage['source_train_ids']+usage['gradient_train']): raise RuntimeError('normalization leakage')
        if set(usage['residual_normalization_ids'])!=set(usage['gradient_train']): raise RuntimeError('residual normalization leakage')
        y=run.selected_truth(run.target_path(usage['column']),usage['test'])
        f=features[usage['column']].loc[table.index];tf=features[usage['column']].loc[usage['gradient_train']]
        for method in [*run.BASELINES,*run.METHODS]:
            pred=table[[f'{method}_V1',f'{method}_V2']].to_numpy()
            extra,parts=sensitivity(y,pred,f,tf,{**keys,'method':method})
            rows.append({**keys,'method':method,**metric(y,pred,scales),**extra});strata.extend(parts)
            for j,target in enumerate(['V1','V2']):
                for flow,group in f.groupby('Flow mL/min'):
                    mask=f['Flow mL/min'].eq(flow).to_numpy();eligible=len(group)>=10 and group.canonical_smiles.nunique()>=3
                    stats=error_summary((pred-y)[mask,j]) if eligible else dict(n=len(group),rmse=np.nan,mae=np.nan,median_absolute_error=np.nan)
                    strata.append({**keys,'method':method,'target':target,'dimension':'flow_ml_min','level':str(flow),'compounds':group.canonical_smiles.nunique(),'eligible':eligible,**stats})
            for target,truth,prediction in [('center',y.mean(1),pred.mean(1)),('width',y[:,1]-y[:,0],pred[:,1]-pred[:,0])]:
                center_width.append({**keys,'method':method,'target':target,**error_summary(prediction-truth)})
        fits=json.loads((out/'fit_audit.json').read_text())
        for method,fit in fits.items():
            if not fit['balanced_batches'] or not fit['shared_head'] or not fit['fit_success']: raise RuntimeError('training contract failed')
            if fit['contract']['target_train_ids']!=usage['gradient_train']: raise RuntimeError('training ID drift')
            training.append({**keys,**{k:v for k,v in fit.items() if not isinstance(v,(list,dict))}})
    metrics=pd.DataFrame(rows)
    if len(metrics)!=120*(len(run.BASELINES)+len(run.METHODS)) or len(training)!=240: raise RuntimeError('missing methods')
    metrics.to_csv(OUT/'all_metrics.csv',index=False)
    numeric=[c for c in metrics.select_dtypes('number') if c not in ['seed','budget','actual_budget']]
    metrics.groupby(['column','protocol','budget','method'])[numeric].agg(['mean','std','median','min','max']).to_csv(OUT/'aggregate_metrics.csv')
    b100=metrics.loc[metrics.budget.eq(100)];b100.to_csv(OUT/'budget100_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'error_stratification.csv',index=False)
    pd.DataFrame(center_width).to_csv(OUT/'center_width_metrics.csv',index=False)
    pd.DataFrame(training).to_csv(OUT/'training_audit.csv',index=False)
    areas=[]
    for keys,g in metrics.groupby(['column','protocol','seed','method']):
        g=g.sort_values('budget')
        if list(g.budget)!=list(run.BUDGETS): raise RuntimeError('budget gap')
        areas.append({**dict(zip(['column','protocol','seed','method'],keys)),'normalized_aulc':np.trapezoid(g.normalized_rmse,g.budget)/70,'actual_budget_aulc':np.trapezoid(g.normalized_rmse,g.actual_budget)/(g.actual_budget.iloc[-1]-g.actual_budget.iloc[0])})
    aulc=pd.DataFrame(areas);aulc.to_csv(OUT/'aulc_by_seed.csv',index=False)
    pairs=pd.concat([paired(aulc,'normalized_aulc','aulc'),paired(b100,'combined_normalized_rmse','budget100')],ignore_index=True)
    pairs.to_csv(OUT/'paired_comparisons.csv',index=False)
    result=decision(pairs);run.atomic_json(OUT/'decision.json',result)
    run.atomic_json(OUT/'execution_audit.json',dict(status='complete',contexts=120,neural_fits=240,residual_fits=120,metric_rows=len(metrics),missing_contexts=0,failed_fits=0,nonfinite_predictions=0,all_predictions_frozen_before_test=True,other_target_labels_used=0,test_labels_used_for_fit=0,normalization_test_rows=0,shared_head=True,balanced_task_weights=[.5,.5],protocol_sha256=run.sha256_file(OUT/'protocol.json'),prediction_freeze_sha256=run.sha256_file(OUT/'all_predictions_frozen.json'),completed_at_unix=time.time()))
    print(json.dumps(result,indent=2))
if __name__=='__main__': evaluate()
