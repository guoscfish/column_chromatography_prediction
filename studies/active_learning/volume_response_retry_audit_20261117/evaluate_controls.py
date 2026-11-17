#!/usr/bin/env python3
"""Score held-out response series only after completed validation-selected fits."""
from pathlib import Path
import json,itertools,sys,hashlib
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'studies/active_learning/decision_mechanism_followup_20261117'))
from model_learnability import classify_pair,pair_numbers,_canonical_conditions
sys.path.insert(0,str(ROOT/'src'))
from qgeognn_al.evaluation.point import point_metrics
SEEDS=[157,887,2357,6101,12203];KINDS=['A_original','B_amount_skip','C_volume_skip']

def series_identity(record,axis):
    return (record['smiles'],)+tuple(v for j,v in enumerate(_canonical_conditions(record)) if j!=axis)

def main():
    rows=[];pairs=[];constraints=[];coverage=[]
    for seed in SEEDS:
        d=json.loads((OUT/f'data/seed_{seed}/identity.json').read_text());v=np.load(OUT/f'data/seed_{seed}/authorized_labels.npz')
        metadata=pd.read_csv(OUT/f'data/seed_{seed}/metadata.csv').set_index('sample_id')
        selected={sid:dict(smiles=metadata.loc[sid,'canonical_smiles'],conditions={k:metadata.loc[sid,k] for k in ['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']}) for sid in d['check_ids']+d['validation_ids']}
        validation_series={axis:{series_identity(selected[sid],axis) for sid in d['validation_ids']} for axis in range(5)}
        for kind in KINDS:
            base=OUT/f'fits/seed_{seed}/{kind}';done=base/'fit_audit.json'
            if not done.exists():coverage.append(dict(seed=seed,kind=kind,complete=False));continue
            audit=json.loads(done.read_text());coverage.append(dict(seed=seed,kind=kind,complete=True))
            for scope,ids,truth in [('validation',d['validation_ids'],v['validation_y']),('check',d['check_ids'],v['check_y'])]:
                table=pd.read_csv(base/f'{scope}_predictions.csv');assert table.sample_id.tolist()==ids
                assert table.columns.tolist()==['sample_id','V1_q10','V1_q50','V1_q90','V2_q10','V2_q50','V2_q90']
                pred=table.drop(columns='sample_id').to_numpy(float);assert np.isfinite(pred).all()
                metric=point_metrics(truth,pred,d['preprocessing']['target_scales'])
                if scope=='validation':assert abs(metric['combined_normalized_rmse']-audit['best_validation_nrmse'])<1e-7
                rows.append(dict(seed=seed,kind=kind,scope=scope,n=len(ids),best_epoch=audit['best_epoch'],epochs_run=audit['epochs_run'],**metric))
                constraints.append(dict(seed=seed,kind=kind,scope=scope,negative_output_fraction=float((pred<0).mean()),
                    V1_quantile_crossing_fraction=float(((pred[:,0]>pred[:,1])|(pred[:,1]>pred[:,2])).mean()),
                    V2_quantile_crossing_fraction=float(((pred[:,3]>pred[:,4])|(pred[:,4]>pred[:,5])).mean()),
                    V2_below_V1_fraction=float((pred[:,4]<pred[:,1]).mean())))
                if scope in ('check','validation'):
                    pos={sid:i for i,sid in enumerate(ids)}
                    for a,b in itertools.combinations(ids,2):
                        match=classify_pair(a,b,selected)
                        if not match:continue
                        axis,lo,hi,lv,hv=match
                        for t in [0,1]:
                            dy=truth[pos[hi],t]-truth[pos[lo],t];dp=pred[pos[hi],3*t+1]-pred[pos[lo],3*t+1]
                            pairs.append(dict(seed=seed,kind=kind,scope=scope,axis=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul'][axis],
                                target=f'V{t+1}',low_id=lo,high_id=hi,low_level=lv,high_level=hv,
                                series_key=hashlib.sha256(repr(series_identity(selected[lo],axis)).encode()).hexdigest(),
                                validation_related_series=series_identity(selected[lo],axis) in validation_series[axis],
                                actual_delta_ml=float(dy),predicted_delta_ml=float(dp),amplitude_error_ml=float(abs(abs(dp)-abs(dy)))))
    pd.DataFrame(coverage).to_csv(OUT/'fit_coverage.csv',index=False)
    if not rows:print('No completed fits yet');return
    pd.DataFrame(rows).to_csv(OUT/'row_metrics.csv',index=False);pd.DataFrame(constraints).to_csv(OUT/'output_constraints.csv',index=False)
    p=pd.DataFrame(pairs);p.to_csv(OUT/'control_response_pairs.csv',index=False)
    stats=[]
    for key,g in p.groupby(['seed','kind','scope','axis','target']):
        for threshold in [.1,.5,1.]:stats.append(dict(zip(['seed','kind','scope','axis','target'],key),n_series=g.series_key.nunique(),amplitude_error_mean_ml=g.amplitude_error_ml.mean(),**pair_numbers(g.actual_delta_ml,g.predicted_delta_ml,threshold,.1)))
    pd.DataFrame(stats).to_csv(OUT/'local_response_metrics.csv',index=False)
    independent=[]
    for key,g in p[(p.scope=='check')&(~p.validation_related_series)].groupby(['seed','kind','scope','axis','target']):
        independent.append(dict(zip(['seed','kind','scope','axis','target'],key),n_series=g.series_key.nunique(),**pair_numbers(g.actual_delta_ml,g.predicted_delta_ml,.5,.1)))
    pd.DataFrame(independent).to_csv(OUT/'validation_unrelated_response_metrics.csv',index=False)
    series=[]
    for key,g in p.groupby(['seed','kind','scope','axis','target','series_key']):
        series.append(dict(zip(['seed','kind','scope','axis','target','series_key'],key),amplitude_error_mean_ml=g.amplitude_error_ml.mean(),**pair_numbers(g.actual_delta_ml,g.predicted_delta_ml,.5,.1)))
    st=pd.DataFrame(series);st.to_csv(OUT/'series_response_metrics.csv',index=False)
    cols=['difference_error_mean','observed_abs_mean','predicted_abs_mean','amplitude_error_mean_ml','direction_accuracy_meaningful','flat_fraction_meaningful']
    st.groupby(['seed','kind','scope','axis','target'])[cols].mean().reset_index().to_csv(OUT/'series_equal_response_metrics.csv',index=False)
    local=pd.DataFrame(stats).query("scope=='check' and axis=='V/ul' and observation_threshold_ml==.5")
    assert all(set(g.target)=={'V1','V2'} for _,g in local.groupby(['seed','kind']))
    local=local.groupby(['seed','kind'])[['difference_error_mean','observed_abs_mean','predicted_abs_mean','flat_fraction_meaningful']].mean().reset_index()
    validation=pd.DataFrame(rows).query("scope=='validation'")
    joined=validation.merge(local,on=['seed','kind']);deltas=[]
    for seed,g in joined.groupby('seed'):
        a=g.set_index('kind')
        for comparison,first,second in [('B-A','B_amount_skip','A_original'),('C-A','C_volume_skip','A_original'),('C-B','C_volume_skip','B_amount_skip')]:
            if first not in a.index or second not in a.index:continue
            x,y=a.loc[first],a.loc[second]
            deltas.append(dict(seed=seed,comparison=comparison,row_nrmse_first=x.combined_normalized_rmse,row_nrmse_second=y.combined_normalized_rmse,
                row_nrmse_delta=x.combined_normalized_rmse-y.combined_normalized_rmse,row_relative_delta=(x.combined_normalized_rmse/y.combined_normalized_rmse)-1,
                local_difference_mae_first=x.difference_error_mean,local_difference_mae_second=y.difference_error_mean,
                local_difference_mae_delta=x.difference_error_mean-y.difference_error_mean,local_relative_delta=x.difference_error_mean/y.difference_error_mean-1))
    pd.DataFrame(deltas).to_csv(OUT/'paired_model_differences.csv',index=False)
    print('Evaluated fits',sum(x['complete'] for x in coverage),'of15',flush=True)

if __name__=='__main__':main()
