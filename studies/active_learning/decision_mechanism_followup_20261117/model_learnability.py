#!/usr/bin/env python3
"""Audit already-revealed same-checkpoint condition responses; no model imports."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,itertools,json,lzma,sys
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101'))
from condition_metrics import _canonical_conditions
AXES=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
FILES={}

def read(p):
    p=Path(p);b=p.read_bytes();assert b==p.read_bytes(),('unstable input',str(p))
    entry=dict(path=str(p.resolve().relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    assert entry['path'] not in FILES or FILES[entry['path']]==entry
    FILES[entry['path']]=entry;return b

def load(p):return json.loads(read(p))
def phase(L):return '333-525' if L<525 else '525-813' if L<813 else '813-1005' if L<1005 else '1005-1485'
def short(v):return hashlib.sha256(v.encode()).hexdigest()[:16]

def classify_pair(a,b,metadata):
    if metadata[a]['smiles']!=metadata[b]['smiles']:return None
    ca,cb=map(lambda i:_canonical_conditions(metadata[i]),[a,b])
    if ca is None or cb is None:return None
    changed=[j for j in range(5) if ca[j]!=cb[j]]
    if len(changed)!=1:return None
    j=changed[0]
    va,vb=(1/(1+float(ca[0])),1/(1+float(cb[0]))) if j==0 else (ca[j],cb[j])
    if va>vb:a,b,va,vb=b,a,vb,va
    return j,a,b,str(va),str(vb)

def pair_numbers(actual_delta,pred_delta,threshold=.5,flat=.1):
    a=np.asarray(actual_delta,float);p=np.asarray(pred_delta,float)
    meaningful=np.abs(a)>threshold;nonflat=np.abs(p)>=flat
    sign=np.sign(a)==np.sign(p)
    return dict(n=len(a),n_meaningful=int(meaningful.sum()),
        observed_abs_mean=float(np.abs(a).mean()),predicted_abs_mean=float(np.abs(p).mean()),
        observed_abs_median=float(np.median(np.abs(a))),predicted_abs_median=float(np.median(np.abs(p))),
        observed_abs_p90=float(np.quantile(np.abs(a),.9)),predicted_abs_p90=float(np.quantile(np.abs(p),.9)),
        difference_error_mean=float(np.abs(a-p).mean()),
        excess_difference_error_over_flat=float((np.abs(a-p)-np.abs(a)).mean()),
        flat_prediction_fraction=float((~nonflat).mean()),
        direction_accuracy_meaningful=float(sign[meaningful].mean()) if meaningful.any() else np.nan,
        direction_correct_nonflat_fraction_meaningful=float((sign&nonflat)[meaningful].mean()) if meaningful.any() else np.nan,
        wrong_direction_nonflat_fraction_meaningful=float(((a*p<0)&nonflat)[meaningful].mean()) if meaningful.any() else np.nan,
        flat_fraction_meaningful=float((~nonflat)[meaningful].mean()) if meaningful.any() else np.nan,
        direction_accuracy_given_nonflat_meaningful=float(sign[meaningful&nonflat].mean()) if (meaningful&nonflat).any() else np.nan,
        observation_threshold_ml=threshold,prediction_flat_threshold_ml=flat)

def aggregate_pairs(pairs,keys,thresholds,flat):
    rows=[]
    for key,g in pairs.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        base=dict(zip(keys,key));base.update(n_unique_pairs=g.pair_key.nunique(),n_molecules=g.molecule_key.nunique(),n_batches=g['round'].nunique())
        for t in thresholds:rows.append(dict(**base,**pair_numbers(g.actual_delta_ml,g.predicted_delta_ml,t,flat)))
    return pd.DataFrame(rows)

def metadata_holdouts(data,cfg):
    rows=[];groupsout=[]
    for m in ['new_free_cw_sol61_xhigh']:
        for seed in [157,887,2357,6101,12203]:
            initial=set(data['initial'][str(seed)])
            for budget in cfg['model_control_budgets']:
                state=next(b for b in data['states'] if b['method']==m and b['seed']==seed and b['L_before']==budget)
                h=state['labeled_ids'];meta=data['metadata'];groups=defaultdict(list)
                for sid in h:
                    c=_canonical_conditions(meta[sid]);key=(meta[sid]['smiles'],)+c[:2]+c[3:]
                    groups[key].append(sid)
                eligible=[];excluded=0
                for key,ids in groups.items():
                    levels={_canonical_conditions(meta[v])[2] for v in ids}
                    if len(levels)<2:continue
                    # Exclude the entire response-series group if any initial row occurs.
                    if set(ids)&initial:excluded+=1;continue
                    digest=hashlib.sha256((cfg['model_control_holdout_salt']+'\0'+repr(key)).encode()).hexdigest()
                    eligible.append((digest,key,ids))
                eligible.sort();n=cfg['model_control_min_pair_groups'];chosen=eligible[:n] if len(eligible)>=n else []
                holdout={sid for _,_,ids in chosen for sid in ids}
                rows.append(dict(method=m,seed=seed,budget=budget,available_response_series=len(eligible),
                    excluded_series_touching_L333=excluded,required_series=n,feasible=len(eligible)>=n,
                    planned_holdout_n=len(holdout),planned_training_n=budget-len(holdout),labels_read_for_planning=False))
                for digest,key,ids in chosen:
                    groupsout.append(dict(method=m,seed=seed,budget=budget,series_key=digest,ids='|'.join(sorted(ids)),
                        levels_ul='|'.join(sorted({str(_canonical_conditions(meta[v])[2]) for v in ids}))))
    pd.DataFrame(rows).to_csv(OUT/'model_control_feasibility.csv',index=False)
    pd.DataFrame(groupsout,columns=['method','seed','budget','series_key','ids','levels_ul']).to_csv(OUT/'model_control_holdouts.csv',index=False)

def main(config):
    cfg=load(config);data=json.loads(lzma.decompress(read(ROOT/cfg['historical_states'])))
    meta=data['metadata'];states=data['states'];protocol=load(ROOT/cfg['fixed_scales_protocol'])
    scales={v['seed']:v['preprocessing']['target_scales'] for v in protocol['initial_states']}
    pairs=[];alloc=[];rowerrors=[];coverage=[];condition_cache={i:_canonical_conditions(v) for i,v in meta.items()}
    for b in states:
        m,s,r=b['method'],b['seed'],b['round'];sel=b['selected_ids'];base=ROOT/b['source']
        f=load(base);assert FILES[str(base.relative_to(ROOT))]['sha256']==b['source_sha256']
        assert len(sel)==len(set(sel))==32 and set(sel)<=set(b['unlabeled_ids'])
        usable='premeasurement_predictions' in f and (base.parent/'feedback.json').is_file()
        recorded={};preds={}
        if usable:
            fb=load(base.parent/'feedback.json');receipt=load(base.parent/'label_access_receipt.json')
            assert f['new_batch_labels_revealed'] is False
            audit=receipt['audit']
            expected_ids=hashlib.sha256(json.dumps(sorted(sel),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert audit['roles']=='u0' and audit['acquisitions_frozen'] is True
            assert audit['requested_rows']==32 and audit['requested_ids_hash']==expected_ids
            assert fb['batch_freeze_sha256']==b['source_sha256']
            semantic_sha=hashlib.sha256(json.dumps(fb,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert receipt['feedback_sha256']==semantic_sha
            assert receipt['batch_freeze_sha256']==b['source_sha256']
            assert [v['candidate_id'] for v in fb['records']]==sel
            assert [v['candidate_id'] for v in f['premeasurement_predictions']]==sel
            preds={v['candidate_id']:v for v in f['premeasurement_predictions']}
            # Truth is projected only after accepted-ID/freeze/receipt checks.
            recorded={v['candidate_id']:v for v in fb['records']}
            for sid,v in recorded.items():
                for t in [1,2]:assert v[f'pred_V{t}_ml']==preds[sid][f'pred_V{t}_ml']
        groups={'all32':sel}
        if m=='free16_lcmd16':groups={'hybrid_llm16':b['llm_ids'],'hybrid_cw16':b['numerical_ids'],'all32':sel}
        hist=defaultdict(list)
        for sid in b['labeled_ids']:hist[meta[sid]['smiles']].append(sid)
        for label,ids in groups.items():
            coverage.append(dict(method=m,source_half=label,seed=s,round=r,L_before=b['L_before'],n_selected=len(ids),
                predictions_and_revealed_feedback_available=usable,reason='' if usable else 'CW runtime has no accepted premeasurement prediction/feedback artifact; no cross-checkpoint substitution',
                checkpoint=b['checkpoint_sha256'],source=b['source']))
            membership={axis:set() for axis in AXES};within={axis:set() for axis in AXES};paircounts={axis:0 for axis in AXES}
            for sid in ids:
                for other in hist[meta[sid]['smiles']]:
                    match=classify_pair(sid,other,meta)
                    if match:membership[AXES[match[0]]].add(sid)
            for a,z in itertools.combinations(ids,2):
                match=classify_pair(a,z,meta)
                if not match:continue
                j,a,z,low,high=match;axis=AXES[j];within[axis]|={a,z};paircounts[axis]+=1
                if usable:
                    for t in [1,2]:
                        actual=recorded[z][f'true_V{t}_ml']-recorded[a][f'true_V{t}_ml']
                        pred=preds[z][f'pred_V{t}_ml']-preds[a][f'pred_V{t}_ml']
                        pairs.append(dict(method=m,source_half=label,seed=s,round=r,L_before=b['L_before'],phase=phase(b['L_before']),
                            target=f'V{t}',axis=axis,pair_key=short('|'.join(sorted([a,z]))),molecule_key=short(meta[a]['smiles']),
                            low_id=a,high_id=z,low_level=low,high_level=high,actual_delta_ml=actual,predicted_delta_ml=pred,
                            checkpoint_sha256=b['checkpoint_sha256'],information_time='predictions before batch; outcomes only after this accepted batch freeze'))
            for axis in AXES:
                flag=membership[axis];both=within[axis]
                entry=dict(method=m,source_half=label,seed=s,round=r,L_before=b['L_before'],phase=phase(b['L_before']),axis=axis,
                    batch_n=len(ids),history_control_slots=len(flag),history_control_fraction=len(flag)/len(ids),
                    within_pair_n=paircounts[axis],within_control_slots=len(both),within_control_fraction=len(both)/len(ids))
                alloc.append(entry)
                for t in [1,2]:
                    errors=np.array([recorded[v][f'pred_V{t}_ml']-recorded[v][f'true_V{t}_ml'] for v in ids]) if usable else np.array([])
                    chosen=np.array([v in flag for v in ids])
                    if usable:
                        marked=errors[chosen];ss=float(np.sum(errors**2));sig=scales[s][f'V{t}']
                        rowerrors.append(dict(**entry,target=f'V{t}',tagged_n=int(chosen.sum()),
                            tagged_mae_ml=float(np.abs(marked).mean()) if len(marked) else np.nan,
                            tagged_normalized_sse=float(np.sum((marked/sig)**2)),
                            fraction_within_source_endpoint_sse=float(np.sum(marked**2)/ss) if ss>0 else np.nan,
                            overall_endpoint_mae_ml=float(np.abs(errors).mean()),
                            overlapping_axis_tags=True,interpretation='selected-row errors, not validation error contribution or causal efficiency'))
        print(m,s,r,'feedback' if usable else 'metadata only',flush=True) if r==0 else None
    pairframe=pd.DataFrame(pairs);pairframe.to_csv(OUT/'condition_response_pairs.csv',index=False)
    pd.DataFrame(alloc).to_csv(OUT/'condition_allocation.csv',index=False)
    pd.DataFrame(rowerrors).to_csv(OUT/'condition_error_tags.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'model_state_coverage.csv',index=False)
    aggregate_pairs(pairframe,['method','source_half','seed','axis','target'],cfg['observed_change_thresholds_ml'],cfg['prediction_flat_threshold_ml']).to_csv(OUT/'response_seed_summary.csv',index=False)
    aggregate_pairs(pairframe,['method','source_half','seed','phase','axis','target'],cfg['observed_change_thresholds_ml'],cfg['prediction_flat_threshold_ml']).to_csv(OUT/'response_phase_summary.csv',index=False)
    volume=pairframe[pairframe.axis=='V/ul']
    aggregate_pairs(volume,['method','source_half','seed','target','low_level','high_level'],[.5],cfg['prediction_flat_threshold_ml']).to_csv(OUT/'sample_volume_levels.csv',index=False)
    new=pairframe[(pairframe.method=='new_free_cw_sol61_xhigh')&(pairframe.source_half=='all32')]
    old=pd.read_csv(ROOT/cfg['reused_model_pairs']);read(ROOT/cfg['reused_model_pairs'])
    grouped=new.assign(actual_abs_delta_ml=lambda x:x.actual_delta_ml.abs(),pred_abs_delta_ml=lambda x:x.predicted_delta_ml.abs()).groupby(['seed','phase','axis','target'])[['actual_abs_delta_ml','pred_abs_delta_ml']].mean().reset_index()
    merged=old.merge(grouped,on=['seed','phase','axis','target'],suffixes=('_old','_new'),validate='one_to_one')
    errors={k:float((merged[k+'_old']-merged[k+'_new']).abs().max()) for k in ['actual_abs_delta_ml','pred_abs_delta_ml']}
    assert max(errors.values())<1e-10
    assert len(new[new.axis=='V/ul'])==330 and len(new[new.axis=='PE/EA'])==1428
    metadata_holdouts(data,cfg)
    actual_feedback=sum(v['predictions_and_revealed_feedback_available'] for v in coverage if v['source_half']=='all32')
    assert actual_feedback==611
    (OUT/'model_inputs.json').write_text(json.dumps(dict(files=list(FILES.values()),new_feedback_states=actual_feedback,
        same_batch_pair_response_events=len(pairframe),old_165_pair_reuse_check_max_error=errors,
        no_new_training=True,no_unobserved_or_test_truth=True,
        boundary='Only true values in the same accepted batch feedback, after matching freeze/receipt; no outcome scores for LCMD counterfactuals'),indent=2))
    print(json.dumps(dict(pair_endpoint_events=len(pairframe),states=len(states),feedback_states=611,reused_165_pairs_verified=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=OUT/'config.json');main(p.parse_args().config)
