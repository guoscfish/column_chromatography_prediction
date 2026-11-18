#!/usr/bin/env python3
"""Score legal identical pairs and metadata-fixed, label-free volume curves."""
from common import *
import gc,itertools,time
from checkpoint_audit import load_formal
from al_checkpoint_audit import load_al
from data_boundary import FormalLabelBoundary,load_metadata,load_formal_split,QUARANTINED_IDS

PRIOR=ROOT/'studies/active_learning/volume_response_retry_audit_20261117'
SEEDS=[157,887,2357,6101,12203];FORMAL=[42,525,1101];BUDGETS=[525,1005,1485]

def series_key(row):
    return sha_text((row['canonical_smiles'],)+tuple(str(row[k]) for k in FIELDS if k!='V/ul'))

def sha_text(value):return hashlib.sha256(repr(value).encode()).hexdigest()

def freeze_pool(metadata,exposure):
    eligible=exposure[(exposure.axis=='V/ul')&exposure.allow_new_scoring&exposure.global_all_three_non_test]
    entries=[];lookup=metadata.set_index('sample_id')
    for al_seed,g in eligible.groupby('al_seed'):
        groups={}
        for row in g.drop_duplicates('pair_key').itertuples():
            key=series_key(lookup.loc[row.low_id]);groups.setdefault(key,set()).update([row.low_id,row.high_id])
        ranked=sorted(groups,key=lambda k:sha_text(('full_training_volume_curves_20261117',int(al_seed),k)))[:3]
        for key in ranked:
            ids=sorted(groups[key]);base=lookup.loc[ids[0]]
            all_series=metadata[metadata.apply(lambda r:series_key(r)==key,axis=1)]
            entries.append(dict(case_id=sha_text((int(al_seed),key))[:12],al_seed=int(al_seed),series_key=key,molecule=base.canonical_smiles,
                base_id=ids[0],observed_ids=ids,local_min_ul=float(all_series['V/ul'].min()),local_max_ul=float(all_series['V/ul'].max()),
                selection='Metadata hash only; no predictions or target values used'))
    assert len(entries)==15 and not any(set(v['observed_ids'])&QUARANTINED_IDS for v in entries)
    dump(OUT/'CURVE_POOL.json',dict(frozen_unix=time.time(),cases=entries,input_metadata_sha256=sha(OUT/'source_metadata.csv'),
        requested_grid_ul=[30,60,90,120,150],new_fits=0,new_labels_used_for_selection=0))
    return entries

def main():
    torch.set_num_threads(2)
    meta=load_metadata();lookup=meta.set_index('sample_id')
    exposure=pd.read_csv(OUT/'al_common_pair_exposure.csv')
    assert not set(exposure[exposure.allow_new_scoring].low_id)&QUARANTINED_IDS
    assert not set(exposure[exposure.allow_new_scoring].high_id)&QUARANTINED_IDS
    pool=freeze_pool(meta,exposure)
    graph_path=ROOT/'experiments/e0_4g_baseline/graph_cache_4g.pt'
    assert sha(graph_path)=='a48f8ea1ce4f89c5a1ea6d17466e7c7ffa0439edb8ec79bd8b765bad3a97c3ee'
    cache=torch.load(graph_path,map_location='cpu',weights_only=False)
    safe=exposure[exposure.allow_new_scoring].copy();predictions={};identities=[]
    for seed in SEEDS:
        group=safe[safe.al_seed==seed];ids=sorted(set(group.low_id)|set(group.high_id))
        assert not set(ids)&QUARANTINED_IDS
        for budget in BUDGETS:
            model,identity=load_al(seed,budget);assert set(ids)<=set(identity['source_history_ids'])
            pred=infer(model,lookup.loc[ids].reset_index(),cache,identity['preprocessing']['scaler'])
            pred.to_csv(OUT/f'al_predictions_seed_{seed}_L{budget}.csv',index=False)
            predictions[(seed,budget)]=pred.set_index('sample_id');identities.append(identity)
            del model;gc.collect()
        print('AL predictions',seed,'safe IDs',len(ids),flush=True)
    rows=[];receipts=[];agreement=[]
    oldcols=['seed','budget','axis','target','low_id','high_id','predicted_delta_ml']
    old=pd.read_csv(PRIOR/'frozen_response_pairs.csv',usecols=oldcols)
    old=old.set_index(['seed','budget','axis','target','low_id','high_id'])
    for seed,g in safe.groupby('formal_row_seed'):
        boundary=FormalLabelBoundary(int(seed));truth=boundary.labels_for_pairs(g).set_index('sample_id')
        full=pd.read_csv(OUT/f'formal_predictions_seed_{seed}.csv').set_index('sample_id')
        for pair in g.itertuples():
            a,b=pair.low_id,pair.high_id;assert a in full.index and b in full.index
            key=series_key(lookup.loc[a]) if pair.axis=='V/ul' else sha_text((lookup.loc[a,'canonical_smiles'],)+tuple(str(lookup.loc[a,k]) for k in FIELDS if k!='PE/EA'))
            for target in ['V1','V2']:
                dy=float(truth.loc[b,target+'_ml'])-float(truth.loc[a,target+'_ml'])
                base=dict(formal_seed=int(seed),al_seed=int(pair.al_seed),axis=pair.axis,exposure=pair.formal_exposure_class,
                    pair_key=pair.pair_key,series_key=key,molecule=pair.canonical_smiles,low_id=a,high_id=b,target=target,delta_true=dy,
                    global_all_three_non_test=bool(pair.global_all_three_non_test))
                rows.append(dict(base,method='Fully_trained',delta_pred=float(full.loc[b,target+'_q50']-full.loc[a,target+'_q50'])))
                for budget in BUDGETS:
                    pred=predictions[(int(pair.al_seed),budget)];dp=float(pred.loc[b,target+'_q50']-pred.loc[a,target+'_q50'])
                    rows.append(dict(base,method=f'AL_L{budget}',delta_pred=dp))
                    if budget in [525,1005]:
                        original=float(old.loc[(pair.al_seed,budget,pair.axis,target,a,b),'predicted_delta_ml'])
                        agreement.append(dict(al_seed=pair.al_seed,budget=budget,target=target,low_id=a,high_id=b,abs_delta_difference_ml=abs(dp-original)))
        receipts.extend(boundary.audit)
    p=pd.DataFrame(rows);p.to_csv(OUT/'matched_response_pairs.csv',index=False)
    summary(p,['formal_seed','al_seed','axis','exposure','method','target']).to_csv(OUT/'matched_seed_metrics.csv',index=False)
    summary(p,['formal_seed','al_seed','axis','exposure','method','target','molecule']).to_csv(OUT/'matched_molecule_metrics.csv',index=False)
    summary(p[p.global_all_three_non_test],['formal_seed','al_seed','axis','exposure','method','target']).to_csv(OUT/'matched_global_nontest_metrics.csv',index=False)
    agreement=pd.DataFrame(agreement).drop_duplicates();assert agreement.abs_delta_difference_ml.max()<5e-4
    agreement.to_csv(OUT/'al_prediction_reconstruction_checks.csv',index=False)
    print('Matched scoring complete',len(p),'endpoint-method events',flush=True)
    curves=[];frames=[]
    for case in pool:
        for volume in [30,60,90,120,150]:
            row=lookup.loc[case['base_id']].to_dict();row['sample_id']=f"curve_{case['case_id']}_{volume}";row['V/ul']=volume
            frames.append(row)
    curve_frame=pd.DataFrame(frames)
    def save_curve_predictions(pred,model_name,model_seed,cases):
        pred=pred.set_index('sample_id')
        for case in cases:
            for volume in [30,60,90,120,150]:
                sid=f"curve_{case['case_id']}_{volume}"
                for target in ['V1','V2']:
                    curves.append(dict(case_id=case['case_id'],al_seed=case['al_seed'],molecule=case['molecule'],method=model_name,model_seed=model_seed,target=target,
                        volume_ul=volume,prediction_ml=float(pred.loc[sid,target+'_q50']),local_min_ul=case['local_min_ul'],local_max_ul=case['local_max_ul'],
                        locally_supported=case['local_min_ul']<=volume<=case['local_max_ul'],scope='label-free sensitivity, synthetic levels are never error-scored'))
    for seed in FORMAL:
        model,identity=load_formal(seed);pred=infer(model,curve_frame,cache,identity['preprocessing']['scaler'])
        save_curve_predictions(pred,'Fully_trained',seed,pool);del model;gc.collect()
    for seed in SEEDS:
        cases=[c for c in pool if c['al_seed']==seed];case_sids={f"curve_{c['case_id']}_{v}" for c in cases for v in [30,60,90,120,150]}
        frame=curve_frame[curve_frame.sample_id.isin(case_sids)]
        for budget in BUDGETS:
            model,identity=load_al(seed,budget);pred=infer(model,frame,cache,identity['preprocessing']['scaler'])
            save_curve_predictions(pred,f'AL_L{budget}',seed,cases);del model;gc.collect()
    pd.DataFrame(curves).to_csv(OUT/'controlled_volume_curves.csv',index=False)
    overlay_ids=sorted({sid for case in pool for sid in case['observed_ids']})
    boundary=FormalLabelBoundary(42);fake=pd.DataFrame({'low_id':overlay_ids,'high_id':overlay_ids})
    overlay=boundary.labels_for_pairs(fake);receipts.extend(boundary.audit)
    overlays=[]
    for case in pool:
        for row in overlay[overlay.sample_id.isin(case['observed_ids'])].itertuples():
            for target in ['V1','V2']:overlays.append(dict(case_id=case['case_id'],sample_id=row.sample_id,target=target,volume_ul=float(lookup.loc[row.sample_id,'V/ul']),value_ml=float(getattr(row,target+'_ml')),label_source='formal42 authorized role; also already in AL H525'))
    pd.DataFrame(overlays).to_csv(OUT/'curve_authorized_observations.csv',index=False)
    inputs=[OUT/'ANALYSIS_PLAN.json',OUT/'LABEL_BOUNDARY_INCIDENT.json',OUT/'al_common_pair_exposure.csv',OUT/'source_metadata.csv',graph_path,PRIOR/'frozen_response_pairs.csv',Path(__file__),OUT/'common.py']
    dump(OUT/'MATCHED_PREDICTION_MANIFEST.json',dict(status='PASS_WITH_DISCLOSED_LABEL_PREVIEW_EXCEPTION',new_fits=0,API_calls=0,
        new_experimental_reveals=0,numerical_source_test_targets_projected=0,quarantine_applied=True,matched_endpoint_method_events=len(p),
        al_reconstruction_max_delta_error_ml=float(agreement.abs_delta_difference_ml.max()),authorized_label_requests=receipts,
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],curve_pool_frozen_before_predictions=True))
    print('Matched and controlled curve analysis complete',flush=True)

if __name__=='__main__':main()
