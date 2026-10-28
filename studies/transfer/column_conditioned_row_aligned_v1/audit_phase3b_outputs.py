"""Independent result/aggregate audit using the single authorized score input snapshot."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_phase3b as b


def close(actual,expected,name):
    b.s.require(np.isclose(actual,expected,rtol=1e-10,atol=1e-10,equal_nan=True),f'result integrity mismatch: {name}')


def audit():
    s,old,study=b.s,b.old,b.STUDY
    receipt=b.verify_completed()
    for rel,digest in s.read_json(study/'PHASE3B_SCORING_PROVENANCE.json')['files'].items():
        s.require(old.sha(s.ROOT/rel)==digest,'scoring implementation provenance changed')
    inputs=s.read_json(study/'PHASE3B_AUTHORIZED_SCORING_INPUTS.json')
    result=pd.read_csv(study/'OUTER_ROW_RESULTS.csv')
    tails=pd.read_csv(study/'OUTER_ROW_TAIL_DIAGNOSTICS.csv')
    paired=pd.read_csv(study/'OUTER_ROW_PAIRED_COMPARISONS.csv')
    summary=pd.read_csv(study/'OUTER_ROW_SUMMARY.csv')
    tailpairs=pd.read_csv(study/'OUTER_ROW_TAIL_PAIRED_COMPARISONS.csv')
    s.require(len(result)==20 and len(tails)==40 and len(paired)==10,'result schedule mismatch')
    expected={(seed,c,arm) for seed in s.SEEDS for c in s.TARGETS for arm in old.ARMS}
    s.require(set(result[b.KEYS].itertuples(index=False,name=None))==expected and not result.duplicated(b.KEYS).any()
              and result.role.eq('test').all(),'primary test coverage mismatch')
    access=[];checked=0
    for context in inputs['contexts']:
        seed,c=context['outer_seed'],context['column'];truth=np.asarray(context['test_truth'])
        s.require(old.digest(truth.tolist())==context['test_truth_sha256'],'cached scoring truth digest mismatch')
        for role,ids in [('gradient_train',context['train_ids']),('test',context['test_ids'])]:
            access.append({'outer_seed':seed,'column':c,'role':role,'count':len(ids),'requested_ids':ids,
                           'requested_ids_sha256':old.digest(sorted(ids)),
                           'canonical_path':str((old.PARENT/f'filtered_canonical_{c}.csv').relative_to(s.ROOT)),
                           'canonical_sha256':old.sha(old.PARENT/f'filtered_canonical_{c}.csv'),
                           'reader':'_read_authorized_truth: sample_id pass then skiprows + V1_ml/V2_ml only',
                           'validation_responses_parsed':False,
                           'call_source':'hash-locked one-time score_phase3b.py; request schedule from frozen scoring input snapshot'})
        for arm in old.ARMS:
            saved=result.loc[result.outer_seed.eq(seed)&result.column.eq(c)&result.arm.eq(arm)].iloc[0]
            path=s.FINAL/f'seed_{seed}/{arm}/{c}_test_predictions_blind.csv'
            frame=pd.read_csv(path,dtype={'sample_id':str})
            s.require(frame.sample_id.tolist()==context['test_ids'] and saved.test_count==len(frame),'score/prediction identity mismatch')
            prediction=frame[old.QUANTILES].to_numpy()
            oracle=old.point_metrics(truth,prediction,context['scales'])
            for key,value in oracle.items():
                if key=='all_outputs_finite':s.require(bool(saved[key])==value,'finite diagnostic mismatch')
                else:close(saved[key],value,key)
            err=prediction[:,[1,4]]-truth
            errors={'V1':err[:,0],'V2':err[:,1],'Center':err.mean(axis=1),'Width':err[:,1]-err[:,0]}
            for endpoint,vector in errors.items():
                close(saved[endpoint+'_max_abs_error'],np.abs(vector).max(),endpoint+' max error')
                s.require(saved[endpoint+'_max_error_sample_id']==context['test_ids'][int(np.argmax(np.abs(vector)))],'maximum error sample identity mismatch')
                if endpoint in ['Center','Width']:close(saved[endpoint+'_rmse'],np.sqrt(np.mean(vector**2)),endpoint+' RMSE')
            for j,endpoint in enumerate(['V1','V2']):
                part=tails.loc[tails.outer_seed.eq(seed)&tails.column.eq(c)&tails.arm.eq(arm)&tails.endpoint.eq(endpoint)]
                s.require(len(part)==1,'tail coverage missing/duplicate')
                row=part.iloc[0];mask=truth[:,j]>=context['thresholds'][j]
                s.require(row.tail_count==mask.sum(),'tail identity mask/count mismatch')
                close(row.threshold,context['thresholds'][j],'tail threshold')
                close(row.tail_sse,float(np.square(err[mask,j]).sum()),'tail SSE')
                close(row.tail_rmse,np.sqrt(np.square(err[mask,j]).mean()) if mask.any() else np.nan,'tail RMSE')
            checked+=1
    for c in s.TARGETS:
        left,right=[result.loc[result.column.eq(c)&result.arm.eq(arm)].set_index('outer_seed').loc[list(s.SEEDS)] for arm in old.ARMS]
        gain=1-right.combined_normalized_rmse/left.combined_normalized_rmse
        pair=paired.loc[paired.column.eq(c)].set_index('outer_seed').loc[list(s.SEEDS)]
        np.testing.assert_allclose(pair.combined_normalized_rmse_relative_gain,gain,rtol=1e-12,atol=1e-12)
        s.require(pair.A2_seed_win.tolist()==gain.gt(0).tolist(),'seed wins mismatch')
        for arm,part in [('A1',left),('A2',right)]:
            row=summary.loc[summary.column.eq(c)&summary.arm.eq(arm)].iloc[0]
            for metric in [*b.LOSS_METRICS,'V1_r2','V2_r2']:
                close(row[metric+'_mean'],part[metric].mean(),'mean '+metric)
                close(row[metric+'_sd'],part[metric].std(ddof=1),'seed descriptive SD '+metric)
            close(row.mean_paired_seed_gain,gain.mean(),'mean paired seed gain')
            close(row.gain_of_means,1-right.combined_normalized_rmse.mean()/left.combined_normalized_rmse.mean(),'gain of means')
    for row in tailpairs.itertuples():
        close(row.tail_relative_gain,1-row.A2_tail_rmse/row.A1_tail_rmse,'paired tail gain')
    history=pd.read_csv(study/'GROUP_VS_ROW_OUTER_COMPARISON.csv')
    historic_summary=pd.read_csv(study/'GROUP_VS_ROW_OUTER_SUMMARY.csv')
    for row in historic_summary.itertuples():
        group=history.loc[history.column.eq(row.column)]
        left,right=[group.loc[group.arm.eq(arm)].set_index('outer_seed').loc[list(s.SEEDS)] for arm in b.old.ARMS]
        key=row.inner_cv+'_combined_normalized_rmse'
        values=(left if row.arm=='A1' else right)[key]
        close(row.mean_test_nrmse,values.mean(),'historical mean')
        close(row.seed_sd_test_nrmse,values.std(ddof=1),'historical SD')
        close(row.mean_paired_seed_gain,(1-right[key]/left[key]).mean(),'historical paired gain')
        close(row.gain_of_means,1-right[key].mean()/left[key].mean(),'historical gain of means')
        s.require(row.A2_seed_wins==int((right[key]<left[key]).sum()),'historical seed wins mismatch')
    baseline=pd.read_csv(study/'OUTER_ROW_BASELINE_RESULTS.csv')
    baseline_comparisons=pd.read_csv(study/'OUTER_ROW_BASELINE_COMPARISONS.csv')
    s.require(len(baseline)==30 and not baseline.duplicated(['outer_seed','column','method']).any(),'secondary baseline schedule mismatch')
    for row in baseline_comparisons.itertuples():
        ref=baseline.loc[baseline.column.eq(row.column)&baseline.method.eq(row.reference)].set_index('outer_seed').loc[list(s.SEEDS)]
        candidate=result.loc[result.column.eq(row.column)&result.arm.eq(row.arm)].set_index('outer_seed').loc[list(s.SEEDS)]
        close(row.reference_mean_nrmse,ref.combined_normalized_rmse.mean(),'baseline mean')
        close(row.candidate_mean_nrmse,candidate.combined_normalized_rmse.mean(),'baseline candidate mean')
        close(row.mean_paired_seed_gain,(1-candidate.combined_normalized_rmse/ref.combined_normalized_rmse).mean(),'baseline paired gain')
        close(row.gain_of_means,1-candidate.combined_normalized_rmse.mean()/ref.combined_normalized_rmse.mean(),'baseline gain of means')
        s.require(row.candidate_seed_wins==int((candidate.combined_normalized_rmse<ref.combined_normalized_rmse).sum()),'baseline wins mismatch')
    decision=s.read_json(study/'FINAL_SCIENTIFIC_DECISION.json')
    s.require(decision['decision']=='C' and decision['new_training_fits']==0 and decision['new_candidate_tests']==0,'decision/scope inconsistent')
    preflight=s.read_json(study/'PHASE3B_PREFLIGHT_AUDIT.json')
    physical=[]
    for key,check in preflight['prediction_checks'].items():
        seed,arm,c,role=key.split('/')
        physical.append({'outer_seed':int(seed),'arm':arm,'column':c,'role':role,'rows':check['rows'],
                         'V1_quantile_crossing_rows':check['V1_quantile_crossing_rows'],
                         'V2_quantile_crossing_rows':check['V2_quantile_crossing_rows'],
                         **{f'V2_below_V1_{q}_rows':check['V2_below_V1_by_quantile'][i] for i,q in enumerate(['q10','q50','q90'])},
                         'all_six_outputs_finite':check['all_six_outputs_finite'],'existing_output_clamp_verified':check['existing_output_clamp_verified'],
                         'truth_required':False})
    pd.DataFrame(physical).to_csv(study/'PHASE3B_PHYSICAL_VALIDITY_DIAGNOSTICS.csv',index=False)
    old.write_json(study/'PHASE3B_LABEL_ACCESS_LOG.json',{'purpose':'source and response-request accounting for one completed score',
                   'requests':access,'request_contexts':20,'source_4g_responses_parsed':False,'outer_validation_responses_parsed':False,
                   'preflight_sha256':old.sha(study/'PHASE3B_PREFLIGHT_AUDIT.json'),'single_score_manifest_sha256':old.sha(study/'PHASE3B_SCORE_MANIFEST.json')},immutable=True)
    for name in ['phase3b_outer_row_paired_gains','phase3b_group_vs_row_outer','phase3b_endpoint_tail_max_tradeoffs']:
        root=ET.parse(study/(name+'.svg')).getroot()
        s.require(root.tag.endswith('svg') and (study/(name+'.png')).stat().st_size>10000,'invalid figure export')
    preserved=s.r.verify_phase1()
    for name,digest in s.read_json(study/'PHASE2_REPORT_MANIFEST.json')['files'].items():s.require(old.sha(study/name)==digest,'Phase 2 evidence changed')
    for name,digest in s.read_json(study/'PHASE3A_REPORT_MANIFEST.json')['files'].items():s.require(old.sha(study/name)==digest,'Phase 3A evidence changed')
    verification=s.verify_freeze()
    output={'all_passed':True,'primary_score_contexts_reproduced':checked,'tail_observations_verified':40,'paired_seed_observations_verified':10,
            'cached_authorized_truth_used_for_verification':True,'raw_test_response_files_not_reread_by_integrity_audit':True,
            'normalization_and_metrics_from_standard_project_function':True,'plot_tables_and_both_gain_definitions_verified':True,
            'png_svg_exports':6,'freeze_verified':verification,'prior_evidence_preserved':preserved,
            'no_new_fit_or_candidate_test':True,'all_phase3a_predictions_unchanged':True,'HIER_exclusion_preserved':True}
    old.write_json(study/'PHASE3B_RESULT_INTEGRITY.json',output,immutable=True)
    print(output)


if __name__=='__main__':audit()
