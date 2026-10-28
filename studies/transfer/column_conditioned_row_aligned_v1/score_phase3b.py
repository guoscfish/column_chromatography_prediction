"""One fixed Row test evaluation after exhaustive freeze checks. No training or prediction edits."""
from copy import deepcopy
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase3a as s

STUDY, ROOT, old = s.STUDY, s.ROOT, s.old
KEYS = ['outer_seed','column','arm']
LOSS_METRICS = ['combined_normalized_rmse','V1_rmse','V2_rmse','V1_mae','V2_mae','Center_rmse','Width_rmse',
                'V1_max_abs_error','V2_max_abs_error','Center_max_abs_error','Width_max_abs_error']


def role_ids(schedule,seed,c,role):
    return schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)&schedule.role.eq(role),'sample_id'].astype(str).tolist()


def preflight():
    torch.set_num_threads(1)
    verified=s.verify_freeze()
    preserved=s.r.verify_phase1()
    for manifest_name in ['PHASE2_REPORT_MANIFEST.json','PHASE3A_REPORT_MANIFEST.json']:
        for name,digest in s.read_json(STUDY/manifest_name)['files'].items():
            s.require(old.sha(STUDY/name)==digest,f'prior phase evidence changed: {name}')
    freeze=s.read_json(STUDY/'PREDICTION_FREEZE_MANIFEST.json')
    protocol=s.read_json(STUDY/'PHASE3A_PROTOCOL.json')
    s.require(freeze['evaluation_plan_sha256']==protocol['evaluation_plan_sha256']==old.sha(STUDY/'EVALUATION_PLAN.md'),'evaluation plan changed')
    s.require(protocol['training']==old.TRAINING and protocol['source_sha256']==old.SOURCE_SHA
              and protocol['original_split_sha256']==old.SPLIT_SHA,'training/source/split changed')
    s.require(all((STUDY/name).stat().st_mode&0o222==0 for name in freeze['files'])
              and (STUDY/'PREDICTION_FREEZE_MANIFEST.json').stat().st_mode&0o222==0,'freeze is no longer read-only')
    schedule=s.p.row_schedule()
    refits=pd.read_csv(STUDY/'FINAL_REFIT_MANIFEST.csv')
    histories=pd.read_csv(STUDY/'FINAL_TRAINING_HISTORY.csv')
    expected={(seed,arm) for seed in s.SEEDS for arm in old.ARMS}
    s.require(len(refits)==10 and set(refits[['outer_seed','arm']].itertuples(index=False,name=None))==expected,'refit coverage mismatch')
    s.require(len(freeze['contexts'])==10 and {(item['outer_seed'],item['arm']) for item in freeze['contexts']}==expected,'frozen context coverage mismatch')
    predictions,checks,fits={},{},{}
    for seed in s.SEEDS:
        shared_init=None; arm_histories={}
        for arm in old.ARMS:
            run=s.FINAL/f'seed_{seed}/{arm}'
            s.check_complete(run)
            fit=s.read_json(run/'fit.json');fits[seed,arm]=fit
            budget=next(b for b in protocol['budgets'] if b['outer_seed']==seed and b['arm']==arm)
            row=refits.loc[refits.outer_seed.eq(seed)&refits.arm.eq(arm)].iloc[0]
            s.require(fit['fixed_epochs']==budget['fixed_epochs']==int(row.fixed_epochs)
                      and fit['five_joint_best_epochs']==budget['five_joint_best_epochs'],'epoch budget changed')
            torch.manual_seed(seed)
            model=old.build_multitask_model(old.source_path(),old.ARMS[arm])
            initial=deepcopy(model.state_dict())
            checkpoint=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
            s.require(checkpoint['arm']==arm and checkpoint['outer_seed']==seed and checkpoint['source_sha256']==old.SOURCE_SHA
                      and checkpoint['output_columns']==old.QUANTILES
                      and checkpoint['phase3a_protocol_sha256']==old.sha(STUDY/'PHASE3A_PROTOCOL.json'),'checkpoint authority mismatch')
            s.require(checkpoint['fixed_epochs']==fit['fixed_epochs'] and checkpoint['optimizer_steps']==fit['optimizer_steps'],'checkpoint budget mismatch')
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            state=s.state_check(model,initial,fit['optimizer_steps'])
            s.require(state['model_state_sha256']==fit['model_state_sha256']==checkpoint['model_state_sha256']==row.model_state_sha256,'checkpoint state mismatch')
            h=pd.read_csv(run/'history.csv');arm_histories[arm]=h
            s.require(len(h)==fit['fixed_epochs']==fit['epochs_run'] and int(h.optimizer_steps.iloc[-1])==fit['optimizer_steps'],'training history budget mismatch')
            np.testing.assert_array_equal(h.optimizer_steps,np.arange(1,len(h)+1)*fit['steps_per_epoch'])
            aggregate=histories.loc[histories.outer_seed.eq(seed)&histories.arm.eq(arm)].drop(columns=['outer_seed','arm']).reset_index(drop=True)
            pd.testing.assert_frame_equal(h,aggregate)
            s.require(np.isfinite(h[[c for c in h if c.startswith('train_loss_')]].to_numpy()).all(),'nonfinite frozen loss history')
            init=s.read_json(run/'initialization.json')
            if shared_init is None:shared_init=init
            else:s.require(init==shared_init,'unpaired initialization')
            identity=pd.read_csv(run/'training_identities.csv',dtype={'sample_id':str})
            source_split=pd.read_csv(old.QUALIFIED/'splits/row_seed_42.csv',usecols=['sample_id','split'])
            access=s.read_json(run/'label_access.json')
            for c in old.TASKS:
                wanted=source_split.loc[source_split.split.eq('train'),'sample_id'].astype(str).tolist() if c=='4g' else role_ids(schedule,seed,c,'gradient_train')
                actual=identity.loc[identity.column.eq(c),'sample_id'].tolist()
                s.require(actual==wanted and len(actual)==len(set(actual))==fit['training_identity'][c]['count'],'training identity mismatch')
                s.require(old.digest(sorted(actual))==fit['training_identity'][c]['sorted_ids_sha256']==row[f'{c}_sorted_ids_sha256'],'training identity digest mismatch')
                receipt=next(item for item in access if item['column']==c)
                s.require(receipt['authorized_ids']==wanted and receipt['outer_response_cells_parsed']==0,'label authorization changed')
            for c in s.TARGETS:
                data={c:{'roles':{role:role_ids(schedule,seed,c,role) for role in ['gradient_train','validation','test']}}}
                for role in ['validation','test']:
                    name=f'{c}_{role}_predictions_blind.csv'
                    path=run/name
                    frame=pd.read_csv(path,dtype={'sample_id':str})
                    check=s.check_prediction(frame,data,seed,arm,c,role)
                    s.require(check==fit['prediction_checks'][name],'frozen prediction audit mismatch')
                    predictions[seed,c,arm,role]=frame
                    checks[f'{seed}/{arm}/{c}/{role}']={**check,'path':str(path.relative_to(ROOT)),'sha256':old.sha(path)}
        common=min(map(len,arm_histories.values()))
        np.testing.assert_array_equal(arm_histories['A1'].batch_identity_draws_sha256.iloc[:common],arm_histories['A2'].batch_identity_draws_sha256.iloc[:common])
    s.require(len(predictions)==40 and sum(len(p) for p in predictions.values())==1852,'prediction schedule/row count mismatch')
    audit={'all_passed':True,'preflight_outer_truth_reads':0,'fits':10,'prediction_contexts':40,'prediction_rows':1852,
           'freeze_sha256':old.sha(STUDY/'PREDICTION_FREEZE_MANIFEST.json'),'evaluation_plan_sha256':old.sha(STUDY/'EVALUATION_PLAN.md'),
           'verified':verified,'preservation':preserved,'prediction_checks':checks,'scored_role':'test',
           'validation_truth_will_not_be_read':True,'plan_conflicts':[],'model_training_or_prediction_rewrite_allowed':False}
    old.write_json(STUDY/'PHASE3B_PREFLIGHT_AUDIT.json',audit,immutable=True)
    return schedule,predictions,fits


def historical_inputs(schedule):
    """Verify historic workflow identities and evidence before including any comparison."""
    historical=old.STUDY
    freeze=s.read_json(historical/'ROW_PREDICTION_FREEZE_MANIFEST.json')
    score=s.read_json(historical/'ROW_SCORE_MANIFEST.json')
    protocol=s.read_json(historical/'protocol.json')
    s.require(score['prediction_freeze_sha256']==old.sha(historical/'ROW_PREDICTION_FREEZE_MANIFEST.json'),'historical score/freeze linkage mismatch')
    for name,digest in score['files'].items():s.require(old.sha(historical/name)==digest,'historical scores changed')
    for rel,digest in freeze['files'].items():s.require(old.sha(historical/rel)==digest,'historical prediction freeze changed')
    s.require(protocol['source_sha256']==old.SOURCE_SHA and protocol['split_manifest_sha256']==old.SPLIT_SHA
              and protocol['training']==old.TRAINING and protocol['inner_cv']['folds']==5
              and protocol['inner_cv']['group']=='canonical_smiles','historical protocol mismatch')
    current=s.read_json(STUDY/'protocol.json')
    for key in ['population_sha256','features_sha256','source_data_sha256','source_graph_cache_sha256','target_graph_cache_sha256','source_split_sha256','film','bn_policy']:
        s.require(protocol[key]==current[key],f'historical/current data/model mismatch: {key}')
    data,records={},[]
    selections=pd.read_csv(historical/'ROW_INNER_RESULTS.csv')
    for seed in s.SEEDS:
        for arm in old.ARMS:
            run=historical/f'runtime/final/row/seed_{seed}/{arm}'
            receipt=s.read_json(run/'complete.json')
            s.require(receipt['protocol_sha256']==old.sha(historical/'protocol.json')
                      and receipt['execution_provenance_sha256']==old.sha(historical/'EXECUTION_PROVENANCE.json'),'historical fit authority mismatch')
            for name,digest in receipt['files'].items():s.require(old.sha(run/name)==digest,'historical fit changed')
            fit=s.read_json(run/'fit.json')
            selected=selections.loc[selections.outer_seed.eq(seed)&selections.arm.eq(arm)&selections.column.eq('25g'),'best_epoch']
            s.require(len(selected)==5 and int(np.floor(selected.median()+.5))==fit['best_epoch']==fit['epochs_run'],'historical refit median mismatch')
            checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            s.require(checkpoint['source_sha256']==old.SOURCE_SHA and checkpoint['arm']==arm
                      and checkpoint['best_epoch']==fit['best_epoch'],'historical checkpoint mismatch')
            frame=pd.read_csv(run/'predictions_blind.csv',dtype={'sample_id':str})
            for c in s.TARGETS:
                for role in ['validation','test']:
                    ids=role_ids(schedule,seed,c,role)
                    part=frame.loc[frame.column.eq(c)&frame.role.eq(role)]
                    s.require(part.sample_id.is_unique and len(part)==len(ids) and set(part.sample_id)==set(ids),'historical test/validation identities differ')
                    aligned=part.set_index('sample_id').loc[ids]
                    s.require(np.isfinite(aligned[old.QUANTILES].to_numpy()).all(),'nonfinite historical prediction')
                    if role=='test':data[seed,c,arm]=(aligned[old.QUANTILES].to_numpy(),fit)
                records.append({'outer_seed':seed,'column':c,'arm':arm,'source_sha256':old.SOURCE_SHA,'original_split_sha256':old.SPLIT_SHA,
                                'gradient_train_ids_sha256':old.digest(sorted(role_ids(schedule,seed,c,'gradient_train'))),
                                'test_ids_sha256':old.digest(sorted(role_ids(schedule,seed,c,'test'))),'test_count':len(role_ids(schedule,seed,c,'test')),
                                'historical_epochs':fit['epochs_run'],'prediction_sha256':old.sha(run/'predictions_blind.csv'),
                                'matched_population_and_target':True,'matched_standard_metric_implementation':True,
                                'training_identity_evidence':'same split authority and hash-verified historical final_context/train-only code',
                                'interpretation':'different inner selection/full-refit workflow; not a same-step module effect'})
    audit={'matched':True,'contexts':20,'before_test_truth_read':True,'score_file_sha256':old.sha(historical/'ROW_OUTER_RESULTS.csv'),
           'historic_freeze_sha256':old.sha(historical/'ROW_PREDICTION_FREEZE_MANIFEST.json'),'records':records}
    old.write_json(STUDY/'PHASE3B_HISTORICAL_MATCH_AUDIT.json',audit,immutable=True)
    return data,pd.read_csv(historical/'ROW_OUTER_RESULTS.csv')


def baseline_inputs(schedule):
    authority=s.read_json(STUDY/'PHASE3A_BASELINE_COMPATIBILITY.json')
    frames={}
    for ref in authority['references']:
        if ref['eligibility'].startswith('EXCLUDED'):continue
        seed,c,method=ref['outer_seed'],ref['column'],ref['method']
        path=ROOT/ref['prediction_path']
        s.require(old.sha(path)==ref['prediction_sha256'],'compatible baseline prediction changed')
        if method=='scale_only':
            frozen=s.read_json(path.parent/'frozen.json')
            global_freeze=s.read_json(old.PARENT/'all_predictions_frozen.json')
            rel=str((path.parent/'frozen.json').relative_to(old.PARENT))
            s.require(global_freeze['files'][rel]==old.sha(path.parent/'frozen.json'),'scale freeze changed')
            for name,digest in frozen['files'].items():s.require(old.sha(path.parent/name)==digest,'scale frozen file changed')
            for role in ['gradient_train','validation','test']:
                s.require(frozen['contract'][role+'_ids_hash']==old.digest(sorted(role_ids(schedule,seed,c,role))),'scale identity mismatch')
            frame=pd.read_csv(path,usecols=ref['prediction_usecols'],dtype={'sample_id':str})
            six=np.zeros((len(frame),6));six[:,[1,4]]=frame[[method+'_V1',method+'_V2']].to_numpy()
        else:
            base=ROOT/'studies/transfer/traditional_transfer_converged_baseline_v1'
            protocol=s.read_json(base/'PROTOCOL.json')
            s.require(protocol['source_checkpoint_sha256']==old.SOURCE_SHA and protocol['split_manifest_sha256']==old.SPLIT_SHA,'P0/P1 authority mismatch')
            run=path.parent.parent;frozen=s.read_json(base/'PREDICTION_FREEZE_MANIFEST.json')
            s.require(frozen['context_fit_complete_sha256'][str((run/'context_fit_complete.json').relative_to(base))]==old.sha(run/'context_fit_complete.json'),'P0/P1 freeze changed')
            for name,digest in s.read_json(run/'context_fit_complete.json')['files'].items():s.require(old.sha(run/name)==digest,'P0/P1 frozen file changed')
            frame=pd.read_csv(path,usecols=['sample_id',*old.QUANTILES],dtype={'sample_id':str})
            six=frame[old.QUANTILES].to_numpy()
        ids=role_ids(schedule,seed,c,'test')
        s.require(frame.sample_id.tolist()==ids and frame.sample_id.is_unique and np.isfinite(six).all(),'baseline test prediction mismatch')
        frames[seed,c,method]=six
    s.require(len(frames)==30 and {k[2] for k in frames}=={'scale_only','P0','P1'},'fixed baseline schedule changed')
    old.write_json(STUDY/'PHASE3B_BASELINE_MATCH_AUDIT.json',{'all_eligible_references_reverified':True,'contexts':30,
                   'authority_sha256':old.sha(STUDY/'PHASE3A_BASELINE_COMPATIBILITY.json'),'HIER_still_excluded':True,
                   'methods':['scale_only','P0','P1'],'before_test_truth_read':True,'reference_selection_based_on_new_scores':False},immutable=True)
    return frames


def metrics(truth,prediction,scales,thresholds,regimes,ids):
    values,tails,_=old.metric_rows(truth,prediction,scales,np.asarray(thresholds),regimes)
    error=prediction[:,[1,4]]-truth
    errs={'V1':error[:,0],'V2':error[:,1],'Center':error.mean(axis=1),'Width':error[:,1]-error[:,0]}
    for endpoint,vector in errs.items():
        idx=int(np.argmax(np.abs(vector)))
        values[endpoint+'_max_abs_error']=float(np.abs(vector[idx]))
        values[endpoint+'_max_error_sample_id']=ids[idx]
    values.update(V1_quantile_crossing_rows=int(np.any(np.diff(prediction[:,:3],axis=1)<0,axis=1).sum()),
                  V2_quantile_crossing_rows=int(np.any(np.diff(prediction[:,3:],axis=1)<0,axis=1).sum()),
                  V2_below_V1_q50_rows=int((prediction[:,4]<prediction[:,1]).sum()),
                  negative_output_rows=int(np.any(prediction<0,axis=1).sum()),
                  predictions_were_repaired=False)
    return values,tails


def pairing(results):
    out=[]
    for (seed,c),group in results.groupby(['outer_seed','column']):
        a,b=[group.loc[group.arm.eq(arm)].iloc[0] for arm in old.ARMS]
        s.require(a.test_ids_sha256==b.test_ids_sha256 and a.scale_V1==b.scale_V1 and a.scale_V2==b.scale_V2,'unpaired score identity/scale')
        out.append({'outer_seed':int(seed),'column':c,'role':'test','A2_seed_win':bool(b.combined_normalized_rmse<a.combined_normalized_rmse),
                    **{f'{arm}_{metric}':float(row[metric]) for arm,row in [('A1',a),('A2',b)] for metric in LOSS_METRICS},
                    **{metric+'_relative_gain':float(1-b[metric]/a[metric]) if a[metric]!=0 else np.nan for metric in LOSS_METRICS},
                    'V1_r2_A2_minus_A1':float(b.V1_r2-a.V1_r2),'V2_r2_A2_minus_A1':float(b.V2_r2-a.V2_r2)})
    return pd.DataFrame(out)


def verify_completed():
    receipt=s.read_json(STUDY/'PHASE3B_SCORE_MANIFEST.json')
    s.require(receipt['prediction_freeze_sha256']==old.sha(STUDY/'PREDICTION_FREEZE_MANIFEST.json')
              and receipt['evaluation_plan_sha256']==old.sha(STUDY/'EVALUATION_PLAN.md'),'score/freeze evidence changed')
    for name,digest in receipt['files'].items():s.require(old.sha(STUDY/name)==digest,'completed score artifact changed')
    s.verify_freeze()
    return receipt


def score_once():
    if (STUDY/'PHASE3B_SCORE_MANIFEST.json').exists():
        verify_completed();print('Existing one-time score verified; no labels reread or metrics recomputed.');return
    schedule,predictions,fits=preflight()
    historical,historical_scores=historical_inputs(schedule)
    baselines=baseline_inputs(schedule)
    provenance_paths=[Path(__file__),STUDY/'test_phase3b.py',ROOT/'src/qgeognn_al/evaluation/point.py',
                      ROOT/'scripts/studies/run_column_conditioned_multitask.py',ROOT/'scripts/studies/run_filtered_full_data_benchmark.py']
    old.write_json(STUDY/'PHASE3B_SCORING_PROVENANCE.json',{'files':{str(p.relative_to(ROOT)):old.sha(p) for p in provenance_paths},
                   'standard_metric_function':old.point_metrics.__module__+'.point_metrics','auxiliary_metric_function':old.metric_rows.__module__+'.metric_rows',
                   'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'test_role_only':True,
                   'evaluation_plan_sha256':old.sha(STUDY/'EVALUATION_PLAN.md')},immutable=True)
    # No outer responses have been parsed above. Recalculate only permitted train statistics first.
    contexts=[]
    for seed in s.SEEDS:
        for c in s.TARGETS:
            ids=role_ids(schedule,seed,c,'gradient_train')
            train=old._read_authorized_truth(old.PARENT/f'filtered_canonical_{c}.csv',ids)
            scales=old.scales(train);thresholds=np.quantile(train,.8,axis=0)
            for arm in old.ARMS:
                authority=fits[seed,arm]['training_identity'][c]
                s.require(old.digest(train.tolist())==authority['label_values_sha256'],'training label identity/value digest changed')
                for endpoint in ['V1','V2']:s.require(np.isclose(scales[endpoint],authority['metric_scales'][endpoint],atol=1e-12,rtol=1e-12),'frozen scale mismatch')
                np.testing.assert_allclose(thresholds,authority['tail_thresholds_80pct'],atol=1e-12,rtol=1e-12)
            contexts.append({'outer_seed':seed,'column':c,'train_ids':ids,'test_ids':role_ids(schedule,seed,c,'test'),
                             'scales':scales,'thresholds':thresholds.tolist(),'regimes':np.quantile(train.mean(axis=1),[1/3,2/3]).tolist(),
                             'training_labels_sha256':old.digest(train.tolist())})
    old.write_json(STUDY/'PHASE3B_TRAIN_SCALE_AUDIT.json',{'all_passed':True,'before_test_truth_read':True,'contexts':contexts},immutable=True)
    print('ALL FREEZE/IDENTITY/BUDGET/REFERENCE/TRAIN-SCALE CHECKS PASSED BEFORE TEST TRUTH',flush=True)
    # The single authorized outer response access boundary.
    for context in contexts:
        seed,c=context['outer_seed'],context['column']
        truth=old._read_authorized_truth(old.PARENT/f'filtered_canonical_{c}.csv',context['test_ids'])
        s.require(truth.shape==(len(context['test_ids']),2) and np.isfinite(truth).all(),'invalid authorized test responses')
        context['test_truth']=truth.tolist()
        context['test_truth_sha256']=old.digest(truth.tolist())
    old.write_json(STUDY/'PHASE3B_AUTHORIZED_SCORING_INPUTS.json',{'scored_role':'test','validation_truth_parsed':False,'contexts':contexts,
                   'preflight_sha256':old.sha(STUDY/'PHASE3B_PREFLIGHT_AUDIT.json'),'original_canonical_sha256':s.read_json(STUDY/'protocol.json')['population_sha256']},immutable=True)
    rows,tails,history_rows,baseline_rows,baseline_tails=[],[],[],[],[]
    for context in contexts:
        seed,c,ids=context['outer_seed'],context['column'],context['test_ids']
        truth=np.asarray(context['test_truth']);scales=context['scales'];thresholds=context['thresholds']
        meta={'protocol':'row','outer_seed':seed,'column':c,'role':'test','test_count':len(ids),
              'test_ids_sha256':old.digest(sorted(ids)),'gradient_train_ids_sha256':old.digest(sorted(context['train_ids'])),
              'scale_V1':scales['V1'],'scale_V2':scales['V2'],'classification':'DEVELOPMENTAL CONFIRMATION'}
        for arm in old.ARMS:
            pred=predictions[seed,c,arm,'test'][old.QUANTILES].to_numpy()
            values,tail=metrics(truth,pred,scales,thresholds,context['regimes'],ids)
            fit=fits[seed,arm]
            rows.append({**meta,'arm':arm,'fixed_epochs':fit['fixed_epochs'],'optimizer_steps':fit['optimizer_steps'],**values})
            tails.extend({**meta,'arm':arm,**item} for item in tail)
            hist_pred,hist_fit=historical[seed,c,arm]
            hist_values,hist_tail=metrics(truth,hist_pred,scales,thresholds,context['regimes'],ids)
            saved=historical_scores.loc[historical_scores.outer_seed.eq(seed)&historical_scores.column.eq(c)&historical_scores.method.eq(arm)]
            s.require(len(saved)==1,'historical scores missing/duplicate')
            for metric in ['combined_normalized_rmse','V1_rmse','V2_rmse','V1_mae','V2_mae','V1_r2','V2_r2','Center_rmse','Width_rmse']:
                s.require(np.isclose(saved.iloc[0][metric],hist_values[metric],atol=1e-10,rtol=1e-10),'historical metric not reproducible under identical test identities/scales')
            history_rows.append({**meta,'arm':arm,'Group_CV_epochs':hist_fit['epochs_run'],'Row_CV_epochs':fit['fixed_epochs'],
                                 'Group_CV_steps':hist_fit['epochs_run']*hist_fit['steps_per_epoch'],'Row_CV_steps':fit['optimizer_steps'],
                                 **{f'Group_CV_{metric}':hist_values[metric] for metric in LOSS_METRICS},
                                 **{f'Row_CV_{metric}':values[metric] for metric in LOSS_METRICS},
                                 **{f'Row_vs_Group_{metric}_relative_gain':float(1-values[metric]/hist_values[metric]) for metric in LOSS_METRICS},
                                 'historic_saved_metrics_reproduced':True})
        for method in ['scale_only','P0','P1']:
            values,tail=metrics(truth,baselines[seed,c,method],scales,thresholds,context['regimes'],ids)
            # Scale has only q50 endpoints; no quantile validity claim from zero placeholders.
            if method=='scale_only':
                for name in ['V1_quantile_crossing_rows','V2_quantile_crossing_rows','negative_output_rows']:values[name]=np.nan
            baseline_rows.append({**meta,'method':method,'reference_role':'secondary complete workflow comparison',**values})
            baseline_tails.extend({**meta,'method':method,**item} for item in tail)
    result=pd.DataFrame(rows)
    paired=pairing(result)
    files={'OUTER_ROW_RESULTS.csv':result,'OUTER_ROW_PAIRED_COMPARISONS.csv':paired,'OUTER_ROW_TAIL_DIAGNOSTICS.csv':pd.DataFrame(tails),
           'GROUP_VS_ROW_OUTER_COMPARISON.csv':pd.DataFrame(history_rows),'OUTER_ROW_BASELINE_RESULTS.csv':pd.DataFrame(baseline_rows),
           'OUTER_ROW_BASELINE_TAIL_DIAGNOSTICS.csv':pd.DataFrame(baseline_tails)}
    for name,frame in files.items():frame.to_csv(STUDY/name,index=False)
    s.verify_freeze();s.r.verify_phase1()
    names=[*files,'PHASE3B_PREFLIGHT_AUDIT.json','PHASE3B_HISTORICAL_MATCH_AUDIT.json','PHASE3B_BASELINE_MATCH_AUDIT.json',
           'PHASE3B_SCORING_PROVENANCE.json','PHASE3B_TRAIN_SCALE_AUDIT.json','PHASE3B_AUTHORIZED_SCORING_INPUTS.json']
    receipt={'status':'ONE_TIME_FIXED_TEST_SCORE_COMPLETE','prediction_freeze_sha256':old.sha(STUDY/'PREDICTION_FREEZE_MANIFEST.json'),
             'evaluation_plan_sha256':old.sha(STUDY/'EVALUATION_PLAN.md'),'files':{name:old.sha(STUDY/name) for name in names},
             'classification':'DEVELOPMENTAL CONFIRMATION','primary_observations':20,'paired_observations':10,
             'test_truth_contexts':10,'test_truth_records_across_seeds':sum(len(c['test_ids']) for c in contexts),
             'validation_truth_parsed':False,'scoring_runs':1,'new_training_fits':0,'frozen_predictions_modified':False,
             'HIER_excluded':True,'historical_absolute_metrics_reproduced':True,'stopped_before_new_candidates':True}
    old.write_json(STUDY/'PHASE3B_SCORE_MANIFEST.json',receipt,immutable=True)
    for name in [*names,'PHASE3B_SCORE_MANIFEST.json']:(STUDY/name).chmod(0o444)
    print(result[['outer_seed','column','arm','combined_normalized_rmse']].to_string(index=False),flush=True)
    print(paired[['outer_seed','column','combined_normalized_rmse_relative_gain','A2_seed_win']].to_string(index=False),flush=True)


if __name__=='__main__':
    score_once()
