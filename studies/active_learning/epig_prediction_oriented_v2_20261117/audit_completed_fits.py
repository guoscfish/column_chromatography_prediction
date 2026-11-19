"""Independent read-only audit of a completed-fit snapshot; never reads targets."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from qgeognn_al.models import build_predictor
from qgeognn_al.schemas.conditions import ConditionNormalization
BASE=HERE.with_name('qgeognn_v2_row_sequential_b32')
ARMS=('posterior_epig_conditional_greedy32','posterior_ivr_conditional_greedy32')
SEEDS=(157,887,2357,6101,12203)
OUT=HERE/'evidence/runner'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def ih(ids):return stable(sorted(map(str,ids)))
def state_hash(state):
    h=hashlib.sha256()
    for name,value in state.items():
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()
def check(condition,message):
    if not condition:raise AssertionError(message)
def close(a,b,message,atol=2e-12):check(np.isclose(float(a),float(b),atol=atol,rtol=2e-12),message)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    started=time.time();errors=[];fit_rows=[];selection_rows=[];inputs={};metric_rows=[]
    status=read(HERE/'exploratory_status.json')
    receipts=sorted((HERE/'runtime').glob('seed_*/*/round_*/fit_receipt.json'))
    snapshot={str(p.relative_to(HERE)):sha(p) for p in receipts}
    frozen=read(HERE/'freeze.json');frozen_hash=sha(HERE/'freeze.json')
    frozen_count=0; excluded_target_bearing_paths=[]
    for name in ('freeze.json','../epig_prediction_oriented_20261117/freeze.json','../epig_prediction_oriented_20261117/stage2_nested_calibration/freeze.json'):
        f=read(HERE/name)
        for relative,digest in {**f.get('files',{}),**f.get('dependencies',{}),**f.get('data_hashes',{})}.items():
            if relative.startswith('dataset/'):
                excluded_target_bearing_paths.append(relative);continue
            check(sha(ROOT/relative)==digest,f'frozen dependency drift {relative}');frozen_count+=1
    # Initial and completed states only; split files contain identities, no targets.
    for seed in SEEDS:
        sp=BASE/f'splits/row_seed_{seed}.csv'
        split=pd.read_csv(sp,dtype={'sample_id':str});ids=split.sample_id.tolist()
        l0=np.flatnonzero(split.role.eq('l0')).tolist();u0=np.flatnonzero(split.role.eq('u0')).tolist()
        valid_ids=split.loc[split.role.eq('validation'),'sample_id'].tolist()
        seed_paths=[p for p in receipts if f'/seed_{seed}/' in str(p)]
        if not seed_paths:continue
        context=read(HERE/f'runtime/seed_{seed}/context/context.json')['contract']
        baseline=read(BASE/f'runtime/seed_{seed}/context.json')['contract']
        check(context['preprocessing']==baseline['preprocessing'],f'{seed} preprocessing drift')
        check(context['normalization']==baseline['normalization'],f'{seed} normalization drift')
        initial_model=BASE/f'runtime/seed_{seed}/lcmd/round_00/model'
        old_audit=read(initial_model/'fit_audit.json')
        expected_config=read(BASE/f'runtime/seed_{seed}/lcmd/round_00/input_contract.json')['training_config']
        random.seed(seed+2000003);np.random.seed(seed+2000003);torch.manual_seed(seed+2000003)
        initial=build_predictor(ConditionNormalization(**context['normalization']))
        actual_initial_hash=state_hash(initial.state_dict())
        check(actual_initial_hash==old_audit['initialization_hash'],f'{seed} independent initialization reconstruction drift')
        parameter_shapes={n:tuple(v.shape) for n,v in initial.named_parameters()}
        check(sum(v.numel() for v in initial.parameters())==458952,f'{seed} actual parameter count')
        for arm in ARMS:
            trajectory=HERE/f'runtime/seed_{seed}/{arm}'
            arm_paths=[p for p in seed_paths if f'/{arm}/' in str(p)]
            if not arm_paths:continue
            rounds={int(read(p)['round']):p for p in arm_paths}
            check(sorted(rounds)==list(range(max(rounds)+1)),f'{seed} {arm} noncontiguous completed fits')
            labeled=l0.copy();unlabeled=u0.copy();chosen_history=[];curve=[]
            scales=context['preprocessing']['target_scales']
            for r,p in sorted(rounds.items()):
                receipt=read(p);rd=p.parent;model=ROOT/receipt['model_directory']
                try:
                    check(sha(p)==snapshot[str(p.relative_to(HERE))],f'fit receipt changed {p}')
                    check(receipt['seed']==seed and receipt['arm']==arm and receipt['budget']==333+32*r,'fit identity drift')
                    if r==0:
                        check(receipt['reused'] and receipt['checkpoint_sha256']==old_audit['checkpoint_sha256'],'initial reuse differs')
                        check(receipt['fit_audit_sha256']==sha(model/'fit_audit.json'),'initial audit hash differs')
                        files={n:sha(model/n) for n in ('best.pt','predictions.csv.gz','fit_audit.json','history.csv')}
                    else:
                        source=trajectory/f'round_{r-1:02d}';selection_path=source/'selection.json'
                        selection=read(selection_path);selection_hash=sha(selection_path)
                        check(read(source/'selection.sha256.json')['sha256']==selection_hash,'selection byte hash drift')
                        check(selection['selected_ids_hash']==stable(selection['selected_ids']),'ordered selected ID hash drift')
                        inp=selection['input'];chosen=selection['selected_ids'];old_l=[ids[i] for i in labeled];old_u=[ids[i] for i in unlabeled]
                        check(inp['source_round']==r-1 and inp['freeze_sha256']==frozen_hash,'selection round/freeze drift')
                        check(inp['L_ids_hash']==ih(old_l) and inp['U_ids_hash']==ih(old_u),'selection L/U hash drift')
                        prev=read(rounds[r-1]);check(inp['source_checkpoint_sha256']==prev['checkpoint_sha256'],'selection source checkpoint differs')
                        check(len(chosen)==32 and len(set(chosen))==32 and set(chosen)<=set(old_u) and not set(chosen)&set(old_l),'illegal B32 transition')
                        outer=labeled+unlabeled
                        check([ids[outer[i]] for i in selection['diagnostics']['selected_positions']]==chosen,'posterior-position/ID alignment drift')
                        posterior=selection['posterior'];aux=ROOT/posterior['model_path'];check(read(aux.parent/'complete.json')==posterior,'aux complete receipt drift')
                        for name,digest in posterior['files'].items():check(sha(aux.parent/name)==digest,f'aux hash drift {name}')
                        audit_aux=read(aux.parent/'fit_audit.json')
                        check((audit_aux['kernel'],audit_aux['n_train'],audit_aux['n_outputs'],audit_aux['maxiter'],audit_aux['restarts'])==('matern32',len(labeled),2,60,1),'surrogate rule drift')
                        check(audit_aux['L_ids_hash']==ih(old_l) and audit_aux['U_ids_hash']==ih(old_u),'aux training identity drift')
                        check(audit_aux['fixed_L333_scales_ml']==[scales['V1'],scales['V2']],'proxy fixed scales drift')
                        check(audit_aux['validation_labels_used']==audit_aux['future_U_labels_used']==audit_aux['test_labels_used']==0,'aux forbidden label claim')
                        acquisition=selection['diagnostics']['audit']
                        check(acquisition['reference_rows']==3330 and acquisition['target_distribution']=='P1_full' and acquisition['target_semantics']=='independent_future_noisy_replicate','full P1/replicate semantics drift')
                        if arm==ARMS[0]:close(sum(t['conditional_score'] for t in selection['diagnostics']['trace']),acquisition['batch_target_information_nats'],'EPIG chain gains/MI drift',atol=1e-9)
                        outcomes=list(source.glob('label_access_attempt_*.outcome.json'))
                        matching=[]
                        for op in outcomes:
                            outcome=read(op)
                            if outcome['status']=='REVEALED' and outcome['selection_ids']==chosen:
                                check(outcome['selection_sha256']==selection_hash,'label access selection binding drift')
                                request=read(Path(str(op).replace('.outcome.json','.json')))
                                check(request['status']=='REQUESTED' and request['selection_ids']==chosen,'missing pre-reveal request')
                                check(request['before_reveal_unix']<=outcome['ended_unix'],'reveal time reversed')
                                check(outcome['test_labels_requested']==0,'test label request')
                                calls=outcome['calls'];check(calls[-1]['purpose']=='after_acquisition_fit' and calls[-1]['requested_ids_hash']==ih(chosen) and calls[-1]['requested_rows']==32 and calls[-1]['acquisitions_frozen'],'reveal authorization mismatch')
                                check(all(c['roles'] in ('l0','validation','u0') and c['purpose'] in ('initial_fit','after_acquisition_fit') for c in calls),'invalid target access role/purpose')
                                matching.append(op)
                        check(bool(matching),'missing durable successful label-access outcome')
                        lookup={ids[i]:i for i in unlabeled};labeled+= [lookup[i] for i in chosen]
                        unlabeled=[i for i in unlabeled if ids[i] not in set(chosen)];chosen_history+=chosen
                        state=read(rd/'state.json')
                        check(stable(state)==receipt['state_hash'],'fit state receipt hash drift')
                        check(state['ordered_labeled_indices']==labeled and state['ordered_unlabeled_indices']==unlabeled,'ordered L/U transition differs')
                        check(state['L_ids_hash']==ih([ids[i] for i in labeled]) and state['U_ids_hash']==ih([ids[i] for i in unlabeled]),'state L/U identity drift')
                        check(state['target_scales']==scales and state['training_config']==expected_config and state['freeze_sha256']==frozen_hash,'state scales/config/freeze differs')
                        files=receipt['files']
                        for name,digest in files.items():check(sha(model/name)==digest,f'fit file drift {name}')
                        predictions=pd.read_csv(model/'predictions.csv.gz',dtype={'sample_id':str})
                        check(predictions.sample_id.tolist()==[ids[i] for i in unlabeled],'prediction ID/order differs from legal U')
                        check(np.isfinite(predictions.iloc[:,1:].to_numpy(float)).all(),'nonfinite predictions')
                        selection_rows.append({'seed':seed,'arm':arm,'source_round':r-1,'budget_before':333+32*(r-1),'selection_sha256':selection_hash,'selected_count':32,'reference_rows':3330,'successful_access_outcomes':len(matching),'status':'PASS'})
                    audit=read(model/'fit_audit.json');history=pd.read_csv(model/'history.csv')
                    check(audit['checkpoint_sha256']==sha(model/'best.pt') and audit['prediction_sha256']==sha(model/'predictions.csv.gz'),'model audit hashes differ')
                    check(audit['initialization_seed']==seed+2000003 and audit['initialization_hash']==actual_initial_hash and receipt['initialization_hash']==actual_initial_hash,'scratch initialization drift')
                    check(audit['train_rows']==333+32*r and audit['validation_rows']==416 and audit['train_ids_hash']==ih([ids[i] for i in labeled]) and audit['validation_ids_hash']==ih(valid_ids),'fit train/validation rows or IDs drift')
                    check(audit['parameter_count']==458952 and audit['model_variant']=='qgeognn_v2' and audit['test_labels_used_for_fit_or_checkpoint_selection']==0 and audit['test_graph_label_state']=='zero_sentinel','model architecture/target boundary differs')
                    checkpoint=torch.load(model/'best.pt',map_location='cpu',weights_only=False)
                    check(checkpoint['training_config']==expected_config and checkpoint['preprocessing']==context['preprocessing'] and checkpoint['normalization']==context['normalization'],'checkpoint training/preprocessing differs')
                    check(checkpoint['parameter_count']==458952 and checkpoint['model_variant']=='qgeognn_v2','checkpoint architecture differs')
                    check(all(tuple(checkpoint['model_state_dict'][n].shape)==shape for n,shape in parameter_shapes.items()),'checkpoint parameter shapes differ')
                    check(sum(checkpoint['model_state_dict'][n].numel() for n in parameter_shapes)==458952,'checkpoint actual parameter count differs')
                    check(state_hash(checkpoint['model_state_dict'])==audit['checkpoint_state_hash'],'checkpoint tensor-state hash differs')
                    check(len(history)==audit['epochs_run']<=1000,'epoch count differs')
                    best=history.loc[history.epoch.eq(audit['best_epoch'])];check(len(best)==1,'best epoch absent/nonunique')
                    best=best.iloc[0]
                    close(best.validation_selection_score,history.validation_selection_score.min(),'wrong checkpoint selection')
                    close(audit['best_validation_combined_normalized_rmse'],best.validation_combined_normalized_rmse,'audit score/history mismatch')
                    score=np.sqrt(((float(best.validation_V1_rmse)/scales['V1'])**2+(float(best.validation_V2_rmse)/scales['V2'])**2)/2)
                    close(score,best.validation_combined_normalized_rmse,'independent combined NRMSE mismatch',atol=2e-7)
                    values={'combined_nrmse':float(best.validation_combined_normalized_rmse),'V1_RMSE':float(best.validation_V1_rmse),'V2_RMSE':float(best.validation_V2_rmse),'V1_MAE':float(best.validation_V1_mae),'V2_MAE':float(best.validation_V2_mae)}
                    for key,value in values.items():close(value,receipt['validation'][key],f'receipt {key}/history differs')
                    check(checkpoint['provenance']['epoch']==audit['best_epoch'],'checkpoint provenance epoch differs')
                    fit_rows.append({'seed':seed,'arm':arm,'round':r,'budget':333+32*r,**values,'best_epoch':audit['best_epoch'],'epochs_run':audit['epochs_run'],'actual_parameter_count':458952,'initialization_hash':actual_initial_hash,'fit_receipt_sha256':sha(p),'status':'PASS'})
                    curve.append({'budget':333+32*r,**values})
                    inputs[str(p.relative_to(HERE))]=sha(p)
                except Exception as exc:
                    errors.append({'path':str(p.relative_to(HERE)),'type':type(exc).__name__,'error':str(exc)})
                    break
            if len(curve)==22 and seed in (157,887,2357):
                seal=read(trajectory/'trajectory_freeze.json')
                check(seal['selected_ids_hash']==stable(chosen_history),'trajectory selected history hash differs')
                for rel,digest in seal['fit_receipts'].items():check(sha(HERE/rel)==digest,'trajectory fit receipt hash differs')
                f=pd.DataFrame(curve);rec={'seed':seed,'arm':arm,'complete_L1005':True}
                for lo,hi,key in ((333,1005,'full'),(333,525,'early'),(525,813,'middle'),(813,1005,'late')):
                    part=f.loc[f.budget.between(lo,hi)]
                    # Explicit trapezoid sum, independent from reporting helper.
                    x=part.budget.to_numpy(float);y=part.combined_nrmse.to_numpy(float)
                    rec['AULC_'+key]=float(np.sum(np.diff(x)*(y[1:]+y[:-1])/2)/(hi-lo))
                for budget in (525,813,1005):rec[f'NRMSE_L{budget}']=float(f.loc[f.budget==budget,'combined_nrmse'].iloc[0])
                metric_rows.append(rec)
    # Only complete trajectories in this snapshot are compared with the live report.
    published=pd.read_csv(HERE/'seed_metrics.csv')
    metric_checks=[]
    for rec in metric_rows:
        row=published.loc[(published.seed==rec['seed'])&(published.arm==rec['arm'])].iloc[0]
        for name,value in rec.items():
            if name in ('seed','arm','complete_L1005'):continue
            difference=abs(value-float(row[name]));close(value,row[name],f'published {name} differs')
            metric_checks.append({'seed':rec['seed'],'arm':rec['arm'],'metric':name,'independent_history_value':value,'published_value':float(row[name]),'absolute_difference':difference})
    OUT.mkdir(parents=True,exist_ok=True)
    stamp='20261118'
    pd.DataFrame(fit_rows).to_csv(OUT/f'in_progress_learning_QA_{stamp}_fits.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(OUT/f'in_progress_learning_QA_{stamp}_selections.csv',index=False)
    pd.DataFrame(metric_checks).to_csv(OUT/f'in_progress_learning_QA_{stamp}_metrics.csv',index=False)
    result={'schema':'epig-v2-in-progress-independent-learning-QA','started_unix':started,'completed_unix':time.time(),
            'script_sha256':sha(__file__),'snapshot_status':status,'snapshot_fit_receipts':len(receipts),
            'snapshot_receipt_hashes':snapshot,'verified_fit_receipts':len(fit_rows),
            'verified_downstream_fits':sum(r['round']>0 for r in fit_rows),'verified_selections':len(selection_rows),
            'metric_audit_seed_cohort':[157,887,2357], 'complete_trajectory_metrics':metric_rows,'complete_seed_pairs':sum(sum(r['seed']==s for r in metric_rows)==2 for s in SEEDS),
            'metric_comparisons':len(metric_checks),'max_published_metric_absolute_difference':max((m['absolute_difference'] for m in metric_checks),default=0.),
            'frozen_file_checks':frozen_count,'excluded_data_hash_paths_to_avoid_source_access':sorted(set(excluded_target_bearing_paths)),'errors':errors,'status':'PASS' if not errors else 'FAIL',
            'independent_initialization_reconstruction':True,'checkpoint_tensor_hashes_verified':True,
            'source_targets_read':0,'future_U_targets_read':0,'test_targets_read':0,'new_training_fits':0,
            'notes':['Read only completed-fit artifacts, frozen identity/split files, and protected source code.',
                     'Training truth numeric values were not re-read from the canonical CSV; verified state truth hashes, legal ID transitions, exact checkpoints and persisted RestrictedLabelStore receipts instead.',
                     'Physical endpoint metrics are independently read from validation history, not independently recomputed from raw validation labels.',
                     'AULC comparison cohort is fixed to the three fully sealed seed pairs requested at task start (157,887,2357); later completed pairs remain fit/transition-audited only in this snapshot.',
                     'This is an in-progress snapshot. Three complete overlapping outer seeds do not establish the required five-seed efficacy verdict.',
                     'Training worker/supervisor were neither started, stopped, locked, nor modified.']}
    write(OUT/f'in_progress_learning_QA_{stamp}.json',result)
    md=['# Independent in-progress learning QA','',f'Snapshot: {result["verified_downstream_fits"]} completed downstream fits, {result["verified_selections"]} acquisitions, {result["complete_seed_pairs"]} complete seed pairs. Status: **{result["status"]}**.',
        '',f'Checked checkpoint/audit/prediction/history hashes, actual458952-parameter tensor shapes and state hashes, independently reconstructed per-seed initialization, original training config, fixedL333 preprocessing/scales, exact orderedL/U transitions and B32 membership, auxiliary hashes/fullP1 semantics, immutable selection SHA before restricted access and durable request/outcome receipts. Frozen file/dependency checks: {frozen_count}.',
        '',f'For completed trajectories, independently integrated raw history scores with explicit trapezoid sums. {len(metric_checks)} full/segmented AULC and milestone comparisons agree with the published report; maximum absolute difference {result["max_published_metric_absolute_difference"]:.3g}. Physical V1/V2 MAE/RMSE receipt values agree with history.',
        '', 'No source labels were read and no training/process state was changed. The artifact audit does not independently recover numerical ground truth from source CSV. This partial snapshot is not a five-seed efficacy result.']
    (OUT/f'in_progress_learning_QA_{stamp}.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('snapshot_receipt_hashes','complete_trajectory_metrics')},ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
