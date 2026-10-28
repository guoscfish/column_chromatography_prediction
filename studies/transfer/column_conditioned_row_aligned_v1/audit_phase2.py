"""Per-seed identity, update, checkpoint, authorized-error and zero-label replay audit."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase2 as r


def audit_seed(seed):
    torch.set_num_threads(1)
    r.verify_phase1()
    data,access=r.load_data(seed)
    schedules=r.p.fold_schedules(data)
    out=r.PHASE2/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    results,tails,regimes,predictions,histories,identities,checks,replay=[],[],[],[],[],[],[],[]
    for fold,(train,valid) in enumerate(schedules,1):
        expected=r.old.identity_records(data,train,valid,'row_cv',seed,fold)
        for arm in r.old.ARMS:
            run=r.STUDY/f'runtime/inner/seed_{seed}/fold_{fold}/{arm}'
            r.check_receipt(run,phase2=seed!=r.p.PILOT_SEED)
            actual=pd.read_csv(run/'identities.csv')
            pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True))
            result=pd.read_csv(run/'results.csv');h=pd.read_csv(run/'history.csv')
            best=int(result.best_epoch.iloc[0]);steps=int(result.steps_per_epoch.iloc[0])
            if best!=int(h.loc[h.joint_validation_score.idxmin(),'epoch']):
                raise RuntimeError('best epoch is not earliest joint argmin')
            if len(h)!=int(result.epochs_run.iloc[0]) or int(h.optimizer_steps.iloc[-1])!=int(result.optimizer_steps.iloc[0]):
                raise RuntimeError('optimizer-step/history count mismatch')
            np.testing.assert_array_equal(h.optimizer_steps,np.arange(1,len(h)+1)*steps)
            if int(result.optimizer_steps_at_best.iloc[0])!=best*steps:
                raise RuntimeError('checkpoint optimizer budget mismatch')
            numeric=h[[c for c in h.columns if c.startswith('train_loss_') or c in ['validation_loss_25g','validation_loss_40g','joint_validation_score']]]
            if not np.isfinite(numeric.to_numpy()).all():
                raise RuntimeError('nonfinite training or target validation diagnostics')
            torch.manual_seed(seed+fold*1009)
            model=r.old.build_multitask_model(r.old.source_path(),r.old.ARMS[arm])
            initial=deepcopy(model.state_dict())
            checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            if checkpoint['best_epoch']!=best or checkpoint['source_sha256']!=r.old.SOURCE_SHA:
                raise RuntimeError('checkpoint authority mismatch')
            model.load_state_dict(checkpoint['model_state_dict'])
            for name,param in model.named_parameters():
                if not param.requires_grad and not torch.equal(model.state_dict()[name],initial[name]):
                    raise RuntimeError('frozen checkpoint parameter changed')
            for name,module in model.named_modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                    expected_delta=best*steps if any(p.requires_grad for p in module.parameters(recurse=False)) else 0
                    if int(model.state_dict()[name+'.num_batches_tracked']-initial[name+'.num_batches_tracked'])!=expected_delta:
                        raise RuntimeError('BN checkpoint counter mismatch')
                    if expected_delta==0:
                        for buffer in ['running_mean','running_var']:
                            if not torch.equal(model.state_dict()[name+'.'+buffer],initial[name+'.'+buffer]):
                                raise RuntimeError('frozen BN checkpoint buffer changed')
            state=json.loads((run/'state_audit.json').read_text())
            if not all(state[k] for k in ['frozen_parameters_unchanged','bn_policy_valid','finite_predictions_and_gradients']):
                raise RuntimeError('fit state audit failed')
            pred=pd.read_csv(run/'predictions.csv')
            for c in r.TARGETS:
                part=pred.loc[pred.column.eq(c)]
                ids=data[c]['frame'].iloc[valid[c]].sample_id.tolist()
                if part.sample_id.tolist()!=ids:
                    raise RuntimeError('OOF prediction identities differ from frozen fold')
                truth=data[c]['truth'][valid[c]]
                saved=part[r.old.QUANTILES].to_numpy()
                metric_scales=r.old.scales(data[c]['truth'][train[c]])
                metrics,_,_=r.old.metric_rows(truth,saved,metric_scales,data[c]['tail'],data[c]['regimes'])
                row=result.loc[result.column.eq(c)].iloc[0]
                for metric in ['V1_rmse','V2_rmse','V1_mae','V2_mae','combined_normalized_rmse','Center_rmse','Width_rmse']:
                    if not np.isclose(float(row[metric]),metrics[metric],atol=1e-10,rtol=1e-10):
                        raise RuntimeError(f'saved metric mismatch: {metric}')
                if not np.isclose(float(row.combined_normalized_rmse),h.loc[h.epoch.eq(best),f'{c}_combined_normalized_rmse'].iloc[0],atol=1e-12):
                    raise RuntimeError('best-history metric mismatch')
                err=saved[:,[1,4]]-truth
                maxima={'V1_max_abs_error':np.abs(err[:,0]).max(),'V2_max_abs_error':np.abs(err[:,1]).max(),
                    'Center_max_abs_error':np.abs(err.mean(axis=1)).max(),'Width_max_abs_error':np.abs(err[:,1]-err[:,0]).max()}
                for name,value in maxima.items():
                    result.loc[result.column.eq(c),name]=float(value)
                # Independent model replay with cloned zero-label inputs, excluding all outer records.
                zero={c:{**data[c],'atoms':[a.clone() for a in data[c]['atoms']]}}
                for atom in zero[c]['atoms']:atom.y.zero_()
                actual_pred=r.old.predict(model,zero,c,valid[c])
                diff=float(np.max(np.abs(actual_pred-saved)))
                if not np.allclose(actual_pred,saved,rtol=1e-6,atol=1e-5):
                    raise RuntimeError('checkpoint prediction replay mismatch')
                replay.append(dict(outer_seed=seed,inner_fold=fold,arm=arm,column=c,rows=len(part),max_abs_difference_ml=diff,replay_passed=True))
            checks.append(dict(outer_seed=seed,inner_fold=fold,arm=arm,best_epoch=best,epochs_run=len(h),
                optimizer_steps=int(h.optimizer_steps.iloc[-1]),steps_per_epoch=steps,optimizer_steps_at_best=best*steps,
                argmin_verified=True,identities_verified=True,frozen_parameters_unchanged=True,bn_valid=True))
            result['source_validation_labels_available']=seed==r.p.PILOT_SEED
            results.append(result);tails.append(pd.read_csv(run/'tails.csv'));regimes.append(pd.read_csv(run/'regimes.csv'))
            predictions.append(pred);histories.append(h.assign(outer_seed=seed,inner_fold=fold,arm=arm))
            if arm=='A1':identities.append(actual)
        # Shared initial tensors, scales, counts, and every common-epoch draw must match.
        left,right=results[-2].set_index('column'),results[-1].set_index('column')
        for key in ['inner_train_ids_sha256','inner_validation_ids_sha256','inner_train_count','inner_validation_count',
                    'scale_V1','scale_V2','training_seed','per_task_batch_size','steps_per_epoch']:
            np.testing.assert_array_equal(left[key],right[key])
        ha,hb=histories[-2].set_index('epoch'),histories[-1].set_index('epoch');common=ha.index.intersection(hb.index)
        np.testing.assert_array_equal(ha.loc[common,'batch_draws_sha256'],hb.loc[common,'batch_draws_sha256'])
        first= r.STUDY/f'runtime/inner/seed_{seed}/fold_{fold}/A1/initialization.json'
        second=first.parent.parent/'A2/initialization.json'
        if json.loads(first.read_text())!=json.loads(second.read_text()):raise RuntimeError('initial pairing mismatch')
    identity=pd.concat(identities,ignore_index=True)
    for c in r.TARGETS:
        oof=identity.loc[identity.column.eq(c)&identity.role.eq('inner_validation')]
        if not oof.sample_id.is_unique or set(oof.sample_id)!=set(data[c]['roles']['gradient_train']):
            raise RuntimeError('five-fold OOF coverage mismatch')
    overlap,counts=[] ,[]
    for c in r.old.TASKS:
        idx=data[c]['positions']['gradient_train'];keys=[data[c]['input_keys'][i] for i in idx]
        counts.append(dict(outer_seed=seed,column=c,gradient_train_rows=len(idx),unique_inputs=len(set(keys)),extra_same_input_records=len(keys)-len(set(keys))))
    for fold,(train,valid) in enumerate(schedules,1):
        for c in r.TARGETS:
            for donor in r.old.TASKS:
                donor_ids=set(data[donor]['frame'].iloc[train[donor]].sample_id)
                donor_keys={data[donor]['input_keys'][i] for i in train[donor]}
                mols=set(data[donor]['frame'].iloc[train[donor]].canonical_smiles)
                overlap.append(dict(outer_seed=seed,inner_fold=fold,column=c,donor=donor,heldout_rows=len(valid[c]),
                    sample_id_overlap=len(donor_ids&set(data[c]['frame'].iloc[valid[c]].sample_id)),
                    backbone_input_overlap_rows=sum(data[c]['input_keys'][i] in donor_keys for i in valid[c]),
                    full_model_input_overlap_rows=sum(data[c]['input_keys'][i] in donor_keys for i in valid[c]) if c==donor else 0,
                    canonical_overlap_rows=sum(data[c]['frame'].iloc[i].canonical_smiles in mols for i in valid[c])))
    frames={'INNER_RESULTS.csv':pd.concat(results,ignore_index=True),'TAIL_METRICS.csv':pd.concat(tails,ignore_index=True),
        'RETENTION_REGIMES.csv':pd.concat(regimes,ignore_index=True),'INNER_PREDICTIONS.csv':pd.concat(predictions,ignore_index=True),
        'TRAINING_HISTORY.csv':pd.concat(histories,ignore_index=True),'ROW_SPLIT_MANIFEST.csv':identity,
        'FIT_AUDIT.csv':pd.DataFrame(checks),'CHECKPOINT_REPLAY.csv':pd.DataFrame(replay),
        'INPUT_OVERLAP_AUDIT.csv':pd.DataFrame(overlap),'INPUT_DUPLICATE_COUNTS.csv':pd.DataFrame(counts)}
    for name,frame in frames.items():frame.to_csv(out/name,index=False)
    r.save(out/'LABEL_ACCESS.json',access)
    r.save(out/'COMPLETE_AUDIT.json',{'outer_seed':seed,'models':10,'all_passed':True,
        'base_protocol_sha256':r.old.sha(r.STUDY/'protocol.json'),
        'phase2_protocol_sha256':r.old.sha(r.STUDY/'PHASE2_PROTOCOL.json'),
        'files':{name:r.old.sha(out/name) for name in frames},'phase1_reused':seed==r.p.PILOT_SEED})
    return frames
