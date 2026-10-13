"""Phase 2 only: extend frozen Row-CV to four seeds, with no outer evaluation."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase1 as p

STUDY, ROOT, old, TARGETS = p.STUDY, p.ROOT, p.old, p.TARGETS
authorize_fit = p.authorize_fit
SEEDS = (p.PILOT_SEED, 536279090, 1362771960, 1425370602, 2767143051)
NEW_SEEDS = SEEDS[1:]
PHASE2 = STUDY / 'phase2'


def save(path, value, immutable=False):
    old.write_json(Path(path), value, immutable=immutable)


def verify_phase1():
    archive = STUDY/'phase1_archive'
    manifest = json.loads((archive/'ARCHIVE_MANIFEST.json').read_text())
    for name, expected in manifest['root_files'].items():
        if old.sha(archive/name) != expected:
            raise RuntimeError(f'Phase 1 archived artifact changed: {name}')
    for rel, expected in manifest['runtime_files'].items():
        if old.sha(STUDY/rel) != expected:
            raise RuntimeError(f'Phase 1 runtime changed: {rel}')
    for name in ['protocol.json','EXECUTION_PROVENANCE.json','run_phase1.py','test_protocol.py',
                 'summarize_phase1.py','verify_inner_predictions.py']:
        if old.sha(STUDY/name) != manifest['root_files'][name]:
            raise RuntimeError(f'frozen Phase 1 authority changed: {name}')
    p.specification()
    historical = json.loads((STUDY/'HISTORICAL_HASHES_BEFORE.json').read_text())
    current = {str(f.relative_to(ROOT)):old.sha(f) for f in old.STUDY.rglob('*') if f.is_file()}
    if historical != current:
        raise RuntimeError('historical experiment artifacts changed')
    return {'archived_files':len(manifest['root_files']), 'runtime_files':len(manifest['runtime_files']),
            'historical_files':len(historical), 'all_unchanged':True}


def freeze():
    torch.set_num_threads(1)
    preserved = verify_phase1()
    base = json.loads((STUDY/'protocol.json').read_text())
    if base['training'] != old.TRAINING or base['inner_cv']['seed'] != 42 or base['inner_cv']['folds'] != 5:
        raise RuntimeError('Phase 1 scientific settings mismatch')
    payload = {'phase':'PHASE_2_INTERNAL_ONLY','base_protocol_sha256':old.sha(STUDY/'protocol.json'),
        'base_protocol_archive': 'phase1_archive/protocol.json', 'protocol_document_sha256':old.sha(STUDY/'PROTOCOL.md'),
        'observed_before_execution':['Phase 1 internal results','historical Row outer results'],
        'interpretation':'developmental method diagnosis; not independent external confirmation',
        'seeds':list(SEEDS),'new_seeds':list(NEW_SEEDS),'reused_fits':10,'new_fits':40,
        'training':base['training'],'inner_cv':base['inner_cv'],'film':base['film'],'epoch_refit':base['epoch_refit'],
        'source_checkpoint':base['source_checkpoint'],'source_sha256':base['source_sha256'],
        'source_label_permission':'source_train only; source validation labels not parsed; no observed source validation metrics',
        'target_label_permission':'each seed outer gradient_train only',
        'gain_A':'mean(1-NRMSE_A2_fold/NRMSE_A1_fold)',
        'gain_B':'1-mean(NRMSE_A2)/mean(NRMSE_A1)',
        'source_diagnostic_change':'omit label diagnostics only; training and target selection unchanged; synthetic equivalence required',
        'forbidden_actions':['FULL refit','outer validation/test label access or scoring','optimizer intervention','hyperparameter search'],
        'workers':5,'threads_per_worker':1}
    save(STUDY/'PHASE2_PROTOCOL.json',payload,immutable=True)
    files = [Path(__file__),STUDY/'run_phase1.py',STUDY/'test_phase2.py']
    files += [ROOT/rel for rel in json.loads((STUDY/'EXECUTION_PROVENANCE.json').read_text())['files']]
    provenance = {'files':{str(f.relative_to(ROOT)):old.sha(f) for f in files},
        'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'threads_per_worker':1,
        'base_protocol_sha256':old.sha(STUDY/'protocol.json'), 'phase2_protocol_sha256':old.sha(STUDY/'PHASE2_PROTOCOL.json')}
    save(STUDY/'PHASE2_EXECUTION_PROVENANCE.json',provenance,immutable=True)
    save(PHASE2/'BEFORE_PRESERVATION_CHECK.json',preserved,immutable=True)
    split = pd.read_csv(old.QUALIFIED/'splits/row_seed_42.csv',usecols=['sample_id','split'])
    receipts = [{'outer_seed':None,'column':'4g','authorized_ids':split.loc[split.split.eq('train'),'sample_id'].astype(str).tolist()}]
    schedule = p.row_schedule()
    for seed in SEEDS:
        for c in TARGETS:
            ids = schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)&schedule.role.eq('gradient_train'),'sample_id'].astype(str).tolist()
            receipts.append({'outer_seed':seed,'column':c,'authorized_ids':ids})
    for receipt in receipts:
        receipt['count']=len(receipt['authorized_ids']); receipt['ids_sha256']=old.digest(sorted(receipt['authorized_ids']))
    save(STUDY/'PHASE2_LABEL_AUTHORIZATION.json',{'receipts':receipts,'outer_labels_authorized':False,
        'source_validation_labels_authorized':False,'feature_usecols':old.FEATURES},immutable=True)


def load_data(seed, *, truth_reader=None):
    if seed not in SEEDS:
        raise ValueError('seed outside frozen schedule')
    torch.set_num_threads(1)
    p.specification()
    reader = truth_reader or old._read_authorized_truth
    schedule = p.row_schedule()
    cache_source = torch.load(old.SOURCE_GRAPH_CACHE,map_location='cpu',weights_only=False)
    checkpoint=old.source_path()
    scaler=torch.load(checkpoint,map_location='cpu',weights_only=False)['preprocessing']['scaler']
    data, access = {}, []
    for c in old.TASKS:
        if c=='4g':
            split=pd.read_csv(old.QUALIFIED/'splits/row_seed_42.csv',usecols=['sample_id','split'])
            roles={r:split.loc[split.split.eq(s),'sample_id'].astype(str).tolist()
                   for r,s in [('gradient_train','train'),('source_validation','validation')]}
            feature_path=canonical_path=old.SOURCE_DATA
            feature_ids=roles['gradient_train']+roles['source_validation']
        else:
            split=schedule.loc[schedule.column.eq(c)&schedule.outer_seed.eq(seed)]
            roles={r:split.loc[split.role.eq(r),'sample_id'].astype(str).tolist() for r in ['gradient_train','validation','test']}
            feature_path=old.PARENT/f'filtered_features_{c}.csv'
            canonical_path=old.PARENT/f'filtered_canonical_{c}.csv'
            feature_ids=roles['gradient_train']
        allowed_labels=roles['gradient_train']  # Source validation labels are explicitly excluded.
        frame=pd.read_csv(feature_path,usecols=old.FEATURES)
        frame.sample_id=frame.sample_id.astype(str)
        if not frame.sample_id.is_unique or not set(feature_ids)<=set(frame.sample_id):
            raise RuntimeError('feature identity mismatch')
        if c!='4g' and set(frame.sample_id)!=set().union(*map(set,roles.values())):
            raise RuntimeError('outer population mismatch')
        frame=frame.loc[frame.sample_id.isin(feature_ids)].reset_index(drop=True)
        lookup={sid:i for i,sid in enumerate(frame.sample_id)}
        positions={r:[lookup[sid] for sid in ids] for r,ids in roles.items() if set(ids)<=set(feature_ids)}
        labels=reader(canonical_path,allowed_labels)
        if not np.isfinite(labels).all():
            raise RuntimeError('nonfinite authorized training labels')
        frame['V1_ml'],frame['V2_ml']=0.,0.
        frame.loc[[lookup[sid] for sid in allowed_labels],['V1_ml','V2_ml']]=labels
        access.append({'outer_seed':seed,'column':c,'count':len(allowed_labels),
                       'ids_sha256':old.digest(sorted(allowed_labels)),
                       'source_validation_labels_parsed':False,'outer_labels_parsed':False})
        cache=dict(cache_source)
        if c!='4g':
            cache.update(torch.load(ROOT/f'studies/transfer/cross_column/data_audit/graph_cache_{c}_only.pt',map_location='cpu',weights_only=False))
        atoms,angles=p.build_model_data(frame,cache,None,scaler)
        keys=[p.input_hash(a,b) for a,b in zip(atoms,angles)]
        old.cache_frozen_prefix(p.load_predictor_checkpoint(checkpoint),atoms,angles)
        truth=frame[['V1_ml','V2_ml']].to_numpy(float)
        if c=='4g':
            truth[positions['source_validation']]=np.nan  # Fail closed if someone tries to score placeholder labels.
        training_truth=truth[positions['gradient_train']]
        data[c]=dict(column=c,frame=frame,positions=positions,roles=roles,truth=truth,atoms=atoms,angles=angles,
            input_keys=keys,scales=old.scales(training_truth),tail=np.quantile(training_truth,.8,axis=0),
            regimes=np.quantile(training_truth.mean(axis=1),[1/3,2/3]),source_validation_labels_available=False)
    return data,access


def run_fold(seed,fold):
    if seed not in NEW_SEEDS or fold not in range(1,6):
        raise ValueError('Phase 2 fits only remaining four seeds and folds 1..5')
    torch.set_num_threads(1)
    data,access=load_data(seed)
    train,valid=p.fold_schedules(data)[fold-1]
    training_seed=seed+fold*1009
    init=p.initialization_audit(training_seed)
    identities=old.identity_records(data,train,valid,'row_cv',seed,fold)
    for arm in old.ARMS:
        run=STUDY/f'runtime/inner/seed_{seed}/fold_{fold}/{arm}'
        run.mkdir(parents=True,exist_ok=True)
        if (run/'complete.json').exists():
            check_receipt(run,phase2=True)
            continue
        identities.to_csv(run/'identities.csv',index=False)
        save(run/'initialization.json',init); save(run/'label_access.json',access)
        model,fitted,scales=fit_model(data,train,valid,arm,training_seed,run)
        rows,tails,regimes,predictions=[],[],[],[]
        for c in old.TASKS:
            pred=old.predict(model,data,c,valid[c])
            meta=dict(protocol='row_cv',outer_seed=seed,inner_fold=fold,arm=arm,column=c)
            if c in TARGETS:
                values,tail,strata=old.metric_rows(data[c]['truth'][valid[c]],pred,scales[c],data[c]['tail'],data[c]['regimes'])
                errors=pred[:,[1,4]]-data[c]['truth'][valid[c]]
                values.update(V1_max_abs_error=float(np.max(np.abs(errors[:,0]))),V2_max_abs_error=float(np.max(np.abs(errors[:,1]))),
                    Center_max_abs_error=float(np.max(np.abs(errors.mean(axis=1)))),Width_max_abs_error=float(np.max(np.abs(errors[:,1]-errors[:,0]))))
                tails.extend([{**meta,**item} for item in tail]); regimes.extend([{**meta,**item} for item in strata])
            else:
                values={'all_outputs_finite':bool(np.isfinite(pred).all()),'source_validation_labels_available':False}
            rows.append({**meta,**fitted,**values,'inner_train_count':len(train[c]),'inner_validation_count':len(valid[c]),
                'inner_train_ids_sha256':old.digest(sorted(data[c]['frame'].iloc[train[c]].sample_id.tolist())),
                'inner_validation_ids_sha256':old.digest(sorted(data[c]['frame'].iloc[valid[c]].sample_id.tolist())),
                'scale_V1':scales[c]['V1'],'scale_V2':scales[c]['V2']})
            predictions.extend([{**meta,'sample_id':str(data[c]['frame'].iloc[i].sample_id),
                **dict(zip(old.QUANTILES,map(float,values)))} for i,values in zip(valid[c],pred)])
        for name,values in [('results.csv',rows),('tails.csv',tails),('regimes.csv',regimes),('predictions.csv',predictions)]:
            pd.DataFrame(values).to_csv(run/name,index=False)
        files=[f.name for f in run.iterdir() if f.is_file() and f.name!='complete.json']
        save(run/'complete.json',{'protocol_sha256':old.sha(STUDY/'protocol.json'),
            'phase2_protocol_sha256':old.sha(STUDY/'PHASE2_PROTOCOL.json'),
            'provenance_sha256':old.sha(STUDY/'PHASE2_EXECUTION_PROVENANCE.json'),
            'files':{name:old.sha(run/name) for name in files}},immutable=True)
        print(f'COMPLETED seed={seed} fold={fold} {arm}',flush=True)
    return seed,fold


def check_receipt(run,phase2):
    receipt=json.loads((run/'complete.json').read_text())
    if receipt['protocol_sha256']!=old.sha(STUDY/'protocol.json'):
        raise RuntimeError('base protocol changed')
    if phase2:
        if receipt['phase2_protocol_sha256']!=old.sha(STUDY/'PHASE2_PROTOCOL.json') or receipt['provenance_sha256']!=old.sha(STUDY/'PHASE2_EXECUTION_PROVENANCE.json'):
            raise RuntimeError('Phase 2 execution authority changed')
    else:
        if receipt['provenance_sha256']!=old.sha(STUDY/'EXECUTION_PROVENANCE.json'):
            raise RuntimeError('Phase 1 execution authority changed')
    for name,expected in receipt['files'].items():
        if old.sha(run/name)!=expected:
            raise RuntimeError(f'fit artifact changed: {run/name}')


def main():
    freeze()
    from audit_phase2 import audit_seed
    # Verify and reuse the original pilot; this action has no fit path.
    audit_seed(p.PILOT_SEED)
    completed={seed:set() for seed in NEW_SEEDS}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(run_fold,seed,fold) for seed in NEW_SEEDS for fold in range(1,6)]
        for future in as_completed(futures):
            seed,fold=future.result(); completed[seed].add(fold)
            if len(completed[seed])==5:
                audit_seed(seed)
                print(f'SEED AUDITED seed={seed}: 10 fits, pairing, identities, steps, checkpoint and epoch passed',flush=True)
    from summarize_phase2 import summarize
    summarize()


def fit_model(data, train, valid, arm, seed, run):
    """Phase 1 fit recipe; omit source-validation label diagnostics under Phase 2 permissions."""
    authorize_fit(data, train, valid)
    torch.manual_seed(seed)
    model = old.build_multitask_model(old.source_path(), old.ARMS[arm])
    initial = deepcopy(model.state_dict())
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = model.optimizer(pretrained_lr=old.TRAINING['pretrained_lr'], new_lr=old.TRAINING['new_lr'],
                                weight_decay=old.TRAINING['weight_decay'])
    size = min(old.TRAINING['per_task_batch_cap'], max(len(train[c]) for c in TARGETS))
    batcher = old.BalancedTaskBatcher(train, per_task_batch_size=size, seed=seed)
    metric_scales = {c: old.scales(data[c]['truth'][train[c]]) for c in old.TASKS}
    history, diagnostics, best, best_epoch, best_state, steps = [], [], float('inf'), 0, None, 0
    start = time.monotonic()
    for epoch in range(1, old.TRAINING['maximum_epochs'] + 1):
        old.set_training_mode(model)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if module.training != any(p.requires_grad for p in module.parameters(recurse=False)):
                    raise RuntimeError(f'BN mode mismatch: {name}')
        losses_epoch = {c: [] for c in old.TASKS}
        draws = []
        for step, indices in enumerate(batcher.epoch_batches(epoch)):
            draws.append({c: list(map(int, indices[c])) for c in old.TASKS})
            atoms, angles, tasks = old.make_batch(data, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(atoms, angles, tasks)
            if not torch.isfinite(prediction).all():
                raise RuntimeError('nonfinite training prediction')
            losses = {c: old.target_loss_recipe(atoms.y[tasks.eq(j), 0], atoms.y[tasks.eq(j), 1],
                      prediction[tasks.eq(j)], {}, recipe='raw_quantile') for j, c in enumerate(old.TASKS)}
            if epoch == 1 or epoch % old.TRAINING['gradient_diagnostic_interval'] == 0:
                diagnostics.append({'epoch': epoch, 'step': step, **old.gradient_geometry(losses, model.shared_gradient_parameters())})
            objective = torch.stack(list(losses.values())).mean()
            if not torch.isfinite(objective):
                raise RuntimeError('nonfinite joint training loss')
            objective.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('nonfinite gradient')
            optimizer.step(); steps += 1
            for c in old.TASKS:
                losses_epoch[c].append(float(losses[c].detach()))
        row = {'epoch': epoch, 'optimizer_steps': steps, 'batch_draws_sha256': old.digest(draws),
               **{f'train_loss_{c}': float(np.mean(v)) for c, v in losses_epoch.items()}}
        for c in old.TASKS:
            pred = old.predict(model, data, c, valid[c])
            if c == '4g':
                row['4g_all_outputs_finite'] = bool(np.isfinite(pred).all())
                row['source_validation_labels_available'] = False
                continue
            truth = data[c]['truth'][valid[c]]
            values = old.point_metrics(truth, pred, metric_scales[c])
            err = pred[:, [1, 4]] - truth
            values.update(Center_rmse=float(np.sqrt(np.square(err.mean(axis=1)).mean())),
                          Width_rmse=float(np.sqrt(np.square(err[:, 1] - err[:, 0]).mean())))
            row.update({f'{c}_{key}': value for key, value in values.items()})
            row[f'validation_loss_{c}'] = float(old.target_loss_recipe(
                torch.tensor(truth[:, 0], dtype=torch.float32), torch.tensor(truth[:, 1], dtype=torch.float32),
                torch.tensor(pred), {}, recipe='raw_quantile'))
        row['joint_validation_score'] = float(np.mean([row[f'{c}_combined_normalized_rmse'] for c in TARGETS]))
        if row['joint_validation_score'] < best:
            best, best_epoch, best_state = row['joint_validation_score'], epoch, deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 50 == 0:
            pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
            print(json.dumps({'run': str(run.relative_to(STUDY)), 'epoch': epoch, 'best_epoch': best_epoch,
                              'steps': steps, 'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
        if epoch - best_epoch >= old.TRAINING['patience']:
            break
    model.load_state_dict(best_state)
    if any(not torch.equal(model.state_dict()[name], initial[name]) for name in frozen_names):
        raise RuntimeError('frozen parameter changed')
    bn_deltas = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            key = f'{name}.num_batches_tracked'
            delta = int(model.state_dict()[key] - initial[key])
            expected = best_epoch * batcher.steps_per_epoch if any(p.requires_grad for p in module.parameters(recurse=False)) else 0
            if delta != expected:
                raise RuntimeError(f'BN step counter mismatch: {name}')
            if expected == 0 and any(not torch.equal(model.state_dict()[f'{name}.{b}'], initial[f'{name}.{b}']) for b in ['running_mean', 'running_var']):
                raise RuntimeError('frozen BN statistics changed')
            bn_deltas[name] = delta
    pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(run / 'gradients.csv', index=False)
    torch.save({'model_state_dict': model.state_dict(), 'arm': arm, 'best_epoch': best_epoch,
                'source_sha256': old.SOURCE_SHA, 'protocol_sha256': old.sha(STUDY / 'protocol.json')}, run / 'best.pt')
    fitted = dict(best_epoch=best_epoch, epochs_run=epoch, runtime_seconds=time.monotonic()-start,
                  per_task_batch_size=size, steps_per_epoch=batcher.steps_per_epoch, optimizer_steps=steps,
                  optimizer_steps_at_best=best_epoch*batcher.steps_per_epoch, training_seed=seed,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  stopped_by_patience=epoch < old.TRAINING['maximum_epochs'])
    old.write_json(run / 'state_audit.json', {'frozen_parameters_unchanged': True, 'bn_policy_valid': True,
                   'bn_deltas_at_best': bn_deltas, 'finite_predictions_and_gradients': True,
                   'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad],
                   'optimizer_groups': [{'name': g['name'], 'lr': g['lr'], 'weight_decay': g['weight_decay']} for g in optimizer.param_groups]})
    return model, fitted, metric_scales

if __name__=='__main__':
    main()
