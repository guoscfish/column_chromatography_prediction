"""Fixed-budget FULL Row refits and feature-only blind freeze; no scoring entry point."""
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
import run_phase2 as r

STUDY, ROOT, old, p = r.STUDY, r.ROOT, r.old, r.p
SEEDS, TARGETS = r.SEEDS, r.TARGETS
FINAL = STUDY / 'runtime/final_row_aligned'
PRED_COLUMNS = ['column', 'outer_seed', 'arm', 'role', 'sample_id', *old.QUANTILES]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def budget_from_results(results, seed, arm):
    part = results.loc[results.outer_seed.eq(seed) & results.arm.eq(arm) & results.column.isin(TARGETS)]
    epochs = []
    for fold in range(1, 6):
        rows = part.loc[part.inner_fold.eq(fold)]
        require(len(rows) == 2 and set(rows.column) == set(TARGETS), 'missing/duplicate joint epoch rows')
        values = rows.best_epoch.to_numpy(float)
        require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
                and len(set(values)) == 1 and 1 <= values[0] <= 500, 'invalid joint best_epoch')
        epochs.append(int(values[0]))
    require(len(part) == 10, 'unexpected fold observations')
    return epochs, int(np.floor(np.median(epochs) + .5))


def state_check(model, initial, steps):
    state = model.state_dict()
    require(all(torch.isfinite(value).all() for value in state.values()), 'nonfinite model state')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            require(torch.equal(state[name], initial[name]), f'frozen parameter changed: {name}')
    deltas = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            expected = steps if any(q.requires_grad for q in module.parameters(recurse=False)) else 0
            delta = int(state[name + '.num_batches_tracked'] - initial[name + '.num_batches_tracked'])
            require(delta == expected, f'BN counter mismatch: {name}')
            if expected == 0:
                for buf in ('running_mean', 'running_var'):
                    require(torch.equal(state[name + '.' + buf], initial[name + '.' + buf]), 'frozen BN changed')
            deltas[name] = delta
    return {'frozen_parameters_unchanged': True, 'bn_policy_valid': True, 'state_finite': True,
            'bn_deltas': deltas, 'model_state_sha256': p.tensor_digest(state)}


def audit_inputs():
    """Read-only audit of the fifty already-completed fits; never writes old evidence."""
    torch.set_num_threads(1)
    preserved = r.verify_phase1()
    report = read_json(STUDY/'PHASE2_REPORT_MANIFEST.json')
    require(report['total_fits'] == 50 and not report['outer_scoring_executed'], 'Phase 2 completion mismatch')
    for name, digest in report['files'].items():
        require(old.sha(STUDY/name) == digest, f'Phase 2 report evidence changed: {name}')
    require(report['phase2_protocol_sha256'] == old.sha(STUDY/'PHASE2_PROTOCOL.json'), 'Phase 2 protocol changed')
    require(report['phase2_provenance_sha256'] == old.sha(STUDY/'PHASE2_EXECUTION_PROVENANCE.json'), 'Phase 2 provenance changed')
    for provenance_name in ('EXECUTION_PROVENANCE.json', 'PHASE2_EXECUTION_PROVENANCE.json'):
        authority = read_json(STUDY/provenance_name)
        for rel, digest in authority['files'].items():
            require(old.sha(ROOT/rel) == digest, 'input implementation changed')
        require(authority['torch'] == torch.__version__ and authority['numpy'] == np.__version__
                and authority['python'] == sys.version, 'runtime changed since Row-CV')
    base, phase2 = read_json(STUDY/'protocol.json'), read_json(STUDY/'PHASE2_PROTOCOL.json')
    require(base['training'] == phase2['training'] == old.TRAINING and base['film'] == phase2['film']
            and base['inner_cv'] == phase2['inner_cv'], 'scientific protocol mismatch')
    schedule = p.row_schedule()
    source_split = pd.read_csv(old.QUALIFIED/'splits/row_seed_42.csv', usecols=['sample_id','split'])
    source_roles = {role: source_split.loc[source_split.split.eq(value),'sample_id'].astype(str).tolist()
                    for role, value in [('inner_train','train'),('inner_validation','validation')]}
    results = pd.read_csv(STUDY/'INNER_RESULTS.csv')
    paired = pd.read_csv(STUDY/'PAIRED_COMPARISONS.csv')
    expected = {(s,f,a,c) for s in SEEDS for f in range(1,6) for a in old.ARMS for c in TARGETS}
    target = results.loc[results.column.isin(TARGETS)]
    actual = list(target[['outer_seed','inner_fold','arm','column']].itertuples(index=False,name=None))
    require(len(actual) == 100 and set(actual) == expected and len(paired) == 50, 'incomplete Row-CV population')
    selections = pd.read_csv(STUDY/'EPOCH_SELECTION.csv')
    budgets, receipts, seed_audits = [], {}, {}
    for seed in SEEDS:
        seed_dir = r.PHASE2/f'seed_{seed}'
        audit = read_json(seed_dir/'COMPLETE_AUDIT.json')
        require(audit['all_passed'] and audit['models'] == 10, 'seed audit failed')
        for name, digest in audit['files'].items():
            require(old.sha(seed_dir/name) == digest, 'audited seed artifact changed')
        seed_audits[str(seed)] = old.sha(seed_dir/'COMPLETE_AUDIT.json')
        ids, cols = [], []
        for c in TARGETS:
            allowed = schedule.loc[schedule.outer_seed.eq(seed)&schedule.column.eq(c)&schedule.role.eq('gradient_train'),'sample_id'].astype(str).tolist()
            ids.extend(allowed); cols.extend([c]*len(allowed))
        folds = p.row_target_folds(ids, cols)
        for fold, (train, valid) in enumerate(folds, 1):
            histories = {}
            for arm in old.ARMS:
                run = STUDY/f'runtime/inner/seed_{seed}/fold_{fold}/{arm}'
                r.check_receipt(run, phase2=seed != p.PILOT_SEED)
                receipts[str(run.relative_to(STUDY))] = old.sha(run/'complete.json')
                identity = pd.read_csv(run/'identities.csv', dtype={'sample_id':str})
                for c in old.TASKS:
                    for role, idx in [('inner_train',train),('inner_validation',valid)]:
                        wanted = set(source_roles[role]) if c=='4g' else {ids[i] for i in idx if cols[i]==c}
                        part = identity.loc[identity.column.eq(c)&identity.role.eq(role),'sample_id']
                        require(part.is_unique and set(part)==wanted, 'inner identities disagree with deterministic folds')
                rows = pd.read_csv(run/'results.csv')
                agg = results.loc[results.outer_seed.eq(seed)&results.inner_fold.eq(fold)&results.arm.eq(arm)&results.column.isin(TARGETS)].set_index('column')
                best = int(rows.best_epoch.iloc[0]); history = pd.read_csv(run/'history.csv')
                require(best == int(history.loc[history.joint_validation_score.idxmin(),'epoch']), 'joint argmin mismatch')
                for c in TARGETS:
                    raw = rows.loc[rows.column.eq(c)].iloc[0]
                    for key in ('best_epoch','inner_train_ids_sha256','inner_validation_ids_sha256','combined_normalized_rmse'):
                        require(raw[key] == agg.loc[c,key] or (key=='combined_normalized_rmse' and np.isclose(raw[key],agg.loc[c,key],atol=1e-12)), 'aggregate/raw mismatch')
                    pair = paired.loc[paired.outer_seed.eq(seed)&paired.inner_fold.eq(fold)&paired.column.eq(c)]
                    require(len(pair)==1 and np.isclose(pair.iloc[0][arm+'_combined_normalized_rmse'],raw.combined_normalized_rmse), 'paired results mismatch')
                steps = int(rows.steps_per_epoch.iloc[0])
                require(len(history) == int(rows.epochs_run.iloc[0]), 'history length mismatch')
                np.testing.assert_array_equal(history.optimizer_steps, np.arange(1,len(history)+1)*steps)
                require(int(history.optimizer_steps.iloc[-1]) == int(rows.optimizer_steps.iloc[0]), 'actual steps mismatch')
                torch.manual_seed(seed+fold*1009)
                model = old.build_multitask_model(old.source_path(),old.ARMS[arm])
                initial = deepcopy(model.state_dict())
                checkpoint = torch.load(run/'best.pt', map_location='cpu',weights_only=False)
                require(checkpoint['best_epoch']==best and checkpoint['source_sha256']==old.SOURCE_SHA, 'inner checkpoint authority mismatch')
                model.load_state_dict(checkpoint['model_state_dict'],strict=True)
                state_check(model,initial,best*steps)
                histories[arm] = history
            common = min(len(histories['A1']),len(histories['A2']))
            np.testing.assert_array_equal(histories['A1'].batch_draws_sha256.iloc[:common],histories['A2'].batch_draws_sha256.iloc[:common])
        for arm in old.ARMS:
            five, epochs = budget_from_results(results,seed,arm)
            selected = selections.loc[selections.outer_seed.eq(seed)&selections.arm.eq(arm)]
            require(len(selected)==1 and selected.iloc[0].cv=='Row-CV' and int(selected.iloc[0].final_epoch_selected)==epochs, 'frozen median budget mismatch')
            budgets.append({'outer_seed':seed,'arm':arm,'five_joint_best_epochs':five,'fixed_epochs':epochs})
        print(f'INPUT AUDIT PASSED seed={seed}: 10 checkpoints, five folds and budgets',flush=True)
    return {'all_passed':True,'fits':50,'target_observations':100,'report_artifacts_verified':len(report['files']),
            'preservation':preserved,'inner_receipts':receipts,'seed_audits':seed_audits,'budgets':budgets,
            'outer_response_cells_parsed':0,'classification':'DEVELOPMENTAL__HISTORIC_OUTER_TEST_EXPOSED'}


def load_data(seed, *, parse_training_labels=True, truth_reader=None):
    """Only whitelisted inputs for outer roles; graph labels there are zero, truth is NaN."""
    require(seed in SEEDS, 'unauthorized seed')
    torch.set_num_threads(1)
    p.specification()
    schedule = p.row_schedule()
    source_cache = torch.load(old.SOURCE_GRAPH_CACHE,map_location='cpu',weights_only=False)
    checkpoint = old.source_path()
    scaler = torch.load(checkpoint,map_location='cpu',weights_only=False)['preprocessing']['scaler']
    reader = truth_reader or old._read_authorized_truth
    data, access = {}, []
    for c in old.TASKS:
        if c=='4g':
            split = pd.read_csv(old.QUALIFIED/'splits/row_seed_42.csv',usecols=['sample_id','split'])
            roles = {role:split.loc[split.split.eq(value),'sample_id'].astype(str).tolist()
                     for role,value in [('gradient_train','train'),('source_validation','validation')]}
            feature_path = canonical_path = old.SOURCE_DATA
            feature_ids = sum(roles.values(),[])
        else:
            split = schedule.loc[schedule.column.eq(c)&schedule.outer_seed.eq(seed)]
            roles = {role:split.loc[split.role.eq(role),'sample_id'].astype(str).tolist()
                     for role in ['gradient_train','validation','test']}
            feature_path,canonical_path = old.PARENT/f'filtered_features_{c}.csv',old.PARENT/f'filtered_canonical_{c}.csv'
            feature_ids = sum(roles.values(),[])
        require(len(feature_ids)==len(set(feature_ids)), 'role identity overlap')
        frame = pd.read_csv(feature_path,usecols=old.FEATURES)
        frame.sample_id = frame.sample_id.astype(str)
        require(frame.sample_id.is_unique and set(feature_ids)<=set(frame.sample_id), 'feature identity mismatch')
        if c!='4g':
            require(set(frame.sample_id)==set(feature_ids), 'filtered population mismatch')
        frame = frame.loc[frame.sample_id.isin(feature_ids)].reset_index(drop=True)
        lookup = {sid:i for i,sid in enumerate(frame.sample_id)}
        positions = {role:[lookup[sid] for sid in ids] for role,ids in roles.items()}
        frame['V1_ml'],frame['V2_ml'] = 0.,0.
        truth = np.full((len(frame),2),np.nan)
        if parse_training_labels:
            ids = roles['gradient_train']
            labels = reader(canonical_path,ids)
            require(labels.shape==(len(ids),2) and np.isfinite(labels).all(), 'invalid permitted training labels')
            frame.loc[positions['gradient_train'],['V1_ml','V2_ml']] = labels
            truth[positions['gradient_train']] = labels
            access.append({'outer_seed':seed,'column':c,'path':str(canonical_path.relative_to(ROOT)),
                           'label_columns':['V1_ml','V2_ml'],'purpose':'source_train' if c=='4g' else 'gradient_train',
                           'authorized_ids':ids,'count':len(ids),'ids_sha256':old.digest(sorted(ids)),
                           'outer_response_cells_parsed':0,'source_validation_response_cells_parsed':0})
        cache = dict(source_cache)
        if c!='4g':
            cache.update(torch.load(ROOT/f'studies/transfer/cross_column/data_audit/graph_cache_{c}_only.pt',map_location='cpu',weights_only=False))
        atoms,angles = p.build_model_data(frame,cache,None,scaler)
        old.cache_frozen_prefix(p.load_predictor_checkpoint(checkpoint),atoms,angles)
        data[c] = dict(column=c,frame=frame,roles=roles,positions=positions,truth=truth,atoms=atoms,angles=angles)
        for role,idx in positions.items():
            if role!='gradient_train' or not parse_training_labels:
                require(np.isnan(truth[idx]).all() and all(torch.count_nonzero(atoms[i].y)==0 for i in idx), 'heldout graph label not scrubbed')
    return data,access


def authorize_full(data, train):
    require(set(train)==set(old.TASKS), 'all three training tasks required')
    training_ids, heldout_ids = set(), set()
    for c in old.TASKS:
        idx = train[c]
        require(idx and len(idx)==len(set(idx)) and idx==data[c]['positions']['gradient_train'], 'FULL training must exactly equal permitted gradient_train')
        require(np.isfinite(data[c]['truth'][idx]).all(), 'training labels unavailable')
        training_ids.update(data[c]['frame'].iloc[idx].sample_id)
        for role,other in data[c]['positions'].items():
            if role!='gradient_train':
                heldout_ids.update(data[c]['frame'].iloc[other].sample_id)
    require(not training_ids & heldout_ids, 'same sample_id crosses train and heldout, including across columns')


def fit_fixed(data, train, arm, seed, run, epochs, protocol_sha256):
    """Original fixed_epochs update recipe, plus read-only finite/identity/state diagnostics."""
    authorize_full(data,train)
    require(arm in old.ARMS and isinstance(epochs,int) and 1<=epochs<=500, 'invalid fixed refit budget')
    run.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(int(seed))
    model = old.build_multitask_model(old.source_path(),old.ARMS[arm])
    initial = deepcopy(model.state_dict())
    optimizer = model.optimizer(pretrained_lr=old.TRAINING['pretrained_lr'],new_lr=old.TRAINING['new_lr'],weight_decay=old.TRAINING['weight_decay'])
    size = min(old.TRAINING['per_task_batch_cap'],max(len(train[c]) for c in TARGETS))
    batcher = old.BalancedTaskBatcher(train,per_task_batch_size=size,seed=seed)
    history,diagnostics,steps = [],[],0
    start = time.monotonic()
    for epoch in range(1,epochs+1):
        old.set_training_mode(model)
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                require(module.training==any(q.requires_grad for q in module.parameters(recurse=False)), f'BN mode mismatch: {name}')
        losses_epoch,draws = {c:[] for c in old.TASKS},[]
        for step,indices in enumerate(batcher.epoch_batches(epoch)):
            draws.append({c:data[c]['frame'].iloc[indices[c]].sample_id.tolist() for c in old.TASKS})
            atoms,angles,tasks = old.make_batch(data,indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(atoms,angles,tasks)
            require(torch.isfinite(prediction).all(), 'nonfinite training prediction')
            losses = {c:old.target_loss_recipe(atoms.y[tasks.eq(j),0],atoms.y[tasks.eq(j),1],prediction[tasks.eq(j)],{},recipe='raw_quantile')
                      for j,c in enumerate(old.TASKS)}
            if epoch==1 or epoch % old.TRAINING['gradient_diagnostic_interval']==0:
                diagnostics.append({'epoch':epoch,'step':step,**old.gradient_geometry(losses,model.shared_gradient_parameters())})
            objective = torch.stack(list(losses.values())).mean()
            require(torch.isfinite(objective), 'nonfinite joint training loss')
            objective.backward()
            require(all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters()), 'nonfinite gradient')
            optimizer.step(); steps += 1
            for c in old.TASKS:
                losses_epoch[c].append(float(losses[c].detach()))
        history.append({'epoch':epoch,'optimizer_steps':steps,'batch_identity_draws_sha256':old.digest(draws),
                        **{f'train_loss_{c}':float(np.mean(v)) for c,v in losses_epoch.items()}})
        if epoch==1 or epoch % 50==0 or epoch==epochs:
            pd.DataFrame(history).to_csv(run/'history.csv',index=False)
            print(json.dumps({'seed':int(seed),'arm':arm,'epoch':epoch,'fixed_epochs':epochs,'optimizer_steps':steps,
                              'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
    require(steps==epochs*batcher.steps_per_epoch, 'actual optimizer budget mismatch')
    state = state_check(model,initial,steps)
    state.update(finite_training_predictions_and_gradients=True,
                 trainable_names=[name for name,q in model.named_parameters() if q.requires_grad],
                 optimizer_groups=[{'name':g['name'],'lr':g['lr'],'weight_decay':g['weight_decay']} for g in optimizer.param_groups])
    pd.DataFrame(history).to_csv(run/'history.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(run/'gradients.csv',index=False)
    old.write_json(run/'state_audit.json',state)
    torch.save({'model_state_dict':model.state_dict(),'arm':arm,'outer_seed':int(seed),'fixed_epochs':epochs,
                'optimizer_steps':steps,'source_sha256':old.SOURCE_SHA,'phase3a_protocol_sha256':protocol_sha256,
                'model_state_sha256':state['model_state_sha256'],'output_columns':old.QUANTILES},run/'final.pt')
    return model,dict(fixed_epochs=epochs,epochs_run=epoch,optimizer_steps=steps,steps_per_epoch=batcher.steps_per_epoch,
                      per_task_batch_size=size,training_seed=int(seed),runtime_seconds=time.monotonic()-start,
                      trainable_parameters=sum(q.numel() for q in model.parameters() if q.requires_grad),
                      total_parameters=sum(q.numel() for q in model.parameters()),model_state_sha256=state['model_state_sha256'])


def training_records(data, seed, arm):
    rows,summary = [],{}
    for c in old.TASKS:
        ids = data[c]['roles']['gradient_train']
        truth = data[c]['truth'][data[c]['positions']['gradient_train']]
        summary[c] = {'count':len(ids),'sorted_ids_sha256':old.digest(sorted(ids)),'ordered_ids_sha256':old.digest(ids),
                      'label_values_sha256':old.digest(truth.tolist()),'metric_scales':old.scales(truth),
                      'tail_thresholds_80pct':np.quantile(truth,.8,axis=0).tolist()}
        rows.extend({'outer_seed':seed,'arm':arm,'column':c,'role':'gradient_train','sample_id':sid} for sid in ids)
    return pd.DataFrame(rows),summary


def check_prediction(frame, data, seed, arm, c, role):
    require(frame.columns.tolist()==PRED_COLUMNS, 'prediction schema/order mismatch')
    require(frame.sample_id.astype(str).tolist()==data[c]['roles'][role], 'prediction identity/order mismatch')
    require(frame.sample_id.is_unique and len(frame)>0, 'invalid prediction count/duplicates')
    for name,value in [('column',c),('outer_seed',seed),('arm',arm),('role',role)]:
        require(frame[name].eq(value).all(), 'prediction metadata mismatch')
    values = frame[old.QUANTILES].to_numpy(float)
    require(values.shape==(len(frame),6) and np.isfinite(values).all(), 'nonfinite/invalid predictions')
    # The architecture's existing physical clamp is retained; do not recalibrate it.
    require((values>=0).all() and (values<=1e8).all(), 'existing nonnegative/bounded output clamp failed')
    return {'rows':len(frame),'ordered_ids_sha256':old.digest(frame.sample_id.astype(str).tolist()),
            'sorted_ids_sha256':old.digest(sorted(frame.sample_id.astype(str))),'columns':PRED_COLUMNS,
            'identity_order_verified':True,'all_six_outputs_finite':True,'existing_output_clamp_verified':True,
            'V2_below_V1_by_quantile':(values[:,3:]<values[:,:3]).sum(axis=0).tolist(),
            'V1_quantile_crossing_rows':int(np.any(np.diff(values[:,:3],axis=1)<0,axis=1).sum()),
            'V2_quantile_crossing_rows':int(np.any(np.diff(values[:,3:],axis=1)<0,axis=1).sum())}


def check_complete(run):
    receipt = read_json(run/'complete.json')
    require(receipt['phase3a_protocol_sha256']==old.sha(STUDY/'PHASE3A_PROTOCOL.json'), 'refit protocol changed')
    require(receipt['provenance_sha256']==old.sha(STUDY/'PHASE3A_EXECUTION_PROVENANCE.json'), 'refit code changed')
    for name,digest in receipt['files'].items():
        require(old.sha(run/name)==digest, f'completed refit artifact changed: {run/name}')
    return receipt


def run_seed(seed):
    torch.set_num_threads(1)
    protocol = read_json(STUDY/'PHASE3A_PROTOCOL.json')
    selected = {item['arm']:item for item in protocol['budgets'] if item['outer_seed']==seed}
    data,access = load_data(seed)
    train = {c:data[c]['positions']['gradient_train'] for c in old.TASKS}
    authorize_full(data,train)
    init = p.initialization_audit(seed)
    for arm in old.ARMS:
        run = FINAL/f'seed_{seed}/{arm}'
        if (run/'complete.json').exists():
            check_complete(run);continue
        run.mkdir(parents=True,exist_ok=True)
        old.write_json(run/'initialization.json',init)
        old.write_json(run/'label_access.json',access)
        identities,identity_summary = training_records(data,seed,arm)
        identities.to_csv(run/'training_identities.csv',index=False)
        model,fitted = fit_fixed(data,train,arm,seed,run,selected[arm]['fixed_epochs'],old.sha(STUDY/'PHASE3A_PROTOCOL.json'))
        checks = {}
        for c in TARGETS:
            for role in ('validation','test'):
                idx = data[c]['positions'][role]
                prediction = old.predict(model,data,c,idx)
                frame = pd.DataFrame([dict(column=c,outer_seed=seed,arm=arm,role=role,sample_id=sid,**dict(zip(old.QUANTILES,map(float,values))))
                                      for sid,values in zip(data[c]['roles'][role],prediction)],columns=PRED_COLUMNS)
                name = f'{c}_{role}_predictions_blind.csv'
                checks[name] = check_prediction(frame,data,seed,arm,c,role)
                frame.to_csv(run/name,index=False)
        old.write_json(run/'fit.json',{**fitted,**selected[arm],'training_identity':identity_summary,
                       'prediction_checks':checks,'outer_response_cells_parsed':0,'epoch_selection_used_outer_labels':False})
        files = {f.name:old.sha(f) for f in run.iterdir() if f.is_file() and f.name!='complete.json'}
        old.write_json(run/'complete.json',{'phase3a_protocol_sha256':old.sha(STUDY/'PHASE3A_PROTOCOL.json'),
                       'provenance_sha256':old.sha(STUDY/'PHASE3A_EXECUTION_PROVENANCE.json'),'files':files},immutable=True)
        print(f'FINAL REFIT COMPLETE seed={seed} arm={arm} epochs/steps={fitted["fixed_epochs"]}/{fitted["optimizer_steps"]}',flush=True)
    left,right = [pd.read_csv(FINAL/f'seed_{seed}/{arm}/history.csv') for arm in old.ARMS]
    common = min(len(left),len(right))
    np.testing.assert_array_equal(left.batch_identity_draws_sha256.iloc[:common],right.batch_identity_draws_sha256.iloc[:common])
    require(read_json(FINAL/f'seed_{seed}/A1/initialization.json')==read_json(FINAL/f'seed_{seed}/A2/initialization.json'), 'final initialization pairing mismatch')
    return seed


def prepare():
    audit = audit_inputs()  # Inconsistency raises before any real fit.
    old.write_json(STUDY/'PHASE3A_INPUT_AUDIT.json',audit,immutable=True)
    base = read_json(STUDY/'protocol.json')
    payload = {'phase':'PHASE3A_FULL_REFIT_BLIND_ONLY','classification':audit['classification'],'seeds':list(SEEDS),'arms':old.ARMS,
               'base_row_cv_protocol_sha256':old.sha(STUDY/'protocol.json'),'phase2_protocol_sha256':old.sha(STUDY/'PHASE2_PROTOCOL.json'),
               'phase2_report_manifest_sha256':old.sha(STUDY/'PHASE2_REPORT_MANIFEST.json'),
               'input_audit_sha256':old.sha(STUDY/'PHASE3A_INPUT_AUDIT.json'),'selection_results_sha256':old.sha(STUDY/'INNER_RESULTS.csv'),
               'evaluation_plan_sha256':old.sha(STUDY/'EVALUATION_PLAN.md'),'original_split_sha256':old.SPLIT_SHA,
               'source_checkpoint':base['source_checkpoint'],'source_sha256':old.SOURCE_SHA,'training':old.TRAINING,'film':base['film'],
               'bn_policy':base['bn_policy'],'feature_usecols':old.FEATURES,'output_columns':old.QUANTILES,
               'budgets':audit['budgets'],'training_seed':'outer_seed; reinitialize each arm from same source',
               'training_labels':'source_train and per-seed target gradient_train only','outer_labels_authorized':False,
               'fixed_budget':'round-half-up median of five valid joint best epochs; no validation selector or patience in refit',
               'effect':'complete Row-CV selection plus FULL refit pipeline; not an equal-step FiLM module effect',
               'workers':5,'threads_per_worker':1,'observed_before_execution':['historical Row outer results','Phase 1/2 internal results']}
    old.write_json(STUDY/'PHASE3A_PROTOCOL.json',payload,immutable=True)
    files = [Path(__file__),STUDY/'test_phase3a.py',STUDY/'EVALUATION_PLAN.md',STUDY/'run_phase2.py',STUDY/'run_phase1.py']
    files += [ROOT/rel for rel in read_json(STUDY/'EXECUTION_PROVENANCE.json')['files']]
    provenance = {'files':{str(f.relative_to(ROOT)):old.sha(f) for f in files},'python':sys.version,'torch':torch.__version__,
                  'numpy':np.__version__,'cpu_threads':1,'phase3a_protocol_sha256':old.sha(STUDY/'PHASE3A_PROTOCOL.json')}
    old.write_json(STUDY/'PHASE3A_EXECUTION_PROVENANCE.json',provenance,immutable=True)
    return payload


def freeze_predictions():
    require(not (STUDY/'PREDICTION_FREEZE_MANIFEST.json').exists(), 'global freeze already exists; use verify only')
    rows,histories,replays,contexts,files = [],[],[],[],{}
    for seed in SEEDS:
        # Independent reload using only inputs; all graph labels zero, all truth NaN.
        zero,access = load_data(seed,parse_training_labels=False)
        require(not access, 'feature-only replay parsed responses')
        for arm in old.ARMS:
            run = FINAL/f'seed_{seed}/{arm}'
            receipt = check_complete(run)
            fit = read_json(run/'fit.json')
            torch.manual_seed(seed)
            model = old.build_multitask_model(old.source_path(),old.ARMS[arm])
            initial = deepcopy(model.state_dict())
            checkpoint = torch.load(run/'final.pt',map_location='cpu',weights_only=False)
            require(checkpoint['arm']==arm and checkpoint['outer_seed']==seed and checkpoint['source_sha256']==old.SOURCE_SHA
                    and checkpoint['phase3a_protocol_sha256']==old.sha(STUDY/'PHASE3A_PROTOCOL.json'), 'final checkpoint provenance mismatch')
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            state = state_check(model,initial,fit['optimizer_steps'])
            require(state['model_state_sha256']==fit['model_state_sha256']==checkpoint['model_state_sha256'], 'state tensor digest mismatch')
            history = pd.read_csv(run/'history.csv')
            require(len(history)==fit['fixed_epochs']==fit['epochs_run']==checkpoint['fixed_epochs'], 'final epoch mismatch')
            require(int(history.optimizer_steps.iloc[-1])==fit['optimizer_steps']==checkpoint['optimizer_steps'], 'final optimizer step mismatch')
            np.testing.assert_array_equal(history.optimizer_steps,np.arange(1,len(history)+1)*fit['steps_per_epoch'])
            checks = {}
            for c in TARGETS:
                for role in ('validation','test'):
                    name = f'{c}_{role}_predictions_blind.csv'
                    saved = pd.read_csv(run/name,dtype={'sample_id':str})
                    checks[name] = check_prediction(saved,zero,seed,arm,c,role)
                    require(checks[name]==fit['prediction_checks'][name], 'reread prediction checks disagree')
                    replay = old.predict(model,zero,c,zero[c]['positions'][role])
                    diff = float(np.max(np.abs(replay-saved[old.QUANTILES].to_numpy())))
                    require(np.allclose(replay,saved[old.QUANTILES].to_numpy(),rtol=1e-6,atol=1e-5), 'zero-label checkpoint replay mismatch')
                    replays.append({'outer_seed':seed,'arm':arm,'column':c,'role':role,'rows':len(saved),'max_abs_difference_ml':diff,'passed':True})
            identity = pd.read_csv(run/'training_identities.csv',dtype={'sample_id':str})
            for c in old.TASKS:
                ids = identity.loc[identity.column.eq(c),'sample_id'].tolist()
                require(ids==zero[c]['roles']['gradient_train'] and old.digest(sorted(ids))==fit['training_identity'][c]['sorted_ids_sha256'], 'final training identity mismatch')
            rows.append({'outer_seed':seed,'arm':arm,'five_joint_best_epochs':','.join(map(str,fit['five_joint_best_epochs'])),
                         **{k:fit[k] for k in ['fixed_epochs','epochs_run','optimizer_steps','steps_per_epoch','per_task_batch_size','training_seed','model_state_sha256','trainable_parameters','total_parameters']},
                         'checkpoint':str((run/'final.pt').relative_to(STUDY)),'checkpoint_sha256':old.sha(run/'final.pt'),
                         **{f'{c}_{key}':fit['training_identity'][c][key] for c in old.TASKS for key in ['count','sorted_ids_sha256']},
                         'label_access_sha256':old.sha(run/'label_access.json'),'prediction_files':4,'outer_response_cells_parsed':0,
                         'checkpoint_replay_passed':True,'paired_initialization_and_batches_verified':True})
            histories.append(history.assign(outer_seed=seed,arm=arm))
            contexts.append({**fit,'checkpoint':str((run/'final.pt').relative_to(STUDY)),'checkpoint_sha256':old.sha(run/'final.pt'),
                             'receipt_sha256':old.sha(run/'complete.json'),'feature_only_replay_passed':True})
            for name,digest in receipt['files'].items():
                files[str((run/name).relative_to(STUDY))] = digest
            files[str((run/'complete.json').relative_to(STUDY))] = old.sha(run/'complete.json')
        ha,hb = [pd.read_csv(FINAL/f'seed_{seed}/{arm}/history.csv') for arm in old.ARMS]
        common = min(len(ha),len(hb))
        np.testing.assert_array_equal(ha.batch_identity_draws_sha256.iloc[:common],hb.batch_identity_draws_sha256.iloc[:common])
        print(f'FINAL REPLAY PASSED seed={seed}: 2 complete states, 8 blind files',flush=True)
    pd.DataFrame(rows).to_csv(STUDY/'FINAL_REFIT_MANIFEST.csv',index=False)
    pd.concat(histories,ignore_index=True).to_csv(STUDY/'FINAL_TRAINING_HISTORY.csv',index=False)
    pd.DataFrame(replays).to_csv(STUDY/'FINAL_CHECKPOINT_REPLAY.csv',index=False)
    for name in ['FINAL_REFIT_MANIFEST.csv','FINAL_TRAINING_HISTORY.csv','FINAL_CHECKPOINT_REPLAY.csv','EVALUATION_PLAN.md',
                 'PHASE3A_PROTOCOL.json','PHASE3A_INPUT_AUDIT.json','PHASE3A_EXECUTION_PROVENANCE.json']:
        files[name] = old.sha(STUDY/name)
    require(len(contexts)==10 and len(replays)==40, 'incomplete final schedule')
    r.verify_phase1()
    for name,digest in read_json(STUDY/'PHASE2_REPORT_MANIFEST.json')['files'].items():
        require(old.sha(STUDY/name)==digest, 'Phase 2 changed during refit')
    payload = {'status':'FROZEN_BEFORE_PHASE3B_OUTER_RESPONSE_ACCESS','classification':'DEVELOPMENTAL__HISTORIC_OUTER_TEST_EXPOSED',
               'phase3a_protocol_sha256':old.sha(STUDY/'PHASE3A_PROTOCOL.json'),'row_cv_protocol_sha256':old.sha(STUDY/'protocol.json'),
               'phase2_protocol_sha256':old.sha(STUDY/'PHASE2_PROTOCOL.json'),'original_split_sha256':old.SPLIT_SHA,
               'source_checkpoint_sha256':old.SOURCE_SHA,'evaluation_plan_sha256':old.sha(STUDY/'EVALUATION_PLAN.md'),
               'final_refits':10,'prediction_files':40,'prediction_rows':sum(item['rows'] for item in replays),
               'output_columns':old.QUANTILES,'outer_response_cells_parsed_in_phase3a':0,'outer_scoring_executed':False,
               'all_checkpoints_replayed_with_zero_labels':True,'replay_max_abs_difference_ml':max(item['max_abs_difference_ml'] for item in replays),
               'contexts':contexts,'files':files,'freeze_policy':'immutable JSON write; completed receipts never overwritten; all frozen files read-only; verify before any later evaluation'}
    old.write_json(STUDY/'PREDICTION_FREEZE_MANIFEST.json',payload,immutable=True)
    for name in files:
        (STUDY/name).chmod(0o444)
    (STUDY/'PREDICTION_FREEZE_MANIFEST.json').chmod(0o444)
    return verify_freeze()


def verify_freeze():
    freeze = read_json(STUDY/'PREDICTION_FREEZE_MANIFEST.json')
    require(freeze['status']=='FROZEN_BEFORE_PHASE3B_OUTER_RESPONSE_ACCESS' and freeze['final_refits']==10
            and freeze['prediction_files']==40 and not freeze['outer_scoring_executed'], 'invalid global freeze')
    for name,digest in freeze['files'].items():
        require(old.sha(STUDY/name)==digest, f'frozen file changed: {name}')
    provenance = read_json(STUDY/'PHASE3A_EXECUTION_PROVENANCE.json')
    for name,digest in provenance['files'].items():
        require(old.sha(ROOT/name)==digest, f'frozen implementation changed: {name}')
    require(old.sha(old.PARENT/'split_manifest.csv')==freeze['original_split_sha256']
            and old.sha(old.source_path())==freeze['source_checkpoint_sha256'], 'frozen split/source changed')
    return {k:freeze[k] for k in ['status','final_refits','prediction_files','prediction_rows','replay_max_abs_difference_ml','outer_response_cells_parsed_in_phase3a']}


def main():
    if (STUDY/'PREDICTION_FREEZE_MANIFEST.json').exists():
        print(json.dumps(verify_freeze(),indent=2));return
    prepare()
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_seed,seed) for seed in SEEDS]
        for future in as_completed(futures):
            print(f'PAIRED FULL REFITS VERIFIED seed={future.result()}',flush=True)
    print(json.dumps(freeze_predictions(),indent=2))


if __name__=='__main__':
    main()
