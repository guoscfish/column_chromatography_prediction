"""Only the frozen single-seed Row-CV pilot; no refit or outer scoring entry point."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.studies import run_column_conditioned_multitask as old
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models import load_predictor_checkpoint

STUDY = Path(__file__).resolve().parent
PILOT_SEED = 769539383
FOLD_SEED = 42
TARGETS = old.TASKS[1:]


def save_json(name, payload, immutable=False):
    old.write_json(STUDY / name, payload, immutable=immutable)


def row_target_folds(sample_ids, columns, *, n_splits=5, seed=FOLD_SEED):
    """Independent deterministic per-column KFold; shared IDs must be resolved first."""
    ids, cols = np.asarray(sample_ids, str), np.asarray(columns, str)
    if ids.ndim != 1 or cols.ndim != 1 or not len(ids) or len(ids) != len(cols):
        raise ValueError('sample_ids and columns must be nonempty aligned vectors')
    if set(cols) != set(TARGETS) or len(set(ids)) != len(ids):
        raise ValueError('both target columns and globally unique sample_id are required')
    if int(n_splits) < 2:
        raise ValueError('at least two folds are required')
    per_column = {}
    for c in TARGETS:
        positions = np.flatnonzero(cols == c)
        positions = positions[np.argsort(ids[positions], kind='stable')]
        if len(positions) < n_splits:
            raise ValueError('each target requires at least n_splits rows')
        per_column[c] = [(positions[t], positions[v]) for t, v in
                         KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(positions)]
    return [(np.concatenate([per_column[c][f][0] for c in TARGETS]),
             np.concatenate([per_column[c][f][1] for c in TARGETS])) for f in range(n_splits)]


def row_schedule():
    schedule = old.verify_schedule()
    schedule = schedule.loc[schedule.protocol.eq('row')].copy()
    for (_, _), frame in schedule.groupby(['column', 'outer_seed']):
        if not frame.sample_id.is_unique or set(frame.role) != {'gradient_train', 'validation', 'test'}:
            raise RuntimeError('invalid frozen outer identity partition')
    return schedule


def input_hash(atom, angle, *, task=None):
    h = hashlib.sha256()
    # These are every input consumed by the original graph forward; exclude y and position.
    for value in (atom.x, atom.edge_index, atom.edge_attr, angle.edge_index, angle.edge_attr):
        value = value.detach().cpu().contiguous()
        h.update(str((str(value.dtype), tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    if task is not None:
        h.update(str(task).encode())
    return h.hexdigest()


def authorized_data(column, specification):
    if column == '4g':
        split = pd.read_csv(old.QUALIFIED / 'splits/row_seed_42.csv')
        roles = {r: split.loc[split.split.eq(s), 'sample_id'].astype(str).tolist()
                 for r, s in [('gradient_train', 'train'), ('source_validation', 'validation')]}
        feature_path = canonical_path = old.SOURCE_DATA
        allowed = roles['gradient_train'] + roles['source_validation']
    else:
        schedule = row_schedule()
        split = schedule.loc[schedule.column.eq(column) & schedule.outer_seed.eq(PILOT_SEED)]
        roles = {r: split.loc[split.role.eq(r), 'sample_id'].astype(str).tolist()
                 for r in ('gradient_train', 'validation', 'test')}
        feature_path = old.PARENT / f'filtered_features_{column}.csv'
        canonical_path = old.PARENT / f'filtered_canonical_{column}.csv'
        allowed = roles['gradient_train']
    # Never parse time columns or endpoints as features. Exclude held-out graphs entirely.
    frame = pd.read_csv(feature_path, usecols=old.FEATURES)
    frame.sample_id = frame.sample_id.astype(str)
    if not frame.sample_id.is_unique or not set(allowed) <= set(frame.sample_id):
        raise RuntimeError('missing/duplicate feature identity')
    if column != '4g' and set(frame.sample_id) != set().union(*map(set, roles.values())):
        raise RuntimeError('filtered population and outer authority differ')
    frame = frame.loc[frame.sample_id.isin(allowed)].reset_index(drop=True)
    lookup = {sid: i for i, sid in enumerate(frame.sample_id)}
    positions = {r: [lookup[sid] for sid in ids] for r, ids in roles.items() if set(ids) <= set(allowed)}
    truth = old._read_authorized_truth(canonical_path, allowed)
    if not np.isfinite(truth).all():
        raise RuntimeError('nonfinite authorized training truth')
    frame['V1_ml'], frame['V2_ml'] = 0., 0.
    frame.loc[[lookup[sid] for sid in allowed], ['V1_ml', 'V2_ml']] = truth
    cache = dict(torch.load(old.SOURCE_GRAPH_CACHE, map_location='cpu', weights_only=False))
    if column != '4g':
        cache.update(torch.load(ROOT / f'studies/transfer/cross_column/data_audit/graph_cache_{column}_only.pt',
                                map_location='cpu', weights_only=False))
    checkpoint = old.source_path()
    preprocessing = torch.load(checkpoint, map_location='cpu', weights_only=False)['preprocessing']
    atoms, angles = build_model_data(frame, cache, None, preprocessing['scaler'])
    keys = [input_hash(a, b) for a, b in zip(atoms, angles)]
    old.cache_frozen_prefix(load_predictor_checkpoint(checkpoint), atoms, angles)
    truth = frame[['V1_ml', 'V2_ml']].to_numpy(float)
    training_truth = truth[positions['gradient_train']]
    return dict(column=column, frame=frame, positions=positions, roles=roles, truth=truth,
                atoms=atoms, angles=angles, input_keys=keys, scales=old.scales(training_truth),
                tail=np.quantile(training_truth, .8, axis=0),
                regimes=np.quantile(training_truth.mean(axis=1), [1/3, 2/3]))


def fold_schedules(data):
    identities = [(c, i) for c in TARGETS for i in data[c]['positions']['gradient_train']]
    folds = row_target_folds([str(data[c]['frame'].iloc[i].sample_id) for c, i in identities],
                             [c for c, _ in identities])
    result = []
    for t, v in folds:
        train = {'4g': data['4g']['positions']['gradient_train']}
        valid = {'4g': data['4g']['positions']['source_validation']}
        for c in TARGETS:
            train[c] = [identities[i][1] for i in t if identities[i][0] == c]
            valid[c] = [identities[i][1] for i in v if identities[i][0] == c]
        authorize_fit(data, train, valid)
        result.append((train, valid))
    return result


def authorize_fit(data, train, valid):
    if set(train) != set(old.TASKS) or set(valid) != set(old.TASKS):
        raise RuntimeError('all three tasks required')
    train_ids, valid_ids = set(), set()
    for c in old.TASKS:
        allowed = set(data[c]['positions']['gradient_train'])
        va = set(data[c]['positions']['source_validation']) if c == '4g' else allowed
        if not train[c] or not valid[c] or len(train[c]) != len(set(train[c])) or len(valid[c]) != len(set(valid[c])):
            raise RuntimeError('empty or duplicate fold identities')
        if not set(train[c]) <= allowed or not set(valid[c]) <= va or set(train[c]) & set(valid[c]):
            raise RuntimeError('outer heldout/unauthorized identity in fit or selection')
        if c in TARGETS and set(train[c]) | set(valid[c]) != allowed:
            raise RuntimeError('incomplete inner partition')
        train_ids.update(data[c]['frame'].iloc[train[c]].sample_id)
        valid_ids.update(data[c]['frame'].iloc[valid[c]].sample_id)
    if train_ids & valid_ids:
        raise RuntimeError('same sample_id crosses train/validation, including across columns')


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialization_audit(seed):
    models = {}
    for arm in old.ARMS:
        torch.manual_seed(seed)
        models[arm] = old.build_multitask_model(old.source_path(), old.ARMS[arm])
    first, second = [m.state_dict() for m in models.values()]
    if not all(torch.equal(value, second[key]) for key, value in first.items()):
        raise RuntimeError('A1/A2 shared initial tensors differ')
    if not all(old.head_copies_are_exact(m) for m in models.values()) or not old.film_is_zero_initialized(models['A2']):
        raise RuntimeError('source heads or zero FiLM initialization invalid')
    return {'shared_initial_state_sha256': tensor_digest(first), 'heads_exact': True, 'film_zero': True,
            'common_tensors': len(first)}


def specification():
    payload = json.loads((old.STUDY / 'protocol.json').read_text())
    historical_code = json.loads((old.STUDY / 'EXECUTION_PROVENANCE.json').read_text())['files']
    for rel, expected in historical_code.items():
        if old.sha(ROOT / rel) != expected:
            raise RuntimeError(f'historical implementation changed: {rel}')
    checks = {old.source_path(): payload['source_sha256'],
              old.PARENT / 'split_manifest.csv': payload['split_manifest_sha256'],
              old.SOURCE_DATA: payload['source_data_sha256'],
              old.SOURCE_GRAPH_CACHE: payload['source_graph_cache_sha256'],
              old.QUALIFIED / 'splits/row_seed_42.csv': payload['source_split_sha256']}
    for c in TARGETS:
        checks[old.PARENT / f'filtered_features_{c}.csv'] = payload['features_sha256'][c]
        checks[old.PARENT / f'filtered_canonical_{c}.csv'] = payload['population_sha256'][c]
        checks[ROOT / f'studies/transfer/cross_column/data_audit/graph_cache_{c}_only.pt'] = payload['target_graph_cache_sha256'][c]
    for path, expected in checks.items():
        if old.sha(path) != expected:
            raise RuntimeError(f'frozen input digest changed: {path}')
    if payload['training'] != old.TRAINING:
        raise RuntimeError('historical training settings changed')
    return payload


def prepare():
    payload = specification()
    watched = {str(p.relative_to(ROOT)): old.sha(p) for p in old.STUDY.rglob('*') if p.is_file()}
    save_json('HISTORICAL_HASHES_BEFORE.json', watched, immutable=True)
    protocol = {**payload, 'study': 'COLUMN_CONDITIONED_ROW_ALIGNED_V1', 'primary_protocol': 'row',
                'secondary_protocol': None, 'pilot_seed': PILOT_SEED, 'execution_phase': 'PHASE_1_ONLY',
                'historical_protocol_sha256': old.sha(old.STUDY / 'protocol.json'),
                'inner_cv': {'folds': 5, 'unit': 'sample_id', 'splitter': 'per-column shuffled KFold',
                             'seed': FOLD_SEED, 'order': 'sample_id sorted', 'selector': payload['inner_cv']['selector'],
                             'tie': 'earliest epoch'},
                'pilot_gate': 'implementation validity only; no efficacy gate',
                'feature_usecols': old.FEATURES, 'outer_actions': 'not implemented; separate authorization required',
                'protocol_document_sha256': old.sha(STUDY / 'PROTOCOL.md')}
    protocol.pop('gate'); protocol.pop('promotion'); protocol.pop('freeze')
    save_json('protocol.json', protocol, immutable=True)
    schedule = row_schedule()
    schedule.to_csv(STUDY / 'OUTER_ROW_IDENTITIES.csv', index=False)
    provenance = {'files': {str(p.relative_to(ROOT)): old.sha(p) for p in [
        Path(__file__), ROOT / 'scripts/studies/run_column_conditioned_multitask.py',
        *[ROOT / p for p in json.loads((old.STUDY / 'EXECUTION_PROVENANCE.json').read_text())['files']]]},
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'cpu_threads': 1, 'protocol_sha256': old.sha(STUDY / 'protocol.json')}
    save_json('EXECUTION_PROVENANCE.json', provenance, immutable=True)
    receipts = []
    for c in old.TASKS:
        if c == '4g':
            split = pd.read_csv(old.QUALIFIED / 'splits/row_seed_42.csv')
            ids = split.loc[split.split.isin(['train', 'validation']), 'sample_id'].astype(str).tolist()
            purpose = 'source train and source diagnostic validation'
        else:
            ids = schedule.loc[schedule.column.eq(c) & schedule.outer_seed.eq(PILOT_SEED) & schedule.role.eq('gradient_train'), 'sample_id'].astype(str).tolist()
            purpose = 'pilot outer gradient_train only'
        receipts.append({'column': c, 'purpose': purpose, 'authorized_ids': ids,
                         'count': len(ids), 'ids_sha256': old.digest(sorted(ids)), 'outer_labels_authorized': False})
    save_json('LABEL_ACCESS_AUTHORIZATION.json', receipts, immutable=True)
    return payload


def fit_model(data, train, valid, arm, seed, run):
    """Original fit recipe with additional read-only loss/step/state diagnostics."""
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


def run_fold(fold):
    if fold not in range(1, 6):
        raise ValueError('Phase 1 accepts only folds 1..5 of the pilot seed')
    torch.set_num_threads(1)
    payload = specification()
    data = {c: authorized_data(c, payload) for c in old.TASKS}
    train, valid = fold_schedules(data)[fold - 1]
    seed = PILOT_SEED + fold * 1009
    init = initialization_audit(seed)
    identities = old.identity_records(data, train, valid, 'row_cv', PILOT_SEED, fold)
    for arm in old.ARMS:
        run = STUDY / f'runtime/inner/seed_{PILOT_SEED}/fold_{fold}/{arm}'
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'complete.json').exists():
            receipt = json.loads((run / 'complete.json').read_text())
            if receipt['protocol_sha256'] != old.sha(STUDY / 'protocol.json') or receipt['provenance_sha256'] != old.sha(STUDY / 'EXECUTION_PROVENANCE.json'):
                raise RuntimeError('completed run authority changed')
            for name, expected in receipt['files'].items():
                if old.sha(run / name) != expected:
                    raise RuntimeError('completed artifact changed')
            continue
        identities.to_csv(run / 'identities.csv', index=False)
        old.write_json(run / 'initialization.json', init)
        model, fitted, metric_scales = fit_model(data, train, valid, arm, seed, run)
        rows, tails, regimes, predictions = [], [], [], []
        for c in old.TASKS:
            pred = old.predict(model, data, c, valid[c])
            values, tail, strata = old.metric_rows(data[c]['truth'][valid[c]], pred, metric_scales[c], data[c]['tail'], data[c]['regimes'])
            meta = dict(protocol='row_cv', outer_seed=PILOT_SEED, inner_fold=fold, arm=arm, column=c)
            rows.append({**meta, **fitted, **values, 'inner_train_count': len(train[c]), 'inner_validation_count': len(valid[c]),
                         'inner_train_ids_sha256': old.digest(sorted(data[c]['frame'].iloc[train[c]].sample_id.tolist())),
                         'inner_validation_ids_sha256': old.digest(sorted(data[c]['frame'].iloc[valid[c]].sample_id.tolist())),
                         'scale_V1': metric_scales[c]['V1'], 'scale_V2': metric_scales[c]['V2']})
            tails.extend([{**meta, **item} for item in tail]); regimes.extend([{**meta, **item} for item in strata])
            predictions.extend([{**meta, 'sample_id': str(data[c]['frame'].iloc[i].sample_id),
                                  **dict(zip(old.QUANTILES, map(float, p)))} for i, p in zip(valid[c], pred)])
        for name, values in [('results.csv', rows), ('tails.csv', tails), ('regimes.csv', regimes), ('predictions.csv', predictions)]:
            pd.DataFrame(values).to_csv(run / name, index=False)
        files = [p.name for p in run.iterdir() if p.is_file()]
        old.write_json(run / 'complete.json', {'protocol_sha256': old.sha(STUDY / 'protocol.json'),
            'provenance_sha256': old.sha(STUDY / 'EXECUTION_PROVENANCE.json'),
            'files': {f: old.sha(run / f) for f in files}}, immutable=True)
        print(f'COMPLETED fold={fold} {arm}', flush=True)
    return fold


def preflight():
    torch.set_num_threads(1)
    payload = prepare()
    data = {c: authorized_data(c, payload) for c in old.TASKS}
    manifest, overlaps, counts = [], [], []
    for fold, (train, valid) in enumerate(fold_schedules(data), 1):
        frame = old.identity_records(data, train, valid, 'row_cv', PILOT_SEED, fold)
        manifest.append(frame)
        for c in old.TASKS:
            donor_keys = {data[c]['input_keys'][i] for i in train[c]}
            donor_mols = set(data[c]['frame'].iloc[train[c]].canonical_smiles)
            overlaps.append(dict(inner_fold=fold, column=c, donor=c, heldout_rows=len(valid[c]),
                sample_id_overlap=len(set(data[c]['frame'].iloc[train[c]].sample_id) & set(data[c]['frame'].iloc[valid[c]].sample_id)),
                backbone_input_overlap_rows=sum(data[c]['input_keys'][i] in donor_keys for i in valid[c]),
                full_model_input_overlap_rows=sum(data[c]['input_keys'][i] in donor_keys for i in valid[c]),
                canonical_overlap_rows=sum(str(data[c]['frame'].iloc[i].canonical_smiles) in donor_mols for i in valid[c])))
            for donor in old.TASKS:
                if donor == c:
                    continue
                keys = {data[donor]['input_keys'][i] for i in train[donor]}
                mols = set(data[donor]['frame'].iloc[train[donor]].canonical_smiles)
                overlaps.append(dict(inner_fold=fold, column=c, donor=donor, heldout_rows=len(valid[c]),
                    sample_id_overlap=len(set(data[donor]['frame'].iloc[train[donor]].sample_id) & set(data[c]['frame'].iloc[valid[c]].sample_id)),
                    backbone_input_overlap_rows=sum(data[c]['input_keys'][i] in keys for i in valid[c]),
                    full_model_input_overlap_rows=0,
                    canonical_overlap_rows=sum(str(data[c]['frame'].iloc[i].canonical_smiles) in mols for i in valid[c])))
    pd.concat(manifest).to_csv(STUDY / 'ROW_SPLIT_MANIFEST.csv', index=False)
    pd.DataFrame(overlaps).to_csv(STUDY / 'INPUT_OVERLAP_AUDIT.csv', index=False)
    for c in old.TASKS:
        indices = data[c]['positions']['gradient_train']
        ids = data[c]['frame'].iloc[indices].sample_id.tolist()
        keys = [data[c]['input_keys'][i] for i in indices]
        counts.append(dict(column=c, gradient_train_rows=len(ids), unique_sample_ids=len(set(ids)),
                           unique_model_inputs=len(set(keys)), extra_same_input_records=len(keys)-len(set(keys))))
    pd.DataFrame(counts).to_csv(STUDY / 'INPUT_DUPLICATE_COUNTS.csv', index=False)
    cross = []
    for left, right in [('4g', '25g'), ('4g', '40g'), ('25g', '40g')]:
        li, ri = [data[c]['positions']['gradient_train'] for c in [left, right]]
        lk, rk = [{data[c]['input_keys'][i] for i in idx} for c, idx in [(left, li), (right, ri)]]
        cross.append(dict(left=left, right=right,
            shared_sample_ids=len(set(data[left]['frame'].iloc[li].sample_id) & set(data[right]['frame'].iloc[ri].sample_id)),
            shared_backbone_input_keys=len(lk & rk), shared_full_model_input_keys=0,
            right_records_matching_left=sum(data[right]['input_keys'][i] in lk for i in ri),
            left_records_matching_right=sum(data[left]['input_keys'][i] in rk for i in li)))
    pd.DataFrame(cross).to_csv(STUDY / 'CROSS_COLUMN_INPUT_AUDIT.csv', index=False)
    headers = {c: list(pd.read_csv(old.PARENT / f'filtered_features_{c}.csv', nrows=0)) for c in TARGETS}
    init = {str(f): initialization_audit(PILOT_SEED + f * 1009) for f in range(1, 6)}
    save_json('PREFLIGHT.json', {'pilot_seed': PILOT_SEED, 'fold_seed': FOLD_SEED,
        'input_headers_metadata_only': headers, 'batch_or_replicate_metadata': 'not identified; physical independence unknown',
        'initialization_by_fold': init, 'outer_label_requests': 0,
        'manifest_sha256': old.sha(STUDY / 'ROW_SPLIT_MANIFEST.csv')}, immutable=True)
    print('PREFLIGHT PASSED: identities, whitelist, initialization, and input overlaps recorded', flush=True)


def main():
    torch.set_num_threads(1)
    preflight()
    # Three independent CPU workers, one thread each; every fit has its own RNG and process.
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run_fold, range(1, 6)))
    from summarize_phase1 import summarize
    summarize()


if __name__ == '__main__':
    main()
