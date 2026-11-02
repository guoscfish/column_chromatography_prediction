"""Unchanged QGeoGNN training primitives behind the V3 freeze/reveal boundary.

Imported lazily: label-safe round-zero staging never creates a label store.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from ..schemas.conditions import ConditionNormalization
from ..models import load_predictor_checkpoint
from ..active_learning_v2.benchmark_protocol import load_features
from ..active_learning_v2.protocol import RestrictedLabelStore, ids_hash, validate_row_protocol
from ..active_learning_v2.short_sequential_runner import ShortSequentialContext
from ..active_learning_v2.gradient_transforms import center_width_transform
from ..active_learning_v2.llm_screen import banks
from ..active_learning_v2.lcmd import lcmd_tp_select, _squared_distances
from ..active_learning_v2.runner import predict_outputs
from ..active_learning_v2.dialog_study import _card, _observed
from .artifacts import once, read, file_hash
from .schema import VERSION
from .study import directory, audit_batch, state, METHODS
from . import protocol


class Context(ShortSequentialContext):
    def __init__(self, root, seed, method):
        self.seed, self.study = seed, root
        self.runtime = root/f'runtime/seed_{seed}'
        self.runtime.mkdir(parents=True, exist_ok=True)
        frozen = read(root/'protocol.json')
        initial = frozen['initial'][str(seed)]
        self.source = initial['source_features']
        self.partition = pd.read_csv(initial['split'])
        validate_row_protocol(self.partition)
        self.data = load_features(self.source)  # feature-only usecols
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError('split/source alignment drift')
        self.roles = {r: self.partition.loc[self.partition.role.eq(r), 'canonical_index'].to_numpy(int)
                      for r in ('l0', 'u0', 'validation', 'test')}
        old = read(initial['context'])['contract']
        self.preprocessing = old['preprocessing']
        self.normalization = ConditionNormalization(**old['normalization'])
        self.atom, self.angle = torch.load(initial['scrubbed_graphs'], weights_only=False)
        if any(torch.count_nonzero(row.y) for row in self.atom):
            raise RuntimeError('scrubbed graphs contain labels')
        observed = read(root/f'initial/seed_{seed}/{method}.json')['observed']
        if [r['id'] for r in observed] != self.ids(self.roles['l0']):
            raise RuntimeError('initial observation alignment mismatch')
        self.l0_truth = np.array([[r['V1_ml'], r['V2_ml']] for r in observed], dtype=np.float32)
        self.validation_truth = None
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = file_hash(root/'protocol.json')

    def new_store(self):
        return RestrictedLabelStore(self.source, self.partition)


def arrays(context, current):
    lookup = dict(zip(context.ids(np.arange(len(context.data))), range(len(context.data))))
    acquired = np.array([lookup[i] for i in current['acquired_ids']], dtype=int)
    labeled = np.r_[context.roles['l0'], acquired]
    unlabeled = np.array([i for i in context.roles['u0'] if i not in set(acquired)], dtype=int)
    new_truth = [[r['true_V1_ml'], r['true_V2_ml']] for h in current['history'] for r in h['records']]
    truth = np.vstack([context.l0_truth, np.asarray(new_truth).reshape(-1, 2)])
    return labeled, unlabeled, truth


def build_catalog(root, seed, method, r, current):
    context = Context(root, seed, method)
    labeled, unlabeled, truth = arrays(context, current)
    trained = read(directory(root, seed, method, r-1)/'training_complete.json')
    checkpoint = trained['checkpoint']
    runtime = context.runtime/method/f'round_{r:02d}/acquisition'
    runtime.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cw, _ = banks(context, np.r_[labeled, unlabeled], checkpoint, r, runtime)
    n = len(labeled)
    pending_positions = lcmd_tp_select(cw[n:], cw[:n], 16).selected_pool_positions if method == METHODS[0] else []
    pending_ids = context.ids(unlabeled[np.asarray(pending_positions, dtype=int)])
    predictions, order = predict_outputs(load_predictor_checkpoint(checkpoint), context.atom, context.angle, unlabeled)
    if not np.array_equal(order, unlabeled) or predictions.shape != (len(unlabeled), 6):
        raise RuntimeError('prediction order/shape drift')
    percentiles = pd.Series(np.sqrt(_squared_distances(cw[n:], cw[:n]).min(axis=1))).rank(pct=True).to_numpy()
    cards = []
    for index, pred, distance in zip(unlabeled, predictions, percentiles):
        row = _card(context, index, pred, distance)
        row.update({f'pred_{target}_{quantile}_ml': float(pred[j]) for j, (target, quantile) in enumerate(
            (t, q) for t in ('V1', 'V2') for q in ('q10', 'q50', 'q90'))})
        cards.append(row)
    by_id = {r['id']: r for r in cards}
    return {'candidates': [row for row in cards if row['id'] not in set(pending_ids)],
            'pending': [by_id[i] for i in pending_ids], 'observed': _observed(context, labeled, truth, current['history']),
            'salt': f'{VERSION}:{seed}:{method}:{r}',
            'target_scales': [context.preprocessing['target_scales'][t] for t in ('V1', 'V2')]}, checkpoint


def measure_and_fit(root, seed, method, r, current):
    protocol.validate(root)
    d = directory(root, seed, method, r)
    frozen = audit_batch(root, seed, method, r)
    context = Context(root, seed, method)
    store = context.new_store()
    store.freeze_acquisitions(list(dict.fromkeys(current['acquired_ids']+frozen['batch_ids'])))
    if not (d/'measurement.json').exists():
        values = store.reveal(frozen['batch_ids'], 'after_acquisition_fit')
        records = []
        for pred, truth in zip(frozen['premeasurement_predictions'], values):
            records.append({**pred, 'true_V1_ml': float(truth[0]), 'true_V2_ml': float(truth[1]),
                'error_V1_ml': pred['pred_V1_ml']-float(truth[0]),
                'error_V2_ml': pred['pred_V2_ml']-float(truth[1]),
                'source': 'CW' if pred['candidate_id'] in frozen['pending_ids'] else 'LLM'})
        once(d/'measurement.json', {'seed': seed, 'method': method, 'round': r,
            'batch_freeze_sha256': file_hash(d/'batch_freeze.json'), 'records': records,
            'label_access_receipt': store.audit[-1]})
    current = state(root, seed, method)  # verifies measurement/receipt before training or resumed fit
    labeled, _, truth = arrays(context, current)
    context.validation_truth = store.reveal(context.ids(context.roles['validation']), 'initial_fit')
    target = context.runtime/method/f'round_{r+1:02d}/model'
    torch.set_num_threads(2)
    context.fit(method, r+1, labeled, truth, target)
    protocol.validate(root)
    fit = read(target/'fit_audit.json')
    if fit['train_ids_hash'] != ids_hash(context.ids(labeled)) or fit['train_rows'] != len(labeled):
        raise RuntimeError('training labels drift')
    initial = read(root/'protocol.json')['initial'][str(seed)]
    if fit['initialization_hash'] != read(initial['checkpoint_audit'])['initialization_hash']:
        raise RuntimeError('training initialization drift')
    once(d/'training_complete.json', {'measurement_sha256': file_hash(d/'measurement.json'),
        'checkpoint': str(target/'best.pt'), 'artifact_hashes': {str(target/name): file_hash(target/name)
             for name in ('best.pt', 'fit_audit.json', 'predictions.csv.gz')},
        'test_truth_access_count': 0})
    return {'status': 'ONE_BATCH_TRAINED', 'active_label_count': len(labeled), 'next_round': r+1,
            'next_selection_started': False}
