"""Single pure acquisition boundary with explicit identities and numerical signals."""
from __future__ import annotations
from dataclasses import dataclass
import json
from collections.abc import Mapping
import numpy as np
from ...infra.contracts import budget_value, canonical_bytes, sample_ids
from ..representations.base import FeatureSnapshot
from .lcmd import lcmd_tp_select
from .maxdet import conditional_gradient_maxdet
from .ivr import conditional_batch_ivr
from .hybrid import hybrid_select
from .random import random_select


@dataclass(frozen=True)
class Selection:
    selected_ids: tuple[str, ...]
    acquisition: str
    seed: int
    budget: int
    feature_hash: str
    trace_json: str
    audit_json: str = '{}'

    def __post_init__(self):
        object.__setattr__(self, 'selected_ids', sample_ids(self.selected_ids, name='selection'))
        if len(self.selected_ids) != self.budget:
            raise ValueError('selection length differs from budget')
        canonical_bytes(self.trace)
        canonical_bytes(self.audit)

    @property
    def trace(self):
        return json.loads(self.trace_json)

    @property
    def audit(self):
        return json.loads(self.audit_json)

    def to_dict(self):
        return {'selected_ids': list(self.selected_ids), 'acquisition': self.acquisition,
                'seed': self.seed, 'budget': self.budget, 'feature_hash': self.feature_hash,
                'trace': self.trace, 'audit': self.audit}


def select(snapshot: FeatureSnapshot, *, acquisition: str, budget: int, seed: int,
           numerical_signals: Mapping[str, object] | None = None) -> Selection:
    """Consume L/U vectors and explicitly authorized signals; never load resources.

    ``seed`` is the effective acquisition RNG seed, not an implicit outer seed.
    Historical random controls use random_control_seed(outer_seed, control).
    Hybrid alone requires numerical_signals with exactly candidate_ids,
    ensemble_predictions and target_scales, all in the snapshot U order.
    """
    snapshot.verify()
    budget_value(budget, len(snapshot.candidate_ids))
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    matrix = np.vstack([snapshot.train, snapshot.candidates])
    labeled = np.arange(len(snapshot.train), dtype=np.int64)
    candidates = np.arange(len(snapshot.train), len(matrix), dtype=np.int64)
    audit = {'normalization': 'method-specific historical definition; no shared rescaling',
             'tie_breaking': 'stable supplied candidate order', 'effective_acquisition_seed': seed}
    if acquisition != 'hybrid' and numerical_signals:
        raise ValueError('numerical signals are only supported by the hybrid selector')
    if acquisition == 'lcmd':
        result = lcmd_tp_select(snapshot.candidates, snapshot.train, budget)
        positions = result.selected_pool_positions.tolist()
        trace = [dict(row) for row in result.trace]
        audit.update(initial_nearest_center=result.initial_nearest_center.tolist(),
                     initial_nearest_sq_distance=result.initial_nearest_sq_distance.tolist())
    elif acquisition == 'random':
        positions = random_select(len(snapshot.candidate_ids), budget, seed).tolist()
        trace = [{'selection_order': i, 'pool_position': p} for i, p in enumerate(positions)]
    elif acquisition == 'maxdet':
        result = conditional_gradient_maxdet(matrix, labeled, candidates, budget)
        positions = result.selected_candidate_positions.tolist()
        trace = [dict(row) for row in result.trace]
        audit.update(result.audit)
        audit['initial_marginal_gains'] = result.initial_marginal_gains.tolist()
    elif acquisition == 'ivr':
        result = conditional_batch_ivr(matrix, labeled, candidates, budget)
        positions = result['selected_candidate_positions']
        trace = [dict(row) for row in result['trace']]
        audit.update(result['audit'])
    elif acquisition == 'hybrid':
        required = {'candidate_ids', 'ensemble_predictions', 'target_scales'}
        if not isinstance(numerical_signals, Mapping) or set(numerical_signals) != required:
            raise ValueError('hybrid requires only identity-bound ensemble predictions and scales')
        if sample_ids(numerical_signals['candidate_ids'], name='signal candidate IDs') != snapshot.candidate_ids:
            raise ValueError('numerical signals differ from authorized candidate identity/order')
        values = np.asarray(numerical_signals['ensemble_predictions'], dtype=float)
        if values.shape != (3, len(snapshot.candidate_ids), 6):
            raise ValueError('ensemble signal must be 3 x authorized U x six outputs')
        result = hybrid_select(matrix, values, len(labeled), numerical_signals['target_scales'], budget)
        positions = result['selected_positions'].tolist()
        shortlist = result['shortlist_positions'].tolist()
        rank = {p: i for i, p in enumerate(shortlist)}
        trace = [{'selection_order': i, 'pool_position': p, 'shortlist_rank': rank[p],
                  'uncertainty_score': float(result['uncertainty_scores'][p])} for i, p in enumerate(positions)]
        audit.update({key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in result.items() if key != 'selected_positions'})
    else:
        raise ValueError('unsupported acquisition: ' + str(acquisition))
    if len(set(positions)) != budget or any(p < 0 or p >= len(snapshot.candidate_ids) for p in positions):
        raise ValueError('selector escaped authorized candidate identities or budget')
    for row, position in zip(trace, positions):
        row['sample_id'] = snapshot.candidate_ids[position]
    snapshot.verify()
    return Selection(tuple(snapshot.candidate_ids[p] for p in positions), acquisition, seed, budget,
                     snapshot.sha256, canonical_bytes(trace).decode(), canonical_bytes(audit).decode())
