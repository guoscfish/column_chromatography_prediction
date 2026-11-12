"""A single-workflow backend for explicit, immutable per-round recordings.

No training or provider execution is possible here. A recording carries the
original fit/feature/metric identity, never a promise that a new fit is equal.
"""
from __future__ import annotations
import numpy as np
from ...infra.contracts import BaselineProvenance, content_hash
from ..representations import FeatureSnapshot, RepresentationConfig


class ReplayBackend:
    kind = 'replay'

    def __init__(self, recording):
        import json
        self.recording = json.loads(json.dumps(recording,allow_nan=False))
        self.initial_state_hash = self.recording['initial_state_hash']
        self._hash = content_hash(self.recording)
        self._rounds = {}
        for row in self.recording['rounds']:
            key = content_hash(row['labeled_ids'])
            if key in self._rounds:
                raise ValueError('duplicate recorded fit context')
            self._rounds[key] = row

    @property
    def contract_hash(self):
        if content_hash(self.recording) != self._hash:
            raise ValueError('replay recording was mutated')
        return self._hash

    def _row(self, fit):
        row = self._rounds.get(fit['labeled_ids_hash'])
        if row is None or fit != row['fit']:
            raise ValueError('recorded fit identity drift')
        return row

    def fit(self, *, seed, labeled_ids, labels, validation_ids, validation_labels):
        row = self._rounds.get(content_hash(list(labeled_ids)))
        if row is None:
            raise LookupError('BLOCKED_MISSING_FIXTURE: recorded fit unavailable for current ordered L')
        context = {'seed':seed,'labeled_ids':list(labeled_ids),'labels':labels,
                   'validation_ids':list(validation_ids),'validation_labels':validation_labels}
        if row['fit_input_sha256'] != content_hash(context):
            raise ValueError('replay fit inputs differ from recorded context')
        return dict(row['fit'])

    def represent(self, *, fit, authorization):
        row = self._row(fit)
        if tuple(row['labeled_ids']) != authorization.train_ids or tuple(row['candidate_ids']) != authorization.candidate_ids:
            raise ValueError('replay feature L/U order differs from workflow')
        features = row['features']
        return FeatureSnapshot(authorization=authorization,train_ids=authorization.train_ids,
            candidate_ids=authorization.candidate_ids,train_features=features['train'],candidate_features=features['candidates'],
            config=RepresentationConfig(**features['config']),predictor=BaselineProvenance(**features['provenance']))

    def validate(self, *, fit, validation_ids, validation_labels):
        row = self._row(fit)
        context = {'validation_ids':list(validation_ids),'validation_labels':validation_labels}
        if row['validation_input_sha256'] != content_hash(context):
            raise ValueError('replay validation context drift')
        metrics = dict(row['validation_metrics'])
        if not metrics or not all(np.isfinite(float(v)) for v in metrics.values()):
            raise ValueError('invalid recorded validation metrics')
        return metrics

    def numerical_signals(self, *, fit, authorization, acquisition):
        return self._row(fit).get('numerical_signals')
