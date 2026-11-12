"""Shared predictor representation backend and a synthetic lifecycle fixture.

Real Row fitting uses the single transaction in row_backend/row_fit. No model
optimization is available from the earlier generic backend scaffold.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from ...infra.contracts import BaselineProvenance, content_hash
from ...infra.artifacts import sha256_file
from ..representations import FeatureSnapshot, RepresentationConfig
from ..evaluation.point import point_metrics


class SyntheticBackend:
    """Deterministic fixture, no optimizer, QGeoGNN fit, provider or real data."""
    kind = 'synthetic'

    def __init__(self, *, features, validation_predictions, initial_state_hash, dataset_hash, split_hash, code_commit, protocol_hash):
        self.features = {key: np.asarray(value, float).copy() for key, value in features.items()}
        self.predictions = {key: np.asarray(value, float).copy() for key, value in validation_predictions.items()}
        self.initial_state_hash, self.dataset_hash, self.split_hash, self.code_commit = initial_state_hash, dataset_hash, split_hash, code_commit
        self.protocol_hash = protocol_hash
        self.calls = []

    @property
    def contract_hash(self):
        return content_hash({'kind': self.kind, 'initial_state': self.initial_state_hash,
                             'features': {k: v.tolist() for k, v in self.features.items()},
                             'validation_predictions': {k: v.tolist() for k, v in self.predictions.items()},
                             'dataset_hash': self.dataset_hash, 'split_hash': self.split_hash,
                             'code_commit': self.code_commit, 'protocol_hash': self.protocol_hash})

    def fit(self, *, seed, labeled_ids, labels, validation_ids, validation_labels):
        self.calls.append(('fit_fixture', tuple(labeled_ids)))
        # This receipt represents a synthetic fixture state, not optimized weights.
        return {'seed': seed, 'initial_state_hash': self.initial_state_hash,
                'labeled_ids_hash': content_hash(list(labeled_ids)),
                'checkpoint_hash': content_hash({'synthetic': True, 'seed': seed, 'labels': labels}),
                'training_performed': False, 'backend': 'synthetic_fixture'}

    def represent(self, *, fit, authorization):
        self.calls.append(('represent', authorization.candidate_ids))
        dimension = len(next(iter(self.features.values())))
        return FeatureSnapshot(authorization=authorization, train_ids=authorization.train_ids,
                               candidate_ids=authorization.candidate_ids,
                               train_features=[self.features[i] for i in authorization.train_ids],
                               candidate_features=[self.features[i] for i in authorization.candidate_ids],
                               config=RepresentationConfig('latent', dimension),
                               predictor=BaselineProvenance(self.code_commit, self.dataset_hash, self.split_hash,
                                                          fit['checkpoint_hash'], self.protocol_hash, 'synthetic'))

    def numerical_signals(self, *, fit, authorization, acquisition):
        if acquisition != 'hybrid':
            return None
        values = np.asarray([self.features[i] for i in authorization.candidate_ids])
        predictions = np.zeros((3, len(values), 6))
        for member in range(3):
            predictions[member, :, 1] = values[:, 0] + member * np.linalg.norm(values, axis=1)
            predictions[member, :, 4] = 2 * values[:, 0] + member * np.linalg.norm(values, axis=1)
        return {'candidate_ids': authorization.candidate_ids,
                'ensemble_predictions': predictions, 'target_scales': (1., 1.)}

    def validate(self, *, fit, validation_ids, validation_labels):
        self.calls.append(('validate', tuple(validation_ids)))
        true = np.asarray([validation_labels[i] for i in validation_ids])
        prediction = np.asarray([self.predictions[i] for i in validation_ids])
        # Synthetic fixed scales are part of this fixture, not historical AL scales.
        return point_metrics(true, prediction, {'V1': 1., 'V2': 1.})


class QGeoGNNBackend:
    """Shared frozen-initialization predictor representation/validation backend.

    Row fitting is implemented only by QGeoGNNRowBackend through the source
    recipe and its complete transaction; no earlier scaffold fit path remains.
    Feature graphs contain no labels; only the Row fitter injects L/validation.
    """
    kind = 'qgeognn'

    def __init__(self, *, normalization, initial_state, initial_state_hash, features, preprocessing,
                 training_config, representation: RepresentationConfig, provenance: BaselineProvenance,
                 output_root, allow_training=False, l0_ids=(), l0_targets=None, forbidden_output_roots=(), ensemble_snapshot_supplier=None):
        import torch
        from ...predictor import build_predictor
        from ..representations.latent import predictor_state_hash
        self.normalization, self.preprocessing = normalization, preprocessing
        self.initial_state = {name: value.detach().cpu().clone() for name, value in initial_state.items()}
        model = build_predictor(normalization)
        model.load_state_dict(self.initial_state, strict=True); model.eval()
        if predictor_state_hash(model) != initial_state_hash:
            raise ValueError('frozen initialization fingerprint mismatch')
        self.initial_state_hash = initial_state_hash
        self.features, self.representation, self.provenance = features, representation, provenance
        self.config, self.allow_training = dict(training_config), bool(allow_training)
        self.l0_ids, self.l0_targets = tuple(l0_ids), l0_targets
        self.root = Path(output_root).absolute()
        if any(p.is_symlink() for p in [self.root, *self.root.parents]):
            raise ValueError('backend output symlink forbidden')
        if any(self.root.resolve() == Path(p).resolve() or Path(p).resolve() in self.root.resolve().parents for p in forbidden_output_roots):
            raise ValueError('backend cannot write in a protected running tree')
        for identity, row in features.items():
            if row.sample_id != identity or 'y' in row.atom or 'y' in row.angle:
                raise ValueError('backend features must be identity-aligned and label-free')
        self._models = {}
        self.ensemble_snapshot_supplier = ensemble_snapshot_supplier

    @property
    def contract_hash(self):
        return content_hash({'kind': self.kind, 'initial_state_hash': self.initial_state_hash,
                             'normalization': asdict(self.normalization), 'preprocessing': self.preprocessing,
                             'training_config': self.config, 'representation': asdict(self.representation),
                             'provenance': self.provenance.to_dict(), 'feature_ids': list(self.features),
                             'l0_ids': self.l0_ids,
                             'l0_targets': None if self.l0_targets is None else np.asarray(self.l0_targets).tolist()})

    def _graphs(self, ids, labels=None):
        import torch
        atom, angle = [], []
        for position, identity in enumerate(ids):
            row = self.features[identity]
            a, b = row.atom.clone(), row.angle.clone()
            a.canonical_position = torch.tensor(position)
            if labels is not None:
                a.y = torch.tensor([labels[identity]], dtype=torch.float32)
            atom.append(a); angle.append(b)
        return atom, angle

    def fit(self, *, seed, labeled_ids, labels, validation_ids, validation_labels):
        if not self.allow_training:
            raise PermissionError('new scientific training requires separate explicit authorization')
        raise NotImplementedError('use QGeoGNNRowBackend for the single source-qualified Row fit transaction')

    def _model(self, fit):
        from ...predictor import load_predictor_checkpoint
        path = Path(fit['checkpoint_path'])
        if not path.resolve().is_relative_to(self.root.resolve()) or sha256_file(path) != fit['checkpoint_hash']:
            raise ValueError('fitted checkpoint path/hash drift')
        if fit['checkpoint_hash'] not in self._models:
            self._models[fit['checkpoint_hash']] = load_predictor_checkpoint(path)
        return self._models[fit['checkpoint_hash']]

    def represent(self, *, fit, authorization):
        from ..representations.latent import extract_latent, predictor_state_hash
        from ..representations.gradient import extract_gradient
        from ..representations.fused import fuse_snapshots
        model = self._model(fit)
        rows = [self.features[i] for i in authorization.train_ids + authorization.candidate_ids]
        provenance = BaselineProvenance(**{**self.provenance.to_dict(), 'model_checkpoint_hash': fit['checkpoint_hash']})
        kwargs = dict(expected_state_hash=predictor_state_hash(model), authorization=authorization, rows=rows, provenance=provenance)
        parameters = json.loads(self.representation.parameters_json)
        kind = self.representation.kind
        if kind == 'latent':
            return extract_latent(model, **kwargs)
        scales = tuple(self.preprocessing['target_scales'][name] for name in ['V1', 'V2'])
        gradient = extract_gradient(model, **kwargs, sketch_seed=parameters['sketch_seed'],
                                    kind='center_width' if kind == 'center_width' else 'gradient',
                                    **({'l0_ids': self.l0_ids, 'l0_targets': self.l0_targets} if kind == 'center_width' else {'scales': scales}))
        if kind == 'fused':
            return fuse_snapshots(gradient, extract_latent(model, **kwargs), alpha=parameters['alpha'])
        return gradient

    def numerical_signals(self, *, fit, authorization, acquisition):
        if acquisition != 'hybrid':
            return None
        if self.ensemble_snapshot_supplier is None:
            raise ValueError('Hybrid requires existing same-seed K3 snapshots; no automatic training fallback')
        packet = self.ensemble_snapshot_supplier(fit, authorization.candidate_ids)
        if packet.get('outer_seed') != fit['seed'] or packet.get('labeled_ids_hash') != fit['labeled_ids_hash']:
            raise ValueError('ensemble snapshots belong to another seed or label budget')
        return {key: packet[key] for key in ['candidate_ids', 'ensemble_predictions', 'target_scales']}

    def validate(self, *, fit, validation_ids, validation_labels):
        from ...predictor.inference import predict_outputs
        atom, angle = self._graphs(validation_ids)
        prediction = predict_outputs(self._model(fit), atom, angle, list(range(len(validation_ids))))
        return point_metrics(np.asarray([validation_labels[i] for i in validation_ids]), prediction,
                             self.preprocessing['target_scales'])
