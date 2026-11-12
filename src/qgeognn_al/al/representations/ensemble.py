"""Identity-bound current-L K=3 inference; never trains an ensemble implicitly.

The sequential numerical Hybrid protocol reuses evaluation member 0 and trains
members 1/2 from their own fixed initializations on exactly the same current L.
This module consumes already frozen, explicitly authorized checkpoints. It does
not substitute checkpoints from different outer seeds or different budgets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Sequence

import numpy as np

from ...infra.contracts import canonical_bytes, content_hash, sample_ids
from ...infra.resources import SnapshotFile
from .base import Authorization, FeatureSnapshot


SOURCE_COMMIT = 'b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28'
SOURCE_PATH = 'src/qgeognn_al/active_learning_v2/sequential_runner.py'
ENSEMBLE_K = 3


def member_seeds(outer_seed: int, member: int) -> tuple[int, int]:
    """Return (initialization, fit RNG) from benchmark_protocol, unchanged."""
    if type(outer_seed) is not int or outer_seed < 0 or type(member) is not int or not 0 <= member < 3:
        raise ValueError('nonnegative outer seed and member 0/1/2 required')
    return (outer_seed + 2_000_003 + member * 1_000_033,
            outer_seed + 3_000_017 + member * 1_000_033)


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(name + ' must be a SHA-256')
    return value


@dataclass(frozen=True)
class FrozenEnsembleMember:
    member: int
    outer_seed: int
    resource: SnapshotFile
    checkpoint_state_hash: str
    train_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    dataset_hash: str
    split_hash: str
    protocol_hash: str
    shared_fit_config_hash: str
    initialization_seed: int
    training_seed: int
    source_kind: str

    def __post_init__(self):
        if (self.initialization_seed, self.training_seed) != member_seeds(self.outer_seed, self.member):
            raise ValueError('ensemble initialization/training seeds differ from the pinned member profile')
        for name in ('checkpoint_state_hash', 'dataset_hash', 'split_hash', 'protocol_hash', 'shared_fit_config_hash'):
            _sha(getattr(self, name), name)
        _sha(self.resource.sha256, 'checkpoint file hash')
        authorization = Authorization(self.train_ids, self.candidate_ids)
        object.__setattr__(self, 'train_ids', authorization.train_ids)
        object.__setattr__(self, 'candidate_ids', authorization.candidate_ids)
        if self.source_kind not in ('synthetic', 'authorized_snapshot'):
            raise ValueError('ensemble source_kind must distinguish synthetic and authorized snapshots')

    def metadata(self):
        return {'member': self.member, 'outer_seed': self.outer_seed,
                'checkpoint_sha256': self.resource.sha256, 'checkpoint_state_hash': self.checkpoint_state_hash,
                'train_ids_hash': content_hash(self.train_ids), 'candidate_ids_hash': content_hash(self.candidate_ids),
                'dataset_hash': self.dataset_hash, 'split_hash': self.split_hash, 'protocol_hash': self.protocol_hash,
                'shared_fit_config_hash': self.shared_fit_config_hash, 'initialization_seed': self.initialization_seed,
                'training_seed': self.training_seed, 'source_kind': self.source_kind}


@dataclass(frozen=True, init=False)
class K3EnsembleSnapshot:
    members: tuple[FrozenEnsembleMember, ...]
    predictions: np.ndarray
    sha256: str

    def __init__(self, members: Sequence[FrozenEnsembleMember], predictions):
        ordered = tuple(members)
        if len(ordered) != ENSEMBLE_K or tuple(m.member for m in ordered) != (0, 1, 2):
            raise ValueError('exactly three ordered member identities 0,1,2 are required')
        first = ordered[0]
        same = ('outer_seed', 'train_ids', 'candidate_ids', 'dataset_hash', 'split_hash',
                'protocol_hash', 'shared_fit_config_hash', 'source_kind')
        if any(any(getattr(m, field) != getattr(first, field) for field in same) for m in ordered[1:]):
            raise ValueError('K3 members must share the same outer seed/current L/current U/split/fit protocol')
        if len({m.resource.sha256 for m in ordered}) != ENSEMBLE_K:
            raise ValueError('K3 requires three distinct frozen checkpoints, not one reused model')
        values = np.array(predictions, dtype=np.float64, copy=True, order='C')
        if values.shape != (3, len(first.candidate_ids), 6) or not np.isfinite(values).all():
            raise ValueError('finite K3 by current-U by six-output predictions required')
        values.setflags(write=False)
        object.__setattr__(self, 'members', ordered)
        object.__setattr__(self, 'predictions', values)
        object.__setattr__(self, 'sha256', self._digest())

    def _digest(self):
        digest = hashlib.sha256(canonical_bytes([m.metadata() for m in self.members]))
        digest.update(self.predictions.tobytes(order='C'))
        return digest.hexdigest()

    def verify(self):
        if self._digest() != self.sha256:
            raise ValueError('frozen K3 predictions or member metadata changed')

    def numerical_signals(self, snapshot: FeatureSnapshot, *, outer_seed: int, target_scales):
        """Bind current-member0 latent features and fixed original-L0 scales."""
        self.verify(); snapshot.verify()
        first = self.members[0]
        provenance = snapshot.predictor
        if snapshot.config.kind != 'latent' or snapshot.config.dimension != 128:
            raise ValueError('numerical Hybrid requires current member-0 latent128 features')
        if (first.outer_seed != outer_seed or first.train_ids != snapshot.train_ids
                or first.candidate_ids != snapshot.candidate_ids):
            raise ValueError('ensemble does not describe the current authorized round')
        for field in ('dataset_hash', 'split_hash', 'protocol_hash', 'source_kind'):
            if getattr(first, field) != getattr(provenance, field):
                raise ValueError('ensemble and latent predictor provenance differ: ' + field)
        if first.resource.sha256 != provenance.model_checkpoint_hash:
            raise ValueError('Hybrid latent features must use evaluation checkpoint member 0')
        if snapshot.extraction is not None and snapshot.extraction.predictor_state_hash != first.checkpoint_state_hash:
            raise ValueError('Hybrid latent extraction state differs from member 0')
        scale = np.asarray(target_scales, dtype=float)
        if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('two positive fixed original-L0 endpoint scales required')
        return {'candidate_ids': self.members[0].candidate_ids,
                'ensemble_predictions': self.predictions, 'target_scales': tuple(scale)}


def collect_checkpoint_ensemble(
    members: Sequence[FrozenEnsembleMember], *,
    load_predictor: Callable, predict: Callable, state_hash: Callable,
) -> K3EnsembleSnapshot:
    """Infer all three immutable members through an explicit trusted model adapter.

    ``predict(model, current_u_ids)`` returns ``(six_outputs, ordered_ids)``.
    Checkpoints must be independent read-only SnapshotFiles; both their bytes
    and predictor state are checked before and after inference. No fit, label
    store, provider or automatic resource discovery is accepted at this boundary.
    """
    ordered = tuple(members)
    # Validate the full identity contract before loading any model resource.
    first = ordered[0] if ordered else None
    rows = len(first.candidate_ids) if first else 0
    K3EnsembleSnapshot(ordered, np.zeros((3, rows, 6)))
    predictions = []
    for member in ordered:
        path = member.resource.verify()
        model = load_predictor(path)
        if state_hash(model) != member.checkpoint_state_hash:
            raise ValueError('loaded ensemble state differs from the pinned checkpoint state')
        values, ids = predict(model, member.candidate_ids)
        if sample_ids(ids, name='ensemble prediction IDs') != member.candidate_ids:
            raise ValueError('ensemble predictions are not in exact current-U order')
        if state_hash(model) != member.checkpoint_state_hash:
            raise ValueError('ensemble predictor changed during inference')
        member.resource.verify()
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (rows, 6) or not np.isfinite(values).all():
            raise ValueError('ensemble member must return finite six-output current-U rows')
        predictions.append(values)
    return K3EnsembleSnapshot(ordered, np.stack(predictions))


def infer_frozen_graph_ensemble(members, *, authorization, rows, provenance):
    """Run three real frozen QGeoGNN checkpoints on qualified label-free graphs.

    Returns ``(K3EnsembleSnapshot, member0_latent_snapshot)``. Original checkpoint
    bytes stay in external readonly SnapshotFiles. Graph rows cover exactly the
    authorized current L then U; no validation/test role or target field enters
    any predictor. This is inference, never an implicit ensemble fit.
    """
    import torch
    from ...predictor.model import load_predictor_checkpoint
    from ...predictor.inference import predict_outputs
    from .latent import _qualified_graph, extract_latent, predictor_state_hash
    ordered = tuple(members)
    if any(m.source_kind != 'authorized_snapshot' for m in ordered):
        raise ValueError('real graph ensemble inference requires explicitly authorized frozen members')
    ids = sample_ids([row.sample_id for row in rows], name='K3 graph rows')
    if ids != authorization.train_ids + authorization.candidate_ids:
        raise ValueError('K3 graph rows must match exact authorized current L then U order')
    # Own independent graph copies; canonical_position is an ordering receipt,
    # not a learned feature. Local candidate inference uses its own zero-based
    # order while latent extraction retains the original canonical receipt.
    candidate_rows = rows[len(authorization.train_ids):]
    atom, angle = [], []
    for position, row in enumerate(candidate_rows):
        a, b = _qualified_graph(row.atom, atom=True), _qualified_graph(row.angle, atom=False)
        a.canonical_position = torch.tensor(position)
        atom.append(a); angle.append(b)
    models = {}
    def load(path):
        model = load_predictor_checkpoint(path)
        models[str(path)] = model
        return model
    def predict(model, candidate_ids):
        if candidate_ids != authorization.candidate_ids:
            raise ValueError('K3 model inference escaped the authorized current-U order')
        return predict_outputs(model, atom, angle, list(range(len(atom)))), candidate_ids
    ensemble = collect_checkpoint_ensemble(ordered, load_predictor=load, predict=predict,
                                           state_hash=predictor_state_hash)
    first = ordered[0]
    model0 = models[str(first.resource.path.absolute())]
    latent = extract_latent(model0, expected_state_hash=first.checkpoint_state_hash,
                           authorization=authorization, rows=rows, provenance=provenance)
    # Validate shared provenance and member0 state before exporting the pair.
    ensemble.numerical_signals(latent, outer_seed=first.outer_seed, target_scales=(1., 1.))
    return ensemble, latent
