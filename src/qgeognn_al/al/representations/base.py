"""A restricted snapshot boundary for latent and gradient representations.

    This module validates precomputed vectors. It does not run the predictor,
    compute gradients, normalize targets or load datasets/checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import hashlib
from typing import Sequence

import numpy as np

from ...infra.contracts import BaselineProvenance, canonical_bytes, content_hash, sample_ids


@dataclass(frozen=True)
class Authorization:
    train_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_ids", sample_ids(self.train_ids, name="authorized L"))
        object.__setattr__(self, "candidate_ids", sample_ids(self.candidate_ids, name="authorized U"))
        if not self.train_ids or not self.candidate_ids:
            raise ValueError("authorized L and U must both be nonempty")
        if set(self.train_ids) & set(self.candidate_ids):
            raise ValueError("authorized L and U must be disjoint")


@dataclass(frozen=True)
class RepresentationConfig:
    kind: str
    dimension: int
    parameters_json: str = '{}'

    def __post_init__(self) -> None:
        if self.kind not in ("latent", "gradient", "center_width", "fused"):
            raise ValueError("registered representation kind required")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("representation dimension must be a positive integer")
        if not isinstance(json.loads(self.parameters_json), dict):
            raise ValueError("representation parameters must be a JSON object")
        canonical_bytes(json.loads(self.parameters_json))


@dataclass(frozen=True)
class ExtractionProvenance:
    method: str
    predictor_state_hash: str
    predictor_state_after_hash: str
    graph_snapshot_hash: str
    graph_snapshot_after_hash: str
    batch_size: int
    details_json: str = '{}'


def _matrix(values: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    source = np.asarray(values)
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(source.dtype, np.complexfloating):
        raise ValueError("features must be real numeric vectors")
    # Normalize layout/dtype only; no scaling or feature mathematics changes.
    array = np.array(source, dtype=np.float64, order="C", copy=True)
    if array.shape != (rows, dimension) or not np.isfinite(array).all():
        raise ValueError("feature shape, sample ID alignment or finite-value check failed")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, init=False)
class FeatureSnapshot:
    authorization: Authorization
    train_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    train: np.ndarray
    candidates: np.ndarray
    config: RepresentationConfig
    predictor: BaselineProvenance
    extraction: ExtractionProvenance | None
    sha256: str

    def __init__(
        self, *, authorization: Authorization, train_ids: Sequence[str], candidate_ids: Sequence[str],
        train_features: np.ndarray, candidate_features: np.ndarray,
        config: RepresentationConfig, predictor: BaselineProvenance,
        extraction: ExtractionProvenance | None = None,
    ) -> None:
        train_order = sample_ids(train_ids, name="feature L")
        candidate_order = sample_ids(candidate_ids, name="feature U")
        if train_order != authorization.train_ids or candidate_order != authorization.candidate_ids:
            raise ValueError("feature IDs must exactly match authorized L/U order")
        for key, value in dict(
            authorization=authorization, train_ids=train_order, candidate_ids=candidate_order,
            train=_matrix(train_features, len(train_order), config.dimension),
            candidates=_matrix(candidate_features, len(candidate_order), config.dimension),
            config=config, predictor=predictor, extraction=extraction,
        ).items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "sha256", self._digest())

    def metadata(self) -> dict[str, object]:
        return {"schema_version": 2, "representation": asdict(self.config),
                "train_ids": list(self.train_ids), "candidate_ids": list(self.candidate_ids),
                "predictor": self.predictor.to_dict(), "dtype": "float64",
                "feature_source": "predictor_extraction" if self.extraction else "precomputed_snapshot",
                "extraction": asdict(self.extraction) if self.extraction else None}

    def _digest(self) -> str:
        digest = hashlib.sha256(canonical_bytes(self.metadata()))
        digest.update(self.train.tobytes(order="C"))
        digest.update(self.candidates.tobytes(order="C"))
        return digest.hexdigest()

    def verify(self) -> None:
        if self._digest() != self.sha256:
            raise ValueError("frozen representation hash changed")

    def provenance(self) -> dict[str, object]:
        self.verify()
        return {**self.metadata(), "feature_hash": self.sha256,
                "authorization_hash": content_hash(asdict(self.authorization))}
