"""Frozen-source feature caching and residual target-side fusion.

The cache contains source predictions and (optionally) the frozen active
QGeoGNN-V2 representation.  It never reads target endpoint cells and is keyed
by both sample identity and the source checkpoint digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from ..models import extract_representation


CACHE_SCHEMA_VERSION = 1
REPRESENTATION_CONTRACT = (
    "QGeoGNNV2.extract_representation: global_add_pool(backbone node embeddings) "
    "+ frozen source condition_branch residual; pre-six-output head"
)


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_cache_path(cache_root: Path, *, column: str, source_checkpoint_sha256: str) -> Path:
    if len(source_checkpoint_sha256) != 64:
        raise ValueError("source checkpoint SHA256 must be a 64-character digest")
    return Path(cache_root) / source_checkpoint_sha256 / f"{column}_source_features.pt"


def _predict_source_features(source_model: nn.Module, atom_data: Sequence[Any], angle_data: Sequence[Any],
                             *, batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Generate source q50 and representation without accessing target labels."""

    if len(atom_data) != len(angle_data):
        raise ValueError("atom and angle data lengths differ")
    source_model.eval()
    predictions: list[np.ndarray] = []
    representations: list[np.ndarray] = []
    atom_loader = DataLoader(list(atom_data), batch_size=int(batch_size), shuffle=False)
    angle_loader = DataLoader(list(angle_data), batch_size=int(batch_size), shuffle=False)
    with torch.no_grad():
        for atom_batch, angle_batch in zip(atom_loader, angle_loader):
            output = source_model(atom_batch, angle_batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 2 or output.shape[1] != 6:
                raise ValueError("source model must emit six quantile columns")
            latent = extract_representation(source_model, atom_batch, angle_batch)
            if latent.ndim != 2 or latent.shape[0] != output.shape[0] or latent.shape[1] != 128:
                raise ValueError("source representation contract mismatch")
            predictions.append(output[:, (1, 4)].detach().cpu().numpy().astype(np.float32, copy=False))
            representations.append(latent.detach().cpu().numpy().astype(np.float32, copy=False))
    if not predictions:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)
    return np.vstack(predictions), np.vstack(representations)


def build_or_load_source_cache(
    cache_root: Path,
    *,
    column: str,
    sample_ids: Iterable[str],
    feature_table_sha256: str,
    source_checkpoint_path: Path,
    source_model: nn.Module,
    atom_data: Sequence[Any],
    angle_data: Sequence[Any],
    source_target_scales: dict[str, float],
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Load a validated cache or create an atomic label-free one.

    ``sample_ids`` must have the same deterministic order as ``atom_data``.
    The feature table digest prevents silently reusing a source feature cache
    after a target population/data-table change.
    """

    ids = [str(value) for value in sample_ids]
    if len(ids) != len(atom_data) or len(ids) != len(angle_data) or len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique and align with graph data")
    if set(source_target_scales) != {"V1", "V2"} or any(float(source_target_scales[k]) <= 0 for k in source_target_scales):
        raise ValueError("source target scales must contain positive V1 and V2 values")
    source_sha = checkpoint_sha256(Path(source_checkpoint_path))
    path = source_cache_path(cache_root, column=column, source_checkpoint_sha256=source_sha)
    expected_ids_hash = stable_hash(ids)
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        required = {"cache_schema_version", "column", "sample_ids", "sample_ids_sha256", "feature_table_sha256",
                    "source_checkpoint_sha256", "source_prediction_q50", "source_representation",
                    "source_target_scales", "representation_contract"}
        missing = required - set(cache)
        if missing:
            raise RuntimeError(f"source cache missing fields: {sorted(missing)}")
        if (cache["cache_schema_version"] != CACHE_SCHEMA_VERSION or cache["column"] != column
                or cache["sample_ids"] != ids or cache["sample_ids_sha256"] != expected_ids_hash
                or cache["feature_table_sha256"] != feature_table_sha256
                or cache["source_checkpoint_sha256"] != source_sha
                or cache["representation_contract"] != REPRESENTATION_CONTRACT
                or {name: float(value) for name, value in cache["source_target_scales"].items()}
                   != {name: float(value) for name, value in source_target_scales.items()}):
            raise RuntimeError("source cache provenance does not match the requested context")
        prediction = torch.as_tensor(cache["source_prediction_q50"])
        latent = torch.as_tensor(cache["source_representation"])
        if prediction.shape != (len(ids), 2) or latent.shape != (len(ids), 128):
            raise RuntimeError("source cache tensors do not match the target population")
        return cache
    predictions, representations = _predict_source_features(source_model, atom_data, angle_data, batch_size=batch_size)
    cache = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "column": str(column),
        "sample_ids": ids,
        "sample_ids_sha256": expected_ids_hash,
        "feature_table_sha256": str(feature_table_sha256),
        "source_checkpoint_path": str(Path(source_checkpoint_path)),
        "source_checkpoint_sha256": source_sha,
        "source_model_variant": getattr(source_model, "model_variant", type(source_model).__name__),
        "source_prediction_contract": "eval-mode source q50 columns [V1_q50,V2_q50], raw mL",
        "source_prediction_q50": torch.as_tensor(predictions),
        "source_representation": torch.as_tensor(representations),
        "representation_contract": REPRESENTATION_CONTRACT,
        "source_target_scales": {name: float(value) for name, value in source_target_scales.items()},
        "target_labels_read": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Distinct temporary names make concurrent independent context workers
    # harmless: both may compute the same cache, but neither can overwrite the
    # other's partial write.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(cache, temporary)
    temporary.replace(path)
    return cache


def attach_source_features(atom_data: Sequence[Any], cache: dict[str, Any]) -> None:
    """Attach cached, label-free source features to graph objects in place."""

    prediction = torch.as_tensor(cache["source_prediction_q50"], dtype=torch.float32)
    representation = torch.as_tensor(cache["source_representation"], dtype=torch.float32)
    if prediction.shape != (len(atom_data), 2) or representation.shape != (len(atom_data), 128):
        raise ValueError("source cache and graph data lengths/shapes differ")
    for index, atom in enumerate(atom_data):
        # Two-dimensional tensors concatenate along their first dimension in a
        # PyG Batch, yielding [batch, feature_dim] for downstream fusion.
        atom.source_prediction_q50 = prediction[index:index + 1].clone()
        atom.source_representation = representation[index:index + 1].clone()


def _batch_feature(atom_batch: Any, name: str, width: int) -> torch.Tensor:
    value = getattr(atom_batch, name, None)
    if value is None:
        raise ValueError(f"batched graph has no {name} source cache feature")
    value = torch.as_tensor(value, device=atom_batch.x.device, dtype=torch.float32)
    if value.ndim == 1 and width == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} has unexpected batched shape {tuple(value.shape)}")
    return value


class SourcePredictionFusion(nn.Module):
    """Zero-initialized residual fusion of frozen source point predictions."""

    def __init__(self, source_target_scales: dict[str, float], *, hidden_dim: int = 128):
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("the active representation width is fixed at 128")
        scales = torch.tensor([float(source_target_scales["V1"]), float(source_target_scales["V2"])], dtype=torch.float32)
        if not torch.isfinite(scales).all() or bool(torch.any(scales <= 0)):
            raise ValueError("source target scales must be finite and positive")
        self.register_buffer("source_target_scales", scales)
        self.hidden = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.output = nn.Linear(32, hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, atom_batch: Any) -> torch.Tensor:
        values = _batch_feature(atom_batch, "source_prediction_q50", 2)
        return self.output(self.hidden(values / self.source_target_scales))


class SourcePredictionAndRepresentationFusion(nn.Module):
    """Independent zero-initialized residual fusions for q50 and frozen latent."""

    def __init__(self, source_target_scales: dict[str, float], *, hidden_dim: int = 128):
        super().__init__()
        self.prediction = SourcePredictionFusion(source_target_scales, hidden_dim=hidden_dim)
        self.representation_norm = nn.LayerNorm(hidden_dim)
        self.representation_hidden = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU())
        self.representation_output = nn.Linear(32, hidden_dim)
        nn.init.zeros_(self.representation_output.weight)
        nn.init.zeros_(self.representation_output.bias)

    def forward(self, atom_batch: Any) -> torch.Tensor:
        latent = _batch_feature(atom_batch, "source_representation", 128)
        return self.prediction(atom_batch) + self.representation_output(
            self.representation_hidden(self.representation_norm(latent))
        )
