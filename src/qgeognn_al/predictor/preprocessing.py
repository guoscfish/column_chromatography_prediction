"""Frozen AL preprocessing: outer-training X, L0-only target scales."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from .data import eluent_descriptor
from .conditions import ConditionNormalization
from .training import atomic_json
from ..infra.artifacts import sha256_file
from ..infra.identity import ids_hash

def fit_al_preprocessing(
    feature_frame: pd.DataFrame,
    graph_cache: Mapping[str, Mapping[str, object]],
    outer_train_indices: Sequence[int],
    l0_ids: Sequence[str],
    l0_truth: np.ndarray,
    scaler_path: Path,
) -> tuple[ConditionNormalization, dict[str, object]]:
    """Fit label-free X preprocessing on outer train and target scales on L0."""

    positions = np.asarray(outer_train_indices, dtype=int)
    outer = feature_frame.iloc[positions]
    descriptors = np.vstack(
        [graph_cache[str(smiles)]["descriptor"] for smiles in outer.canonical_smiles]
    ).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in outer["PE/EA"]]).astype(np.float32)
    scaler = {
        "descriptor": {"min": descriptors.min(0).tolist(), "max": descriptors.max(0).tolist()},
        "eluent": {"min": eluents.min(0).tolist(), "max": eluents.max(0).tolist()},
    }
    atomic_json(scaler_path, scaler)
    amount = (
        outer["Density g/ml"].to_numpy(dtype=np.float32)
        * outer["V/ul"].to_numpy(dtype=np.float32)
    )
    volume = outer["Volume of loading solvent/ul"].to_numpy(dtype=np.float32)
    normalization = ConditionNormalization(
        loading_amount_min=float(amount.min()),
        loading_amount_max=float(amount.max()),
        loading_volume_min=float(volume.min()),
        loading_volume_max=float(volume.max()),
        fit_dataset="4g",
        fit_role="source_train",
        fit_row_count=len(outer),
        fit_ids_hash=ids_hash(outer.sample_id.astype(str)),
        eluent_scaler_sha256=sha256_file(scaler_path),
    )
    normalization.validate()
    truth = np.asarray(l0_truth, dtype=np.float32)
    if truth.shape != (len(l0_ids), 2) or not np.isfinite(truth).all():
        raise ValueError("L0 target-scale labels must align and be finite")
    scales = truth.std(axis=0, ddof=0)
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("L0 target scales must be finite and positive")
    preprocessing: dict[str, object] = {
        "scaler": scaler,
        "target_scales": {"V1": float(scales[0]), "V2": float(scales[1])},
        "fit_role": "source_train",
        "fit_rows": len(outer),
        "fit_ids_hash": ids_hash(outer.sample_id.astype(str)),
        "covariate_fit_role": "outer_train_X_only",
        "covariate_fit_rows": len(outer),
        "covariate_fit_ids_hash": ids_hash(outer.sample_id.astype(str)),
        "target_scale_fit_role": "l0_labels_only",
        "target_scale_fit_rows": len(l0_ids),
        "target_scale_fit_ids_hash": ids_hash(l0_ids),
        "validation_rows_used_for_scaling": 0,
        "u0_rows_used_for_scaling": 0,
        "test_rows_used": 0,
    }
    return normalization, preprocessing
