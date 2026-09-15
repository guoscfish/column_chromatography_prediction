"""Training helpers that keep hidden labels out of V2 active learning."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..data import build_model_data, eluent_descriptor
from ..models import build_predictor, load_predictor_checkpoint
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json, loader_pair, seed_everything, train_source
from .gradient_features import state_dict_hash
from .cache import array_hash
from .protocol import ids_hash


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


def make_label_scrubbed_graphs(
    feature_frame: pd.DataFrame,
    graph_cache: Mapping[str, Mapping[str, object]],
    scaler: Mapping[str, object],
) -> tuple[list[object], list[object]]:
    scrubbed = feature_frame.copy().reset_index(drop=True)
    scrubbed["V1_ml"] = 0.0
    scrubbed["V2_ml"] = 0.0
    return build_model_data(scrubbed, dict(graph_cache), pd.DataFrame(), dict(scaler))


def set_revealed_labels(
    atom_base: Sequence[object], indices: Sequence[int], truth: np.ndarray
) -> list[object]:
    result = list(atom_base)
    positions = [int(value) for value in indices]
    values = np.asarray(truth, dtype=np.float32)
    if values.shape != (len(positions), 2) or not np.isfinite(values).all():
        raise ValueError("revealed targets must align with graph positions")
    for position, target in zip(positions, values):
        item = result[position].clone()
        item.y = torch.tensor([target], dtype=torch.float32)
        result[position] = item
    return result


def predict_outputs(
    model: torch.nn.Module,
    atom: Sequence[object],
    angle: Sequence[object],
    indices: Sequence[int],
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    output: list[np.ndarray] = []
    order: list[np.ndarray] = []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, indices, batch_size)):
            prediction = model(atom_batch, angle_batch)
            if prediction.ndim != 2 or prediction.shape[1] != 6:
                raise ValueError("QGeoGNN-V2 prediction must have six outputs")
            output.append(prediction.detach().cpu().numpy())
            order.append(atom_batch.canonical_position.detach().cpu().numpy().reshape(-1))
    if not output:
        return np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=int)
    values = np.vstack(output)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite frozen prediction")
    return values, np.concatenate(order).astype(int)


def write_predictions(
    path: Path,
    sample_ids: Sequence[str],
    predictions: np.ndarray,
) -> str:
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(sample_ids), 6):
        raise ValueError("prediction table must align with six-output model contract")
    frame = pd.DataFrame(
        values,
        columns=[f"{target}_{quantile}" for target in ("V1", "V2") for quantile in ("q10", "q50", "q90")],
    )
    frame.insert(0, "sample_id", [str(value) for value in sample_ids])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return sha256_file(path)


def fit_from_same_initialization(
    *,
    atom_base: Sequence[object],
    angle: Sequence[object],
    normalization: ConditionNormalization,
    preprocessing: Mapping[str, object],
    train_indices: Sequence[int],
    train_truth: np.ndarray,
    validation_indices: Sequence[int],
    validation_truth: np.ndarray,
    prediction_indices: Sequence[int],
    prediction_sample_ids: Sequence[str],
    initialization_seed: int,
    training_config: Mapping[str, object],
    contract: Mapping[str, object],
    runtime: Path,
) -> dict[str, object]:
    """Train one arm from the frozen initialization, with no test-truth input."""

    runtime = Path(runtime)
    checkpoint_path = runtime / "best.pt"
    prediction_path = runtime / "predictions.csv.gz"
    audit_path = runtime / "fit_audit.json"
    runtime.mkdir(parents=True, exist_ok=True)

    seed_everything(int(initialization_seed))
    initial_model = build_predictor(normalization)
    initialization_hash = state_dict_hash(initial_model)
    expected = {
        **dict(contract),
        "initialization_seed": int(initialization_seed),
        "initialization_hash": initialization_hash,
        "train_ids_hash": ids_hash(str(value) for value in contract["train_sample_ids"]),
        "validation_ids_hash": ids_hash(str(value) for value in contract["validation_sample_ids"]),
        "training_config": dict(training_config),
        "preprocessing": dict(preprocessing),
        "normalization": asdict(normalization),
        "train_truth_hash": array_hash(np.asarray(train_truth, dtype=np.float32)),
        "validation_truth_hash": array_hash(np.asarray(validation_truth, dtype=np.float32)),
        "ordered_train_ids": list(contract["train_sample_ids"]),
        "ordered_validation_ids": list(contract["validation_sample_ids"]),
        "ordered_train_indices": [int(i) for i in train_indices],
        "ordered_validation_indices": [int(i) for i in validation_indices],
        "prediction_sample_ids": list(prediction_sample_ids),
        "prediction_indices": [int(i) for i in prediction_indices],
    }
    expected_hash = hashlib_sha(expected)
    if audit_path.exists() and checkpoint_path.exists() and prediction_path.exists():
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("fit_contract_hash") != expected_hash
            or audit.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or audit.get("prediction_sha256") != sha256_file(prediction_path)
            or audit.get("initialization_hash") != initialization_hash
        ):
            raise RuntimeError(f"refusing incompatible completed fit: {runtime}")
        return audit

    atom = set_revealed_labels(atom_base, train_indices, train_truth)
    atom = set_revealed_labels(atom, validation_indices, validation_truth)
    started = time.perf_counter()
    history, best_epoch = train_source(
        initial_model,
        atom,
        angle,
        np.asarray(train_indices, dtype=int),
        np.asarray(validation_indices, dtype=int),
        dict(preprocessing),
        dict(training_config),
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(runtime / "history.csv", index=False)
    model = load_predictor_checkpoint(checkpoint_path)
    prediction, canonical_order = predict_outputs(
        model, atom_base, angle, prediction_indices, int(training_config["batch_size"])
    )
    expected_order = np.asarray(prediction_indices, dtype=int)
    if not np.array_equal(canonical_order, expected_order):
        raise RuntimeError("frozen prediction order changed")
    prediction_hash = write_predictions(prediction_path, prediction_sample_ids, prediction)
    audit: dict[str, object] = {
        "fit_contract_hash": expected_hash,
        "initialization_seed": int(initialization_seed),
        "initialization_hash": initialization_hash,
        "train_ids_hash": expected["train_ids_hash"],
        "validation_ids_hash": expected["validation_ids_hash"],
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "best_validation_combined_normalized_rmse": float(
            min(row["validation_selection_score"] for row in history)
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_state_hash": state_dict_hash(model),
        "prediction_sha256": prediction_hash,
        "prediction_rows": len(prediction_indices),
        "training_seconds": time.perf_counter() - started,
        "test_labels_used_for_fit_or_checkpoint_selection": 0,
        "test_graph_label_state": "zero_sentinel",
        "model_variant": "qgeognn_v2",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_json(audit_path, audit)
    return audit


def hashlib_sha(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
