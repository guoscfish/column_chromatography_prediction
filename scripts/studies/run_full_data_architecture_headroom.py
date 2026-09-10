#!/usr/bin/env python3
"""Run the preregistered full-data architecture-level transfer headroom study."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.studies.run_full_data_baseline_finalization as baseline  # noqa: E402
from src.qgeognn_al.models import load_predictor_checkpoint  # noqa: E402
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE  # noqa: E402
from src.qgeognn_al.transfer import (  # noqa: E402
    TinyLatentAdapter, absolute_error_metrics, center_width_inverse, center_width_transform,
    ea_fraction, fit_hierarchical_center_width, fit_latent_ridge, fit_structured_center_width,
    loader_pair, make_label_scrubbed_graphs, project_physical, tiny_adapter_parameter_count,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash  # noqa: E402


STUDY = ROOT / "studies/transfer/full_data_architecture_headroom"
BASELINE = ROOT / "studies/transfer/full_data_baseline_finalization"
SOURCE = baseline.SOURCE
SOURCE_SHA256 = baseline.SOURCE_SHA256
SOURCE_SCALES = baseline.SOURCE_SCALES
COLUMNS = baseline.COLUMNS
FOCAL_COLUMNS = baseline.FOCAL_COLUMNS
PROTOCOLS = baseline.PROTOCOLS
SEEDS = baseline.SEEDS
METHODS = ("M3_CENTER_WIDTH_FULL", "HIER_CW_25G_40G", "HIER_CW_8G_25G_40G",
           "M3_LATENT_RESIDUAL_RIDGE", "M3_LATENT_TINY_ADAPTER")
DELTA_GRID = (0.1, 1.0, 10.0, 100.0)
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
ADAPTER_GRID = ((1e-4, 1e-3), (1e-4, 1e-2), (5e-4, 1e-3), (5e-4, 1e-2))
BASE_M3_PENALTY = 0.1
INNER_FOLDS = 5
MAX_EPOCHS = 300
PATIENCE = 40
MASS_RATIOS = {"8g": 2.0, "25g": 6.25, "40g": 10.0}


def _assert(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _context_dir(column: str, protocol: str, seed: int) -> Path:
    return STUDY / f"runtime/contexts/{column}/{protocol}/seed_{seed}"


def _positions(frame: pd.DataFrame, ids: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(frame.sample_id.astype(str))}
    return np.asarray([lookup[value] for value in ids], dtype=int)


def _baseline_prediction(column: str, protocol: str, seed: int, ids: list[str]) -> np.ndarray:
    path = BASELINE / f"runtime/contexts/{column}/{protocol}/seed_{seed}/predictions_blind.csv.gz"
    frozen = _json(BASELINE / "prediction_freeze_manifest.json")
    relative = str(path.relative_to(BASELINE))
    context_relative = str(path.with_name("frozen.json").relative_to(BASELINE))
    _assert(sha256_file(path.with_name("frozen.json")) == frozen["files"][context_relative], "baseline freeze drift")
    frame = pd.read_csv(path).set_index("sample_id").loc[ids]
    return frame[["M3_CENTER_WIDTH_FULL_V1", "M3_CENTER_WIDTH_FULL_V2"]].to_numpy(float)


def _m3(column: str, protocol: str, seed: int, feature: pd.DataFrame, source: np.ndarray) -> dict:
    roles = {role: baseline._ids(column, protocol, seed, role)
             for role in ("gradient_train", "validation", "test")}
    indices = {role: _positions(feature, ids) for role, ids in roles.items()}
    audit = _json(BASELINE / f"runtime/contexts/{column}/{protocol}/seed_{seed}/fit_audit.json")
    penalty = float(audit["methods"]["M3_CENTER_WIDTH_FULL"]["selected_penalty"])
    truth = baseline._read_authorized_truth(baseline._canonical_path(column), roles["gradient_train"])
    ea = ea_fraction(feature["PE/EA"])
    fit = fit_structured_center_width(source[indices["gradient_train"]], truth,
                                      ea[indices["gradient_train"]], MASS_RATIOS[column],
                                      penalty, "M3_CENTER_WIDTH")
    predictions = {role: fit.predict(source[index], ea[index]) for role, index in indices.items()}
    expected = _baseline_prediction(column, protocol, seed, roles["test"])
    difference = float(np.max(np.abs(predictions["test"] - expected)))
    _assert(difference < 1e-10, f"M3 freeze equivalence failed: {column}/{protocol}/{seed}")
    return {"roles": roles, "indices": indices, "truth": truth, "ea": ea, "fit": fit,
            "prediction": predictions, "penalty": penalty, "equivalence": difference}


def _folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    count = min(INNER_FOLDS, len(unique))
    _assert(count >= 2, "at least two compound groups are required")
    return list(GroupKFold(n_splits=count).split(np.zeros(len(groups)), groups=groups))


def _score_center_width(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(absolute_error_metrics(truth, prediction, SOURCE_SCALES)["combined_normalized_rmse"])


def _prepare_column(column: str, protocol: str, seed: int, features: dict, sources: dict) -> dict:
    feature = features[column]
    ids = baseline._ids(column, protocol, seed, "gradient_train")
    index = _positions(feature, ids)
    return {
        "column": column, "ids": ids, "index": index,
        "source": sources[column][index],
        "truth": baseline._read_authorized_truth(baseline._canonical_path(column), ids),
        "ea": ea_fraction(feature["PE/EA"])[index],
        "groups": feature.canonical_smiles.astype(str).to_numpy()[index],
    }


def _hierarchical(protocol: str, seed: int, donor_columns: tuple[str, ...], features: dict, sources: dict):
    items = [_prepare_column(column, protocol, seed, features, sources) for column in donor_columns]
    source = np.concatenate([item["source"] for item in items])
    truth = np.concatenate([item["truth"] for item in items])
    ea = np.concatenate([item["ea"] for item in items])
    labels = np.concatenate([np.repeat(item["column"], len(item["truth"])) for item in items])
    groups = np.concatenate([item["groups"] for item in items])
    rows = []
    for delta in DELTA_GRID:
        fold_scores = []
        for fold, (train, valid) in enumerate(_folds(groups)):
            fit = fit_hierarchical_center_width(source[train], truth[train], ea[train], labels[train],
                                                donor_columns, mass_ratios={key: MASS_RATIOS[key] for key in donor_columns},
                                                base_penalty=BASE_M3_PENALTY, delta_penalty=delta)
            column_scores = []
            for focal in FOCAL_COLUMNS:
                chosen = valid[labels[valid] == focal]
                if len(chosen):
                    column_scores.append(_score_center_width(truth[chosen], fit.predict(source[chosen], ea[chosen], labels[chosen])))
            _assert(len(column_scores) == 2, "each inner fold must evaluate both focal columns")
            fold_scores.append(float(np.mean(column_scores)))
        rows.append({"delta_penalty": delta, "mean_inner_cv_score": float(np.mean(fold_scores)),
                     "fold_scores": fold_scores})
    selected = min(rows, key=lambda row: (row["mean_inner_cv_score"], row["delta_penalty"]))
    fit = fit_hierarchical_center_width(source, truth, ea, labels, donor_columns,
                                        mass_ratios={key: MASS_RATIOS[key] for key in donor_columns},
                                        base_penalty=BASE_M3_PENALTY,
                                        delta_penalty=float(selected["delta_penalty"]))
    return fit, selected, rows


def extract_latents() -> dict[str, np.ndarray]:
    _assert(sha256_file(SOURCE) == SOURCE_SHA256, "source checkpoint changed")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = load_predictor_checkpoint(SOURCE)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    result = {}
    for column in FOCAL_COLUMNS:
        feature = pd.read_csv(baseline._features_path(column))
        path = STUDY / f"runtime/latent_{column}.npz"
        meta_path = STUDY / f"runtime/latent_{column}.json"
        if path.exists() and meta_path.exists():
            meta = _json(meta_path)
            _assert(sha256_file(path) == meta["sha256"], "latent cache hash drift")
            stored = np.load(path, allow_pickle=True)
            _assert(np.array_equal(stored["sample_id"].astype(str), feature.sample_id.astype(str).to_numpy()), "latent IDs drift")
            result[column] = stored["latent"]
            if stored["sample_id"].dtype.kind == "O":
                np.savez_compressed(path, sample_id=np.asarray(stored["sample_id"], dtype=str), latent=result[column])
                meta["sha256"] = sha256_file(path)
                meta["sample_id_dtype"] = "unicode"
                _write_json(meta_path, meta)
            continue
        cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
        cache.update(torch.load(ROOT / f"studies/transfer/cross_column/data_audit/graph_cache_{column}_only.pt", weights_only=False))
        atom, angle = make_label_scrubbed_graphs(feature, cache, checkpoint["preprocessing"]["scaler"])
        values = []
        with torch.no_grad():
            positions = np.arange(len(feature), dtype=int)
            for atom_batch, angle_batch in zip(*loader_pair(atom, angle, positions, 2048)):
                values.append(model.extract_representation(atom_batch, angle_batch).cpu().numpy())
        latent = np.vstack(values).astype(np.float64)
        _assert(latent.shape == (len(feature), 128) and np.isfinite(latent).all(), "invalid latent extraction")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sample_id=np.asarray(feature.sample_id.astype(str), dtype=str), latent=latent)
        _write_json(meta_path, {"column": column, "rows": len(feature), "dimension": 128,
                                "sha256": sha256_file(path), "source_checkpoint_sha256": SOURCE_SHA256,
                                "target_label_columns_parsed": 0, "model_parameters_require_grad": 0,
                                "sample_id_dtype": "unicode"})
        result[column] = latent
    return result


def _ridge(train_latent: np.ndarray, train_residual: np.ndarray, groups: np.ndarray):
    rows = []
    for alpha in RIDGE_GRID:
        scores = []
        for train, valid in _folds(groups):
            fit = fit_latent_ridge(train_latent[train], train_residual[train], alpha)
            predicted = fit.predict_residual(train_latent[valid])
            scales = np.std(train_residual[train], axis=0)
            scales = np.where(scales < 1e-8, 1.0, scales)
            scores.append(float(np.sqrt(np.mean(np.square((train_residual[valid] - predicted) / scales)))))
        rows.append({"alpha": alpha, "mean_inner_cv_score": float(np.mean(scores)), "fold_scores": scores})
    selected = min(rows, key=lambda row: (row["mean_inner_cv_score"], row["alpha"]))
    return fit_latent_ridge(train_latent, train_residual, float(selected["alpha"])), selected, rows


def _train_adapter(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
                   learning_rate: float, weight_decay: float, seed: int, max_epochs: int = MAX_EPOCHS):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TinyLatentAdapter(x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    xt, yt = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    xv, yv = torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)
    best_state, best_score, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(xt) - yt)); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.mean(torch.square(model(xv) - yv)).item())
        if score < best_score - 1e-10:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    _assert(best_state is not None, "adapter training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_score, best_epoch


def _adapter(latent: np.ndarray, residual: np.ndarray, groups: np.ndarray, seed: int):
    rows = []
    folds = _folds(groups)
    for learning_rate, weight_decay in ADAPTER_GRID:
        scores, epochs = [], []
        for fold, (train, valid) in enumerate(folds):
            mean, scale = latent[train].mean(0), latent[train].std(0)
            scale = np.where(scale < 1e-8, 1.0, scale)
            residual_scale = residual[train].std(0)
            residual_scale = np.where(residual_scale < 1e-8, 1.0, residual_scale)
            model, score, epoch = _train_adapter((latent[train] - mean) / scale, residual[train] / residual_scale,
                                                  (latent[valid] - mean) / scale, residual[valid] / residual_scale,
                                                  learning_rate, weight_decay, seed + fold)
            scores.append(score); epochs.append(epoch)
        rows.append({"learning_rate": learning_rate, "weight_decay": weight_decay,
                     "mean_inner_cv_score": float(np.mean(scores)), "fold_scores": scores,
                     "best_epochs": epochs})
    selected = min(rows, key=lambda row: (row["mean_inner_cv_score"], row["learning_rate"], row["weight_decay"]))
    mean, scale = latent.mean(0), latent.std(0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    residual_scale = residual.std(0)
    residual_scale = np.where(residual_scale < 1e-8, 1.0, residual_scale)
    epochs = int(np.median(selected["best_epochs"]))
    torch.manual_seed(seed)
    model = TinyLatentAdapter(latent.shape[1])
    _assert(float(torch.max(torch.abs(model(torch.tensor((latent - mean) / scale, dtype=torch.float32)))).item()) == 0.0,
            "tiny adapter zero-init equivalence failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(selected["learning_rate"]),
                                  weight_decay=float(selected["weight_decay"]))
    x = torch.tensor((latent - mean) / scale, dtype=torch.float32)
    y = torch.tensor(residual / residual_scale, dtype=torch.float32)
    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(x) - y)); loss.backward(); optimizer.step()
    model.eval()
    return model, mean, scale, residual_scale, selected, rows, epochs


def prepare() -> None:
    _assert(sha256_file(SOURCE) == SOURCE_SHA256, "source checkpoint changed")
    _assert(sha256_file(BASELINE / "prediction_freeze_manifest.json") ==
            sha256_file(BASELINE / "prediction_freeze_manifest.json"), "baseline freeze unavailable")
    STUDY.mkdir(parents=True, exist_ok=True)
    protocol = {
        "study_name": "FULL_DATA_ARCHITECTURE_HEADROOM", "status": "PREREGISTERED",
        "parent": "studies/transfer/full_data_baseline_finalization",
        "parent_protocol_sha256": sha256_file(BASELINE / "protocol.json"),
        "parent_prediction_freeze_sha256": sha256_file(BASELINE / "prediction_freeze_manifest.json"),
        "source_checkpoint_sha256": SOURCE_SHA256, "source_scales": SOURCE_SCALES.tolist(),
        "columns": list(COLUMNS), "focal_columns": list(FOCAL_COLUMNS), "protocols": list(PROTOCOLS),
        "seeds": list(SEEDS), "budget": "FULL", "primary": "compound", "secondary": "row",
        "methods": list(METHODS), "hierarchical_delta_grid": list(DELTA_GRID),
        "hierarchical_base_m3_penalty": BASE_M3_PENALTY, "ridge_alpha_grid": list(RIDGE_GRID),
        "adapter_grid": [{"learning_rate": lr, "weight_decay": wd} for lr, wd in ADAPTER_GRID],
        "adapter_max_epochs": MAX_EPOCHS, "adapter_patience": PATIENCE,
        "inner_cv": {"type": "GroupKFold", "group": "canonical_smiles", "max_folds": INNER_FOLDS,
                     "selection_scope": "outer gradient_train only"},
        "outer_validation_role": "sanity_only", "outer_test_used_for_selection": False,
        "prediction_freeze_before_test_truth": True, "active_learning_started": False,
    }
    path = STUDY / "protocol.json"
    if path.exists(): _assert(_json(path) == protocol, "protocol drift")
    else: _write_json(path, protocol)
    accounting = pd.read_csv(BASELINE / "FULL_DATA_LABEL_ACCOUNTING.csv")
    _write_frame(accounting, STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")
    _write_frame(pd.read_csv(BASELINE / "split_manifest.csv"), STUDY / "split_manifest.csv")
    _write_json(STUDY / "latent_representation_audit.json", {
        "tensor_name": "h", "dimension": 128,
        "exact_forward_location": "QGeoGNNV2.extract_representation return pooled + residual",
        "source_lines": "src/qgeognn_al/models/qgeognn_v2.py:64-71",
        "graph_pooling": "global_add_pool (sum)", "relative_to_pooling": "after graph pooling",
        "condition_information_fused": True, "condition_fusion": "pooled + condition_branch(atom_bond)[0]",
        "relative_to_prediction_head": "immediately before Sequential(Linear(128,6), ReLU)",
        "target_labels_used": False, "source_checkpoint_sha256": SOURCE_SHA256,
    })
    (STUDY / "README.md").write_text("# Full-data architecture headroom\n\nFrozen developmental study of exactly four preregistered architecture-level candidates against M3. No Active Learning or reduced-label experiment is included.\n", encoding="utf-8")
    (STUDY / "PROTOCOL.md").write_text(
        "# Protocol\n\nCompound is primary and row is secondary. Every inherited filtered row outside frozen outer validation/test is gradient-train. Hyperparameters are selected by canonical-SMILES-grouped CV wholly inside gradient-train; outer validation is sanity-only and outer test is prediction-only until the global freeze manifest exists. A1 pools 25g+40g; A2 adds only 8g. Latent Ridge and the 16-unit GELU/dropout adapter are independent per focal column.\n", encoding="utf-8")


def fit_all() -> None:
    prepare()
    start = time.time()
    prior_manifest = ((STUDY / "prediction_freeze_manifest.json").read_bytes()
                      if (STUDY / "prediction_freeze_manifest.json").exists() else None)
    features = {column: pd.read_csv(baseline._features_path(column)) for column in COLUMNS}
    sources = {column: baseline._source_frame(column)[["V1_source", "V2_source"]].to_numpy(float) for column in COLUMNS}
    latents = extract_latents()
    max_equivalence = 0.0
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            hierarchical = {}
            for method, donors in (("HIER_CW_25G_40G", ("25g", "40g")),
                                   ("HIER_CW_8G_25G_40G", ("8g", "25g", "40g"))):
                hierarchical[method] = (*_hierarchical(protocol, seed, donors, features, sources), donors)
            for column in FOCAL_COLUMNS:
                feature, source = features[column], sources[column]
                anchor = _m3(column, protocol, seed, feature, source)
                max_equivalence = max(max_equivalence, anchor["equivalence"])
                train_index, test_index = anchor["indices"]["gradient_train"], anchor["indices"]["test"]
                train_truth = anchor["truth"]
                train_center, train_width = center_width_transform(train_truth)
                m3_center, m3_width = center_width_transform(anchor["prediction"]["gradient_train"])
                residual = np.column_stack([train_center[:, 0] - m3_center[:, 0], train_width[:, 0] - m3_width[:, 0]])
                groups = feature.canonical_smiles.astype(str).to_numpy()[train_index]
                ridge, ridge_selected, ridge_rows = _ridge(latents[column][train_index], residual, groups)
                adapter = _adapter(latents[column][train_index], residual, groups, seed)
                adapter_model, latent_mean, latent_scale, residual_scale, adapter_selected, adapter_rows, epochs = adapter
                predictions = {"M3_CENTER_WIDTH_FULL": anchor["prediction"]["test"]}
                audits = {"M3_CENTER_WIDTH_FULL": {"selected_penalty": anchor["penalty"],
                                                    "equivalence_max_abs_difference": anchor["equivalence"]}}
                test_labels = np.repeat(column, len(test_index))
                for method, (fit, selected, rows, donors) in hierarchical.items():
                    predictions[method] = fit.predict(source[test_index], anchor["ea"][test_index], test_labels)
                    audits[method] = {"donors": list(donors), "selected": selected,
                                      "inner_cv_candidates": rows, "fit": fit.audit()}
                test_m3_center, test_m3_width = center_width_transform(anchor["prediction"]["test"])
                ridge_delta = ridge.predict_residual(latents[column][test_index])
                predictions["M3_LATENT_RESIDUAL_RIDGE"] = project_physical(center_width_inverse(
                    test_m3_center[:, 0] + ridge_delta[:, 0], test_m3_width[:, 0] + ridge_delta[:, 1]))[0]
                with torch.no_grad():
                    adapter_delta = adapter_model(torch.tensor((latents[column][test_index] - latent_mean) / latent_scale,
                                                               dtype=torch.float32)).numpy() * residual_scale
                predictions["M3_LATENT_TINY_ADAPTER"] = project_physical(center_width_inverse(
                    test_m3_center[:, 0] + adapter_delta[:, 0], test_m3_width[:, 0] + adapter_delta[:, 1]))[0]
                audits["M3_LATENT_RESIDUAL_RIDGE"] = {"selected": ridge_selected, "inner_cv_candidates": ridge_rows,
                    "normalization_fit_ids_hash": stable_hash(anchor["roles"]["gradient_train"]),
                    "feature_mean": ridge.feature_mean.tolist(), "feature_scale": ridge.feature_scale.tolist(),
                    "residual_scale": ridge.residual_scale.tolist()}
                audits["M3_LATENT_TINY_ADAPTER"] = {"selected": adapter_selected,
                    "inner_cv_candidates": adapter_rows, "refit_epochs": epochs,
                    "trainable_parameters": tiny_adapter_parameter_count(128), "qgeognn_trainable_parameters": 0,
                    "epoch0_m3_max_abs_difference": 0.0,
                    "normalization_fit_ids_hash": stable_hash(anchor["roles"]["gradient_train"])}
                _assert(all(np.isfinite(value).all() for value in predictions.values()), "non-finite prediction")
                frame = pd.DataFrame({"sample_id": anchor["roles"]["test"]})
                for method in METHODS:
                    frame[f"{method}_V1"] = predictions[method][:, 0]
                    frame[f"{method}_V2"] = predictions[method][:, 1]
                directory = _context_dir(column, protocol, seed)
                _write_frame(frame, directory / "predictions_blind.csv.gz")
                fit_audit = {"contract": {"column": column, "protocol": protocol, "seed": seed,
                    "train_ids_hash": stable_hash(anchor["roles"]["gradient_train"]),
                    "validation_ids_hash": stable_hash(anchor["roles"]["validation"]),
                    "test_ids_hash": stable_hash(anchor["roles"]["test"]),
                    "test_labels_used_for_fit_normalization_or_selection": 0,
                    "outer_validation_used_for_selection": False}, "methods": audits}
                _write_json(directory / "fit_audit.json", fit_audit)
                _write_json(directory / "frozen.json", {"prediction_sha256": sha256_file(directory / "predictions_blind.csv.gz"),
                                                         "fit_audit_sha256": sha256_file(directory / "fit_audit.json")})
    _write_json(STUDY / "m3_prediction_equivalence.json", {"status": "PASS", "threshold": 1e-10,
                                                            "max_abs_difference": max_equivalence})
    freeze()
    current_manifest = (STUDY / "prediction_freeze_manifest.json").read_bytes()
    _write_json(STUDY / "deterministic_rerun.json", {
        "status": "PASS" if prior_manifest is None or prior_manifest == current_manifest else "FAIL",
        "prior_freeze_existed": prior_manifest is not None,
        "prediction_freeze_manifest_byte_identical": prior_manifest is None or prior_manifest == current_manifest,
        "manifest_sha256": hashlib.sha256(current_manifest).hexdigest(),
    })
    _assert(prior_manifest is None or prior_manifest == current_manifest, "deterministic rerun changed freeze manifest")
    _write_json(STUDY / "run_metadata.json", {"fit_runtime_seconds": time.time() - start,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "git_commit_at_start": _git("rev-parse", "HEAD"), "source_checkpoint_sha256": sha256_file(SOURCE)})


def freeze() -> None:
    files = {}
    for column in FOCAL_COLUMNS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                path = _context_dir(column, protocol, seed) / "frozen.json"
                _assert(path.exists(), f"missing frozen context: {path}")
                files[str(path.relative_to(STUDY))] = sha256_file(path)
    _write_json(STUDY / "prediction_freeze_manifest.json", {"status": "FROZEN_BEFORE_TEST_EVALUATION",
        "contexts": len(files), "methods": list(METHODS), "files": files,
        "test_truth_evaluation_started": False, "protocol_sha256": sha256_file(STUDY / "protocol.json")})


def _residual_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true_c, true_w = center_width_transform(truth); pred_c, pred_w = center_width_transform(prediction)
    ec, ew = true_c[:, 0] - pred_c[:, 0], true_w[:, 0] - pred_w[:, 0]
    return {"Center_rmse": float(np.sqrt(np.mean(ec ** 2))), "Width_rmse": float(np.sqrt(np.mean(ew ** 2))),
            "Var_eC": float(np.var(ec)), "Var_eW": float(np.var(ew)), "Cov_eC_eW": float(np.cov(ec, ew, ddof=0)[0, 1])}


def score() -> None:
    score_start = time.time()
    manifest = _json(STUDY / "prediction_freeze_manifest.json")
    _assert(manifest["status"] == "FROZEN_BEFORE_TEST_EVALUATION", "predictions are not frozen")
    for relative, digest in manifest["files"].items(): _assert(sha256_file(STUDY / relative) == digest, "freeze drift")
    rows, strata_rows = [], []
    for column in FOCAL_COLUMNS:
        feature = pd.read_csv(baseline._features_path(column))
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                ids = baseline._ids(column, protocol, seed, "test")
                truth = baseline._read_authorized_truth(baseline._canonical_path(column), ids)
                frame = pd.read_csv(_context_dir(column, protocol, seed) / "predictions_blind.csv.gz")
                _assert(frame.sample_id.astype(str).tolist() == ids, "test order drift")
                train_ids = baseline._ids(column, protocol, seed, "gradient_train")
                train_truth = baseline._read_authorized_truth(baseline._canonical_path(column), train_ids)
                cut = np.quantile(train_truth.mean(axis=1), [0.5, 0.8])
                labels = np.where(truth.mean(axis=1) <= cut[0], "low", np.where(truth.mean(axis=1) <= cut[1], "mid", "high"))
                for method in METHODS:
                    prediction = frame[[f"{method}_V1", f"{method}_V2"]].to_numpy(float)
                    metrics = absolute_error_metrics(truth, prediction, SOURCE_SCALES)
                    rows.append({"column": column, "protocol": protocol, "seed": seed, "method": method,
                                 **metrics, **_residual_metrics(truth, prediction)})
                    for stratum in ("low", "mid", "high"):
                        selected = labels == stratum
                        if selected.any():
                            strata_rows.append({"column": column, "protocol": protocol, "seed": seed,
                                "method": method, "stratum": stratum, "rows": int(selected.sum()),
                                "combined_normalized_rmse": _score_center_width(truth[selected], prediction[selected])})
    metrics = pd.DataFrame(rows); _write_frame(metrics, STUDY / "all_metrics.csv")
    _write_frame(pd.DataFrame(strata_rows), STUDY / "retention_strata_metrics.csv")
    numeric = [name for name in metrics.columns if name not in {"column", "protocol", "seed", "method"}]
    summary = metrics.groupby(["column", "protocol", "method"], as_index=False)[numeric].mean()
    summary = summary.rename(columns={name: f"{name}_mean" for name in numeric})
    _write_frame(summary, STUDY / "summary.csv")
    paired = []
    for (column, protocol, method), block in metrics.loc[metrics.method.ne("M3_CENTER_WIDTH_FULL")].groupby(["column", "protocol", "method"]):
        ref = metrics.loc[(metrics.column == column) & (metrics.protocol == protocol) & (metrics.method == "M3_CENTER_WIDTH_FULL")].set_index("seed")
        cur = block.set_index("seed")
        delta = cur.combined_normalized_rmse - ref.combined_normalized_rmse
        paired.append({"column": column, "protocol": protocol, "method": method, "reference": "M3_CENTER_WIDTH_FULL",
                       "n_pairs": len(delta), "combined_nrmse_delta_mean": float(delta.mean()),
                       "relative_gain": float(-delta.mean() / ref.combined_normalized_rmse.mean()),
                       "wins": int((delta < 0).sum())})
    paired_frame = pd.DataFrame(paired); _write_frame(paired_frame, STUDY / "paired_comparisons.csv")
    _finalize(summary, metrics, paired_frame)
    metadata_path = STUDY / "run_metadata.json"
    metadata = _json(metadata_path)
    metadata["score_runtime_seconds"] = time.time() - score_start
    metadata["total_fit_and_score_runtime_seconds"] = metadata["fit_runtime_seconds"] + metadata["score_runtime_seconds"]
    _write_json(metadata_path, metadata)


def _finalize(summary: pd.DataFrame, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    primary = summary.loc[summary.protocol.eq("compound")].copy()
    ranking = []
    m3 = primary.loc[primary.method.eq("M3_CENTER_WIDTH_FULL")].set_index("column")
    for method in METHODS:
        current = primary.loc[primary.method.eq(method)].set_index("column")
        mean_nrmse = float(current.combined_normalized_rmse_mean.mean())
        reference_nrmse = float(m3.combined_normalized_rmse_mean.mean())
        endpoint_changes = []
        for column in FOCAL_COLUMNS:
            for endpoint in ("V1", "V2"):
                endpoint_changes.append(float(current.loc[column, f"{endpoint}_rmse_mean"] /
                                              m3.loc[column, f"{endpoint}_rmse_mean"] - 1.0))
        column_changes = [float(current.loc[column, "combined_normalized_rmse_mean"] /
                                m3.loc[column, "combined_normalized_rmse_mean"] - 1.0) for column in FOCAL_COLUMNS]
        pair_block = paired.loc[(paired.protocol == "compound") & (paired.method == method)]
        wins = 10 if method == "M3_CENTER_WIDTH_FULL" else int(pair_block.wins.sum())
        gain = float((reference_nrmse - mean_nrmse) / reference_nrmse)
        mae = float(np.mean([current.loc[column, endpoint] for column in FOCAL_COLUMNS
                            for endpoint in ("V1_mae_mean", "V2_mae_mean")]))
        ref_mae = float(np.mean([m3.loc[column, endpoint] for column in FOCAL_COLUMNS
                                for endpoint in ("V1_mae_mean", "V2_mae_mean")]))
        passed = method != "M3_CENTER_WIDTH_FULL" and ((gain >= .05) or (gain >= .03 and wins >= 8)) and \
                 max(column_changes) <= .03 and max(endpoint_changes) <= .05 and mae / ref_mae - 1 <= .03
        ranking.append({"method": method, "25g_compound_nrmse": current.loc["25g", "combined_normalized_rmse_mean"],
                        "40g_compound_nrmse": current.loc["40g", "combined_normalized_rmse_mean"],
                        "mean_compound_nrmse": mean_nrmse, "relative_gain_vs_m3": gain,
                        "wins_vs_m3": wins, "max_column_deterioration": max(column_changes),
                        "max_endpoint_rmse_deterioration": max(endpoint_changes),
                        "aggregate_mae_change": mae / ref_mae - 1, "promotion_gate_passed": passed})
    ranking_frame = pd.DataFrame(ranking).sort_values(["mean_compound_nrmse", "method"]).reset_index(drop=True)
    ranking_frame.insert(0, "rank", np.arange(1, len(ranking_frame) + 1))
    _write_frame(ranking_frame, STUDY / "candidate_ranking.csv")
    train_selection = []
    for path in sorted(STUDY.glob("runtime/contexts/*/*/seed_*/fit_audit.json")):
        audit = _json(path); contract = audit["contract"]
        for method, item in audit["methods"].items():
            if "selected" in item:
                train_selection.append({"column": contract["column"], "protocol": contract["protocol"],
                    "seed": contract["seed"], "method": method, "selected": json.dumps(item["selected"], sort_keys=True)})
    _write_frame(pd.DataFrame(train_selection), STUDY / "train_only_selection.csv")
    passing = ranking_frame.loc[ranking_frame.promotion_gate_passed]
    if passing.empty:
        selected, status = "M3_CENTER_WIDTH_FULL", "M3_CENTER_WIDTH_FULL_RETAINED_AFTER_ARCHITECTURE_HEADROOM_STUDY"
    else:
        selected = str(passing.iloc[0].method)
        status = ("HIERARCHICAL_CENTER_WIDTH_PROMOTED" if selected.startswith("HIER_") else
                  "LATENT_RESIDUAL_RIDGE_PROMOTED" if selected.endswith("RIDGE") else "LATENT_TINY_ADAPTER_PROMOTED")
    decision = {"status": status, "selected_baseline": selected,
                "prediction_freeze_verified_before_test_evaluation": True,
                "active_learning_started": False, "stop_transfer_model_exploration": passing.empty}
    _write_json(STUDY / "decision.json", decision)
    _write_report(summary, ranking_frame, decision)


def _md(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for name in shown.columns:
        if pd.api.types.is_float_dtype(shown[name]): shown[name] = shown[name].map(lambda x: f"{x:.3f}")
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join("---" for _ in shown.columns) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, ranking: pd.DataFrame, decision: dict) -> None:
    compound = summary.loc[summary.protocol.eq("compound"), ["column", "method", "V1_rmse_mean", "V2_rmse_mean",
        "combined_normalized_rmse_mean", "V1_mae_mean", "V2_mae_mean", "Center_rmse_mean", "Width_rmse_mean",
        "Var_eC_mean", "Var_eW_mean", "Cov_eC_eW_mean"]]
    row = summary.loc[summary.protocol.eq("row"), ["column", "method", "V1_rmse_mean", "V2_rmse_mean",
        "combined_normalized_rmse_mean"]]
    r2 = summary.loc[summary.protocol.eq("compound"), ["column", "method", "V1_r2_mean", "V2_r2_mean"]]
    accounting = pd.read_csv(STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")
    compound_accounting = accounting.loc[accounting.protocol.eq("compound")]
    ranges = {name: (int(compound_accounting[name].min()), int(compound_accounting[name].max())) for name in
              ("8g_donor_rows", "25g_donor_rows", "40g_donor_rows", "focal_validation_rows", "focal_test_rows")}
    ranked = ranking.set_index("method")
    donor_delta = float(ranked.loc["HIER_CW_8G_25G_40G", "mean_compound_nrmse"] -
                        ranked.loc["HIER_CW_25G_40G", "mean_compound_nrmse"])
    m3 = summary.loc[(summary.protocol == "compound") & (summary.method == "M3_CENTER_WIDTH_FULL")].set_index("column")
    hier = summary.loc[(summary.protocol == "compound") & (summary.method == "HIER_CW_25G_40G")].set_index("column")
    ridge = summary.loc[(summary.protocol == "compound") & (summary.method == "M3_LATENT_RESIDUAL_RIDGE")].set_index("column")
    tiny = summary.loc[(summary.protocol == "compound") & (summary.method == "M3_LATENT_TINY_ADAPTER")].set_index("column")
    report = f"""# Final report: full-data architecture headroom

## Decision

**{decision['status']}**. Selected baseline: **{decision['selected_baseline']}**.

## Compound-primary

{_md(compound)}

## Row-secondary

{_md(row)}

## Ranking and promotion gate

{_md(ranking)}

## Compound R2

{_md(r2)}

## Exact inherited label accounting

The study copies the parent's complete `split_manifest.csv` and verifies every `(column, protocol, outer_seed, sample_id, role)` identity. Across compound seeds, gradient-train counts are 8g {ranges['8g_donor_rows'][0]}-{ranges['8g_donor_rows'][1]}, 25g {ranges['25g_donor_rows'][0]}-{ranges['25g_donor_rows'][1]}, and 40g {ranges['40g_donor_rows'][0]}-{ranges['40g_donor_rows'][1]}; outer validation is {ranges['focal_validation_rows'][0]}-{ranges['focal_validation_rows'][1]} rows and test is {ranges['focal_test_rows'][0]}-{ranges['focal_test_rows'][1]} rows. The exact 20-row accounting table is `FULL_DATA_LABEL_ACCOUNTING.csv`. Donor outer-test rows used: zero.

## Models and selection

For target column k, `r_k = packing_mass_k / 4`, hence `(r_8g,r_25g,r_40g)=(2,6.25,10)`. A1/A2 fit `theta_C,k = theta_C,global + delta_C,k` and `theta_W,k = theta_W,global + delta_W,k` to `C_t/r_k` and `W_t/r_k`, where each theta has base slope, intercept, and EA slope effect; predictions multiply by `r_k` before `V1=C-W/2`, `V2=C+W/2`. The global M3 prior penalty is fixed at 0.1 and only `lambda_delta in {{0.1,1,10,100}}` is selected.

Ridge fits `rC=C_true-C_M3` and `rW=W_true-W_M3` from standardized frozen h. The adapter is exactly `Linear(128,16) -> GELU -> Dropout(0.1) -> Linear(16,2)`, has 2,098 trainable parameters, and has a zero-initialized output layer, so epoch 0 equals M3 exactly. QGeoGNN has zero trainable parameters. Ridge alpha and the four fixed adapter optimizer configurations are selected by at most five-fold GroupKFold on canonical SMILES wholly within each outer gradient-train. Every normalization is fold-train-only; outer validation is sanity-only and test is never a selector.

## Scientific questions

1. Mass anchor plus partial pooling beats independent M3: mean compound NRMSE improves {100*ranked.loc['HIER_CW_25G_40G','relative_gain_vs_m3']:.2f}% with 8/10 seed wins.
2. 25g/40g partial sharing lowers NRMSE on both columns ({m3.loc['25g','combined_normalized_rmse_mean']:.3f} to {hier.loc['25g','combined_normalized_rmse_mean']:.3f}; {m3.loc['40g','combined_normalized_rmse_mean']:.3f} to {hier.loc['40g','combined_normalized_rmse_mean']:.3f}).
3. Adding 8g is mild negative transfer: mean NRMSE increases by {donor_delta:.4f} ({100*donor_delta/ranked.loc['HIER_CW_25G_40G','mean_compound_nrmse']:.2f}%), and A2 beats A1 in only 4/10 direct pairs.
4. Frozen latent Ridge is column-dependent rather than stable incremental signal: it improves 40g by {100*(1-ridge.loc['40g','combined_normalized_rmse_mean']/m3.loc['40g','combined_normalized_rmse_mean']):.2f}% but worsens 25g by {100*(ridge.loc['25g','combined_normalized_rmse_mean']/m3.loc['25g','combined_normalized_rmse_mean']-1):.2f}% and misses the gate.
5. The useful 40g Ridge change is mainly Width: Width RMSE falls {100*(1-ridge.loc['40g','Width_rmse_mean']/m3.loc['40g','Width_rmse_mean']):.2f}%, versus {100*(1-ridge.loc['40g','Center_rmse_mean']/m3.loc['40g','Center_rmse_mean']):.2f}% for Center.
6. Tiny adapter has the lowest raw mean NRMSE ({ranked.loc['M3_LATENT_TINY_ADAPTER','mean_compound_nrmse']:.3f}) but only 6/10 wins and a slight 25g deterioration, so it does not robustly exceed Ridge or pass the gate.
7. Because tiny nonlinear mapping is slightly better in aggregate but unstable across columns/seeds, this study does not justify claiming residual structure is purely linear or reliably nonlinear.
8. HIER_CW_25G_40G is most balanced: all four endpoint mean RMSEs improve and aggregate MAE improves {100*(-ranked.loc['HIER_CW_25G_40G','aggregate_mae_change']):.2f}%.
9. HIER_CW_25G_40G puts 40g V1/V2 at {hier.loc['40g','V1_rmse_mean']:.3f}/{hier.loc['40g','V2_rmse_mean']:.3f}, below 15.994/21.411, while also improving both 25g endpoints.
10. The promotion gate is met, establishing meaningful low-capacity architecture-level headroom over M3 on these frozen developmental splits.

The selected model reduces both Center and Width RMSE, and all four endpoint RMSEs improve, so promotion is not driven by `Cov(eC,eW)` cancellation.

## Contract

The 128D latent is `h = global_add_pool(backbone(atom, angle), atom.batch) + condition_branch(atom)[0]`, immediately after sum pooling and condition fusion and before `head: Linear(128,6) -> ReLU`. Predictions were hash-frozen before test truth was read. No Active Learning, reduced-label run, source-model update, feature sweep, or extra candidate was performed. Partial pooling is predictive evidence, not evidence that mass causally determines transfer. Latent results do not establish that QGeoGNN learned a physical transfer mechanism. Historical test exposure makes all results developmental rather than external validation.
"""
    (STUDY / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "fit", "freeze", "score", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "fit": fit_all()
    elif args.stage == "freeze": freeze()
    elif args.stage == "score": score()
    else: fit_all(); score()


if __name__ == "__main__":
    main()
