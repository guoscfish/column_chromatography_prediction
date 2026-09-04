#!/usr/bin/env python3
"""Run the frozen E0-split point-predictor regression ladder."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.studies.run_clean_4g_baseline_qualification import fit_molecular_scaler
from src.qgeognn_al.condition_complete_v2 import (
    ConditionCompleteQGeoGNNV2,
    fit_condition_normalization,
)
from src.qgeognn_al.data import build_model_data, qg
from src.qgeognn_al.model import build_model as build_legacy_model
from src.qgeognn_al.model import quantile_target_loss
from src.qgeognn_al.models.clean_fusion import build_clean_model
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE, SOURCE_SCALER
from src.qgeognn_al.schemas.clean import (
    CleanConditionBatch,
    fit_clean_condition_normalization,
    parse_clean_conditions,
)


STUDY = ROOT / "studies/predictor/performance_regression_audit"
RESULTS = STUDY / "results"
RUNTIME = STUDY / "runtime"
FROZEN_SPLIT = ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
HISTORICAL_METRICS = ROOT / "experiments/e0_4g_baseline/metrics.json"
EXPECTED_SPLIT_SHA = "9a758e115c63cc9de2491d483b224d2e4c4b88fd6aadabaf7a45d4e73263b198"
EXPECTED_COUNTS = {"train": 3330, "validation": 416, "test": 417}
VARIANTS = (
    "R0_LEGACY_E0_EXACT",
    "R1_LEGACY_CLEAN_TRAINING_PROTOCOL",
    "R2_CONDITION_COMPLETE_V2",
    "R3_CLEAN_CURRENT",
)
R0_TOLERANCE = 0.03
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    if sha256_file(FROZEN_SPLIT) != EXPECTED_SPLIT_SHA:
        raise RuntimeError("historical E0 split hash mismatch; refusing to regenerate")
    data = pd.read_csv(SOURCE_DATA).reset_index(drop=True)
    split = pd.read_csv(FROZEN_SPLIT)
    split["split"] = split["split"].replace({"valid": "validation"})
    if len(data) != 4163 or len(split) != 4163:
        raise RuntimeError("historical E0 domain must contain exactly 4163 rows")
    if data["sample_id"].duplicated().any() or split["sample_id"].duplicated().any():
        raise RuntimeError("sample IDs must be unique")
    if set(data["sample_id"].astype(str)) != set(split["sample_id"].astype(str)):
        raise RuntimeError("historical split does not exactly cover canonical E0 sample IDs")
    if float(data["V1_ml"].max()) > 60 or float(data["V2_ml"].max()) > 120:
        raise RuntimeError("historical E0 threshold domain mismatch")
    joined = data[["sample_id"]].merge(
        split[["sample_id", "source_row_1based", "canonical_index", "split"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if joined["split"].isna().any() or not np.array_equal(
        joined["canonical_index"].to_numpy(dtype=int), np.arange(len(data))
    ):
        raise RuntimeError("historical split ordering/index mapping mismatch")
    indices = {
        role: joined.index[joined["split"].eq(role)].to_numpy(dtype=int)
        for role in EXPECTED_COUNTS
    }
    counts = {role: len(values) for role, values in indices.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"historical E0 split counts mismatch: {counts}")
    return data, split, indices


def fit_inputs(data: pd.DataFrame, split: pd.DataFrame, indices: dict[str, np.ndarray]):
    graph_cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    scaler = fit_molecular_scaler(data, indices["train"], graph_cache)
    historical_scaler = json.loads(SOURCE_SCALER.read_text(encoding="utf-8"))
    scaler_matches_historical = scaler == {
        "descriptor": historical_scaler["descriptor"], "eluent": historical_scaler["eluent"]
    }
    clean_normalization = fit_clean_condition_normalization(
        data, split[["sample_id", "split"]].assign(
            split=lambda frame: frame["split"].replace({"validation": "valid"})
        )
    )
    v2_normalization = fit_condition_normalization(
        data,
        split[["sample_id", "split"]].assign(
            split=lambda frame: frame["split"].replace({"validation": "valid"})
        ),
        SOURCE_SCALER,
    )
    atom_data, angle_data = build_model_data(data, graph_cache, pd.DataFrame(), scaler)
    labels = data.iloc[indices["train"]][["V1_ml", "V2_ml"]]
    scales = {
        "V1": max(float(labels["V1_ml"].std(ddof=0)), 1e-8),
        "V2": max(float(labels["V2_ml"].std(ddof=0)), 1e-8),
    }
    return atom_data, angle_data, scaler, scaler_matches_historical, clean_normalization, v2_normalization, scales


def variant_config(variant: str) -> dict[str, Any]:
    clean_protocol = variant != "R0_LEGACY_E0_EXACT"
    architecture = {
        "R0_LEGACY_E0_EXACT": "legacy_qgeognn_executed_path",
        "R1_LEGACY_CLEAN_TRAINING_PROTOCOL": "legacy_qgeognn_executed_path",
        "R2_CONDITION_COMPLETE_V2": "qgeognn_condition_complete_v2",
        "R3_CLEAN_CURRENT": "qgeognn_clean_fusion_v1_preflight_revision_2",
    }[variant]
    config = {
        "scientific_role": "DEVELOPMENTAL_REGRESSION_DIAGNOSTIC",
        "variant": variant,
        "architecture": architecture,
        "seed": SEED,
        "split_path": str(FROZEN_SPLIT.relative_to(ROOT)),
        "split_sha256": EXPECTED_SPLIT_SHA,
        "split_generation_called": False,
        "source_data": str(SOURCE_DATA.relative_to(ROOT)),
        "source_data_sha256": sha256_file(SOURCE_DATA),
        "graph_cache": str(SOURCE_GRAPH_CACHE.relative_to(ROOT)),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0 if clean_protocol else 1e-5,
        "batch_size": 2048,
        "maximum_epochs": 1000,
        "patience": 100,
        "shuffle": "deterministic_each_epoch" if clean_protocol else False,
        "loss_weights": {"V1": 1.0, "V2": 1.0 if clean_protocol else 0.5},
        "checkpoint_selection": (
            "minimum_validation_combined_normalized_RMSE" if clean_protocol
            else "minimum_validation_V1_RMSE_squared_plus_0.5_V2_RMSE_squared"
        ),
        "normalization_fit_role": "train_only",
        "validation_rows_used_for_normalization": 0,
        "test_rows_used_for_normalization": 0,
        "8g_rows_used": 0,
        "test_during_training": False,
    }
    config["config_hash"] = stable_hash(config)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_variant(variant: str, clean_normalization, v2_normalization, device: torch.device):
    seed_everything(SEED)
    if variant in {"R0_LEGACY_E0_EXACT", "R1_LEGACY_CLEAN_TRAINING_PROTOCOL"}:
        return build_legacy_model(device)
    if variant == "R2_CONDITION_COMPLETE_V2":
        legacy = build_legacy_model(device)
        return ConditionCompleteQGeoGNNV2(legacy, v2_normalization).to(device)
    if variant == "R3_CLEAN_CURRENT":
        return build_clean_model(clean_normalization, device)
    raise ValueError(variant)


def loader_pair(atom_data: list, angle_data: list, indices: Iterable[int]):
    positions = [int(value) for value in indices]
    return (
        DataLoader([atom_data[index] for index in positions], batch_size=2048, shuffle=False),
        DataLoader([angle_data[index] for index in positions], batch_size=2048, shuffle=False),
    )


def clean_conditions(data: pd.DataFrame, positions: np.ndarray, normalization, device: torch.device):
    rows = data.iloc[positions]
    return parse_clean_conditions(
        rows, normalization, sample_ids=tuple(rows["sample_id"].astype(str))
    ).to(device)


def forward_prediction(model, variant: str, atom_batch, angle_batch, data, normalization, device):
    if variant == "R3_CLEAN_CURRENT":
        positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
        return model(atom_batch, angle_batch, clean_conditions(data, positions, normalization, device))
    return model(atom_batch, angle_batch)[0]


def metrics(y_true: np.ndarray, prediction: np.ndarray, scales: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, target in enumerate(("V1", "V2")):
        residual = y_true[:, index] - prediction[:, index * 3 + 1]
        total = np.square(y_true[:, index] - y_true[:, index].mean()).sum()
        result[f"{target}_rmse"] = float(np.sqrt(np.square(residual).mean()))
        result[f"{target}_mae"] = float(np.abs(residual).mean())
        result[f"{target}_r2"] = float(1 - np.square(residual).sum() / total)
    result["combined_normalized_rmse"] = float(math.sqrt(0.5 * sum(
        (result[f"{target}_rmse"] / scales[target]) ** 2 for target in ("V1", "V2")
    )))
    result["all_outputs_finite"] = bool(np.isfinite(prediction).all())
    return result


def evaluate(model, variant, data, atom_data, angle_data, indices, normalization, scales, device):
    model.eval()
    truths, predictions, positions = [], [], []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loader_pair(atom_data, angle_data, indices)):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            batch_positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
            prediction = forward_prediction(
                model, variant, atom_batch, angle_batch, data, normalization, device
            )
            truths.append(atom_batch.y.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            positions.append(batch_positions)
    truth = np.vstack(truths)
    prediction = np.vstack(predictions)
    return metrics(truth, prediction, scales), truth, prediction, np.concatenate(positions)


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        "min": float(array.min()), "max": float(array.max()),
    }


def clean_extra_diagnostics(model, data, atom_data, angle_data, indices, normalization, scales, device):
    result: dict[str, Any] = {"by_split": {}}
    for role, role_indices in indices.items():
        truth_chunks, prediction_chunks, raw_chunks = [], [], []
        pooled_chunks, pre_norm_chunks, molecular_chunks, condition_chunks = [], [], [], []
        with torch.no_grad():
            model.eval()
            for atom_batch, angle_batch in zip(*loader_pair(atom_data, angle_data, role_indices)):
                atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
                positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
                conditions = clean_conditions(data, positions, normalization, device)
                nodes = model.molecular_backbone(atom_batch, angle_batch)
                pooled = global_add_pool(nodes, atom_batch.batch)
                pre_norm = model.molecular_projection[0](pooled)
                molecular = model.molecular_projection[1](pre_norm)
                condition = model.condition_encoder(conditions)
                fused = torch.cat([molecular, condition], dim=1)
                raw = model.quantile_head.linear(fused)
                ordered = model.quantile_head(fused)
                prediction = torch.clamp(ordered, min=0, max=1e8)
                truth_chunks.append(atom_batch.y.cpu().numpy())
                prediction_chunks.append(prediction.cpu().numpy())
                raw_chunks.append(raw.cpu().numpy())
                pooled_chunks.append(pooled.norm(dim=1).cpu().numpy())
                pre_norm_chunks.append(pre_norm.norm(dim=1).cpu().numpy())
                molecular_chunks.append(molecular.norm(dim=1).cpu().numpy())
                condition_chunks.append(condition.norm(dim=1).cpu().numpy())
        truth, prediction, raw = np.vstack(truth_chunks), np.vstack(prediction_chunks), np.vstack(raw_chunks)
        split_result: dict[str, Any] = {
            "representation_l2_mean": {
                "pooled_molecular_128d_before_projection": float(np.concatenate(pooled_chunks).mean()),
                "molecular_64d_after_projection_before_layernorm": float(np.concatenate(pre_norm_chunks).mean()),
                "molecular_64d_after_layernorm": float(np.concatenate(molecular_chunks).mean()),
                "condition_64d_after_layernorm": float(np.concatenate(condition_chunks).mean()),
            },
            "clamp": {
                "all_quantiles_clamped_to_zero_rate": float((prediction == 0).mean()),
                "V1_q50_clamped_to_zero_rate": float((prediction[:, 1] == 0).mean()),
                "V2_q50_clamped_to_zero_rate": float((prediction[:, 4] == 0).mean()),
            },
            "targets": {},
        }
        for target_index, target in enumerate(("V1", "V2")):
            target_truth = truth[:, target_index]
            q50 = prediction[:, target_index * 3 + 1]
            raw_median = raw[:, target_index * 3]
            std_ratio = float(q50.std(ddof=0) / max(target_truth.std(ddof=0), 1e-8))
            split_result["targets"][target] = {
                "truth": distribution(target_truth),
                "prediction_q50": distribution(q50),
                "raw_median_softplus_input": distribution(raw_median),
                "q50_near_zero_rate_threshold_1e-6": float((q50 <= 1e-6).mean()),
                "prediction_to_truth_std_ratio": std_ratio,
                "extremely_low_variance_flag_std_ratio_below_0_1": bool(std_ratio < 0.1),
            }
        result["by_split"][role] = split_result

    full, _, _, _ = evaluate(model, "R3_CLEAN_CURRENT", data, atom_data, angle_data, indices["test"], normalization, scales, device)
    mode_results = {"full": full}
    for mode in ("condition_disabled", "condition_only"):
        model.eval()
        truths, predictions = [], []
        with torch.no_grad():
            for atom_batch, angle_batch in zip(*loader_pair(atom_data, angle_data, indices["test"])):
                atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
                positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
                conditions = clean_conditions(data, positions, normalization, device)
                predictions.append(model(atom_batch, angle_batch, conditions, mode=mode).cpu().numpy())
                truths.append(atom_batch.y.cpu().numpy())
        candidate = metrics(np.vstack(truths), np.vstack(predictions), scales)
        mode_results[mode] = {**candidate, **{
            f"delta_{key}": candidate[key] - full[key]
            for key in ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2")
        }}
    permutation = np.random.default_rng(900042).permutation(len(indices["test"]))
    test_positions = indices["test"]
    condition_frame = data.iloc[test_positions[permutation]]
    condition_batch = parse_clean_conditions(
        condition_frame, normalization,
        sample_ids=tuple(condition_frame["sample_id"].astype(str)),
    ).to(device)
    atom_loader, angle_loader = loader_pair(atom_data, angle_data, test_positions)
    atom_batch, angle_batch = next(iter(atom_loader)).to(device), next(iter(angle_loader)).to(device)
    with torch.no_grad():
        permuted_prediction = model(atom_batch, angle_batch, condition_batch).cpu().numpy()
    truth = atom_batch.y.cpu().numpy()
    permuted = metrics(truth, permuted_prediction, scales)
    mode_results["condition_permuted"] = {**permuted, **{
        f"delta_{key}": permuted[key] - full[key]
        for key in ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2")
    }}
    result["condition_diagnostics_test"] = mode_results
    return result


def prediction_rows(data, role, positions, truth, prediction):
    source = data.iloc[positions].reset_index(drop=True)
    for index in range(len(source)):
        yield {
            "split": role, "sample_id": source.iloc[index]["sample_id"],
            "source_row_1based": int(source.iloc[index]["source_row_1based"]),
            "V1_true": truth[index, 0], "V2_true": truth[index, 1],
            "V1_q10": prediction[index, 0], "V1_q50": prediction[index, 1], "V1_q90": prediction[index, 2],
            "V2_q10": prediction[index, 3], "V2_q50": prediction[index, 4], "V2_q90": prediction[index, 5],
        }


def run_variant(variant: str, device: torch.device) -> dict[str, Any]:
    data, split, indices = load_frozen_inputs()
    atom_data, angle_data, scaler, scaler_match, clean_norm, v2_norm, scales = fit_inputs(data, split, indices)
    config = variant_config(variant)
    model = build_variant(variant, clean_norm, v2_norm, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    run_dir = RESULTS / variant
    runtime_dir = RUNTIME / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)
    history, best_score, best_epoch, stale = [], float("inf"), None, 0
    started = time.time()
    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        order = indices["train"]
        if config["shuffle"]:
            order = np.random.default_rng(SEED * 10000 + epoch).permutation(order)
        losses = []
        for atom_batch, angle_batch in zip(*loader_pair(atom_data, angle_data, order)):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            prediction = forward_prediction(
                model, variant, atom_batch, angle_batch, data, clean_norm, device
            )
            loss_v1 = quantile_target_loss(atom_batch.y[:, 0], prediction[:, 0:3])
            loss_v2 = quantile_target_loss(atom_batch.y[:, 1], prediction[:, 3:6])
            loss = loss_v1 + config["loss_weights"]["V2"] * loss_v2
            if not torch.isfinite(loss):
                raise RuntimeError(f"{variant} non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation, _, _, _ = evaluate(
            model, variant, data, atom_data, angle_data, indices["validation"], clean_norm, scales, device
        )
        score = (
            validation["V1_rmse"] ** 2 + 0.5 * validation["V2_rmse"] ** 2
            if variant == "R0_LEGACY_E0_EXACT" else validation["combined_normalized_rmse"]
        )
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "validation_selection_score": score,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "validation_selection_score": score, "config_hash": config["config_hash"],
                "split_sha256": EXPECTED_SPLIT_SHA,
            }, runtime_dir / "best.pt")
        else:
            stale += 1
        if stale >= config["patience"]:
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint_path = runtime_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded = build_variant(variant, clean_norm, v2_norm, device)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    split_metrics, all_predictions = {}, []
    for role in ("train", "validation", "test"):
        role_metrics, truth, prediction, positions = evaluate(
            reloaded, variant, data, atom_data, angle_data, indices[role], clean_norm, scales, device
        )
        split_metrics[role] = role_metrics
        all_predictions.extend(prediction_rows(data, role, positions, truth, prediction))
    with gzip.open(run_dir / "predictions.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    atomic_json(run_dir / "metrics.json", split_metrics)
    normalization = {
        "molecular_and_eluent_minmax": scaler,
        "matches_historical_e0_scaler": scaler_match,
        "target_training_scales": scales,
        "fit_row_count": len(indices["train"]),
        "fit_ids_hash": stable_hash(sorted(data.iloc[indices["train"]]["sample_id"].astype(str))),
        "validation_rows_used": 0, "test_rows_used": 0, "8g_rows_used": 0,
    }
    if variant == "R3_CLEAN_CURRENT":
        normalization["clean_condition"] = asdict(clean_norm)
    if variant == "R2_CONDITION_COMPLETE_V2":
        normalization["v2_condition"] = asdict(v2_norm)
    atomic_json(run_dir / "normalization.json", normalization)
    if variant == "R3_CLEAN_CURRENT":
        atomic_json(
            run_dir / "clean_diagnostics.json",
            clean_extra_diagnostics(reloaded, data, atom_data, angle_data, indices, clean_norm, scales, device),
        )
    checkpoint_metadata = {
        "runtime_path": str(checkpoint_path.relative_to(ROOT)),
        "sha256": sha256_file(checkpoint_path), "best_epoch": best_epoch,
        "epochs_run": len(history), "best_validation_selection_score": best_score,
        "checkpoint_reload_validated": True,
        "nominal_parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        "training_seconds": time.time() - started,
    }
    atomic_json(run_dir / "checkpoint_metadata.json", checkpoint_metadata)
    artifact_names = [
        "config.json", "history.csv", "metrics.json", "predictions.csv.gz",
        "normalization.json", "checkpoint_metadata.json",
    ]
    if variant == "R3_CLEAN_CURRENT":
        artifact_names.append("clean_diagnostics.json")
    manifest = {"files": [
        {"path": name, "bytes": (run_dir / name).stat().st_size, "sha256": sha256_file(run_dir / name)}
        for name in artifact_names
    ], "runtime_checkpoint_committed": False}
    atomic_json(run_dir / "artifact_manifest.json", manifest)
    summary = {
        "variant": variant, "best_epoch": best_epoch, "epochs_run": len(history),
        "metrics": split_metrics, "checkpoint": checkpoint_metadata,
    }
    atomic_json(run_dir / "run_summary.json", summary)
    return summary


def r0_sanity(summary: dict[str, Any]) -> dict[str, Any]:
    historical = json.loads(HISTORICAL_METRICS.read_text(encoding="utf-8"))["test"]
    observed = summary["metrics"]["test"]
    differences = {
        target: abs(observed[f"{target}_r2"] - historical[f"{target}_r2"])
        for target in ("V1", "V2")
    }
    return {
        "tolerance": R0_TOLERANCE,
        "historical_test_r2": {target: historical[f"{target}_r2"] for target in ("V1", "V2")},
        "observed_test_r2": {target: observed[f"{target}_r2"] for target in ("V1", "V2")},
        "absolute_difference": differences,
        "status": "PASS" if all(value <= R0_TOLERANCE for value in differences.values()) else "R0_SANITY_CHECK_FAILED",
    }


def write_environment(device: torch.device) -> None:
    packages = {}
    for name in ("numpy", "pandas", "rdkit", "scikit-learn", "torch", "torch-geometric", "mordred"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    atomic_json(STUDY / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "device": str(device), "git_commit_sha": git_sha(),
    })


def execute(selected: list[str], device: torch.device, resume: bool) -> None:
    load_frozen_inputs()
    write_environment(device)
    summaries: dict[str, Any] = {}
    r0_path = RESULTS / "R0_LEGACY_E0_EXACT/run_summary.json"
    for variant in VARIANTS:
        if variant not in selected and not (variant == VARIANTS[0] and not r0_path.exists()):
            continue
        summary_path = RESULTS / variant / "run_summary.json"
        if resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = run_variant(variant, device)
        summaries[variant] = summary
        if variant == VARIANTS[0]:
            sanity = r0_sanity(summary)
            atomic_json(RESULTS / "r0_sanity.json", sanity)
            if sanity["status"] != "PASS":
                raise RuntimeError("R0_SANITY_CHECK_FAILED; R1-R3 are blocked")
        atomic_json(RUNTIME / "progress.json", {"completed": list(summaries)})
    if not r0_path.exists():
        raise RuntimeError("R0 result is required before later ladder variants")
    sanity = r0_sanity(json.loads(r0_path.read_text(encoding="utf-8")))
    if sanity["status"] != "PASS":
        raise RuntimeError("stored R0 sanity failed; R1-R3 are blocked")
    all_summaries = {
        variant: json.loads((RESULTS / variant / "run_summary.json").read_text(encoding="utf-8"))
        for variant in VARIANTS if (RESULTS / variant / "run_summary.json").exists()
    }
    atomic_json(RESULTS / "run_summaries.json", all_summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    execute(args.variants, torch.device(args.device), args.resume)


if __name__ == "__main__":
    main()
