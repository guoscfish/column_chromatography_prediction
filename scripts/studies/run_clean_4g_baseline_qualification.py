#!/usr/bin/env python3
"""Freeze splits and execute the six Clean-QGeoGNN 4g qualification runs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data, eluent_descriptor
from src.qgeognn_al.models.clean_fusion import (
    build_clean_model,
    clean_checkpoint_payload,
    latent_l2_norms,
    load_clean_checkpoint,
    per_target_gradient_contribution,
)
from src.qgeognn_al.model import quantile_target_loss
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.schemas.clean import (
    CleanConditionBatch,
    CleanConditionNormalization,
    fit_clean_condition_normalization,
    parse_clean_conditions,
)


STUDY = ROOT / "studies/predictor/clean_4g_baseline_qualification"
SPLITS = STUDY / "splits"
RESULTS = STUDY / "results"
RUNTIME = STUDY / "runtime"
RAW_SOURCE = ROOT / "dataset/dataset_4g.csv"
EXPECTED_SOURCE_SHA = "d485d3d46a96458d1baac11b2a21cd33374b1be42ba2303e7ae5823cc8ee553a"
EXPECTED_ROWS = 4163
EXPECTED_COMPOUNDS = 217
SEEDS = (42, 525, 1101)
MODES = ("row", "compound")
V1_MAX = 60.0
V2_MAX = 120.0
TARGETS = ("V1", "V2")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_qualification_data() -> pd.DataFrame:
    if sha256_file(RAW_SOURCE) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("4g source SHA mismatch; STOP before training")
    data = pd.read_csv(SOURCE_DATA).copy()
    required = {"sample_id", "source_row_1based", "canonical_smiles", "V1_ml", "V2_ml"}
    if not required.issubset(data.columns):
        raise RuntimeError(f"canonical source is missing: {sorted(required - set(data.columns))}")
    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(f"qualification row mismatch: expected {EXPECTED_ROWS}, observed {len(data)}; STOP")
    if data["canonical_smiles"].nunique() != EXPECTED_COMPOUNDS:
        raise RuntimeError("qualification unique-compound count mismatch; STOP")
    if data["sample_id"].duplicated().any() or data["sample_id"].isna().any():
        raise RuntimeError("qualification sample IDs are not unique and complete; STOP")
    if data["V1_ml"].gt(V1_MAX).any() or data["V2_ml"].gt(V2_MAX).any():
        raise RuntimeError("qualification source exceeds frozen Legacy threshold domain; STOP")
    data["canonical_index"] = np.arange(len(data), dtype=int)
    return data


def generate_split_table(data: pd.DataFrame, split_mode: str, seed: int) -> pd.DataFrame:
    if split_mode not in MODES or seed not in SEEDS:
        raise ValueError("split mode or seed is outside the frozen protocol")
    rng = np.random.default_rng(seed)
    roles = np.empty(len(data), dtype=object)
    if split_mode == "row":
        order = rng.permutation(len(data))
        train_end, valid_end = int(0.8 * len(data)), int(0.9 * len(data))
        roles[order[:train_end]] = "train"
        roles[order[train_end:valid_end]] = "validation"
        roles[order[valid_end:]] = "test"
    else:
        compounds = np.asarray(sorted(data["canonical_smiles"].unique()), dtype=object)
        compounds = compounds[rng.permutation(len(compounds))]
        train_end, valid_end = int(0.8 * len(compounds)), int(0.9 * len(compounds))
        mapping = {compound: "train" for compound in compounds[:train_end]}
        mapping.update({compound: "validation" for compound in compounds[train_end:valid_end]})
        mapping.update({compound: "test" for compound in compounds[valid_end:]})
        roles = data["canonical_smiles"].map(mapping).to_numpy(dtype=object)
    result = pd.DataFrame({
        "sample_id": data["sample_id"].astype(str),
        "source_row": data["source_row_1based"].astype(int),
        "canonical_smiles": data["canonical_smiles"].astype(str),
        "split": roles,
        "seed": int(seed),
        "split_mode": split_mode,
    })
    if result["split"].isna().any() or set(result["split"]) != {"train", "validation", "test"}:
        raise AssertionError("split assignment incomplete")
    return result


def split_audit(table: pd.DataFrame) -> dict[str, Any]:
    role_sets = {
        role: set(table.loc[table["split"].eq(role), "canonical_smiles"])
        for role in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(role_sets["train"] & role_sets["validation"]),
        "train_test": len(role_sets["train"] & role_sets["test"]),
        "validation_test": len(role_sets["validation"] & role_sets["test"]),
    }
    return {
        "row_counts": {role: int(table["split"].eq(role).sum()) for role in role_sets},
        "compound_counts": {role: len(values) for role, values in role_sets.items()},
        "compound_overlap_counts": overlaps,
        "compound_overlap_assertion": (
            "REQUIRED_ZERO_AND_PASS" if table["split_mode"].iloc[0] == "compound" and not any(overlaps.values())
            else "NOT_REQUIRED_FOR_ROW_INTERPOLATION"
        ),
    }


def freeze_splits() -> dict[str, Any]:
    data = load_qualification_data()
    SPLITS.mkdir(parents=True, exist_ok=True)
    dataset_manifest = data[["sample_id", "source_row_1based", "canonical_smiles", "V1_ml", "V2_ml"]].rename(
        columns={"source_row_1based": "source_row"}
    )
    dataset_path = SPLITS / "qualification_dataset_manifest.csv"
    dataset_manifest.to_csv(dataset_path, index=False)
    records = []
    for mode in MODES:
        for seed in SEEDS:
            table = generate_split_table(data, mode, seed)
            path = SPLITS / f"{mode}_seed_{seed}.csv"
            table.to_csv(path, index=False)
            records.append({
                "split_mode": mode,
                "seed": seed,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                **split_audit(table),
            })
    payload = {
        "dataset_source": str(RAW_SOURCE.relative_to(ROOT)),
        "dataset_source_sha256": sha256_file(RAW_SOURCE),
        "canonical_filtered_source": str(SOURCE_DATA.relative_to(ROOT)),
        "canonical_filtered_source_sha256": sha256_file(SOURCE_DATA),
        "qualification_dataset_manifest": str(dataset_path.relative_to(ROOT)),
        "qualification_dataset_manifest_sha256": sha256_file(dataset_path),
        "retained_rows": len(data),
        "unique_compounds": int(data["canonical_smiles"].nunique()),
        "thresholds_ml": {"V1_max_inclusive": V1_MAX, "V2_max_inclusive": V2_MAX},
        "generation_code": str(Path(__file__).resolve().relative_to(ROOT)),
        "generation_code_sha256": sha256_file(Path(__file__).resolve()),
        "splits": records,
    }
    atomic_json(SPLITS / "split_manifest.json", payload)
    return payload


def fit_molecular_scaler(data: pd.DataFrame, train_indices: np.ndarray, graph_cache: dict) -> dict[str, Any]:
    train = data.iloc[train_indices]
    descriptors = np.vstack([graph_cache[value]["descriptor"] for value in train["canonical_smiles"]]).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in train["PE/EA"]]).astype(np.float32)
    return {
        "descriptor": {"min": descriptors.min(axis=0).tolist(), "max": descriptors.max(axis=0).tolist()},
        "eluent": {"min": eluents.min(axis=0).tolist(), "max": eluents.max(axis=0).tolist()},
    }


def loader_pair(atom_data: list, angle_data: list, indices: Iterable[int], batch_size: int) -> tuple[DataLoader, DataLoader]:
    positions = [int(value) for value in indices]
    return (
        DataLoader([atom_data[index] for index in positions], batch_size=batch_size, shuffle=False),
        DataLoader([angle_data[index] for index in positions], batch_size=batch_size, shuffle=False),
    )


def batch_conditions(data: pd.DataFrame, positions: np.ndarray, normalization: CleanConditionNormalization, device: torch.device) -> CleanConditionBatch:
    rows = data.iloc[positions]
    return parse_clean_conditions(rows, normalization, sample_ids=tuple(rows["sample_id"].astype(str))).to(device)


def permute_condition_batch(conditions: CleanConditionBatch, seed: int) -> tuple[CleanConditionBatch, np.ndarray]:
    order = np.random.default_rng(seed).permutation(conditions.continuous.shape[0])
    permuted = CleanConditionBatch(
        continuous=conditions.continuous[order],
        loading_solvent=conditions.loading_solvent[order],
        sample_ids=tuple(conditions.sample_ids[index] for index in order),
    )
    permuted.validate()
    return permuted, order


def combined_normalized_rmse(metrics: dict[str, float], scales: dict[str, float]) -> float:
    return float(math.sqrt(0.5 * ((metrics["V1_rmse"] / scales["V1"]) ** 2 + (metrics["V2_rmse"] / scales["V2"]) ** 2)))


def metric_bundle(y_true: np.ndarray, prediction: np.ndarray, scales: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    crossings = []
    levels = np.asarray([0.1, 0.5, 0.9]).reshape(1, 3)
    for target_index, target in enumerate(TARGETS):
        quantiles = prediction[:, target_index * 3 : target_index * 3 + 3]
        residual = y_true[:, target_index] - quantiles[:, 1]
        denominator = np.square(y_true[:, target_index] - y_true[:, target_index].mean()).sum()
        errors = y_true[:, [target_index]] - quantiles
        crossing = (quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])
        crossings.append(crossing)
        result.update({
            f"{target}_rmse": float(np.sqrt(np.square(residual).mean())),
            f"{target}_mae": float(np.abs(residual).mean()),
            f"{target}_r2": float(1 - np.square(residual).sum() / denominator) if denominator else float("nan"),
            f"{target}_q10_q90_coverage": float(((y_true[:, target_index] >= quantiles[:, 0]) & (y_true[:, target_index] <= quantiles[:, 2])).mean()),
            f"{target}_interval_width": float((quantiles[:, 2] - quantiles[:, 0]).mean()),
            f"{target}_mean_pinball_loss": float(np.maximum(levels * errors, (levels - 1) * errors).mean()),
            f"{target}_quantile_crossing_rate": float(crossing.mean()),
        })
    result["within_target_quantile_crossing_rate"] = float(np.logical_or(crossings[0], crossings[1]).mean())
    result["q50_V1_gt_q50_V2_rate"] = float((prediction[:, 1] > prediction[:, 4]).mean())
    result["V1_q90_gt_V2_q10_rate"] = float((prediction[:, 2] > prediction[:, 3]).mean())
    result["combined_normalized_rmse"] = combined_normalized_rmse(result, scales)
    result["all_outputs_finite"] = bool(np.isfinite(prediction).all())
    return result


def evaluate(
    model: torch.nn.Module,
    data: pd.DataFrame,
    atom_data: list,
    angle_data: list,
    indices: np.ndarray,
    normalization: CleanConditionNormalization,
    scales: dict[str, float],
    device: torch.device,
    *,
    mode: str = "full",
    permutation_seed: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    true_chunks, prediction_chunks, position_chunks = [], [], []
    latent_sums = {"molecular": 0.0, "condition": 0.0, "rows": 0}
    loaders = loader_pair(atom_data, angle_data, indices, 2048)
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loaders):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
            conditions = batch_conditions(data, positions, normalization, device)
            if permutation_seed is not None:
                conditions, _ = permute_condition_batch(conditions, permutation_seed)
            representations = model.representations(atom_batch, angle_batch, conditions, mode=mode)
            prediction = model.quantile_head(representations["fused"])
            prediction = torch.clamp(prediction, min=0, max=1e8)
            norms = latent_l2_norms(representations)
            latent_sums["molecular"] += norms["molecular_latent_l2_mean"] * len(positions)
            latent_sums["condition"] += norms["condition_latent_l2_mean"] * len(positions)
            latent_sums["rows"] += len(positions)
            true_chunks.append(atom_batch.y.cpu().numpy())
            prediction_chunks.append(prediction.cpu().numpy())
            position_chunks.append(positions)
    y_true, predictions, positions = np.vstack(true_chunks), np.vstack(prediction_chunks), np.concatenate(position_chunks)
    latents = {
        "molecular_latent_l2_mean": latent_sums["molecular"] / latent_sums["rows"],
        "condition_latent_l2_mean": latent_sums["condition"] / latent_sums["rows"],
    }
    return metric_bundle(y_true, predictions, scales), y_true, predictions, positions, latents


def metric_delta(candidate: dict[str, float], full: dict[str, float]) -> dict[str, float]:
    keys = [f"{target}_{metric}" for target in TARGETS for metric in ("rmse", "mae", "r2")]
    return {f"delta_{key}": float(candidate[key] - full[key]) for key in keys}


def prediction_frame(data: pd.DataFrame, positions: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    source = data.iloc[positions].reset_index(drop=True)
    return pd.DataFrame({
        "sample_id": source["sample_id"], "source_row": source["source_row_1based"],
        "canonical_smiles": source["canonical_smiles"], "V1_true": truth[:, 0], "V2_true": truth[:, 1],
        "V1_q10": prediction[:, 0], "V1_q50": prediction[:, 1], "V1_q90": prediction[:, 2],
        "V2_q10": prediction[:, 3], "V2_q50": prediction[:, 4], "V2_q90": prediction[:, 5],
    })


def run_one(split_mode: str, seed: int, device: torch.device) -> dict[str, Any]:
    data = load_qualification_data()
    split_path = SPLITS / f"{split_mode}_seed_{seed}.csv"
    split = pd.read_csv(split_path)
    if sha256_file(split_path) != next(
        item["sha256"] for item in json.loads((SPLITS / "split_manifest.json").read_text())["splits"]
        if item["split_mode"] == split_mode and item["seed"] == seed
    ):
        raise RuntimeError("frozen split hash mismatch")
    merged = data[["sample_id", "canonical_index"]].merge(split[["sample_id", "split"]], on="sample_id", validate="one_to_one")
    indices = {role: merged.loc[merged["split"].eq(role), "canonical_index"].to_numpy(dtype=int) for role in ("train", "validation", "test")}
    graph_cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    molecular_scaler = fit_molecular_scaler(data, indices["train"], graph_cache)
    normalization_split = split.rename(columns={"source_row": "source_row_1based"})
    normalization_split["split"] = normalization_split["split"].replace({"validation": "valid"})
    normalization = fit_clean_condition_normalization(data, normalization_split[["sample_id", "split"]])
    atom_data, angle_data = build_model_data(data, graph_cache, pd.DataFrame(), molecular_scaler)
    labels = data.iloc[indices["train"]][["V1_ml", "V2_ml"]]
    scales = {target: max(float(labels[f"{target}_ml"].std(ddof=0)), 1e-8) for target in TARGETS}

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_clean_model(normalization, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    run_dir = RESULTS / split_mode / f"seed_{seed}"
    runtime_dir = RUNTIME / split_mode / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True); runtime_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = runtime_dir / "best.pt"
    config = {
        "model": "qgeognn_clean_fusion_v1", "preflight_revision": 2, "split_mode": split_mode,
        "seed": seed, "split_path": str(split_path.relative_to(ROOT)), "split_sha256": sha256_file(split_path),
        "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 2048, "maximum_epochs": 1000,
        "patience": 100, "scheduler": None, "checkpoint_selection": "validation_only_minimum_combined_normalized_rmse",
        "test_during_training": False, "target_weights": {"V1": 1.0, "V2": 1.0}, "device": str(device),
    }
    atomic_json(run_dir / "config.json", config)
    history, best_score, best_epoch, stale = [], float("inf"), None, 0
    started = time.time()
    for epoch in range(1, 1001):
        model.train()
        order = np.random.default_rng(seed * 10000 + epoch).permutation(indices["train"])
        loaders = loader_pair(atom_data, angle_data, order, 2048)
        losses = []
        for atom_batch, angle_batch in zip(*loaders):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
            conditions = batch_conditions(data, positions, normalization, device)
            prediction = model(atom_batch, angle_batch, conditions)
            loss_v1 = quantile_target_loss(atom_batch.y[:, 0], prediction[:, 0:3])
            loss_v2 = quantile_target_loss(atom_batch.y[:, 1], prediction[:, 3:6])
            loss = loss_v1 + loss_v2
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        validation_metrics, _, _, _, _ = evaluate(model, data, atom_data, angle_data, indices["validation"], normalization, scales, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"validation_{key}": value for key, value in validation_metrics.items()}})
        score = validation_metrics["combined_normalized_rmse"]
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            payload = clean_checkpoint_payload(
                model, gradient_bearing_parameter_count=413732, git_commit_sha=git_sha(),
                source_split_hash=sha256_file(split_path), training_config=config,
            )
            torch.save(payload, checkpoint_path)
        else:
            stale += 1
        if stale >= 100:
            break
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model = load_clean_checkpoint(checkpoint_path, device)
    full, truth, prediction, positions, latents = evaluate(model, data, atom_data, angle_data, indices["test"], normalization, scales, device)
    permuted, _, _, _, _ = evaluate(model, data, atom_data, angle_data, indices["test"], normalization, scales, device, permutation_seed=900000 + seed)
    disabled, _, _, _, _ = evaluate(model, data, atom_data, angle_data, indices["test"], normalization, scales, device, mode="condition_disabled")
    condition_only, _, _, _, _ = evaluate(model, data, atom_data, angle_data, indices["test"], normalization, scales, device, mode="condition_only")

    validation_loaders = loader_pair(atom_data, angle_data, indices["validation"], min(2048, len(indices["validation"])))
    atom_batch, angle_batch = next(iter(validation_loaders[0])), next(iter(validation_loaders[1]))
    atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
    diagnostic_positions = atom_batch.canonical_position.cpu().numpy().reshape(-1).astype(int)
    diagnostic_conditions = batch_conditions(data, diagnostic_positions, normalization, device)
    diagnostic_targets = atom_batch.y
    model.train()
    gradients = per_target_gradient_contribution(model, atom_batch, angle_batch, diagnostic_conditions, diagnostic_targets)
    model.eval()

    metrics = {
        "evaluation_split": "test_after_validation_best_checkpoint_frozen", "full": full,
        "condition_permuted": {**permuted, **metric_delta(permuted, full)},
        "condition_disabled": {**disabled, **metric_delta(disabled, full)},
        "condition_only": {**condition_only, **metric_delta(condition_only, full)},
    }
    atomic_json(run_dir / "metrics.json", metrics)
    prediction_frame(data, positions, truth, prediction).to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    normalization_record = {
        "condition": asdict(normalization), "molecular_and_eluent_minmax": molecular_scaler,
        "target_training_scales": scales, "normalization_fit_row_count": len(indices["train"]),
        "normalization_fit_ids_hash": stable_hash(sorted(data.iloc[indices["train"]]["sample_id"].astype(str))),
        "validation_rows_used": 0, "test_rows_used": 0, "8g_rows_used": 0,
    }
    atomic_json(run_dir / "normalization.json", normalization_record)
    diagnostics = {"representation_test": latents, "gradient_dataset": "validation", "gradient_rows": len(diagnostic_positions), "per_target_gradients": gradients}
    atomic_json(run_dir / "representation_gradient_diagnostics.json", diagnostics)
    checkpoint_metadata = {
        "runtime_path": str(checkpoint_path.relative_to(ROOT)), "sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch, "best_validation_combined_normalized_rmse": best_score,
        "checkpoint_reload_validated": True, "nominal_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_schema_hash": torch.load(checkpoint_path, map_location="cpu", weights_only=False)["input_schema_hash"],
        "training_seconds": time.time() - started,
    }
    atomic_json(run_dir / "checkpoint_metadata.json", checkpoint_metadata)
    manifest_files = ["config.json", "checkpoint_metadata.json", "history.csv", "metrics.json", "predictions.csv.gz", "normalization.json", "representation_gradient_diagnostics.json"]
    artifact_manifest = build_artifact_manifest(run_dir, manifest_files)
    atomic_json(run_dir / "artifact_manifest.json", artifact_manifest)
    return {"split_mode": split_mode, "seed": seed, "best_epoch": best_epoch, "epochs_run": len(history), "metrics": metrics, "diagnostics": diagnostics, "checkpoint": checkpoint_metadata}


def build_artifact_manifest(run_dir: Path, names: Iterable[str]) -> dict[str, Any]:
    return {
        "files": [
            {"path": name, "sha256": sha256_file(run_dir / name), "bytes": (run_dir / name).stat().st_size}
            for name in names
        ],
        "runtime_checkpoint_committed": False,
    }


def execute_all(device_name: str) -> None:
    if not (SPLITS / "split_manifest.json").exists():
        raise RuntimeError("splits are not frozen; STOP")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("formal execution requires a clean committed tree")
    device = torch.device(device_name)
    summaries = []
    for mode in MODES:
        for seed in SEEDS:
            summaries.append(run_one(mode, seed, device))
            atomic_json(RUNTIME / "progress.json", {"completed": [[item["split_mode"], item["seed"]] for item in summaries]})
    atomic_json(RESULTS / "run_summaries.json", summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-splits", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.freeze_splits == args.execute:
        parser.error("choose exactly one of --freeze-splits or --execute")
    if args.freeze_splits:
        freeze_splits()
    else:
        execute_all(args.device)


if __name__ == "__main__":
    main()
