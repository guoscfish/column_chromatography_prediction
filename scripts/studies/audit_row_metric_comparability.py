#!/usr/bin/env python3
"""Re-score existing matched ROW predictions under one train-only scale contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "studies/transfer/filtered_full_data_benchmark"
FORMAL = ROOT / "studies/transfer/traditional_transfer_improvement/formal_row_5seed"
REFERENCE = ROOT / "studies/transfer/row_column_selective_latent_audit"
OUT = ROOT / "studies/transfer/row_metric_harmonization"
COLUMNS = ("25g", "40g")
METHODS = (
    "P0", "P1", "P2", "P3", "paper_style_current_v2",
    "HIER_CW_SHARED_LAMBDA_CORRECTED", "BALANCED_JOINT_FULL128",
)
REFERENCE_METHODS = METHODS[4:]
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
SOURCE_SCALES = {"V1": 7.8796590346317394, "V2": 16.076509553562932}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth(path: Path, ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"])
    frame["sample_id"] = frame.sample_id.astype(str)
    return frame.set_index("sample_id").loc[ids, ["V1_ml", "V2_ml"]].to_numpy(float)


def scales_for_train(path: Path, train_ids: list[str]) -> dict[str, float]:
    values = truth(path, train_ids)
    scales = values.std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError("non-positive train-only scale")
    return {"V1": float(scales[0]), "V2": float(scales[1])}


def metrics(y: np.ndarray, p: np.ndarray, scales: dict[str, float]) -> dict[str, float]:
    errors = y - p
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    r2 = []
    for j in range(2):
        denominator = np.sum((y[:, j] - y[:, j].mean()) ** 2)
        r2.append(float(1 - np.sum(errors[:, j] ** 2) / denominator))
    combined = float(np.sqrt(0.5 * ((rmse[0] / scales["V1"]) ** 2 + (rmse[1] / scales["V2"]) ** 2)))
    source_combined = float(np.sqrt(0.5 * ((rmse[0] / SOURCE_SCALES["V1"]) ** 2 + (rmse[1] / SOURCE_SCALES["V2"]) ** 2)))
    return {"V1_rmse": float(rmse[0]), "V1_mae": float(mae[0]), "V1_r2": r2[0],
            "V2_rmse": float(rmse[1]), "V2_mae": float(mae[1]), "V2_r2": r2[1],
            "combined_nrmse_train_only": combined, "combined_nrmse_source_scale": source_combined}


def formal_predictions(column: str, seed: int) -> tuple[list[str], dict[str, np.ndarray]]:
    base = FORMAL / "runtime" / column / f"seed_{seed}"
    result = {}
    ids = None
    for method in ("P0", "P1", "P2", "P3"):
        frame = pd.read_csv(base / method / "test_predictions_blind.csv.gz")
        current = frame.sample_id.astype(str).tolist()
        ids = current if ids is None else ids
        if current != ids:
            raise RuntimeError(f"formal prediction ID mismatch: {column}/{seed}/{method}")
        result[method] = frame[["V1_q50", "V2_q50"]].to_numpy(float)
    return ids, result


def reference_predictions(column: str, seed: int, ids: list[str]) -> dict[str, np.ndarray]:
    frame = pd.read_csv(REFERENCE / "runtime" / column / "row" / f"seed_{seed}" / "predictions_blind.csv.gz")
    frame["sample_id"] = frame.sample_id.astype(str)
    if frame.sample_id.tolist() != ids:
        frame = frame.set_index("sample_id").loc[ids].reset_index()
    if frame.sample_id.tolist() != ids:
        raise RuntimeError(f"reference prediction ID mismatch: {column}/{seed}")
    return {method: frame[[f"{method}_V1", f"{method}_V2"]].to_numpy(float) for method in REFERENCE_METHODS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    schedule = pd.read_csv(FROZEN / "split_manifest.csv")
    rows = []
    scale_rows = []
    prediction_hashes = {}
    for column in COLUMNS:
        canonical = FROZEN / f"filtered_canonical_{column}.csv"
        for seed in SEEDS:
            block = schedule.loc[(schedule.column == column) & (schedule.protocol == "row") & (schedule.outer_seed == seed)]
            train_ids = block.loc[block.role == "gradient_train", "sample_id"].astype(str).tolist()
            test_ids = block.loc[block.role == "test", "sample_id"].astype(str).tolist()
            scales = scales_for_train(canonical, train_ids)
            scale_rows.append({"column": column, "seed": seed, **scales, "scale_source": "outer_gradient_train_ddof0"})
            ids, predictions = formal_predictions(column, seed)
            if ids != test_ids:
                raise RuntimeError(f"formal test IDs differ from frozen split: {column}/{seed}")
            predictions.update(reference_predictions(column, seed, ids))
            y = truth(canonical, test_ids)
            for method, prediction in predictions.items():
                row = {"column": column, "protocol": "row", "seed": seed, "method": method,
                       "n_test": len(test_ids), "scale_V1": scales["V1"], "scale_V2": scales["V2"], **metrics(y, prediction, scales)}
                rows.append(row)
            for method in ("P0", "P1", "P2", "P3"):
                prediction_hashes[f"formal/{column}/{seed}/{method}"] = sha(FORMAL / "runtime" / column / f"seed_{seed}" / method / "test_predictions_blind.csv.gz")
            prediction_hashes[f"reference/{column}/{seed}"] = sha(REFERENCE / "runtime" / column / "row" / f"seed_{seed}" / "predictions_blind.csv.gz")
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(OUT / "per_seed_metrics.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(OUT / "train_only_scales.csv", index=False)
    summary = metrics_frame.groupby(["column", "method"])[["V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_nrmse_train_only", "combined_nrmse_source_scale"]].agg(["mean", "std", "median", "min", "max"])
    summary.columns = ["_".join(x) for x in summary.columns]
    summary.reset_index().to_csv(OUT / "summary.csv", index=False)
    ranks = []
    for column, group in metrics_frame.groupby("column"):
        for metric in ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_nrmse_train_only"):
            ascending = metric not in ("V1_r2", "V2_r2")
            ordered = group.groupby("method")[metric].mean().sort_values(ascending=ascending)
            for rank, (method, value) in enumerate(ordered.items(), 1):
                ranks.append({"column": column, "metric": metric, "rank": rank, "method": method, "mean": float(value)})
    pd.DataFrame(ranks).to_csv(OUT / "ranks.csv", index=False)
    report = ["# Metric Comparability Audit", "", "## Problem", "", "The formal P0-P3 score used an outer-split, gradient-train-only standard deviation for each target. Historical ROW reference scores used the fixed qualified 4g source-train scales. Test IDs were matched, but the normalization denominator was not.", "", "The discrepancy is visible in the 40g example: paper-style has lower raw V1/V2 RMSE than P1, while its historical source-scale combined NRMSE is much larger. This is a denominator difference, not evidence that P1 dominates paper-style.", "", "## Contract", "", "For every column and outer seed, this audit computes `s_V1` and `s_V2` from that context's `gradient_train` labels with population standard deviation (`ddof=0`). All seven methods use the same scales and the same frozen outer test IDs. No test label is used to define scales or select a method.", "", "Source-scale values retained for diagnosing the old reports: `V1=7.8796590346`, `V2=16.0765095536`.", "", "## Findings", "", "- Raw RMSE, MAE, and R2 were already directly comparable on matched test IDs.", "- Historical combined NRMSE values in `reference_comparison.csv` are not directly comparable to formal P0-P3 combined NRMSE values and must not be used for cross-study ranking.", "- The unified values are in `summary.csv`; per-seed scales and metrics are in `train_only_scales.csv` and `per_seed_metrics.csv`.", "- This is a re-score of frozen prediction artifacts, not a new fit and not a new test-driven method selection.", "", "## Scope limitation", "", "Only methods with local frozen ROW prediction artifacts were included: P0-P3, paper-style, HIER_CW_SHARED_LAMBDA_CORRECTED, and BALANCED_JOINT_FULL128. Other historical methods without matched prediction artifacts are marked unavailable rather than reconstructed.", "", "" ]
    (OUT / "METRIC_COMPARABILITY_AUDIT.md").write_text("\n".join(report))
    protocol = {"status": "RE_SCORE_OF_FROZEN_PREDICTIONS", "columns": list(COLUMNS), "seeds": list(SEEDS), "methods": list(METHODS), "scale_contract": "per column x outer seed gradient_train population std ddof=0", "same_test_ids_required": True, "test_truth_used_for_selection": False, "source_scale": SOURCE_SCALES, "prediction_hashes": prediction_hashes}
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
