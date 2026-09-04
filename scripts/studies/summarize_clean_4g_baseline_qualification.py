#!/usr/bin/env python3
"""Audit and summarize the six frozen Clean 4g qualification runs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/predictor/clean_4g_baseline_qualification"
RESULTS = STUDY / "results"
MODES = ("row", "compound")
SEEDS = (42, 525, 1101)
EXPECTED_MODEL = "qgeognn_clean_fusion_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def numeric_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from numeric_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from numeric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    audit_runs: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    uq_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    metric_by_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode in MODES}

    for mode in MODES:
        for seed in SEEDS:
            run_dir = RESULTS / mode / f"seed_{seed}"
            config = load_json(run_dir / "config.json")
            checkpoint = load_json(run_dir / "checkpoint_metadata.json")
            metrics = load_json(run_dir / "metrics.json")
            normalization = load_json(run_dir / "normalization.json")
            diagnostics = load_json(run_dir / "representation_gradient_diagnostics.json")
            manifest = load_json(run_dir / "artifact_manifest.json")

            split_path = ROOT / config["split_path"]
            with split_path.open(newline="", encoding="utf-8") as handle:
                split_rows = list(csv.DictReader(handle))
            split_counts = {
                role: sum(row["split"] == role for row in split_rows)
                for role in ("train", "validation", "test")
            }

            manifest_ok = all(
                (run_dir / item["path"]).stat().st_size == item["bytes"]
                and sha256(run_dir / item["path"]) == item["sha256"]
                for item in manifest["files"]
            )
            runtime_checkpoint = ROOT / checkpoint["runtime_path"]
            checkpoint_hash_ok = (
                runtime_checkpoint.is_file()
                and sha256(runtime_checkpoint) == checkpoint["sha256"]
            )

            with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            history_columns = list(history[0])
            history_finite = all(
                math.isfinite(float(value))
                for row in history
                for value in row.values()
                if value not in ("True", "False")
            )

            prediction_path = run_dir / "predictions.csv.gz"
            with gzip.open(prediction_path, "rt", newline="", encoding="utf-8") as handle:
                predictions = list(csv.DictReader(handle))
            prediction_numeric_columns = (
                "V1_true", "V2_true", "V1_q10", "V1_q50", "V1_q90",
                "V2_q10", "V2_q50", "V2_q90",
            )
            predictions_finite = all(
                math.isfinite(float(row[key]))
                for row in predictions
                for key in prediction_numeric_columns
            )
            quantile_order_ok = all(
                float(row["V1_q10"]) <= float(row["V1_q50"]) <= float(row["V1_q90"])
                and float(row["V2_q10"]) <= float(row["V2_q50"]) <= float(row["V2_q90"])
                for row in predictions
            )
            metrics_finite = all(math.isfinite(value) for value in numeric_values(metrics))
            normalization_ok = (
                normalization["normalization_fit_row_count"] == split_counts["train"]
                and normalization["validation_rows_used"] == 0
                and normalization["test_rows_used"] == 0
                and normalization["8g_rows_used"] == 0
            )
            config_ok = (
                config["model"] == EXPECTED_MODEL
                and config["split_mode"] == mode
                and config["seed"] == seed
                and config["test_during_training"] is False
                and config["checkpoint_selection"].startswith("validation_only")
                and sha256(split_path) == config["split_sha256"]
            )
            audit = {
                "split_mode": mode,
                "seed": seed,
                "artifact_manifest_valid": manifest_ok,
                "runtime_checkpoint_hash_valid": checkpoint_hash_ok,
                "checkpoint_reload_validated": checkpoint["checkpoint_reload_validated"],
                "config_valid": config_ok,
                "history_rows": len(history),
                "history_has_test_columns": any("test" in name.lower() for name in history_columns),
                "history_all_finite": history_finite,
                "prediction_rows": len(predictions),
                "expected_test_rows": split_counts["test"],
                "prediction_shape_valid": len(predictions) == split_counts["test"],
                "predictions_all_finite": predictions_finite,
                "metrics_all_finite": metrics_finite,
                "quantile_order_valid": quantile_order_ok,
                "within_target_crossing_rate": metrics["full"]["within_target_quantile_crossing_rate"],
                "normalization_train_only_valid": normalization_ok,
                "passed": False,
            }
            audit["passed"] = all(
                (
                    manifest_ok,
                    checkpoint_hash_ok,
                    checkpoint["checkpoint_reload_validated"],
                    config_ok,
                    not audit["history_has_test_columns"],
                    history_finite,
                    audit["prediction_shape_valid"],
                    predictions_finite,
                    metrics_finite,
                    quantile_order_ok,
                    normalization_ok,
                    metrics["full"]["within_target_quantile_crossing_rate"] == 0.0,
                )
            )
            audit_runs.append(audit)

            full = metrics["full"]
            metric_by_mode[mode].append(full)
            point_rows.append({
                "split_mode": mode,
                "seed": seed,
                "V1_r2": full["V1_r2"],
                "V1_rmse": full["V1_rmse"],
                "V1_mae": full["V1_mae"],
                "V2_r2": full["V2_r2"],
                "V2_rmse": full["V2_rmse"],
                "V2_mae": full["V2_mae"],
                "combined_normalized_rmse": full["combined_normalized_rmse"],
            })
            uq_rows.append({
                "split_mode": mode,
                "seed": seed,
                "V1_coverage": full["V1_q10_q90_coverage"],
                "V1_interval_width": full["V1_interval_width"],
                "V1_mean_pinball_loss": full["V1_mean_pinball_loss"],
                "V2_coverage": full["V2_q10_q90_coverage"],
                "V2_interval_width": full["V2_interval_width"],
                "V2_mean_pinball_loss": full["V2_mean_pinball_loss"],
                "within_target_crossing_rate": full["within_target_quantile_crossing_rate"],
                "q50_V1_gt_q50_V2_rate": full["q50_V1_gt_q50_V2_rate"],
                "V1_q90_gt_V2_q10_rate": full["V1_q90_gt_V2_q10_rate"],
            })
            for prediction_mode in ("full", "condition_permuted", "condition_disabled", "condition_only"):
                values = metrics[prediction_mode]
                condition_rows.append({
                    "split_mode": mode,
                    "seed": seed,
                    "prediction_mode": prediction_mode,
                    "V1_rmse": values["V1_rmse"],
                    "V1_mae": values["V1_mae"],
                    "V1_r2": values["V1_r2"],
                    "V2_rmse": values["V2_rmse"],
                    "V2_mae": values["V2_mae"],
                    "V2_r2": values["V2_r2"],
                    "delta_V1_rmse": values.get("delta_V1_rmse", 0.0),
                    "delta_V1_mae": values.get("delta_V1_mae", 0.0),
                    "delta_V1_r2": values.get("delta_V1_r2", 0.0),
                    "delta_V2_rmse": values.get("delta_V2_rmse", 0.0),
                    "delta_V2_mae": values.get("delta_V2_mae", 0.0),
                    "delta_V2_r2": values.get("delta_V2_r2", 0.0),
                })
            representation = diagnostics["representation_test"]
            gradients = diagnostics["per_target_gradients"]
            diagnostic_rows.append({
                "split_mode": mode,
                "seed": seed,
                "molecular_latent_l2_mean": representation["molecular_latent_l2_mean"],
                "condition_latent_l2_mean": representation["condition_latent_l2_mean"],
                "V1_molecular_projection_gradient_l2": gradients["V1"]["molecular_projection_gradient_l2"],
                "V1_condition_encoder_gradient_l2": gradients["V1"]["condition_encoder_gradient_l2"],
                "V2_molecular_projection_gradient_l2": gradients["V2"]["molecular_projection_gradient_l2"],
                "V2_condition_encoder_gradient_l2": gradients["V2"]["condition_encoder_gradient_l2"],
            })

    point_keys = list(point_rows[0])[2:]
    uq_keys = list(uq_rows[0])[2:]
    aggregate = {
        "formal_runs_completed": len(audit_runs),
        "formal_runs_expected": 6,
        "all_runs_pass_numerical_audit": all(run["passed"] for run in audit_runs),
        "point": {
            mode: {key: stats([row[key] for row in point_rows if row["split_mode"] == mode]) for key in point_keys}
            for mode in MODES
        },
        "uq": {
            mode: {key: stats([row[key] for row in uq_rows if row["split_mode"] == mode]) for key in uq_keys}
            for mode in MODES
        },
        "condition_usage": {
            "permutation_increases_both_target_rmse_runs": sum(
                row["prediction_mode"] == "condition_permuted"
                and row["delta_V1_rmse"] > 0
                and row["delta_V2_rmse"] > 0
                for row in condition_rows
            ),
            "disabled_increases_both_target_rmse_runs": sum(
                row["prediction_mode"] == "condition_disabled"
                and row["delta_V1_rmse"] > 0
                and row["delta_V2_rmse"] > 0
                for row in condition_rows
            ),
            "formal_runs": 6,
        },
    }

    audit_document = {
        "gate": "Gate A — Numerical validity",
        "passed": aggregate["all_runs_pass_numerical_audit"],
        "runs": audit_runs,
    }
    (RESULTS / "formal_run_audit.json").write_text(
        json.dumps(audit_document, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(RESULTS / "point_summary.csv", point_rows)
    write_csv(RESULTS / "uq_summary.csv", uq_rows)
    write_csv(RESULTS / "condition_usage_summary.csv", condition_rows)
    write_csv(RESULTS / "representation_gradient_summary.csv", diagnostic_rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
