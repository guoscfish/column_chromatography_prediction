#!/usr/bin/env python3
"""Audit and summarize the completed point-predictor regression ladder."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.studies.run_point_predictor_regression_audit import (
    EXPECTED_COUNTS,
    EXPECTED_SPLIT_SHA,
    RESULTS,
    ROOT,
    VARIANTS,
    r0_sanity,
    sha256_file,
    stable_hash,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numbers(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def main() -> None:
    audit_rows, ladder_rows = [], []
    summaries = {}
    for variant in VARIANTS:
        run_dir = RESULTS / variant
        config = load_json(run_dir / "config.json")
        metrics = load_json(run_dir / "metrics.json")
        checkpoint = load_json(run_dir / "checkpoint_metadata.json")
        manifest = load_json(run_dir / "artifact_manifest.json")
        summary = load_json(run_dir / "run_summary.json")
        summaries[variant] = summary
        config_payload = dict(config)
        recorded_config_hash = config_payload.pop("config_hash")
        config_valid = (
            stable_hash(config_payload) == recorded_config_hash
            and config["variant"] == variant
            and config["split_sha256"] == EXPECTED_SPLIT_SHA
            and config["split_generation_called"] is False
            and config["test_during_training"] is False
            and config["8g_rows_used"] == 0
        )
        manifest_valid = all(
            (run_dir / item["path"]).stat().st_size == item["bytes"]
            and sha256_file(run_dir / item["path"]) == item["sha256"]
            for item in manifest["files"]
        )
        runtime_checkpoint = ROOT / checkpoint["runtime_path"]
        checkpoint_valid = (
            checkpoint["checkpoint_reload_validated"] is True
            and runtime_checkpoint.is_file()
            and sha256_file(runtime_checkpoint) == checkpoint["sha256"]
        )
        with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        history_has_test = any("test" in key.lower() for key in history[0])
        history_finite = all(
            math.isfinite(float(value))
            for row in history for value in row.values()
            if value not in ("True", "False")
        )
        with gzip.open(run_dir / "predictions.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        prediction_counts = {
            role: sum(row["split"] == role for row in predictions)
            for role in EXPECTED_COUNTS
        }
        prediction_ids = [row["sample_id"] for row in predictions]
        prediction_finite = all(
            math.isfinite(float(row[key]))
            for row in predictions
            for key in ("V1_true", "V2_true", "V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90")
        )
        metrics_finite = all(math.isfinite(value) for value in numbers(metrics))
        passed = all((
            config_valid, manifest_valid, checkpoint_valid, not history_has_test,
            history_finite, prediction_counts == EXPECTED_COUNTS,
            len(predictions) == 4163, len(set(prediction_ids)) == 4163,
            prediction_finite, metrics_finite,
        ))
        audit_rows.append({
            "variant": variant, "passed": passed,
            "config_hash_valid": config_valid, "manifest_valid": manifest_valid,
            "checkpoint_valid": checkpoint_valid, "history_rows": len(history),
            "history_has_test_columns": history_has_test, "history_all_finite": history_finite,
            "prediction_rows": len(predictions), "prediction_ids_unique": len(set(prediction_ids)) == 4163,
            "prediction_split_counts_valid": prediction_counts == EXPECTED_COUNTS,
            "predictions_all_finite": prediction_finite, "metrics_all_finite": metrics_finite,
        })
        for role in ("train", "validation", "test"):
            values = metrics[role]
            ladder_rows.append({
                "variant": variant, "split": role,
                "V1_r2": values["V1_r2"], "V1_rmse": values["V1_rmse"], "V1_mae": values["V1_mae"],
                "V2_r2": values["V2_r2"], "V2_rmse": values["V2_rmse"], "V2_mae": values["V2_mae"],
                "combined_normalized_rmse": values["combined_normalized_rmse"],
            })

    sanity = r0_sanity(summaries[VARIANTS[0]])
    test_r2 = {
        variant: {
            target: summaries[variant]["metrics"]["test"][f"{target}_r2"]
            for target in ("V1", "V2")
        }
        for variant in VARIANTS
    }
    deltas = {
        f"{left}_to_{right}": {
            target: test_r2[right][target] - test_r2[left][target]
            for target in ("V1", "V2")
        }
        for left, right in zip(VARIANTS, VARIANTS[1:])
    }
    diagnosis = {
        "scientific_role": "DEVELOPMENTAL_REGRESSION_DIAGNOSTIC",
        "all_artifact_audits_pass": all(row["passed"] for row in audit_rows),
        "r0_sanity": sanity,
        "test_r2": test_r2,
        "adjacent_test_r2_deltas": deltas,
        "earliest_regression_step": "R2_CONDITION_COMPLETE_V2_TO_R3_CLEAN_CURRENT",
        "confirmed": [
            "CLEAN_ARCHITECTURE_REGRESSION_CONFIRMED_AS_A_PACKAGE",
            "R3_UNDERFITS_TRAINING_DATA_RELATIVE_TO_R2",
        ],
        "supported": ["LOSS_OF_LEGACY_EARLY_MOLECULE_CONDITION_INTERACTION_AND_ADDITIVE_LATE_FUSION"],
        "plausible": [
            "UNVALIDATED_128_TO_64_MOLECULAR_BOTTLENECK",
            "LAYERNORM_AFTER_SUM_POOLING_REMOVES_MAGNITUDE_INFORMATION",
            "MONOTONIC_SOFTPLUS_HEAD_INTERACTION_WITH_CLEAN_LATENTS",
        ],
        "not_supported": [
            "DIFFERENT_SPLIT_AS_PRIMARY_CAUSE",
            "TRAINING_PROTOCOL_REGRESSION",
            "CONDITION_COMPLETION_IMPLEMENTATION_REGRESSION",
            "GLOBAL_Q50_LOW_VARIANCE_COMPRESSION",
            "Q50_CLAMP_TO_ZERO",
        ],
        "clean_baseline_status": "NO_POINT_PERFORMANCE_QUALIFICATION_REOPENED",
        "next_stage": "CONTROLLED_MINIMAL_NONLINEAR_LATE_FUSION_TEST_AFTER_HUMAN_APPROVAL",
    }
    (RESULTS / "artifact_audit.json").write_text(json.dumps({"passed": all(row["passed"] for row in audit_rows), "runs": audit_rows}, indent=2) + "\n")
    (RESULTS / "regression_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    with (RESULTS / "ladder_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ladder_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ladder_rows)
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
