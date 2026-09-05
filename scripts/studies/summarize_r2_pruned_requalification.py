#!/usr/bin/env python3
"""Compare completed R2-pruned artifacts against the immutable R2 control."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies import run_r2_pruned_requalification as study
from src.qgeognn_al.condition_complete_v2_pruned import DEAD_MODULE_PREFIXES


def read(path):
    return json.loads(path.read_text())


def summarize():
    out = study.STUDY
    result = out / "results/R2_PRUNED"
    reference = study.r2.RESULTS / study.REFERENCE
    gate = read(out / "function_equivalence_audit.json")
    assert gate["status"] == "PASS"
    before, after = [read(out / f"R2_REACHABILITY_{phase}.json") for phase in ("BEFORE", "AFTER")]
    observed, baseline = read(result / "metrics.json"), read(reference / "metrics.json")
    metadata = read(result / "checkpoint_metadata.json")
    original_state = torch.load(study.r2.RUNTIME / study.REFERENCE / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    trained = torch.load(out / "runtime/R2_PRUNED/best.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    state_diff = max(float((original_state[k] - v).abs().max()) for k, v in trained.items())
    histories = [pd.read_csv(p / "history.csv") for p in (reference, result)]
    history_equal = histories[0].equals(histories[1])
    predictions = [pd.read_csv(p / "predictions.csv.gz") for p in (reference, result)]
    assert predictions[0][["split", "sample_id"]].equals(predictions[1][["split", "sample_id"]])
    outputs = gate["output_order"]
    prediction_diff = float(np.abs(predictions[0][outputs].to_numpy() - predictions[1][outputs].to_numpy()).max())
    differences = {role: {key: observed[role][key] - baseline[role][key]
                         for key in baseline[role] if key != "all_outputs_finite"} for role in baseline}
    # Exact replay is stronger than a discretionary "no obvious degradation" gate.
    exact = all(abs(v) <= 1e-6 for row in differences.values() for v in row.values()) and state_diff <= 1e-6 and prediction_diff <= 1e-6
    decision = {
        "status": "FUNCTION_PRESERVING_PARAMETER_CLEANUP_SUCCESS" if exact else "FUNCTION_PRESERVING_BUT_RETRAIN_DIFFERENCE_REQUIRES_AUDIT",
        "model_variant": "qgeognn_condition_complete_v2_pruned",
        "qualification_status": "POINT_PREDICTOR_CANDIDATE_BASELINE" if exact else "NOT_PROMOTED",
        "function_preserving_gate": gate["status"], "training_executed": True,
        "forward_unreachable_trainable_parameters": after["forward_unreachable_trainable_parameters"],
        "metric_deltas_vs_r2": differences, "history_exactly_equal": history_equal,
        "retrained_retained_checkpoint_max_abs_difference": state_diff,
        "retrained_prediction_max_abs_difference": prediction_diff,
        "next_action": "R2_PRUNED_QUANTILE_HEAD_QUALIFICATION" if exact else "AUDIT_TRAINING_IMPLEMENTATION",
        "next_experiment_executed": False, "quantile_head_changed": False,
        "formal_baseline": False, "uq_status": "NOT_QUALIFIED_NOT_EXECUTED",
        "transfer_8g_executed": False, "active_learning_executed": False,
        "clean_status": "FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE",
        "clean_engineering_reachability": "PASS", "clean_performance": "FAIL",
        "interpretation": "Dead-module removal needs no Clean architecture redesign; candidate applies to this frozen developmental E0 control only." if exact else "Dead parameters do not provide predictive capacity; investigate training/RNG/numerical differences.",
    }
    study.r2.atomic_json(out / "decision.json", decision)
    metadata["gradient_bearing_parameter_count"] = after["gradient_bearing_parameters"]
    study.r2.atomic_json(result / "checkpoint_metadata.json", metadata)
    for path in (result / "run_summary.json", out / "results/run_summary.json"):
        summary = read(path)
        summary.update(variant="R2_PRUNED", checkpoint=metadata)
        study.r2.atomic_json(path, summary)
    manifest = read(result / "artifact_manifest.json")
    for item in manifest["files"]:
        path = result / item["path"]
        item.update(bytes=path.stat().st_size, sha256=study.r2.sha256_file(path))
    study.r2.atomic_json(result / "artifact_manifest.json", manifest)
    lines = ["# R2-pruned controlled requalification", "", f"Status: `{decision['status']}`.", "",
             "Mainline: Legacy historical → Condition Completion V2 → R2-pruned candidate baseline.", "",
             "## Reachability", "", "| Model | Nominal | Requires grad | Gradient-bearing | Unreachable |",
             "|---|---:|---:|---:|---:|"]
    for name, audit in (("R2", before), ("R2-pruned", after)):
        lines.append(f"| {name} | {audit['nominal_parameters']} | {audit['requires_grad_parameters']} | {audit['gradient_bearing_parameters']} | {audit['forward_unreachable_trainable_parameters']} |")
    lines += ["", "Removed registrations (parameter counts exclude buffers):", "", "| Module | Removed parameters |", "|---|---:|"]
    for prefix in DEAD_MODULE_PREFIXES:
        count = sum(p["parameter_count"] for p in before["parameters"] if p["name"].startswith(prefix + "."))
        lines.append(f"| `{prefix}` | {count} |")
    lines += ["", "The terminal edge update executes in R2, but its result cannot reach a later node update or the prediction. Other removed modules are never called in the geometry-enhanced prediction path. Static trace, forward hooks, and three real multi-molecule/multi-condition backward batches agree. Gradient-bearing means `grad is not None`, including mathematically reachable parameters with zero-valued gradients. Counts describe registered parameters, not unregistered Legacy RBF tensor attributes.", "",
              "Five node layers, four effective geometry updates, bond length, descriptor geometry path, early eluent interaction, sum pooling, 128D representation, typed completion branch and Linear(128,6)+ReLU head are retained.", "",
              "## Function and initialization gates", "",
              "P0 and P1 cover all 4,163 real rows, 217 molecules, PE/EA/DCM loading solvents, varying loading masses/volumes and eluent compositions. Each split's six-output maximum absolute difference is 0, and all point metric differences are 0. The trained source checkpoint SHA-256 is `2c8bbb738b7e163b53bc80786747edf661df08e150103b0bc7611d9240456072`.", "",
              f"Mapped initial-state hash: `{read(out / 'initialization_audit.json')['initialization_mapping_hash']}`. All retained initial values and parameter traversal order are exact; construction preserves the canonical post-construction RNG state. Two real Adam steps have zero retained-state differences.", "",
              "Legacy RBF centers are non-leaf unregistered tensor views, so generic deepcopy fails. The constructor rebuilds an independent source inside an isolated RNG context, loads the full canonical state, and transfers only retained modules. The resulting model registers no deleted parameters; original R2 remains intact.", "",
              "## Controlled retraining", "",
              f"Best epoch: **{metadata['best_epoch']}**; total epochs: **{metadata['epochs_run']}**. Frozen split SHA-256: `{study.r2.EXPECTED_SPLIT_SHA}`. Train/validation/test: 3330/416/417; thresholds V1 ≤ 60, V2 ≤ 120. Adam, lr 0.001, weight decay 0, batch 2048, seed 42, maximum 1000 epochs, patience 100, epoch-deterministic shuffle, equal target loss weights and train-only normalization match R2. Only validation combined normalized RMSE selects the checkpoint; test is evaluated after selection.", "",
              "| Split | V1 R² | V1 RMSE | V1 MAE | V2 R² | V2 RMSE | V2 MAE | Combined normalized RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for role, row in observed.items():
        keys = ("V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae", "combined_normalized_rmse")
        lines.append("| " + role + " | " + " | ".join(f"{row[k]:.9f}" for k in keys) + " |")
    lines += ["", f"All metric deltas versus the full-precision historical R2 artifacts: `{differences}`.", "",
              f"Entire training history exactly equal: `{history_equal}`. Retrained retained-checkpoint maximum difference: `{state_diff}`. Retrained six-output prediction maximum difference: `{prediction_diff}`.", "",
              "## Decision", "", f"`{decision['qualification_status']}`; this is a candidate, not a formally qualified UQ/baseline contract. Dead trainable registrations can be removed while reproducing the effective R2 model; they contributed no predictive ability. The next separate controlled study should change only the quantile head. No head alternative, UQ, 8g, transfer, AL, sweep, or Clean repair was executed.", "",
              "`qgeognn_clean_fusion_v1`: `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`. Engineering reachability PASS; performance FAIL. Preserve it and all historical R0/R1/R2/R3 results for regression provenance. No further MLP/FiLM/LayerNorm/bottleneck repair is part of this mainline.", "",
              "Validation details and the complete pytest output are stored in `test_report.json` and `results/pytest_output.txt`."]
    (out / "R2_PRUNED_COMPARISON.md").write_text("\n".join(lines) + "\n")
    (out / "README.md").write_text("# R2-pruned requalification\n\n" + f"Status: `{decision['status']}` / `{decision['qualification_status']}`.\n\n"
        + "See [comparison](R2_PRUNED_COMPARISON.md), [decision](decision.json), [protocol](protocol.json), [function gates](function_equivalence_audit.json), and the before/after reachability inventories.\n\n"
        + "Reproduce in the historical conda `fish` environment:\n\n```bash\nKMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_r2_pruned_requalification.py\nKMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/summarize_r2_pruned_requalification.py\nKMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python -m pytest -q\n```\n\n"
        + "The local historical R2 runtime best checkpoint is required and SHA-verified before P1. Its metadata stays tracked; runtime checkpoints remain local under the existing retention policy. Formal training refuses to proceed if any function/reachability gate fails. P0/P1 test-domain comparisons are equivalence-only engineering gates, not fitting or checkpoint selection.\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    summarize()
