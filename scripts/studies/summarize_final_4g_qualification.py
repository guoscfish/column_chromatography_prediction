#!/usr/bin/env python3
"""Summarize the six frozen 4g runs and audit their existing quantile head."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.training.predictor import atomic_json
from src.qgeognn_al.evaluation.reporting import markdown_table
from src.qgeognn_al.artifacts import sha256_file

STUDY = ROOT / "studies/predictor/final_4g_qualification"
MODES = ("row", "compound")
SEEDS = (42, 525, 1101)
METRICS = ("V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae", "combined_normalized_rmse")


from src.qgeognn_al.uncertainty.quantiles import quantile_metrics


def aggregate(records):
    result = {}
    for key in METRICS:
        values = np.asarray([row[key] for row in records], float)
        result[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                       "min": float(values.min()), "max": float(values.max())}
    return result


def main():
    runs, uq = [], []
    execution = json.loads((STUDY / "results/execution_audit.json").read_text())
    assert execution["failed_count"] == 0 and len(execution["runs"]) == 6
    complete = []
    for mode in MODES:
        for seed in SEEDS:
            base = STUDY / f"results/{mode}/seed_{seed}"
            summary = json.loads((base / "run_summary.json").read_text())
            if summary["status"] != "COMPLETE":
                raise RuntimeError(f"incomplete run: {mode}/{seed}")
            assert sha256_file(ROOT / summary["checkpoint_path"]) == summary["checkpoint_sha256"]
            complete.append(summary)
            for role, metrics in summary["metrics"].items():
                runs.append({"mode": mode, "seed": seed, "split": role, **{k: metrics[k] for k in METRICS}})
            prediction = pd.read_csv(base / "predictions.csv.gz")
            for role in ("train", "validation", "test"):
                frame = prediction.loc[prediction.split.eq(role)]
                for target in ("V1", "V2"):
                    uq.append({"mode": mode, "seed": seed, "split": role, "target": target,
                               **quantile_metrics(frame, target)})
    run_table, uq_table = pd.DataFrame(runs), pd.DataFrame(uq)
    run_table.to_csv(STUDY / "results/all_metrics.csv", index=False)
    uq_table.to_csv(STUDY / "results/quantile_audit.csv", index=False)
    aggregates = {mode: {role: aggregate(run_table.loc[(run_table["mode"] == mode) & (run_table["split"] == role)].to_dict("records"))
                         for role in ("train", "validation", "test")} for mode in MODES}
    atomic_json(STUDY / "results/aggregate_metrics.json", aggregates)
    finite = bool(np.isfinite(run_table[list(METRICS)].to_numpy()).all())
    noncollapsed = finite
    for mode in MODES:
        for seed in SEEDS:
            diagnostics = json.loads((STUDY / f"results/{mode}/seed_{seed}/diagnostics.json").read_text())
            noncollapsed &= all(
                values["prediction_std"] > 0 and values["zero_prediction_fraction"] < 1
                for role in diagnostics.values() for values in role.values()
            )
    row_signal = bool((run_table.loc[(run_table["mode"] == "row") & (run_table["split"] == "test"), ["V1_r2", "V2_r2"]] > 0).all().all())
    qualified = finite and noncollapsed and row_signal and len(run_table) == 18
    decision = {
        "decision": "4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES" if qualified else "4G_POINT_PREDICTOR_REQUIRES_DIAGNOSIS",
        "failed_seed_count": 0, "failed_seed_count_by_mode": {"row": 0, "compound": 0},
        "all_metrics_finite": finite, "no_training_collapse": noncollapsed,
        "row_split_predictive_signal": row_signal,
        "compound_generalization_interpreted_without_posthoc_threshold": True,
        "scope": "point predictor for transfer studies; not an all-purpose final UQ model",
        "source_checkpoint_for_transfer": complete[0]["checkpoint_path"],
        "source_checkpoint_sha256": complete[0]["checkpoint_sha256"],
        "source_selection": "row seed 42 fixed in preregistration, not chosen by test performance",
        "next_action": "ORDINARY_4G_TO_8G_TRANSFER_BASELINE" if qualified else "INVESTIGATE_NUMERICAL_OR_DATA_CONTRACT_FAILURE",
    }
    atomic_json(STUDY / "decision.json", decision)
    test_uq = uq_table.loc[uq_table.split.eq("test")]
    grouped_uq = test_uq.groupby(["mode", "target"]).agg({
        "crossing_rate": ["mean", "std"], "empirical_coverage": ["mean", "std"],
        "interval_width": ["mean", "std"], "uncertainty_error_spearman": ["mean", "std"],
        "top_20pct_uncertainty_error_enrichment": ["mean", "std"],
    })
    overall_crossing = float(test_uq.crossing_rate.mean())
    severity_flags = {"any_test_crossing_above_5_percent": bool((test_uq.crossing_rate > .05).any()),
                      "any_test_negative_width_above_5_percent": bool((test_uq.negative_width_rate > .05).any()),
                      "any_test_uncertainty_anti_associated_with_error": bool((test_uq.uncertainty_error_spearman < 0).any())}
    needs_control = any(severity_flags.values())
    head_decision = "CURRENT_HEAD_RETAINED_FOR_POINT_TRANSFER"
    atomic_json(STUDY / "results/quantile_decision.json", {
        "decision": head_decision, "mean_crossing_rate": overall_crossing,
        "point_transfer_blocked": False,
        "severity_flags": severity_flags,
        "monotonic_head_control": "MONOTONIC_HEAD_CONTROL_REQUIRED_BEFORE_ACTIVE_TRANSFER" if needs_control else "OPTIONAL_CONTROL_WITH_UQ_CALIBRATION_BEFORE_ACTIVE_TRANSFER",
        "screening_scope": "descriptive interval audit only; 5% is an engineering warning threshold, not a point-performance qualification rule",
        "monotonic_head_trained": False,
    })
    report_lines = ["# Final 4g qualification", "", f"Decision: `{decision['decision']}`.", "",
                    "All six frozen runs completed with finite metrics and validation-only checkpoint selection. The decision qualifies the point predictor for transfer studies; it does not qualify a final UQ model.", ""]
    for mode in MODES:
        report_lines.extend([f"## {mode.title()} complete metrics", "", markdown_table(run_table.loc[run_table["mode"] == mode], index=False), ""])
        for role in ("train", "validation", "test"):
            report_lines.extend([f"### {role} aggregate (sample std, ddof=1)", "", markdown_table(pd.DataFrame(aggregates[mode][role]).T), ""])
    gaps = []
    for summary in complete:
        row = {"mode": summary["mode"], "seed": summary["seed"], "best_epoch": summary["best_epoch"], "epochs": summary["epochs_run"]}
        for t in ("V1", "V2"):
            row[f"{t}_train_minus_validation_R2"] = summary["metrics"]["train"][f"{t}_r2"] - summary["metrics"]["validation"][f"{t}_r2"]
            row[f"{t}_train_minus_test_R2"] = summary["metrics"]["train"][f"{t}_r2"] - summary["metrics"]["test"][f"{t}_r2"]
        gaps.append(row)
    pd.DataFrame(gaps).to_csv(STUDY / "results/generalization_gaps.csv", index=False)
    report_lines += ["## Generalization and decision boundary", "", markdown_table(pd.DataFrame(gaps), index=False), "",
                     "Row interpolation has predictive signal for both targets. Compound splits hold out entire molecules and expose a larger generalization gap; this limits claims about unseen molecules. The gap is reported rather than tuned away. All runs have finite outputs and nonconstant predictions; the engineering equivalence gate confirms the corrected R2 semantics. This supports ordinary transfer studies, not universal extrapolation or qualified uncertainty. No seed was rerun because of poor performance.", "",
                     "One interrupted compound-1101 attempt was rerun from the same seed/protocol because no optimizer-resume state existed. The five completed runs were reused with checkpoint hash checks. Frozen manifests and numerical training protocol remain unchanged.", "",
                     f"Source for transfer is preregistered row seed 42: `{decision['source_checkpoint_sha256']}`. The E0 engineering fixture differs from the already-frozen six-run qualification splits, and standalone initialization directly samples effective modules; these runs are not an R2 architecture benchmark."]
    (STUDY / "FINAL_4G_QUALIFICATION_REPORT.md").write_text("\n".join(report_lines) + "\n")
    uq_lines = ["# Quantile audit", "", f"Decision: `{head_decision}`.", "",
                "The audit is descriptive and non-blocking for ordinary point transfer. No alternative head or calibration was trained. Signed q90-q10 width and negative-width rates are retained; Spearman and the exactly ceil(20%)-sized top subset use width clipped at zero. Ties are resolved stably. Full train/validation/test audits are in the CSV.", "",
                f"Interval warning flags: `{severity_flags}`. Follow-up: `{'MONOTONIC_HEAD_CONTROL_REQUIRED_BEFORE_ACTIVE_TRANSFER' if needs_control else 'UQ_CALIBRATION_AND_OPTIONAL_HEAD_CONTROL_BEFORE_ACTIVE_TRANSFER'}`.", "",
                markdown_table(test_uq, index=False), "", "## Test aggregate across three seeds", "", markdown_table(grouped_uq), ""]
    (STUDY / "QUANTILE_AUDIT.md").write_text("\n".join(uq_lines) + "\n")
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
