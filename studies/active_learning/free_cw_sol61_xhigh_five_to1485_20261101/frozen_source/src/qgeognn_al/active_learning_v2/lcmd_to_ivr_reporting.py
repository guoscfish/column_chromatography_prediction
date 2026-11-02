"""Five matched methods; new test truth is gated behind all five continuations."""

import numpy as np
import pandas as pd

from . import lcmd_to_ivr_runner as runner
from . import lcmd_to_ivr_study as study
from .benchmark_reporting import metric_row
from .efficiency_reporting import METRICS, crossings, validate_curves
from .protocol import RestrictedLabelStore
from .sequential_protocol import label_budget


METHODS = ("random", "lcmd", "hybrid", "kernel_ivr", study.METHOD)
METRIC = "combined_normalized_RMSE"


def summarize(curves, full, scales):
    """Pure table transformation, also exercised on synthetic metrics in tests."""
    validate_curves(curves, full)
    expected = {(s, m, b) for s in study.CONFIRMATION_SEEDS for m in METHODS for b in study.ACTIVE_LABEL_BUDGETS}
    if set(zip(curves.outer_seed, curves.method, curves.active_label_count)) != expected:
        raise ValueError("complete five-method five-seed matrix required")
    curves = curves.copy()
    for target in ("V1", "V2"):
        curves[f"{target}_NRMSE"] = curves[f"{target}_RMSE"] / curves.outer_seed.map(lambda s: scales[int(s)][target])
    numeric = [*METRICS, "V1_NRMSE", "V2_NRMSE"]
    means = curves.groupby(["method", "active_label_count"])[numeric].mean().reset_index()
    targets, summaries = [], []
    scopes = [("seed", int(s), rows, full.loc[full.outer_seed.eq(s)].iloc[0]) for s, rows in curves.groupby("outer_seed")]
    scopes.append(("cohort_mean", None, means, full[list(METRICS)].mean()))
    for scope, seed, rows, reference in scopes:
        random = rows.loc[rows.method.eq("random")].sort_values("active_label_count")
        e0, er = random[METRIC].iloc[[0, -1]]
        ef = reference[METRIC]
        thresholds = {"T_R30": er}
        if e0 > ef:
            thresholds.update({f"T{p}": ef + (1 - p / 100) * (e0 - ef) for p in (80, 90, 95)})
        for method, values in rows.groupby("method"):
            values = values.sort_values("active_label_count")
            x, y = values.active_label_count.to_numpy(), values[METRIC].to_numpy()
            result = {"scope": scope, "outer_seed": seed, "method": method,
                      "normalized_AULC": float(np.trapezoid(y, x) / (x[-1] - x[0])),
                      "gap_closure_at_1005": float((e0-y[-1])/(e0-ef)) if e0 > ef else np.nan}
            result.update({f"{k}_at_1005": float(values[k].iloc[-1]) for k in numeric})
            for phase, lo, hi in study.PHASES:
                subset = values.loc[values.active_label_count.between(lo, hi)]
                result[f"{phase}_normalized_partial_AULC"] = float(np.trapezoid(subset[METRIC], subset.active_label_count) / (hi-lo))
            for name in ("T_R30", "T80", "T90", "T95"):
                crossing = crossings(x, y, thresholds[name]) if name in thresholds else {
                    "interpolated_labels": np.nan, "first_observed_labels": np.nan,
                    "sustained_labels": np.nan, "status": "NONPOSITIVE_FULL_GAP", "censor_budget": 1005}
                targets.append({"scope": scope, "outer_seed": seed, "method": method, "target": name,
                                "threshold": thresholds.get(name, np.nan), **crossing})
                result.update({f"{name}_{k}": v for k, v in crossing.items()})
            summaries.append(result)
    summary = pd.DataFrame(summaries)
    per_seed = summary.loc[summary.scope.eq("seed")]
    paired = []
    for seed, rows in per_seed.groupby("outer_seed"):
        rows = rows.set_index("method")
        for comparator in METHODS[:-1]:
            for metric in ("normalized_AULC", "late_normalized_partial_AULC", f"{METRIC}_at_1005", "V1_RMSE_at_1005", "V2_RMSE_at_1005"):
                paired.append({"outer_seed": seed, "comparator": comparator, "metric": metric,
                               "switch_minus_comparator": rows.loc[study.METHOD, metric] - rows.loc[comparator, metric]})
    return {"learning_curve_metrics": curves, "cohort_mean_learning_curve": means,
            "summary": summary, "labels_to_target": pd.DataFrame(targets), "paired_differences": pd.DataFrame(paired)}


def report():
    runner.global_freeze()
    old = pd.read_csv(study.BASELINE / "results/learning_curve_metrics.csv")
    pure = pd.read_csv(study.IVR / "results/learning_curve_metrics.csv")
    prefix = old.loc[old.method.eq("lcmd") & old.active_label_count.lt(study.POLICY.switch_active_labels)].copy()
    prefix["method"] = study.METHOD
    rows, access, costs, scales = [], [], [], {}
    for seed in study.CONFIRMATION_SEEDS:
        runtime = study.STUDY / f"runtime/seed_{seed}"
        lineage = study.read_json(study.STUDY / f"lineage/seed_{seed}.json")
        scales[seed] = lineage["target_scales"]
        partition = study._partition(seed)
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        selected = lineage["labeled_ids"][333:] + [i for r in range(study.switch_round(), len(study.ACTIVE_LABEL_BUDGETS)-1)
                    for i in study.read_json(runtime / f"round_{r:02d}/selection.json")["selected_ids"]]
        store = RestrictedLabelStore(study.SOURCE_DATA, partition)
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        access += [{"outer_seed": seed, **row} for row in store.audit]
        for r in range(study.switch_round(), len(study.ACTIVE_LABEL_BUDGETS)):
            record = runner.verify_record(runtime / f"round_{r:02d}/freeze.json")
            prediction = pd.read_csv(study.ROOT / record["model_directory"] / "predictions.csv.gz")
            if prediction.sample_id.astype(str).tolist() != test_ids:
                raise RuntimeError("test prediction identity/order mismatch")
            rows.append({"outer_seed": seed, "method": study.METHOD, "round": r,
                         **label_budget(study.ACTIVE_LABEL_BUDGETS[r]),
                         **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), scales[seed])})
        fits, acq = pd.read_csv(runtime / "fit_audit.csv"), pd.read_csv(runtime / "acquisition_runtime.csv")
        costs.append({"outer_seed": seed, "method": study.METHOD, "new_fit_count": int((~fits.reused).sum()),
                      "reused_prefix_fit_count": study.switch_round() + int(lineage["checkpoint_reused"]),
                      "new_epochs": int(fits.loc[~fits.reused, "epochs_run"].sum()),
                      "new_training_seconds": float(fits.loc[~fits.reused, "training_seconds"].sum()),
                      "new_gradient_seconds": float(acq.loc[~acq.gradient_reused, "gradient_seconds"].sum()),
                      "new_ivr_seconds": float(acq.ivr_seconds.sum())})
    curves = pd.concat([old, pure.loc[pure.method.eq("kernel_ivr")], prefix, pd.DataFrame(rows)], ignore_index=True)
    tables = summarize(curves, pd.read_csv(study.BASELINE / "results/full_data_reference.csv"), scales)
    tables.update(runtime=pd.DataFrame(costs), test_label_access_audit=pd.DataFrame(access))
    destination = study.STUDY / "results"
    destination.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    outcome = {"status": "COMPLETED_EXPLORATORY", "independent_confirmation": False,
               "primary_contrast": "paired late AULC and endpoint versus continuation with LCMD",
               "automatic_followup_authorized": False}
    study.atomic_json(study.STUDY / "decision.json", outcome)
    return outcome
