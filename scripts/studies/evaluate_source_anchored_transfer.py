#!/usr/bin/env python3
"""Evaluate all blind predictions only after validating the complete global freeze."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies import run_source_anchored_transfer as run
from src.qgeognn_al.evaluation.point import point_metrics

STUDY = run.STUDY
CALIBRATION = list(run.BASELINES[:-1])
MATCHED = {"source_anchored_shallow": "standard_shallow_finetune", "source_anchored_full": "standard_full_finetune"}
POINT = [f"{t}_{m}" for t in ("V1", "V2") for m in ("r2", "rmse", "mae")]+["normalized_rmse", "combined_normalized_rmse"]


def lock_evaluation():
    config = {"evaluator_sha256": run.sha256_file(Path(__file__)),
              "preregistration_sha256": run.sha256_file(STUDY / "MODEL_PREREGISTRATION.md"),
              "relative_error_floor_ml": 1., "numerical_tie_tolerance": 1e-7,
              "source_stabilization_seed_metric": "arithmetic mean source combined NRMSE drift across all four budgets",
              "strata_quantiles": [.33, .67, .90], "EA_edges": [.1, .5],
              "target_magnitude_strata_role": "post-freeze test characterization only"}
    path = STUDY / "evaluation_protocol.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("frozen evaluator changed")
    if not path.exists():
        run.atomic_json(path, config)


def validate_global_freeze():
    run.prepare()
    freeze = run.verify_manifest(STUDY / "all_predictions_frozen.json")
    expected = {str((Path("contexts") / run.context_path(*c) / "frozen.json")) for c in run.expected_contexts()}
    if set(freeze["files"]) != expected or freeze["contexts"] != 120 or freeze["fits"] != 480:
        raise RuntimeError("missing/extra contexts in freeze")
    if freeze["protocol_sha256"] != run.sha256_file(STUDY / "protocol.json"):
        raise RuntimeError("frozen protocol mismatch")
    for name in freeze["files"]:
        record = run.verify_manifest(STUDY / name)
        if record["protocol_sha256"] != freeze["protocol_sha256"]:
            raise RuntimeError("context protocol mismatch")
    return freeze


def metric(truth, pred, scales):
    six = np.column_stack([pred[:, 0]]*3+[pred[:, 1]]*3)
    result = point_metrics(truth, six, scales)
    result["normalized_rmse"] = .5*sum(result[f"{t}_rmse"]/scales[t] for t in ("V1", "V2"))
    return result


def error_summary(error):
    return {"n": len(error), "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
            "mae": float(np.mean(np.abs(error))) if len(error) else np.nan,
            "median_absolute_error": float(np.median(np.abs(error))) if len(error) else np.nan}


def quantile_masks(value, boundaries):
    q33, q67, q90 = boundaries
    return {"low": value <= q33, "mid": (value > q33) & (value <= q67),
            "high": value > q67, "top10": value >= q90}


def sensitivity(truth, prediction, features, train_features, keys):
    extra, rows = {}, []
    for j, target in enumerate(("V1", "V2")):
        error = prediction[:, j]-truth[:, j]
        group = pd.DataFrame({"compound": features.canonical_smiles.to_numpy(), "ae": abs(error), "se": error**2})
        compounds = group.groupby("compound").agg({"ae": "mean", "se": "mean"})
        extra[f"{target}_macro_compound_mae"] = compounds.ae.mean()
        extra[f"{target}_macro_compound_rmse"] = np.sqrt(compounds.se).mean()
        eligible = abs(truth[:, j]) >= 1.
        extra[f"{target}_median_relative_absolute_error"] = float(np.median(abs(error[eligible])/abs(truth[eligible, j]))) if eligible.any() else np.nan
        extra[f"{target}_relative_excluded_rows"] = int((~eligible).sum())
        extra[f"{target}_median_absolute_error"] = float(np.median(abs(error)))
        source_bounds = np.quantile(train_features[f"source_{target}"], [.33, .67, .90])
        magnitude_bounds = np.quantile(truth[:, j], [.33, .67, .90])
        dimensions = {"source_q50": (features[f"source_{target}"].to_numpy(), source_bounds),
                      "target_magnitude": (truth[:, j], magnitude_bounds)}
        for dimension, (values, bounds) in dimensions.items():
            for level, mask in quantile_masks(values, bounds).items():
                rows.append({**keys, "target": target, "dimension": dimension, "level": level,
                             "q33": bounds[0], "q67": bounds[1], "q90": bounds[2], **error_summary(error[mask])})
        ea = features.EA_fraction.to_numpy()
        for level, mask in {"low": ea <= .1, "mid": (ea > .1) & (ea <= .5), "high": ea > .5}.items():
            rows.append({**keys, "target": target, "dimension": "EA", "level": level,
                         "q33": np.nan, "q67": np.nan, "q90": np.nan, **error_summary(error[mask])})
    return extra, rows


def pair_scores(frame, value, endpoint):
    rows = []
    for (column, protocol), group in frame.groupby(["column", "protocol"]):
        pivot = group.pivot(index="seed", columns="method", values=value)
        for method in run.METHODS:
            for reference in [*run.BASELINES, *run.METHODS]:
                if reference == method:
                    continue
                delta = pivot[method]-pivot[reference]
                gain = 1-pivot[method].mean()/pivot[reference].mean()
                no_output_regression = True
                if endpoint == "budget100":
                    means = group.groupby("method")[["V1_rmse", "V2_rmse"]].mean()
                    no_output_regression = bool((means.loc[method] <= means.loc[reference]).all())
                wins = int((delta < -1e-7).sum())
                rows.append({"column": column, "protocol": protocol, "method": method, "reference": reference,
                             "endpoint": endpoint, "relative_gain": gain, "mean_delta": delta.mean(),
                             "median_delta": delta.median(), "std_delta": delta.std(), "wins": wins,
                             "ties": int((abs(delta) <= 1e-7).sum()), "seeds": len(delta),
                             "no_output_regression": no_output_regression,
                             "material": bool(gain >= .05 and delta.median() < 0 and wins >= 4 and no_output_regression),
                             "stronger_10pct": bool(gain >= .10 and delta.median() < 0 and wins >= 4 and no_output_regression)})
    return pd.DataFrame(rows)


def replicated(contexts, pairs, method, refs):
    both = any((c, "row") in contexts and (c, "compound") in contexts for c in run.COLUMNS)
    compound = []
    for c, p in contexts:
        if p != "compound":
            continue
        row = pairs.loc[pairs.column.eq(c) & pairs.protocol.eq("row") & pairs.method.eq(method)
                        & pairs.reference.isin(refs) & pairs.endpoint.eq("aulc")]
        if len(row) == len(refs) and (row.relative_gain >= -.05).all():
            compound.append(c)
    return both or len(set(compound)) >= 2


def material_gate(pairs, method, refs, endpoint):
    frame = pairs.loc[pairs.method.eq(method) & pairs.reference.isin(refs) & pairs.endpoint.eq(endpoint)]
    contexts = [(c, p) for (c, p), g in frame.groupby(["column", "protocol"])
                if len(g) == len(refs) and g.material.all()]
    return {"replicated": bool(replicated(contexts, pairs, method, refs)), "contexts": contexts}


def decision(pairs, aulc, drift):
    gates = {}
    for method in run.METHODS:
        refs = CALIBRATION+([MATCHED[method]] if method in MATCHED else [])
        gates[method] = {endpoint: material_gate(pairs, method, refs, endpoint) for endpoint in ("aulc", "budget100")}
    a = any(gates[m]["aulc"]["replicated"] and gates[m]["budget100"]["replicated"]
            and any(c in ("25g", "40g") and p == "compound" for c, p in gates[m]["budget100"]["contexts"])
            for m in MATCHED)
    anchoring_increment = {m: {e: material_gate(pairs, m, [n], e) for e in ("aulc", "budget100")} for m, n in MATCHED.items()}
    extra_anchor = any(g["aulc"]["replicated"] and g["budget100"]["replicated"] for g in anchoring_increment.values())
    b = any(gates[m]["aulc"]["replicated"] and gates[m]["budget100"]["replicated"] for m in MATCHED.values()) and not extra_anchor
    stability_rows, stabilizes, calibration_stronger_stability = [], {}, {}
    for method, reference in MATCHED.items():
        contexts, calibration_contexts = [], []
        for column in run.COLUMNS:
            for protocol in ("row", "compound"):
                subset = aulc.loc[aulc.column.eq(column) & aulc.protocol.eq(protocol)]
                pivot = subset.pivot(index="seed", columns="method", values="normalized_aulc")
                source = drift.loc[drift.column.eq(column) & drift.protocol.eq(protocol)].groupby(["seed", "method"]).source_combined_nrmse_drift.mean().unstack()
                wins = int((source[method]-source[reference] < -1e-7).sum())
                passed = bool(wins >= 4 and pivot[method].std() < pivot[reference].std() and pivot[method].mean() <= pivot[reference].mean())
                calibration_stronger = bool(pivot[CALIBRATION].mean().min() < pivot[method].mean())
                stability_rows.append({"column": column, "protocol": protocol, "method": method, "reference": reference,
                                       "source_drift_wins": wins, "target_aulc_std": pivot[method].std(),
                                       "reference_aulc_std": pivot[reference].std(), "mean_aulc_delta": (pivot[method]-pivot[reference]).mean(),
                                       "stability_gate": passed, "calibration_lower_mean_aulc": calibration_stronger})
                if passed:
                    contexts.append((column, protocol))
                    if calibration_stronger:
                        calibration_contexts.append((column, protocol))
        stabilizes[method] = {"replicated": bool(replicated(contexts, pairs, method, [reference])), "contexts": contexts}
        calibration_stronger_stability[method] = {"replicated": bool(replicated(calibration_contexts, pairs, method, [reference])), "contexts": calibration_contexts}
    # Stronger calibration means no neural arm meets the joint material endpoints.
    neural_supported = any(g["aulc"]["replicated"] and g["budget100"]["replicated"] for g in gates.values())
    c = any(g["replicated"] for g in calibration_stronger_stability.values()) and not neural_supported
    d = True
    for column in ("25g", "40g"):
        sub = aulc.loc[aulc.column.eq(column) & aulc.protocol.eq("compound")]
        strongest = sub.loc[sub.method.isin(CALIBRATION)].groupby("method").normalized_aulc.mean().idxmin()
        pivot = sub.pivot(index="seed", columns="method", values="normalized_aulc")
        for method in run.METHODS:
            d &= bool(pivot[method].mean() >= 1.05*pivot[strongest].mean() and (pivot[method]-pivot[strongest] > 1e-7).sum() >= 4)
    name = ("SOURCE_ANCHORED_SHARED_REPRESENTATION_SUPPORTED" if a else
            "LATENT_FINETUNING_SUPPORTED_BUT_SOURCE_ANCHORING_NOT_NEEDED" if b else
            "SOURCE_ANCHORING_STABILIZES_NEURAL_TRANSFER_BUT_CALIBRATION_REMAINS_STRONGER" if c else
            "CURRENT_QGEOGNN_REPRESENTATION_NOT_SUFFICIENT_FOR_LOW_LABEL_CROSS_COLUMN_TRANSFER" if d else
            "INCONCLUSIVE_MIXED_EVIDENCE")
    pd.DataFrame(stability_rows).to_csv(STUDY / "stability_comparisons.csv", index=False)
    return {"decision": name, "gates": gates, "anchoring_increment": anchoring_increment,
            "stabilization": stabilizes, "calibration_stronger_stability": calibration_stronger_stability,
            "outcome_flags": {"A": bool(a), "B": bool(b), "C": bool(c), "D": bool(d)},
            "developmental_evidence": True, "active_learning_started": False, "test_driven_models_added": 0}


def evaluate():
    lock_evaluation()
    freeze = validate_global_freeze()
    event = STUDY / "test_evaluation_started.json"
    if not event.exists():
        run.atomic_json(event, {"unix_time": time.time(), "all_predictions_frozen_sha256": run.sha256_file(STUDY / "all_predictions_frozen.json"),
                               "evaluation_protocol_sha256": run.sha256_file(STUDY / "evaluation_protocol.json")})
    # No call to selected_truth appears above the complete freeze verification.
    scales = run.torch.load(run.SOURCE, weights_only=False)["preprocessing"]["target_scales"]
    features = {c: pd.read_csv(run.CONDITIONAL / f"features_{c}.csv",
                            usecols=["sample_id", "canonical_smiles", "EA_fraction", "source_V1", "source_V2"]).set_index("sample_id") for c in run.COLUMNS}
    rows, strata, drifts, training = [], [], [], []
    for name in freeze["files"]:
        output = (STUDY / name).parent
        usage = json.loads((output / "label_usage.json").read_text())
        keys = {k: usage[k] for k in ("column", "protocol", "seed", "budget", "actual_budget")}
        table = pd.read_csv(output / "predictions_blind.csv.gz").set_index("sample_id")
        if list(table.index) != usage["test"]:
            raise RuntimeError("test identity order mismatch")
        truth = run.selected_truth(run.target_path(usage["column"]), table.index.tolist())
        if not np.isfinite(truth).all():
            raise RuntimeError("nonfinite target truth")
        focal = features[usage["column"]]
        for method in [*run.BASELINES, *run.METHODS]:
            pred = table[[f"{method}_V1", f"{method}_V2"]].to_numpy()
            if not np.isfinite(pred).all():
                raise RuntimeError("nonfinite predictions")
            extra, parts = sensitivity(truth, pred, focal.loc[table.index], focal.loc[usage["gradient_train"]], {**keys, "method": method})
            rows.append({**keys, "method": method, **metric(truth, pred, scales), **extra})
            strata.extend(parts)
        audit = json.loads((output / "fit_audit.json").read_text())
        probe = pd.read_csv(output / "source_probe_blind.csv.gz").set_index("sample_id")
        if list(probe.index) != usage["source_probe_ids"]:
            raise RuntimeError("source probe identity mismatch")
        source_truth = run.selected_truth(run.SOURCE_DATA, probe.index.tolist())
        before = metric(source_truth, probe[["initial_V1", "initial_V2"]].to_numpy(), scales)
        for method, fit in audit.items():
            after = metric(source_truth, probe[[f"{method}_V1", f"{method}_V2"]].to_numpy(), scales)
            if not fit["fit_success"] or fit["source_head_l2_drift"] != 0:
                raise RuntimeError("failed fit or source-head drift")
            if fit["contract"]["target_train_ids"] != usage["gradient_train"] or fit["contract"]["target_validation_ids"] != usage["validation"]:
                raise RuntimeError("fit ledger mismatch")
            drifts.append({**keys, "method": method, **{f"before_{k}": v for k, v in before.items()},
                           **{f"after_{k}": v for k, v in after.items()},
                           **{f"delta_{k}": after[k]-before[k] for k in POINT},
                           "source_combined_nrmse_drift": after["combined_normalized_rmse"]-before["combined_normalized_rmse"],
                           **{k: v for k, v in fit.items() if k.endswith("_l2_drift")}})
            training.append({**keys, "method": method, **{k: v for k, v in fit.items() if not isinstance(v, (dict, list))},
                             **{f"best_{k}": v for k, v in fit["best_epoch_losses"].items()},
                             **{f"initial_{k}": v for k, v in fit["initial_epoch_losses"].items()},
                             **{f"final_{k}": v for k, v in fit["final_epoch_losses"].items()},
                             "target_purchased_labels": usage["actual_budget"], "other_target_labels_used": 0})
    metrics, drift = pd.DataFrame(rows), pd.DataFrame(drifts)
    if len(metrics) != 1080 or len(drift) != 480:
        raise RuntimeError("missing method/context")
    metrics.to_csv(STUDY / "all_metrics.csv", index=False)
    numeric = [n for n in metrics.select_dtypes(include="number") if n not in ("seed", "budget", "actual_budget")]
    metrics.groupby(["column", "protocol", "budget", "method"])[numeric].agg(["mean", "std", "median", "min", "max"]).to_csv(STUDY / "aggregate_metrics.csv")
    b100 = metrics.loc[metrics.budget.eq(100)]
    b100.to_csv(STUDY / "budget100_absolute_metrics.csv", index=False)
    pd.DataFrame(strata).to_csv(STUDY / "error_stratification.csv", index=False)
    drift.to_csv(STUDY / "source_drift_metrics.csv", index=False)
    pd.DataFrame(training).to_csv(STUDY / "training_audit.csv", index=False)
    areas = []
    for keys, group in metrics.groupby(["column", "protocol", "seed", "method"]):
        group = group.sort_values("budget")
        if list(group.budget) != list(run.BUDGETS):
            raise RuntimeError("budget gap")
        actual = group.actual_budget.to_numpy()
        areas.append({**dict(zip(["column", "protocol", "seed", "method"], keys)),
                      "normalized_aulc": np.trapezoid(group.normalized_rmse, group.budget)/70,
                      "actual_budget_aulc": np.trapezoid(group.normalized_rmse, actual)/(actual[-1]-actual[0])})
    aulc = pd.DataFrame(areas)
    aulc.to_csv(STUDY / "aulc_by_seed.csv", index=False)
    pairs = pd.concat([pair_scores(aulc, "normalized_aulc", "aulc"),
                       pair_scores(b100, "combined_normalized_rmse", "budget100")], ignore_index=True)
    pairs.to_csv(STUDY / "paired_comparisons.csv", index=False)
    result = decision(pairs, aulc, drift)
    run.atomic_json(STUDY / "decision.json", result)
    audit = {"contexts": 120, "new_fits": 480, "metric_rows": len(metrics), "failed_fits": 0,
             "missing_contexts": 0, "nonfinite_predictions": 0, "all_predictions_frozen_before_test": True,
             "protocol_sha256": run.sha256_file(STUDY / "protocol.json"),
             "prediction_freeze_sha256": run.sha256_file(STUDY / "all_predictions_frozen.json"),
             "evaluation_protocol_sha256": run.sha256_file(STUDY / "evaluation_protocol.json"),
             "other_target_labels_used": 0, "test_labels_used_for_fit_or_selection": 0,
             "KMP_DUPLICATE_LIB_OK": "TRUE", "cpu_threads_per_worker": 1,
             "null_metric_policy": "empty strata or undefined R2/relative subset are NA; never treated as zero error",
             "completed_at_unix": time.time()}
    run.atomic_json(STUDY / "execution_audit.json", audit)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    lock_evaluation() if args.lock else evaluate()
