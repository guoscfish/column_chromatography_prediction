#!/usr/bin/env python3
"""Frozen two-experiment extension; fit all predictions before test evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.studies import run_cross_column_transfer as previous
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.evaluation.reporting import markdown_table
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.training.predictor import atomic_json, predict
from src.qgeognn_al.transfer.calibration import fit_affine, fit_scale_only
from src.qgeognn_al.transfer.residual_diagnostics import (
    donor_training_rows, fit_column_calibration, fit_monotone, validation_score,
)

STUDY = ROOT / "studies/transfer/residual_diagnostics"
COLUMNS = tuple(previous.COLUMNS)
FEATURES = ["sample_id", "canonical_smiles", "PE/EA", "loading solvent", "Density g/ml",
            "V/ul", "Volume of loading solvent/ul", "Flow mL/min"]
SPLINE_STRENGTHS = (.1, 1.)
SHARING_STRENGTHS = (0., .1, 1.)
NEW_METHODS = ("monotone_spline", "nonlinear_policy", "linear_policy",
               "shared_column_affine", "local_identity_shrinkage")
REFERENCES = ("scale_only", "affine", "target_head_only")
POINT_NAMES = ("V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae",
               "normalized_rmse", "combined_normalized_rmse")


def contract():
    old = json.loads((previous.STUDY / "protocol.json").read_text())
    if sha256_file(previous.SOURCE) != old["source_checkpoint_sha256"]:
        raise RuntimeError("qualified source hash changed")
    files = [previous.SOURCE, previous.STUDY / "protocol.json",
             previous.STUDY / "splits/schedule_manifest.csv",
             ROOT / "src/qgeognn_al/models/qgeognn_v2.py",
             ROOT / "src/qgeognn_al/transfer/calibration.py",
             ROOT / "src/qgeognn_al/training/predictor.py",
             ROOT / "src/qgeognn_al/evaluation/point.py"]
    for column in COLUMNS:
        if sha256_file(previous.canonical_path(column)) != old["canonical_sha256"][column]:
            raise RuntimeError("frozen canonical data changed")
        files.extend([previous.canonical_path(column), previous.graph_path(column)])
    if sha256_file(files[2]) != old["schedule_sha256"]:
        raise RuntimeError("frozen roles changed")
    return {
        "base_commit": "61f20c9d10e1e5cfbd523ae0cac448f55de74eb2",
        "protected_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        "design_sha256": sha256_file(ROOT / "NEXT_TRANSFER_MODEL_AUDIT.md"),
        "implementation_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p) for p in (
                Path(__file__), ROOT / "src/qgeognn_al/transfer/residual_diagnostics.py")},
        "columns": list(COLUMNS), "seeds": list(previous.SEEDS), "budgets": list(previous.BUDGETS),
        "protocols": list(previous.PROTOCOLS), "contexts": 120,
        "spline_strengths": list(SPLINE_STRENGTHS), "spline_knots": [1/3, 2/3],
        "sharing_strengths": list(SHARING_STRENGTHS), "selection": "focal validation only; ties prefer simpler",
        "nonlinear_primary": "nonlinear_policy", "nonlinear_required_references": ["scale_only", "affine", "linear_policy"],
        "shared_primary": "shared_column_affine", "shared_required_references": ["scale_only", "affine", "local_identity_shrinkage"],
        "gate": "at least two different compound columns: >=5% relative mean AULC gain, negative median delta, >=4/5 wins; row degradation <=5%; 25g/40g also beat head",
        "shared_additional_gate": "compound equal-column portfolio AULC >=5% better than affine, scale and local shrinkage",
        "budget": "single-column B for nonlinear; shared vs independent three-column portfolio costs sum(actual budgets), including validation",
        "compound_donor_purge": "focal validation and test compounds removed from all donor training",
        "test_tuning": False, "active_learning": False, "adaptive_readout": False,
        "historically_consumed_test": True,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
    }


def lock_contract():
    STUDY.mkdir(parents=True, exist_ok=True)
    current = contract()
    path = STUDY / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise RuntimeError("frozen experiment contract changed; do not silently rerun")
    atomic_json(path, current)
    return current


def source_predictions(config):
    """Explicit feature whitelist excludes times, volumes and all label proxies."""
    payload = torch.load(previous.SOURCE, map_location="cpu", weights_only=False)
    model = load_predictor_checkpoint(previous.SOURCE)
    scales = payload["preprocessing"]["target_scales"]
    frames = {}
    for column in COLUMNS:
        path = STUDY / "runtime" / f"source_{column}.csv"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if record["protocol"] != config or record["sha256"] != sha256_file(path):
                raise RuntimeError("source cache contract changed")
            frames[column] = pd.read_csv(path).set_index("sample_id")
            continue
        data = pd.read_csv(previous.canonical_path(column), usecols=FEATURES)
        placeholder = data.assign(V1_ml=0., V2_ml=0.)
        atom, angle = build_model_data(placeholder, previous.combined_cache(column), pd.DataFrame(), payload["preprocessing"]["scaler"])
        pred = predict(model, atom, angle, np.arange(len(data)))[1][:, [1, 4]]
        data[["source_V1", "source_V2"]] = pred
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        atomic_json(meta, {"protocol": config, "sha256": sha256_file(path), "target_label_cells_parsed": 0})
        frames[column] = data.set_index("sample_id")
        print(f"source inference {column}: {len(data)} label-free rows", flush=True)
    return frames, scales


def directory(column, protocol, seed, budget):
    return STUDY / "contexts" / column / protocol / f"seed_{seed}" / f"budget_{budget}"


def read_truth(column, ids):
    return previous.load_selected_truth(previous.canonical_path(column), set(ids)).set_index("sample_id").loc[
        ids, ["V1_ml", "V2_ml"]].to_numpy(float)


def fit_context(context, column, split, seed, budget, sources, scales):
    output = directory(column, split, seed, budget)
    done = output / "frozen.json"
    if done.exists():
        record = json.loads(done.read_text())
        for name, digest in record["files"].items():
            if sha256_file(output / name) != digest:
                raise RuntimeError("frozen context changed")
        return
    focal = context.loc[context.column.eq(column)]
    ids = {role: sorted(focal.loc[focal.role.eq(role), "sample_id"]) for role in ("gradient_train", "validation", "test")}
    tr, va, te = ids["gradient_train"], ids["validation"], ids["test"]
    if set(tr) & set(va) or (set(tr) | set(va)) & set(te):
        raise RuntimeError("focal identity leakage")
    frame = sources[column]
    def x(index):
        return frame.loc[index, ["source_V1", "source_V2"]].to_numpy(float)
    train_x, valid_x, test_x = x(tr), x(va), x(te)
    train_y, valid_y = read_truth(column, tr), read_truth(column, va)
    scale_vector = np.array([scales[t] for t in ("V1", "V2")])
    valid_predictions = {
        "scale_only": fit_scale_only(train_y, train_x, valid_x).prediction,
        "affine": fit_affine(train_y, train_x, valid_x).prediction,
    }
    predictions = {
        "scale_only": fit_scale_only(train_y, train_x, test_x).prediction,
        "affine": fit_affine(train_y, train_x, test_x).prediction,
    }
    selection = [{"method": name, "strength": None, "score": validation_score(valid_y, value, scale_vector)}
                 for name, value in valid_predictions.items()]
    splines = [fit_monotone(train_x, train_y, scale_vector, value) for value in SPLINE_STRENGTHS]
    spline_scores = [validation_score(valid_y, model.predict(valid_x), scale_vector) for model in splines]
    # Equal scores prefer stronger regularization for the fixed spline family.
    best_spline = min(range(len(splines)), key=lambda i: (spline_scores[i], -SPLINE_STRENGTHS[i]))
    predictions["monotone_spline"] = splines[best_spline].predict(test_x)
    selection.extend({"method": "monotone_spline", "strength": s, "score": v}
                     for s, v in zip(SPLINE_STRENGTHS, spline_scores))
    linear_choice = min(selection[:2], key=lambda item: item["score"])["method"]
    nonlinear_choice = min(
        [(selection[0]["score"], 0, "scale_only"), (selection[1]["score"], 1, "affine"),
         (spline_scores[best_spline], 2, "monotone_spline")])[2]
    predictions["linear_policy"] = predictions[linear_choice]
    predictions["nonlinear_policy"] = predictions[nonlinear_choice]

    donor_rows = donor_training_rows(context, column, split)
    joint_x, joint_y, joint_ids = [], [], {}
    for donor in COLUMNS:
        allowed = donor_rows.loc[donor_rows.column.eq(donor)]
        if allowed.canonical_smiles.nunique() < 2:
            raise RuntimeError("STOP_SHARED_CONTEXT: fewer than two donor compounds")
        allowed_ids = sorted(allowed.sample_id)
        joint_ids[donor] = allowed_ids
        joint_x.append(sources[donor].loc[allowed_ids, ["source_V1", "source_V2"]].to_numpy(float))
        joint_y.append(read_truth(donor, allowed_ids))
    focal_index = COLUMNS.index(column)
    sharing_audit = {}
    for method, shared in (("shared_column_affine", True), ("local_identity_shrinkage", False)):
        models = [fit_column_calibration(joint_x, joint_y, [previous.COLUMNS[c]/4 for c in COLUMNS],
                  scale_vector, strength, shared=shared) for strength in SHARING_STRENGTHS]
        np.testing.assert_allclose(models[0].predict(test_x, focal_index), predictions["affine"], rtol=1e-9, atol=1e-9)
        scores = [validation_score(valid_y, model.predict(valid_x, focal_index), scale_vector) for model in models]
        chosen = min(range(len(scores)), key=lambda i: (scores[i], SHARING_STRENGTHS[i]))
        predictions[method] = models[chosen].predict(test_x, focal_index)
        selection.extend({"method": method, "strength": strength, "score": score}
                         for strength, score in zip(SHARING_STRENGTHS, scores))
        sharing_audit[method] = {"selected_strength": SHARING_STRENGTHS[chosen],
                                "coefficients": models[chosen].coefficients.tolist()}
    output.mkdir(parents=True, exist_ok=True)
    result = frame.loc[te, ["canonical_smiles", "source_V1", "source_V2"]].reset_index()
    for name, value in predictions.items():
        if not np.isfinite(value).all():
            raise RuntimeError("non-finite frozen predictions")
        result[[f"{name}_V1", f"{name}_V2"]] = value
    result.to_csv(output / "predictions_blind.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    portfolio = context.loc[context.role.isin(["gradient_train", "validation"])]
    purged = context.loc[context.role.eq("gradient_train") & ~context.sample_id.isin(donor_rows.sample_id)]
    actual_budget = int(focal.actual_budget.iloc[0])
    # Every comparison portfolio purchases the same per-column train/valid IDs.
    usage = {
        "column": column, "protocol": split, "seed": seed, "planned_budget": budget,
        "actual_budget": actual_budget, "portfolio_planned_budget": 3*budget,
        "portfolio_actual_budget": len(portfolio), "portfolio_revealed_ids": sorted(portfolio.sample_id),
        "gradient_train_ids": tr, "validation_ids": va, "test_ids": te,
        "joint_train_ids": joint_ids, "purged_donor_ids": sorted(purged.sample_id),
        "donor_validation_labels_used": 0, "test_labels_used_for_fit_or_selection": 0,
        "focal_compound_isolation": split == "compound", "lambda_zero_equals_independent_affine": True,
        "shared_cost_interpretation": "three-column portfolio; not a single-column B-label result",
    }
    atomic_json(output / "label_usage.json", usage)
    coverage = {t: {"train_min": float(train_x[:, j].min()), "train_max": float(train_x[:, j].max()),
                   "test_outside_train_fraction": float(np.mean((test_x[:, j] < train_x[:, j].min()) |
                                                                (test_x[:, j] > train_x[:, j].max())))}
                for j, t in enumerate(("V1", "V2"))}
    atomic_json(output / "fit_audit.json", {
        "selection": selection, "linear_policy": linear_choice, "nonlinear_policy": nonlinear_choice,
        "selected_spline_strength": SPLINE_STRENGTHS[best_spline],
        "spline_candidates": [model.audit() for model in splines], "source_range": coverage,
        **sharing_audit,
    })
    names = ("predictions_blind.csv.gz", "fit_audit.json", "label_usage.json")
    atomic_json(done, {"files": {name: sha256_file(output/name) for name in names},
                      "test_truth_read_in_fit": False})


def fit_all():
    torch.set_num_threads(1)
    config = lock_contract()
    sources, scales = source_predictions(config)
    schedule = pd.read_csv(previous.STUDY / "splits/schedule_manifest.csv")
    for (split, seed, budget), context in schedule.groupby(["protocol", "outer_seed", "planned_budget"], sort=True):
        for column in COLUMNS:
            fit_context(context, column, split, int(seed), int(budget), sources, scales)
        print(f"frozen {split} seed={seed} budget={budget} (three columns)", flush=True)
    frozen = sorted((STUDY / "contexts").glob("*/*/seed_*/budget_*/frozen.json"))
    if len(frozen) != config["contexts"]:
        raise RuntimeError("incomplete fit phase: test evaluation forbidden")
    atomic_json(STUDY / "all_predictions_frozen.json", {
        "contexts": len(frozen), "protocol_sha256": sha256_file(STUDY/"protocol.json"),
        "files": {str(p.relative_to(STUDY)): sha256_file(p) for p in frozen},
        "source_scales": scales, "test_evaluation_started": False,
    })


def paired_table(aulc):
    rows = []
    for (column, split), group in aulc.groupby(["column", "protocol"]):
        pivot = group.pivot(index="seed", columns="method", values="normalized_aulc")
        for method in NEW_METHODS:
            for reference in (*REFERENCES, "linear_policy", "local_identity_shrinkage"):
                if method == reference:
                    continue
                delta = pivot[method] - pivot[reference]
                relative = -delta.mean() / pivot[reference].mean()
                rows.append({"column": column, "protocol": split, "method": method, "reference": reference,
                    "mean_delta": delta.mean(), "median_delta": delta.median(), "std_delta": delta.std(ddof=1),
                    "min_delta": delta.min(), "max_delta": delta.max(), "wins": int((delta < 0).sum()),
                    "seeds": len(delta), "relative_mean_gain": relative,
                    "mean_paired_relative_gain": float((-delta/pivot[reference]).mean()),
                    "stable_material": bool(relative >= .05 and delta.median() < 0 and (delta < 0).sum() >= 4)})
    return pd.DataFrame(rows)


def decide(paired, portfolio):
    qualifying = {}
    for method, refs in (("nonlinear_policy", ["affine", "scale_only", "linear_policy"]),
                         ("shared_column_affine", ["affine", "scale_only", "local_identity_shrinkage"])):
        columns = []
        for column in COLUMNS:
            group = paired.loc[paired.column.eq(column) & paired.method.eq(method)]
            comp = group.loc[group.protocol.eq("compound")].set_index("reference")
            row = group.loc[group.protocol.eq("row")].set_index("reference")
            passed = bool(comp.loc[refs, "stable_material"].all() and (row.loc[refs, "relative_mean_gain"] >= -.05).all())
            if column in ("25g", "40g"):
                passed &= bool(comp.loc["target_head_only", "stable_material"])
            if passed:
                columns.append(column)
        qualifying[method] = columns
    portfolio_gate = True
    frame = portfolio.loc[portfolio.protocol.eq("compound")].groupby("method").normalized_aulc.mean()
    for ref in ("affine", "scale_only", "local_identity_shrinkage"):
        portfolio_gate &= bool(1 - frame["shared_column_affine"] / frame[ref] >= .05)
    supported = []
    if len(qualifying["nonlinear_policy"]) >= 2:
        supported.append("NONLINEAR_CALIBRATION_WARRANTED")
    if len(qualifying["shared_column_affine"]) >= 2 and portfolio_gate:
        supported.append("SHARED_COLUMN_CONTEXT_WARRANTED")
    return {
        "decision": supported[0] if supported else "NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA",
        "supported_conclusions": supported, "qualifying_compound_columns": qualifying,
        "shared_portfolio_gate": bool(portfolio_gate), "adaptive_readout_tested": False,
        "adaptive_readout_warranted": False, "additional_methods_after_test": 0,
        "scope": "developmental frozen-split evidence for these two bounded families, not proof of irreducible noise",
        "source_unseen_ood_claim": False, "mass_flow_physical_law_claim": False,
    }


def evaluate():
    config = lock_contract()
    frozen_path = STUDY / "all_predictions_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen["contexts"] != 120 or frozen["protocol_sha256"] != sha256_file(STUDY/"protocol.json"):
        raise RuntimeError("evaluation requires 120 frozen contexts under the same protocol")
    rows, usage_rows, selection_rows, checks = [], [], [], []
    for relative, digest in frozen["files"].items():
        done = STUDY / relative
        if sha256_file(done) != digest:
            raise RuntimeError("frozen manifest changed")
        for name, expected in json.loads(done.read_text())["files"].items():
            if sha256_file(done.parent/name) != expected:
                raise RuntimeError("predictions or fitting audit changed")
    # Only this phase reads focal test truth and historical test-baseline outputs.
    for relative in frozen["files"]:
        output = (STUDY/relative).parent
        usage = json.loads((output/"label_usage.json").read_text())
        keys = {k: usage[k] for k in ("column", "protocol", "seed", "planned_budget", "actual_budget",
                                      "portfolio_planned_budget", "portfolio_actual_budget")}
        old = previous.STUDY / keys["column"] / keys["protocol"] / f"seed_{keys['seed']}" / f"budget_{keys['planned_budget']}"
        completed = json.loads((old/"completion.json").read_text())
        if sha256_file(old/"predictions.csv.gz") != completed["files"]["predictions.csv.gz"]:
            raise RuntimeError("historical baseline predictions hash changed")
        frame = pd.read_csv(output/"predictions_blind.csv.gz").set_index("sample_id")
        baseline = pd.read_csv(old/"predictions.csv.gz").set_index("sample_id").loc[frame.index]
        truth = read_truth(keys["column"], frame.index.tolist())
        np.testing.assert_allclose(truth, baseline[["V1_true", "V2_true"]], rtol=1e-10, atol=1e-10)
        maximum = 0.
        for method in ("scale_only", "affine"):
            columns = [f"{method}_V1", f"{method}_V2"]
            error = float(np.max(np.abs(frame[columns].to_numpy() - baseline[columns].to_numpy())))
            maximum = max(maximum, error)
            # Historical neural inference/labels use float32; the new calibrators use CSV float64.
            np.testing.assert_allclose(frame[columns], baseline[columns], rtol=3e-5, atol=1e-4)
        for method in (*REFERENCES, *NEW_METHODS):
            columns = [f"{method}_V1", f"{method}_V2"]
            values = baseline[columns].to_numpy() if method in REFERENCES else frame[columns].to_numpy()
            metrics = previous.metric_record(truth, values, frozen["source_scales"])
            rows.append({**keys, "method": method, **metrics})
        audit = json.loads((output/"fit_audit.json").read_text())
        selection_rows.append({**keys, "linear_policy": audit["linear_policy"], "nonlinear_policy": audit["nonlinear_policy"],
            "spline_strength": audit["selected_spline_strength"],
            "shared_strength": audit["shared_column_affine"]["selected_strength"],
            "local_strength": audit["local_identity_shrinkage"]["selected_strength"],
            **{f"{t}_test_outside_train_fraction": audit["source_range"][t]["test_outside_train_fraction"] for t in ("V1", "V2")}})
        usage_rows.append({**keys, "train_rows": len(usage["gradient_train_ids"]), "validation_rows": len(usage["validation_ids"]),
            "joint_train_rows": sum(map(len, usage["joint_train_ids"].values())), "purged_donor_rows": len(usage["purged_donor_ids"]),
            "test_rows": len(usage["test_ids"]), "test_labels_used_for_fit_or_selection": 0})
        checks.append({**keys, "max_baseline_reproduction_error_ml": maximum})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(STUDY/"all_metrics.csv", index=False)
    aulc_rows = []
    for (column, split, seed, method), group in metrics.groupby(["column", "protocol", "seed", "method"]):
        ordered = group.sort_values("planned_budget")
        actual = ordered.actual_budget.to_numpy()
        aulc_rows.append({"column": column, "protocol": split, "seed": seed, "method": method,
            "normalized_aulc": float(np.trapezoid(ordered.normalized_rmse, ordered.planned_budget)/70),
            "actual_budget_normalized_aulc": float(np.trapezoid(ordered.normalized_rmse, actual)/(actual[-1]-actual[0])),
            "actual_budget_start": int(actual[0]), "actual_budget_end": int(actual[-1])})
    aulc = pd.DataFrame(aulc_rows)
    aulc.to_csv(STUDY/"aulc_by_seed.csv", index=False)
    summary = aulc.groupby(["column", "protocol", "method"]).normalized_aulc.agg(["mean", "std", "median", "min", "max"]).reset_index()
    summary.to_csv(STUDY/"aulc_summary.csv", index=False)
    paired = paired_table(aulc)
    paired.to_csv(STUDY/"paired_aulc.csv", index=False)
    aggregate = metrics.groupby(["column", "protocol", "planned_budget", "method"])[list(POINT_NAMES)].agg(["mean", "std", "median", "min", "max"])
    aggregate.to_csv(STUDY/"aggregate_metrics.csv")
    portfolio_curve = metrics.groupby(["protocol", "seed", "planned_budget", "portfolio_planned_budget",
                                       "portfolio_actual_budget", "method"]).normalized_rmse.mean().reset_index()
    portfolio_curve.to_csv(STUDY/"portfolio_learning_curves.csv", index=False)
    portfolio_rows = []
    for (split, seed, method), group in portfolio_curve.groupby(["protocol", "seed", "method"]):
        group = group.sort_values("portfolio_planned_budget")
        actual = group.portfolio_actual_budget.to_numpy()
        portfolio_rows.append({"protocol": split, "seed": seed, "method": method,
            "normalized_aulc": float(np.trapezoid(group.normalized_rmse, group.portfolio_planned_budget)/210),
            "actual_budget_normalized_aulc": float(np.trapezoid(group.normalized_rmse, actual)/(actual[-1]-actual[0])),
            "actual_total_budget_start": int(actual[0]), "actual_total_budget_end": int(actual[-1])})
    portfolio = pd.DataFrame(portfolio_rows)
    portfolio.to_csv(STUDY/"portfolio_aulc.csv", index=False)
    pd.DataFrame(usage_rows).to_csv(STUDY/"label_usage_summary.csv", index=False)
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(STUDY/"selection_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(STUDY/"baseline_reproduction.csv", index=False)
    decision = decide(paired, portfolio)
    atomic_json(STUDY/"decision.json", decision)
    atomic_json(STUDY/"execution_audit.json", {
        "contexts": len(checks), "failed": 0, "evaluation_rows": len(metrics), "new_methods": list(NEW_METHODS),
        "all_fits_frozen_before_evaluation": True, "maximum_baseline_reproduction_error_ml": max(r["max_baseline_reproduction_error_ml"] for r in checks),
        "protected_hashes_unchanged": contract() == config,
        "freeze_manifest_sha256": sha256_file(frozen_path),
    })
    make_report(metrics, summary, paired, portfolio, selections, decision)
    print(json.dumps(decision, indent=2), flush=True)


def make_report(metrics, summary, paired, portfolio, selections, decision):
    comparisons = paired.loc[paired.method.isin(["nonlinear_policy", "shared_column_affine"]) & paired.reference.isin(REFERENCES)]
    b100 = metrics.loc[metrics.planned_budget.eq(100)].groupby(["column", "protocol", "method"])[list(POINT_NAMES)].mean().reset_index()
    portfolio_summary = portfolio.groupby(["protocol", "method"])[["normalized_aulc", "actual_budget_normalized_aulc"]].agg(["mean", "std"])
    sections = ["# Next transfer residual diagnostics", "", f"Decision: `{decision['decision']}`.", "",
        "Two preregistered diagnostic families completed on all 120 frozen column/protocol/seed/budget contexts. "
        "The qualified source, baseline code, original results and frozen roles remain unchanged. "
        "No new architecture, fine-tuning, width search, active learning, or post-test candidate was run.", "",
        "The nonlinear primary policy selects scale, affine or a train-only two-knot monotone spline using focal validation. "
        "`monotone_spline` reports the standalone family, including cases where validation rejects it. "
        "`linear_policy` uses the same validation to choose only scale or affine. "
        "Shared column affine uses mass-normalized partial pooling; local identity shrinkage is the regularization control.", "",
        "The shared/independent comparison is a THREE-COLUMN PORTFOLIO at total planned budgets 90/150/210/300, "
        "with identical purchased label IDs (including all validation) for all portfolio methods. Per-column B is an allocation, "
        "not the shared model's total label cost. Actual compound budgets, discarded donor labels and allowed IDs are retained. "
        "Focal compound test/validation compounds are purged from every donor fit. Focal models are separate; "
        "no model or tuned hyperparameter is reused across focal holdouts. Row remains row interpolation.", "",
        "Arithmetic source NRMSE = mean(RMSE_V1/7.8797, RMSE_V2/16.0765); AULC is its trapezoidal average "
        "over B=30..100, exactly as cross-column. `combined_normalized_rmse` is RMS, used for validation. "
        "Exact source scales are in all_predictions_frozen.json. Actual-budget AULC is also provided. "
        "Five outer seeds share one fixed source seed 42 and overlapping data; their dispersion is not five independent external cohorts.", "",
        "## Frozen decision gate", "", json.dumps(decision, indent=2), "",
        "A candidate requires >=5% mean relative AULC improvement, negative median delta, >=4/5 seed wins "
        "against each specified reference in at least two different compound columns, with <=5% row degradation. "
        "25g/40g must also beat head-only. Shared requires >=5% compound portfolio gain over affine, scale and local shrinkage. "
        "Test results implement a preregistered research go/no-go gate; they do not select knots, penalties or model checkpoints.", "",
        "## Paired AULC (positive relative gain is better)", "",
        markdown_table(comparisons[["column", "protocol", "method", "reference", "relative_mean_gain", "mean_delta", "median_delta", "std_delta", "wins", "stable_material"]]), "",
        "## Full AULC seed stability", "", markdown_table(summary), "",
        "## Portfolio total-cost AULC", "", markdown_table(portfolio_summary), "",
        "## Budget 100 point metrics (mean over five seeds)", "", markdown_table(b100), "",
        "## Validation selections", "",
        markdown_table(selections.groupby(["column", "protocol", "nonlinear_policy"]).size().rename("contexts").reset_index()), "",
        markdown_table(selections.groupby(["column", "protocol", "shared_strength"]).size().rename("contexts").reset_index()), "",
        "## Interpretation limits and data", "",
        "Negative results do not establish an irreducible noise floor, prove affine is the true physical mapping, "
        "or identify sum readout as the residual cause. Adaptive readout was not tested and is not warranted by exclusion alone. "
        "A successful diagnostic only motivates independent confirmation of that bounded mechanism. "
        "Only three observed column/flow combinations exist, so mass and flow effects cannot be separated; no new-specification "
        "extrapolation or physical law is claimed. Target-compound holdout is not source-unseen OOD. "
        "These frozen tests have historical exposure: all results are developmental, not pristine confirmatory evidence.", "",
        "Additional data most useful for a stronger conclusion: independent compound/batch validation; matched conditions "
        "with replicated measurements; crossed mass/flow settings; greater high-q50 tail coverage; and genuinely source-unseen "
        "molecules for OOD. Existing data were sufficient to run the two diagnostics, but cannot identify all four hypotheses uniquely.", "",
        "Every per-seed/per-budget R²/RMSE/MAE/NRMSE is in all_metrics.csv; all point-metric mean/std/median/min/max "
        "are in aggregate_metrics.csv. paired_aulc.csv includes linear-policy and local-shrinkage controls. "
        "contexts/* contains train/validation/test IDs, donor purges, all validation candidate scores, slopes, knots, "
        "coefficients and truth-free frozen predictions. No further method is appended after this evaluation.", ""]
    (STUDY/"NEXT_TRANSFER_DIAGNOSTICS_REPORT.md").write_text("\n".join(sections))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.fit or args.run:
        fit_all()
    if args.evaluate or args.run:
        evaluate()
    if not (args.fit or args.evaluate or args.run):
        lock_contract()


if __name__ == "__main__":
    main()
