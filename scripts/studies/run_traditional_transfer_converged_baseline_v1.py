#!/usr/bin/env python3
"""Train-only-selected P0/P1 convergence baseline for frozen ROW contexts.

Fit and freeze never read outer-validation or outer-test endpoint cells.  The
separate score action is deliberately the first code path allowed to read test
truth, after all twenty final predictions and checkpoints are hash-locked.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.transfer import adaptation as a
from scripts.studies.run_filtered_full_data_benchmark import _read_authorized_truth

STUDY = ROOT / "studies/transfer/traditional_transfer_converged_baseline_v1"
FROZEN = ROOT / "studies/transfer/filtered_full_data_benchmark"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
COLUMNS = ("25g", "40g")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
METHODS = ("P0", "P1")
FEATURES = ("sample_id", "canonical_smiles", "PE/EA", "loading solvent", "Density g/ml", "V/ul", "Volume of loading solvent/ul")
CONFIG = {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 1e-5,
          "batch_size": 2048, "maximum_epochs": 500, "patience": 80,
          "bn_policy": "current", "normalized_target_loss": False}
DIAGNOSTIC_EPOCHS = (150, 300, 500)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _protocol() -> dict:
    frozen = json.loads((FROZEN / "protocol.json").read_text())
    return {
        "study": "CONVERGED_TRADITIONAL_PARAMETER_TRANSFER_BASELINE_V1",
        "classification": "DEVELOPMENTAL_CONFIRMATION__HISTORIC_OUTER_TEST_EXPOSED",
        "columns": list(COLUMNS), "outer_protocol": "row", "outer_seeds": list(SEEDS),
        "methods": {"P0": "qualified source -> historical_shallow, current BN, Adam",
                    "P1": "qualified source -> head-only -> historical_shallow, current BN, Adam"},
        "source_checkpoint": str(SOURCE.relative_to(ROOT)), "source_checkpoint_sha256": sha(SOURCE),
        "filtered_protocol_sha256": sha(FROZEN / "protocol.json"),
        "split_manifest_sha256": sha(FROZEN / "split_manifest.csv"),
        "inner_cv": {"folds": 5, "splitter": "GroupKFold", "group": "canonical_smiles",
                     "outer_validation_used_for_selection": False,
                     "outer_test_used_for_fit_or_selection": False},
        "training": {**CONFIG, "loss": "quantile_target_loss", "diagnostic_epochs": list(DIAGNOSTIC_EPOCHS)},
        "epoch_selection": {"P0": "median of five inner-fold Stage-B best epochs",
                            "P1": "separate medians of five inner-fold Stage-A and Stage-B best epochs",
                            "median_rounding": "round-half-up"},
        "final_refit": "fixed epochs; no validation set accepted; P1 Stage B inherits actual Stage-A final state",
        "scale_authority": "inner-train only for inner CV; outer gradient_train only for final evaluation denominator",
        "freeze_requirement": "all 20 final fits, checkpoints and blind validation/test predictions hash-locked before test truth read",
    }


def ensure_protocol() -> dict:
    payload = _protocol()
    path = STUDY / "PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("PROTOCOL.json differs from the frozen preregistered protocol")
    write_json(path, payload)
    return payload


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _prediction_frame(ids: list[str], prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ids, "V1_q10": prediction[:, 0], "V1_q50": prediction[:, 1],
                         "V1_q90": prediction[:, 2], "V2_q10": prediction[:, 3],
                         "V2_q50": prediction[:, 4], "V2_q90": prediction[:, 5]})


def _predict(model, atoms, angles, positions) -> np.ndarray:
    return a.predict_point(model, atoms, angles, positions, batch_size=CONFIG["batch_size"])[1]


def prepare_context(column: str, seed: int):
    """Construct a label-scrubbed outer context exposing only gradient-train truth."""
    frozen = json.loads((FROZEN / "protocol.json").read_text())
    if seed not in frozen["outer_seeds"]:
        raise RuntimeError("outer seed absent from frozen schedule")
    feature = pd.read_csv(FROZEN / f"filtered_features_{column}.csv", usecols=FEATURES)
    canonical_path = FROZEN / f"filtered_canonical_{column}.csv"
    if sha(canonical_path) != frozen["filtered_canonical_sha256"][column]:
        raise RuntimeError("filtered population hash mismatch")
    schedule = pd.read_csv(FROZEN / "split_manifest.csv")
    context = schedule.loc[(schedule.column.eq(column)) & (schedule.protocol.eq("row")) &
                           (schedule.outer_seed.eq(seed)), ["sample_id", "role"]].copy()
    roles = {role: context.loc[context.role.eq(role), "sample_id"].astype(str).tolist()
             for role in ("gradient_train", "validation", "test")}
    if any(not roles[role] for role in roles) or any(set(roles[x]) & set(roles[y]) for x in roles for y in roles if x < y):
        raise RuntimeError("outer roles are incomplete or overlap")
    feature.sample_id = feature.sample_id.astype(str)
    if not feature.sample_id.is_unique or set(feature.sample_id) != set().union(*map(set, roles.values())):
        raise RuntimeError("feature and frozen split identities differ")
    # This is the sole endpoint read in fit: explicitly only outer gradient-train.
    train_truth = _read_authorized_truth(canonical_path, roles["gradient_train"])
    frame = feature.copy()
    frame["V1_ml"] = 0.0
    frame["V2_ml"] = 0.0
    lookup = {sample_id: index for index, sample_id in enumerate(frame.sample_id)}
    for sample_id, endpoint in zip(roles["gradient_train"], train_truth):
        frame.loc[lookup[sample_id], ["V1_ml", "V2_ml"]] = endpoint
    cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
    cache.update(torch.load(ROOT / f"studies/transfer/cross_column/data_audit/graph_cache_{column}_only.pt", weights_only=False))
    if set(frame.canonical_smiles) - set(cache):
        raise RuntimeError("incomplete graph cache")
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_pre = source_payload["preprocessing"]
    atoms, angles = build_model_data(frame, cache, None, source_pre["scaler"])
    positions = {role: [lookup[sample_id] for sample_id in ids] for role, ids in roles.items()}
    outer_scales = a.fit_target_scales(atoms, positions["gradient_train"])
    source_scales = source_pre.get("target_scales")
    if not isinstance(source_scales, dict):
        raise RuntimeError("qualified source preprocessing lacks target scales")
    audit = {
        "column": column, "outer_seed": int(seed), "population_sha256": sha(canonical_path),
        "split_manifest_sha256": sha(FROZEN / "split_manifest.csv"), "source_checkpoint_sha256": sha(SOURCE),
        "roles": {role: {"count": len(ids), "ids_hash": stable_hash(sorted(ids))} for role, ids in roles.items()},
        "gradient_fit_rows": len(roles["gradient_train"]), "validation_selection_rows": 0,
        "test_truth_rows_used_for_fit": 0, "test_truth_rows_used_for_selection": 0,
        "outer_gradient_train_scales": outer_scales, "qualified_source_scales": source_scales,
        "authorized_fit_ids_hash": stable_hash(sorted(roles["gradient_train"])),
        "outer_validation_or_test_labels_loaded": False,
    }
    return frame, roles, positions, atoms, angles, outer_scales, audit


def _history_value(history, epoch: int, key: str) -> float:
    values = {int(row["epoch"]): float(row[key]) for row in history}
    return values.get(epoch, float("nan"))


def _stage_diagnostics(*, column, seed, method, fold, stage, fit, model, atoms, angles,
                       inner_train, inner_valid, inner_train_groups, inner_valid_groups, inner_scales, source_state, bn_before, elapsed):
    prediction = _predict(model, atoms, angles, inner_valid)
    truth = np.vstack([atoms[index].y.detach().cpu().numpy().reshape(-1)[:2] for index in inner_valid])
    metric = point_metrics(truth, prediction, inner_scales)
    tail = list(fit.history)[-min(20, len(fit.history)):]
    slope = float(np.polyfit([row["epoch"] for row in tail], [row["validation_score"] for row in tail], 1)[0]) if len(tail) >= 2 else float("nan")
    crossings = int(((prediction[:, 0] > prediction[:, 1]) | (prediction[:, 1] > prediction[:, 2]) |
                     (prediction[:, 3] > prediction[:, 4]) | (prediction[:, 4] > prediction[:, 5])).sum())
    row = {"column": column, "outer_seed": int(seed), "method": method, "inner_fold": int(fold), "stage": stage,
           "n_inner_train": len(inner_train), "n_inner_validation": len(inner_valid),
           "inner_train_group_hash": stable_hash(sorted(set(inner_train_groups))),
           "inner_validation_group_hash": stable_hash(sorted(set(inner_valid_groups))),
           "best_epoch": int(fit.best_epoch), "epochs_run": int(fit.epochs_run),
           "validation_score": float(fit.validation_score), "last_20_validation_slope": slope,
           "train_loss": float(fit.history[-1]["train_loss"]),
           "parameter_drift": a.parameter_drift(model, source_state, trainable_only=False),
           "bn_buffer_drift": a.bn_buffer_drift(model, bn_before),
           "trainable_parameter_count": int(fit.trainable_parameters),
           "quantile_crossing_count": crossings, "runtime_seconds": float(elapsed),
           "inner_validation_combined_nrmse": float(metric["combined_normalized_rmse"])}
    for epoch in DIAGNOSTIC_EPOCHS:
        row[f"epoch_{epoch}_validation_score"] = _history_value(fit.history, epoch, "validation_score")
    return row, [{"column": column, "outer_seed": int(seed), "method": method, "inner_fold": int(fold), "stage": stage, **dict(item)} for item in fit.history]


def _fit_inner_context(column: str, seed: int, atoms, angles, positions, canonical_by_position):
    train_positions = np.asarray(positions["gradient_train"], dtype=int)
    groups = np.asarray([str(canonical_by_position[index]) for index in train_positions])
    if len(np.unique(groups)) < 5:
        raise RuntimeError("fewer than five canonical-smiles groups in gradient_train")
    histories, summaries = [], []
    splitter = GroupKFold(n_splits=5)
    for fold, (local_train, local_valid) in enumerate(splitter.split(train_positions, groups=groups), start=1):
        inner_train = train_positions[local_train].tolist()
        inner_valid = train_positions[local_valid].tolist()
        inner_train_groups = groups[local_train].tolist()
        inner_valid_groups = groups[local_valid].tolist()
        if set(groups[local_train]) & set(groups[local_valid]):
            raise RuntimeError("canonical-smiles group appears in both inner train and validation")
        scales = a.fit_target_scales(atoms, inner_train)
        for method in METHODS:
            model = load_predictor_checkpoint(SOURCE)
            source_state = deepcopy(model.state_dict())
            started = time.time()
            bn_before = a.snapshot_bn_buffers(model)
            stage_seed = int(seed + 1009 * fold + (0 if method == "P0" else 50000))
            if method == "P0":
                fit = a.train_target_adaptation(model, atoms, angles, inner_train, inner_valid,
                                                {"target_scales": scales}, mode="historical_shallow",
                                                seed=stage_seed, config=CONFIG)
                summary, history = _stage_diagnostics(column=column, seed=seed, method=method, fold=fold, stage="B",
                    fit=fit, model=model, atoms=atoms, angles=angles, inner_train=inner_train, inner_valid=inner_valid,
                    inner_train_groups=inner_train_groups, inner_valid_groups=inner_valid_groups,
                    inner_scales=scales, source_state=source_state, bn_before=bn_before, elapsed=time.time()-started)
                summaries.append(summary); histories.extend(history)
            else:
                fit_a = a.train_target_adaptation(model, atoms, angles, inner_train, inner_valid,
                                                  {"target_scales": scales}, mode="head_only", seed=stage_seed,
                                                  config=CONFIG)
                summary, history = _stage_diagnostics(column=column, seed=seed, method=method, fold=fold, stage="A",
                    fit=fit_a, model=model, atoms=atoms, angles=angles, inner_train=inner_train, inner_valid=inner_valid,
                    inner_train_groups=inner_train_groups, inner_valid_groups=inner_valid_groups,
                    inner_scales=scales, source_state=source_state, bn_before=bn_before, elapsed=time.time()-started)
                summaries.append(summary); histories.extend(history)
                bn_before_b = a.snapshot_bn_buffers(model)
                stage_b_started = time.time()
                fit_b = a.train_target_adaptation(model, atoms, angles, inner_train, inner_valid,
                                                  {"target_scales": scales}, mode="historical_shallow", seed=stage_seed + 1,
                                                  config=CONFIG)
                summary, history = _stage_diagnostics(column=column, seed=seed, method=method, fold=fold, stage="B",
                    fit=fit_b, model=model, atoms=atoms, angles=angles, inner_train=inner_train, inner_valid=inner_valid,
                    inner_train_groups=inner_train_groups, inner_valid_groups=inner_valid_groups,
                    inner_scales=scales, source_state=source_state, bn_before=bn_before_b, elapsed=time.time()-stage_b_started)
                summaries.append(summary); histories.extend(history)
    return pd.DataFrame(histories), pd.DataFrame(summaries)


def _selected_epochs(summary: pd.DataFrame, column: str, seed: int) -> pd.DataFrame:
    rows = []
    for method, stages in (("P0", ("B",)), ("P1", ("A", "B"))):
        for stage in stages:
            values = summary.loc[(summary.method.eq(method)) & (summary.stage.eq(stage)), "best_epoch"].astype(float)
            if len(values) != 5:
                raise RuntimeError("missing inner-fold epoch selections")
            rows.append({"column": column, "outer_seed": int(seed), "method": method, "stage": stage,
                         "inner_fold_count": int(len(values)), "best_epochs": json.dumps(values.astype(int).tolist()),
                         "median_best_epoch": float(values.median()), "selected_epoch": round_half_up(values.median()),
                         "rounding_rule": "round-half-up"})
    return pd.DataFrame(rows)


def _final_refit(column: str, seed: int, roles, positions, atoms, angles, outer_scales, selected: pd.DataFrame, runtime: Path):
    method_records = []
    for method in METHODS:
        model = load_predictor_checkpoint(SOURCE)
        source_state = deepcopy(model.state_dict())
        run = runtime / method
        run.mkdir(parents=True, exist_ok=True)
        selected_for_method = selected.loc[selected.method.eq(method)].set_index("stage")["selected_epoch"].to_dict()
        started = time.time()
        if method == "P0":
            fit_b = a.train_target_adaptation_fixed_epochs(model, atoms, angles, positions["gradient_train"],
                int(selected_for_method["B"]), mode="historical_shallow", seed=int(seed), config=CONFIG)
            fits = {"B": fit_b}
        else:
            fit_a, fit_b = a.train_staged_target_adaptation_fixed_epochs(
                model, atoms, angles, positions["gradient_train"], int(selected_for_method["A"]),
                int(selected_for_method["B"]), stage_b_mode="historical_shallow", seed=int(seed),
                stage_a_config=CONFIG, stage_b_config=CONFIG)
            fits = {"A": fit_a, "B": fit_b}
        validation = _predict(model, atoms, angles, positions["validation"])
        test = _predict(model, atoms, angles, positions["test"])
        _prediction_frame(roles["validation"], validation).to_csv(run / "validation_predictions_blind.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0})
        _prediction_frame(roles["test"], test).to_csv(run / "test_predictions_blind.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0})
        torch.save({"model_state_dict": model.state_dict(), "method": method, "selected_epochs": selected_for_method,
                    "outer_validation_or_test_used_for_selection": False}, run / "final.pt")
        write_json(run / "final_fit_audit.json", {"method": method, "selected_epochs": selected_for_method,
            "gradient_fit_rows": len(positions["gradient_train"]), "validation_selection_rows": 0,
            "test_truth_rows_used_for_fit": 0, "test_truth_rows_used_for_selection": 0,
            "parameter_drift": a.parameter_drift(model, source_state, trainable_only=False),
            "runtime_seconds": time.time()-started, "stages": {stage: fit.as_dict() | {"history": None} for stage, fit in fits.items()}})
        method_records.append(method)
    return method_records


def _context_runtime(column: str, seed: int) -> Path:
    return STUDY / "runtime" / column / f"seed_{seed}"


def _refresh_inner_tables() -> None:
    history, summary, selected = [], [], []
    for column in COLUMNS:
        for seed in SEEDS:
            base = _context_runtime(column, seed)
            for name, sink in (("inner_cv_history.csv", history), ("inner_cv_summary.csv", summary), ("selected_epochs.csv", selected)):
                path = base / name
                if not path.exists():
                    raise RuntimeError(f"missing context inner artifact: {path}")
                sink.append(pd.read_csv(path))
    pd.concat(history, ignore_index=True).to_csv(STUDY / "INNER_CV_HISTORY.csv", index=False)
    pd.concat(summary, ignore_index=True).to_csv(STUDY / "INNER_CV_SUMMARY.csv", index=False)
    pd.concat(selected, ignore_index=True).to_csv(STUDY / "SELECTED_EPOCHS.csv", index=False)


def fit_context(column: str, seed: int) -> None:
    """Fit one independent outer context; safe to run in a separate process."""
    ensure_protocol()
    torch.set_num_threads(1)
    if column not in COLUMNS or seed not in SEEDS:
        raise ValueError("context must be a preregistered column and outer seed")
    runtime = _context_runtime(column, seed)
    complete = runtime / "context_fit_complete.json"
    if complete.exists():
        return
    frame, roles, positions, atoms, angles, outer_scales, audit = prepare_context(column, seed)
    history, summary = _fit_inner_context(column, seed, atoms, angles, positions, frame.canonical_smiles.astype(str).tolist())
    selected = _selected_epochs(summary, column, seed)
    runtime.mkdir(parents=True, exist_ok=True)
    history.to_csv(runtime / "inner_cv_history.csv", index=False)
    summary.to_csv(runtime / "inner_cv_summary.csv", index=False)
    selected.to_csv(runtime / "selected_epochs.csv", index=False)
    write_json(runtime / "context_audit.json", audit)
    _final_refit(column, seed, roles, positions, atoms, angles, outer_scales, selected, runtime)
    required = [runtime / name for name in ("inner_cv_history.csv", "inner_cv_summary.csv", "selected_epochs.csv", "context_audit.json")]
    required.extend(path for method in METHODS for path in (runtime / method / "final.pt", runtime / method / "validation_predictions_blind.csv.gz", runtime / method / "test_predictions_blind.csv.gz", runtime / method / "final_fit_audit.json"))
    write_json(complete, {"protocol_sha256": sha(STUDY / "PROTOCOL.json"), "column": column, "outer_seed": seed,
        "files": {str(path.relative_to(runtime)): sha(path) for path in required}, "methods": list(METHODS),
        "outer_validation_or_test_labels_used": False})


def fit() -> None:
    for column in COLUMNS:
        for seed in SEEDS:
            fit_context(column, seed)
    _refresh_inner_tables()


def freeze() -> None:
    ensure_protocol()
    _refresh_inner_tables()
    records = {}
    for column in COLUMNS:
        for seed in SEEDS:
            path = _context_runtime(column, seed) / "context_fit_complete.json"
            if not path.exists():
                raise RuntimeError(f"missing final fit: {path}")
            payload = json.loads(path.read_text())
            if payload["methods"] != list(METHODS) or payload["outer_validation_or_test_labels_used"]:
                raise RuntimeError("invalid context fit audit")
            for relative, expected in payload["files"].items():
                if sha(path.parent / relative) != expected:
                    raise RuntimeError(f"post-fit change before freeze: {relative}")
            records[str(path.relative_to(STUDY))] = sha(path)
    write_json(STUDY / "PREDICTION_FREEZE_MANIFEST.json", {"contexts": 10, "final_fits": 20,
        "methods": list(METHODS), "protocol_sha256": sha(STUDY / "PROTOCOL.json"), "source_checkpoint_sha256": sha(SOURCE),
        "selected_epochs_sha256": sha(STUDY / "SELECTED_EPOCHS.csv"), "inner_cv_summary_sha256": sha(STUDY / "INNER_CV_SUMMARY.csv"),
        "context_fit_complete_sha256": records, "test_truth_evaluation_started": False})


def _validate_freeze() -> dict:
    path = STUDY / "PREDICTION_FREEZE_MANIFEST.json"
    if not path.exists():
        raise RuntimeError("PREDICTION_FREEZE_MANIFEST.json is required before scoring")
    payload = json.loads(path.read_text())
    if payload["contexts"] != 10 or payload["final_fits"] != 20 or payload["methods"] != list(METHODS):
        raise RuntimeError("incomplete prediction freeze")
    if payload["protocol_sha256"] != sha(STUDY / "PROTOCOL.json") or payload["selected_epochs_sha256"] != sha(STUDY / "SELECTED_EPOCHS.csv"):
        raise RuntimeError("freeze protocol or selected epochs changed")
    for relative, expected in payload["context_fit_complete_sha256"].items():
        if sha(STUDY / relative) != expected:
            raise RuntimeError("context fit complete manifest changed")
    return payload


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse", "combined_source_normalized_rmse")
    rows = []
    for (column, method), group in frame.groupby(["column", "method"], sort=True):
        row = {"column": column, "method": method, "n_seeds": len(group)}
        for metric in metrics:
            value = group[metric].astype(float)
            for stat, result in (("mean", value.mean()), ("std", value.std(ddof=1)), ("median", value.median()), ("min", value.min()), ("max", value.max())):
                row[f"{metric}_{stat}"] = float(result)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse", "combined_source_normalized_rmse")
    for column, group in frame.groupby("column", sort=True):
        p0 = group.loc[group.method.eq("P0")].set_index("outer_seed")
        p1 = group.loc[group.method.eq("P1")].set_index("outer_seed")
        if set(p0.index) != set(p1.index):
            raise RuntimeError("P0/P1 seed contexts are not paired")
        for metric in metrics:
            delta = p1[metric] - p0[metric]
            row = {"column": column, "candidate": "P1", "reference": "P0", "metric": metric,
                   "mean_paired_delta": float(delta.mean()), "std_paired_delta": float(delta.std(ddof=1)),
                   "median_paired_delta": float(delta.median()), "seed_wins": int((delta < 0).sum()), "n": len(delta)}
            row["relative_improvement_percent"] = float(-100 * delta.mean() / p0[metric].mean()) if metric not in ("V1_r2", "V2_r2") else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _convergence_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (column, method, stage), group in summary.groupby(["column", "method", "stage"], sort=True):
        rows.append({"column": column, "method": method, "stage": stage, "inner_folds": len(group),
                     "best_at_protocol_ceiling": int((group.best_epoch == CONFIG["maximum_epochs"]).sum()),
                     "run_reached_protocol_ceiling": int((group.epochs_run == CONFIG["maximum_epochs"]).sum()),
                     "negative_last_20_slope": int((group.last_20_validation_slope < -1e-4).sum()),
                     "mean_best_epoch": float(group.best_epoch.mean()), "mean_epochs_run": float(group.epochs_run.mean()),
                     "mean_last_20_slope": float(group.last_20_validation_slope.mean()),
                     "still_budget_censored": bool((group.best_epoch == CONFIG["maximum_epochs"]).mean() >= .40 and (group.last_20_validation_slope < -1e-4).mean() >= .40)})
    return pd.DataFrame(rows)


def _write_final_report(summary: pd.DataFrame, paired: pd.DataFrame, convergence: pd.DataFrame, reference: pd.DataFrame) -> None:
    promoted = []
    for column in COLUMNS:
        comparison = paired.loc[(paired.column.eq(column)) & (paired.metric.eq("combined_normalized_rmse"))].iloc[0]
        p0 = summary.loc[(summary.column.eq(column)) & (summary.method.eq("P0"))].iloc[0]
        p1 = summary.loc[(summary.column.eq(column)) & (summary.method.eq("P1"))].iloc[0]
        endpoint_ok = all(p1[f"{metric}_mean"] <= 1.02 * p0[f"{metric}_mean"] for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae"))
        censored = bool(convergence.loc[(convergence.column.eq(column)) & (convergence.method.eq("P1")), "still_budget_censored"].any())
        promoted.append({"column": column, "nrmse_improved": bool(comparison.mean_paired_delta < 0), "wins_at_least_3": bool(comparison.seed_wins >= 3), "endpoint_ok": endpoint_ok, "still_budget_censored": censored})
    gate = all(all(item[key] for key in ("nrmse_improved", "wins_at_least_3", "endpoint_ok")) and not item["still_budget_censored"] for item in promoted)
    lines = ["# Converged P0/P1 neural-transfer baseline: final report\n",
        "## Design\n",
        "This is a ROW-only developmental confirmation because the frozen outer tests had historical exposure. P0 and P1 use only Adam/current BN/raw quantile loss/historical-shallow scope. Each outer gradient-train population is split by canonical-smiles GroupKFold; no outer validation or test label reached fitting or epoch selection. Final refits use the median inner-fold selected epoch(s) and a fixed-epoch API with no validation input.\n",
        "## Selected-epoch and convergence result\n", summary.to_markdown(index=False), "\n", convergence.to_markdown(index=False), "\n",
        "The convergence adequacy rule marks a stage `STILL_BUDGET_CENSORED` only where at least 40% of folds both selected epoch 500 and retained a negative last-20 validation slope. It does not silently extend the budget.\n",
        "## P0 vs P1 developmental test comparison\n", paired.to_markdown(index=False), "\n",
        "## Historical paper-style reference\n", reference.to_markdown(index=False), "\n",
        "## Promotion decision\n",
        f"`{'P1_PREFERRED_NEURAL_TRANSFER_BASELINE' if gate else 'NO_UNIVERSAL_STAGED_TRANSFER_GAIN'}`. The frozen gate requires both columns to improve mean train-normalized NRMSE, at least 3/5 paired seed wins per column, no mean endpoint RMSE/MAE deterioration >2%, no systematic opposite R2 direction, and no serious inner-CV budget censoring. Per-column checks: `{json.dumps(promoted)}`.\n",
        "## Next scientific action\n",
        "If convergence is adequate, the next isolated neural ablation is endpoint-normalized quantile loss; q50-only/weak-quantile loss follows to test point-loss mismatch. Normalized L2-SP must first correct its parameter-count scaling and be selected by inner CV. Scope/discriminative-LR work follows those loss/regularization controls. Structured HIER and measured-4g-anchor are separate structured/domain branches, not additions to this neural comparison. COMPOUND and Active Learning remain blocked by this ROW developmental result alone.\n"]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(lines))


def score() -> None:
    manifest = _validate_freeze()
    rows = []
    for column in COLUMNS:
        canonical = FROZEN / f"filtered_canonical_{column}.csv"
        for seed in SEEDS:
            runtime = _context_runtime(column, seed)
            audit = json.loads((runtime / "context_audit.json").read_text())
            schedule = pd.read_csv(FROZEN / "split_manifest.csv")
            test_ids = schedule.loc[(schedule.column.eq(column)) & (schedule.protocol.eq("row")) &
                                    (schedule.outer_seed.eq(seed)) & (schedule.role.eq("test")), "sample_id"].astype(str).tolist()
            truth = _read_authorized_truth(canonical, test_ids)
            for method in METHODS:
                prediction = pd.read_csv(runtime / method / "test_predictions_blind.csv.gz")
                if prediction.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("blind prediction IDs mismatch frozen test order")
                value = prediction.drop(columns=["sample_id"]).to_numpy(float)
                target_metric = point_metrics(truth, value, audit["outer_gradient_train_scales"])
                source_metric = point_metrics(truth, value, audit["qualified_source_scales"])
                rows.append({"column": column, "outer_seed": seed, "method": method, "n_test": len(test_ids),
                             **target_metric, "combined_source_normalized_rmse": source_metric["combined_normalized_rmse"],
                             "metric_scale_source": "outer_gradient_train_ddof0_shared_by_P0_P1"})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(STUDY / "test_metrics.csv", index=False)
    summary = _summary(metrics); summary.to_csv(STUDY / "summary.csv", index=False)
    paired = _paired(metrics); paired.to_csv(STUDY / "paired_comparisons.csv", index=False)
    inner = pd.read_csv(STUDY / "INNER_CV_SUMMARY.csv")
    convergence = _convergence_summary(inner); convergence.to_csv(STUDY / "CONVERGENCE_SUMMARY.csv", index=False)
    historical = pd.read_csv(FROZEN / "all_metrics.csv")
    reference = historical.loc[(historical.protocol.eq("row")) & historical.method.eq("paper_style_current_v2"),
        ["column", "seed", "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse"]].copy()
    reference.rename(columns={"seed": "outer_seed", "combined_normalized_rmse": "historical_source_normalized_rmse"}, inplace=True)
    reference.to_csv(STUDY / "paper_style_current_v2_reference.csv", index=False)
    _write_final_report(summary, paired, convergence, reference.groupby("column", as_index=False).mean(numeric_only=True))
    write_json(STUDY / "TEST_SCORE_MANIFEST.json", {"prediction_freeze_sha256": sha(STUDY / "PREDICTION_FREEZE_MANIFEST.json"),
        "test_metrics_sha256": sha(STUDY / "test_metrics.csv"), "status": "ONE_SHOT_DEVELOPMENTAL_TEST_SCORED"})


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fit", action="store_true")
    group.add_argument("--fit-context", nargs=2, metavar=("COLUMN", "SEED"))
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.fit:
        fit()
    elif args.fit_context:
        fit_context(args.fit_context[0], int(args.fit_context[1]))
    elif args.freeze:
        freeze()
    else:
        score()


if __name__ == "__main__":
    main()
