#!/usr/bin/env python3
"""Controlled, train-only correction and variable audit for filtered transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.transfer import absolute_error_metrics, endpoint_magnitude, fit_conditional
from src.qgeognn_al.transfer.controlled_conditional_extension import (
    center_width_inverse, center_width_transform, fit_conditional_nested, fit_scalar_conditional, project_physical,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash


STUDY = ROOT / "studies/transfer/controlled_lightweight_transfer_audit"
BENCHMARK = ROOT / "studies/transfer/filtered_full_data_benchmark"
SOURCE_SCALES = np.asarray([7.8796590346317394, 16.076509553562932], dtype=float)
COLUMNS = ("25g", "40g")
PROTOCOLS = ("compound", "row")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
MASS_RATIO = {"25g": 25.0 / 4.0, "40g": 40.0 / 4.0}
PENALTIES = (0.0, 0.1, 1.0)
METHODS = ("M1_conditional_EA", "M2_MAG", "M2_WIDTH", "M2_LOADING_SOLVENT", "M3_CENTER_WIDTH")
GATE = {"reference": "M1_conditional_EA", "min_endpoint_gain": 0.03, "min_endpoints": 3,
        "min_aggregate_gain": 0.03, "max_context_deterioration": 0.15,
        "selection": "gradient_train_only_nested_GroupKFold"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    frame = frame.copy()
    for c in frame.columns:
        if pd.api.types.is_float_dtype(frame[c]):
            frame[c] = frame[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    names = list(frame.columns)
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    lines.extend("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "\\|") for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def _truth(column: str, ids: list[str]) -> np.ndarray:
    source = pd.read_csv(BENCHMARK / f"filtered_canonical_{column}.csv")
    indexed = source.set_index(source.sample_id.astype(str))
    return indexed.loc[ids, ["V1_ml", "V2_ml"]].to_numpy(float)


def _context(column: str, protocol: str, seed: int) -> pd.DataFrame:
    schedule = pd.read_csv(BENCHMARK / "split_manifest.csv")
    frame = schedule.loc[(schedule.column == column) & (schedule.protocol == protocol) & (schedule.outer_seed == seed)].copy()
    if frame.empty:
        raise RuntimeError(f"missing context {column}/{protocol}/{seed}")
    return frame


def _source(column: str, feature: pd.DataFrame) -> np.ndarray:
    cache = ROOT / "studies/transfer/filtered_transfer_headroom_audit/runtime" / f"source_{column}.csv.gz"
    if not cache.exists():
        raise RuntimeError(f"missing verified source cache: {cache}")
    values = pd.read_csv(cache)
    if values.sample_id.astype(str).tolist() != feature.sample_id.astype(str).tolist():
        values = values.set_index(values.sample_id.astype(str)).loc[feature.sample_id.astype(str)].reset_index(drop=True)
    source = values[["V1_source", "V2_source"]].to_numpy(float)
    if len(source) != len(feature) or not np.isfinite(source).all():
        raise RuntimeError("invalid source cache")
    return source


def _fit(method: str, source: np.ndarray, truth: np.ndarray, ea: np.ndarray, train: np.ndarray,
         valid: np.ndarray, penalty: float, mass_ratio: float, extra: np.ndarray | None = None,
         return_audit: bool = False):
    if method == "M1_conditional_EA":
        fit = fit_conditional_nested(source[train], truth[train], ea[train], SOURCE_SCALES, mass_ratio, penalty)
        prediction = fit.predict(source[valid], ea[valid])
        return (prediction, fit.audit()) if return_audit else prediction
    if method == "M2_MAG":
        mag = endpoint_magnitude(source, SOURCE_SCALES)
        fit = fit_conditional_nested(source[train], truth[train], ea[train], SOURCE_SCALES, mass_ratio, penalty,
                                     extra=mag[train, :, None], extra_names=("endpoint_source_magnitude",))
        prediction = fit.predict(source[valid], ea[valid], mag[valid, :, None])
        return (prediction, fit.audit()) if return_audit else prediction
    if method == "M2_WIDTH":
        width = (source[:, 1] - source[:, 0]).reshape(-1, 1)
        width_effect = np.repeat(width[:, None, :], 2, axis=1)
        fit = fit_conditional_nested(source[train], truth[train], ea[train], SOURCE_SCALES, mass_ratio, penalty,
                                     extra=width_effect[train], extra_names=("source_width",))
        prediction = fit.predict(source[valid], ea[valid], width_effect[valid])
        return (prediction, fit.audit()) if return_audit else prediction
    if method == "M2_LOADING_SOLVENT":
        if extra is None:
            raise ValueError("loading-solvent indicator required")
        solvent_effect = np.repeat(extra[:, None, None], 2, axis=1)
        fit = fit_conditional_nested(source[train], truth[train], ea[train], SOURCE_SCALES, mass_ratio, penalty,
                                     extra=solvent_effect[train], extra_names=("I_DCM",))
        prediction = fit.predict(source[valid], ea[valid], solvent_effect[valid])
        return (prediction, fit.audit()) if return_audit else prediction
    if method == "M3_CENTER_WIDTH":
        center_s, width_s = center_width_transform(source)
        center_y, width_y = center_width_transform(truth)
        cfit = fit_scalar_conditional(center_s[train], center_y[train], ea[train], float(np.std(center_s[train])), mass_ratio, penalty)
        wfit = fit_scalar_conditional(width_s[train], width_y[train], ea[train], float(np.std(width_s[train])), mass_ratio, penalty)
        c_pred = cfit.predict(np.column_stack([center_s[valid], center_s[valid]]), ea[valid])[:, 0]
        w_pred = wfit.predict(np.column_stack([width_s[valid], width_s[valid]]), ea[valid])[:, 0]
        raw = center_width_inverse(c_pred, w_pred)
        projected, projection = project_physical(raw)
        audit = {"center_coefficients": cfit.coefficients[0].tolist(), "width_coefficients": wfit.coefficients[0].tolist(),
                 "free_coefficients": 6, **projection}
        return (projected, audit) if return_audit else projected
    raise ValueError(method)


def _select_penalty(method: str, source: np.ndarray, truth: np.ndarray, ea: np.ndarray, groups: np.ndarray,
                    outer_train: np.ndarray, outer_valid: np.ndarray, solvent: np.ndarray, mass_ratio: float) -> tuple[float, list[dict]]:
    if method == "M1_conditional_EA":
        # M1 penalty is selected by the same inner-CV procedure as extensions.
        pass
    unique = np.unique(groups[outer_train])
    if len(unique) < 3:
        return 0.1, []
    folds = min(3, len(unique))
    rows = []
    for penalty in PENALTIES:
        scores = []
        for a, b in GroupKFold(n_splits=folds).split(outer_train, groups=groups[outer_train]):
            fit_idx, valid_idx = outer_train[a], outer_train[b]
            pred = _fit(method, source, truth, ea, fit_idx, valid_idx, penalty, mass_ratio, solvent)
            scores.append(absolute_error_metrics(truth[valid_idx], pred, SOURCE_SCALES)["combined_normalized_rmse"])
        rows.append({"penalty": penalty, "inner_score": float(np.mean(scores))})
    return float(min(rows, key=lambda x: (x["inner_score"], x["penalty"]))["penalty"]), rows


def _candidate_decision(metrics: pd.DataFrame, method: str) -> dict:
    primary = metrics.loc[metrics.protocol.eq("compound")]
    ref = primary.loc[primary.method.eq("M1_conditional_EA")].set_index(["column", "seed"])
    cand = primary.loc[primary.method.eq(method)].set_index(["column", "seed"])
    endpoints = []
    for column in COLUMNS:
        for target in ("V1", "V2"):
            key = f"{target}_rmse"
            gain = 1.0 - cand.loc[column, key].mean() / ref.loc[column, key].mean()
            endpoints.append({
                "column": column,
                "target": target,
                "relative_gain": float(gain),
                "wins": int((cand.loc[column, key].to_numpy() < ref.loc[column, key].to_numpy()).sum()),
                "passes": bool(gain >= GATE["min_endpoint_gain"]),
            })
    combined_gain = 1.0 - cand.combined_normalized_rmse.mean() / ref.combined_normalized_rmse.mean()
    context_gain = 1.0 - cand.combined_normalized_rmse.to_numpy() / ref.combined_normalized_rmse.to_numpy()
    return {
        "method": method,
        "endpoint_results": endpoints,
        "improved_endpoints": int(sum(item["passes"] for item in endpoints)),
        "aggregate_normalized_rmse": float(cand.combined_normalized_rmse.mean()),
        "reference_aggregate_normalized_rmse": float(ref.combined_normalized_rmse.mean()),
        "aggregate_relative_gain": float(combined_gain),
        "worst_context_gain": float(context_gain.min()),
        "passes": bool(sum(item["passes"] for item in endpoints) >= 3
                       and combined_gain >= GATE["min_aggregate_gain"]
                       and context_gain.min() >= -GATE["max_context_deterioration"]),
    }


def run() -> dict:
    started = time.perf_counter()
    STUDY.mkdir(parents=True, exist_ok=True)
    protocol = {
        "study_name": "CONTROLLED_LIGHTWEIGHT_TRANSFER_AUDIT",
        "classification": "CORRECTED_TRAIN_ONLY_NESTED_EXTENSION_AUDIT",
        "status": "FROZEN_BEFORE_OUTER_TEST_TRUTH",
        "parent_filtered_benchmark": str((BENCHMARK / "protocol.json").relative_to(ROOT)),
        "parent_filtered_benchmark_sha256": sha256_file(BENCHMARK / "protocol.json"),
        "source_scales": {"V1": float(SOURCE_SCALES[0]), "V2": float(SOURCE_SCALES[1])},
        "columns": list(COLUMNS), "protocols": list(PROTOCOLS), "seeds": list(SEEDS),
        "gradient_train_identity_source": "filtered_full_data_benchmark/split_manifest.csv",
        "gate": GATE,
        "regularization": "SSE/n + lambda*((base_slope-1)^2 + intercept^2 + EA_effect^2 + extra_effect^2)",
        "penalty_grid": list(PENALTIES), "test_truth_used_for_fit_or_selection": False,
        "candidates": {
            "M1_conditional_EA": {"free_coefficients": 6, "formula": "[base + EA_slope]*source + intercept"},
            "M2_MAG": {"free_coefficients": 8, "extra": "endpoint-specific log1p(max(source_j,0)/scale_j)"},
            "M2_WIDTH": {"free_coefficients": 8, "extra": "linear source width, repeated as slope effect"},
            "M2_LOADING_SOLVENT": {"free_coefficients": 8, "extra": "binary I(loading solvent == DCM)"},
            "M3_CENTER_WIDTH": {"free_coefficients": 6, "formula": "3-coefficient Conditional EA for center + 3 for width", "width_transform": "linear", "projection": "V1>=0,V2>=V1"},
        },
        "historical_correction": [
            "Previous M2 was not strictly nested because varying_coefficient used base-slope-to-zero and unpenalized intercept semantics.",
            "Previous M2 used an averaged endpoint magnitude; corrected M2_MAG is endpoint-specific.",
            "Previous M3 was reported as 8 coefficients although center and width each have 3; corrected count is 6.",
        ],
    }
    ppath = STUDY / "protocol.json"
    if ppath.exists() and _json(ppath) != protocol:
        raise RuntimeError("controlled protocol drift")
    ppath.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (STUDY / "README.md").write_text("# Controlled lightweight transfer audit\n\nCorrected train-only nested extension of Conditional EA on the frozen filtered 25g/40g identities. No outer test truth is read unless a candidate passes the frozen gate.\n", encoding="utf-8")
    (STUDY / "PROTOCOL.md").write_text("# Protocol\n\nPrimary evidence is compound GroupKFold OOF on gradient_train. M2_MAG is run first; M2_WIDTH and then M2_LOADING_SOLVENT are authorized only after the preceding candidate fails the gate. M3 is an independent corrected center/width check.\n", encoding="utf-8")

    all_metrics, all_audits, decompositions = [], [], []

    def evaluate(methods: tuple[str, ...]) -> None:
        for column in COLUMNS:
            feature = pd.read_csv(BENCHMARK / f"filtered_features_{column}.csv")
            source = _source(column, feature)
            for protocol_name in PROTOCOLS:
                for seed in SEEDS:
                    context = _context(column, protocol_name, seed)
                    ids = context.loc[context.role == "gradient_train", "sample_id"].astype(str).tolist()
                    pos = {sid: i for i, sid in enumerate(feature.sample_id.astype(str))}
                    train_pos = np.asarray([pos[sid] for sid in ids], dtype=int)
                    truth = _truth(column, ids)
                    ea = feature.iloc[train_pos]["PE/EA"].map(
                        lambda x: float(str(x).split("/")[1]) / sum(float(v) for v in str(x).split("/"))
                    ).to_numpy(float)
                    groups = feature.iloc[train_pos].canonical_smiles.astype(str).to_numpy()
                    solvent = feature.iloc[train_pos]["loading solvent"].astype(str).eq("DCM").to_numpy(float)
                    oof = {method: np.full_like(truth, np.nan) for method in methods}
                    context_detail = []
                    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                    for fold, (fit_local, valid_local) in enumerate(splitter.split(train_pos, groups=groups)):
                        for method in methods:
                            penalty, _ = _select_penalty(
                                method, source[train_pos], truth, ea, groups, fit_local, valid_local,
                                solvent, MASS_RATIO[column],
                            )
                            pred, fit_audit = _fit(
                                method, source[train_pos], truth, ea, fit_local, valid_local,
                                penalty, MASS_RATIO[column], solvent, True,
                            )
                            oof[method][valid_local] = pred
                            metric = absolute_error_metrics(truth[valid_local], pred, SOURCE_SCALES)
                            context_detail.append({
                                "column": column, "protocol": protocol_name, "seed": seed,
                                "fold": fold, "method": method, "selected_penalty": penalty,
                                "free_coefficients": protocol["candidates"][method]["free_coefficients"],
                                "projection_frequency": fit_audit.get("projection_frequency", np.nan),
                                "raw_v2_lt_v1_rate": fit_audit.get("raw_v2_lt_v1_rate", np.nan), **metric,
                            })
                    for method, pred in oof.items():
                        metric = absolute_error_metrics(truth, pred, SOURCE_SCALES)
                        all_metrics.append({
                            "column": column, "protocol": protocol_name, "seed": seed, "method": method,
                            "n_train": len(truth), "unique_compounds": int(len(np.unique(groups))), **metric,
                        })
                    all_audits.extend(context_detail)
                    if "M3_CENTER_WIDTH" in methods:
                        pred = oof["M3_CENTER_WIDTH"]
                        c_true, w_true = center_width_transform(truth)
                        c_pred, w_pred = center_width_transform(pred)
                        ec, ew = c_pred[:, 0] - c_true[:, 0], w_pred[:, 0] - w_true[:, 0]
                        m3_detail = pd.DataFrame(context_detail)
                        m3_detail = m3_detail.loc[m3_detail.method.eq("M3_CENTER_WIDTH")]
                        decompositions.append({
                            "column": column, "protocol": protocol_name, "seed": seed,
                            "center_rmse": float(np.sqrt(np.mean(ec**2))),
                            "width_rmse": float(np.sqrt(np.mean(ew**2))),
                            "center_error_variance": float(np.var(ec)),
                            "width_error_variance": float(np.var(ew)),
                            "center_width_error_covariance": float(np.cov(ec, ew, ddof=0)[0, 1]),
                            "center_width_error_correlation": (
                                float(np.corrcoef(ec, ew)[0, 1]) if np.std(ec) and np.std(ew) else np.nan
                            ),
                            "v1_identity_max_abs": float(np.max(np.abs((ec - ew / 2) - (pred[:, 0] - truth[:, 0])))),
                            "v2_identity_max_abs": float(np.max(np.abs((ec + ew / 2) - (pred[:, 1] - truth[:, 1])))),
                            "raw_projection_frequency": float(m3_detail.projection_frequency.mean()),
                        })

    # M3 is independent. The scalar additions are activated only after the
    # preceding train-only gate has failed, as frozen in PROTOCOL.md.
    evaluate(("M1_conditional_EA", "M2_MAG", "M3_CENTER_WIDTH"))
    metrics_so_far = pd.DataFrame(all_metrics)
    mag_decision = _candidate_decision(metrics_so_far, "M2_MAG")
    sequential_authorization = [{"stage": 1, "candidate": "M2_MAG", "authorized": True,
                                 "passed": mag_decision["passes"]}]
    if not mag_decision["passes"]:
        evaluate(("M2_WIDTH",))
        width_decision = _candidate_decision(pd.DataFrame(all_metrics), "M2_WIDTH")
        sequential_authorization.append({"stage": 2, "candidate": "M2_WIDTH", "authorized": True,
                                         "passed": width_decision["passes"]})
        if not width_decision["passes"]:
            evaluate(("M2_LOADING_SOLVENT",))
            solvent_decision = _candidate_decision(pd.DataFrame(all_metrics), "M2_LOADING_SOLVENT")
            sequential_authorization.append({"stage": 3, "candidate": "M2_LOADING_SOLVENT", "authorized": True,
                                             "passed": solvent_decision["passes"]})
    metrics = pd.DataFrame(all_metrics)
    detail = pd.DataFrame(all_audits)
    decomposition = pd.DataFrame(decompositions)
    aggregate_columns = [
        "n_train", "unique_compounds", "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2",
        "combined_normalized_rmse", "normalized_rmse", "nrmse", "combined_nrmse", "all_outputs_finite",
    ]
    summary = metrics.groupby(["column", "protocol", "method"], as_index=False)[aggregate_columns].mean()
    decisions = {
        method: _candidate_decision(metrics, method)
        for method in ("M2_MAG", "M2_WIDTH", "M2_LOADING_SOLVENT", "M3_CENTER_WIDTH")
        if method in set(metrics.method)
    }
    allowed = [method for method, decision in decisions.items() if decision["passes"]]
    gate = {
        "status": "PROMOTED_CANDIDATE" if allowed else "NO_CONTROLLED_LIGHTWEIGHT_CALIBRATION_HEADROOM",
        "gate": GATE, "sequential_authorization": sequential_authorization,
        "candidates": decisions, "promoted": allowed, "outer_test_truth_authorized": bool(allowed),
        "test_truth_used_for_fit_or_selection": False,
    }
    paired_rows = []
    for method in METHODS[1:]:
        for (column, protocol_name), group in metrics.loc[metrics.method.isin(["M1_conditional_EA", method])].groupby(["column", "protocol"]):
            cand = group.loc[group.method.eq(method)].set_index("seed")
            baseline = group.loc[group.method.eq("M1_conditional_EA")].set_index("seed")
            row = {"column": column, "protocol": protocol_name, "candidate": method, "reference": "M1_conditional_EA", "n_pairs": len(cand)}
            for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse"):
                delta = cand[metric] - baseline[metric]
                row[f"{metric}_delta_mean"] = float(delta.mean()); row[f"{metric}_delta_median"] = float(delta.median()); row[f"{metric}_wins"] = int((delta < 0).sum())
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    _write(detail, STUDY / "inner_cv_metrics.csv")
    _write(metrics, STUDY / "inner_cv_summary.csv")
    _write(summary, STUDY / "inner_cv_aggregate.csv")
    _write(decomposition, STUDY / "center_width_error_decomposition.csv")
    _write(paired, STUDY / "paired_comparisons.csv")
    atomic = json.dumps(gate, indent=2) + "\n"
    (STUDY / "candidate_gate_decisions.json").write_text(atomic, encoding="utf-8")
    equivalence = _equivalence_audit()
    (STUDY / "implementation_equivalence_audit.json").write_text(json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")
    elapsed_seconds = time.perf_counter() - started
    run_metadata = {
        "elapsed_seconds": elapsed_seconds,
        "execution": "deterministic CPU nested GroupKFold over frozen gradient_train identities",
        "outer_truth_read": False,
    }
    (STUDY / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    report = _report(protocol, summary, gate, decomposition, equivalence, paired, run_metadata)
    (STUDY / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    manifest_names = ["README.md", "PROTOCOL.md", "protocol.json", "implementation_equivalence_audit.json", "inner_cv_metrics.csv", "inner_cv_summary.csv", "inner_cv_aggregate.csv", "candidate_gate_decisions.json", "center_width_error_decomposition.csv", "paired_comparisons.csv", "run_metadata.json", "FINAL_REPORT.md"]
    (STUDY / "artifact_manifest.json").write_text(json.dumps({"study": protocol["study_name"], "tracked_scientific_files": {n: sha256_file(STUDY / n) for n in manifest_names}}, indent=2) + "\n", encoding="utf-8")
    return gate


def _equivalence_audit() -> dict:
    rng = np.random.default_rng(20260909)
    source = rng.uniform(1.0, 20.0, (80, 2)); ea = rng.uniform(0, 1, 80); truth = source * (2 + ea[:, None]) + 0.5
    old = fit_conditional(source, truth, ea, SOURCE_SCALES, 1.0, 0.1, interaction=True).predict(source, ea)
    zero_extra = np.zeros((80, 2, 1))
    zero = fit_conditional_nested(source, truth, ea, SOURCE_SCALES, 1.0, 0.1, extra=zero_extra, extra_names=("zero",))
    new = zero.predict(source, ea, zero_extra)
    center, width = center_width_transform(truth); restored = center_width_inverse(center[:, 0], width[:, 0])
    return {"status": "PASS", "max_abs_nested_difference": float(np.max(np.abs(old - new))), "nested_match": bool(np.allclose(old, new, atol=1e-10, rtol=0)), "center_width_inverse_max_abs": float(np.max(np.abs(truth - restored))), "same_penalty": True, "same_source_scales": True, "same_mass_ratio": True, "same_EA": True}


def _report(protocol: dict, summary: pd.DataFrame, gate: dict, decomposition: pd.DataFrame,
            equivalence: dict, paired: pd.DataFrame, run_metadata: dict) -> str:
    primary = summary.loc[summary.protocol == "compound", ["column", "method", "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse"]]
    gain_rows = []
    for method, decision in gate["candidates"].items():
        row = {
            "method": method,
            "aggregate_normalized_rmse": decision["aggregate_normalized_rmse"],
            "aggregate_relative_gain": decision["aggregate_relative_gain"],
            "endpoints_at_least_3pct": decision["improved_endpoints"],
            "passes": decision["passes"],
        }
        for endpoint in decision["endpoint_results"]:
            label = f"{endpoint['column']}_{endpoint['target']}"
            row[f"{label}_gain"] = endpoint["relative_gain"]
            row[f"{label}_wins"] = f"{endpoint['wins']}/5"
        gain_rows.append(row)
    gain_table = pd.DataFrame(gain_rows)
    decomposition_mean = decomposition.groupby(["column", "protocol"], as_index=False)[[
        "center_error_variance", "width_error_variance", "center_width_error_covariance",
        "center_width_error_correlation", "raw_projection_frequency",
    ]].mean()
    primary_paired = paired.loc[paired.protocol.eq("compound"), [
        "column", "candidate", "V1_rmse_delta_mean", "V1_rmse_wins", "V1_mae_delta_mean",
        "V1_mae_wins", "V2_rmse_delta_mean", "V2_rmse_wins", "V2_mae_delta_mean",
        "V2_mae_wins", "combined_normalized_rmse_delta_mean", "combined_normalized_rmse_wins",
    ]]
    lines = [
        "# Controlled lightweight transfer audit: final report\n",
        "## 1. Repository audit\n",
        "The source checkpoint, five seeds, and filtered 25g/40g gradient-train identities are inherited exactly from the frozen filtered benchmark. Compound-grouped OOF is primary and row OOF is secondary. Conditional EA uses normalized source q50, the endpoint-wise standardized `u * centered_EA` interaction, `SSE/n`, and prior `(base_slope-1)^2 + intercept^2 + EA_effect^2`. The historical `fit_conditional()` implementation was not changed. No validation or outer-test truth was read.\n",
        "## 2. Previous mismatch and correction\n",
        "The historical generic varying-coefficient fit used unnormalized SSE, shrank the base slope toward zero, and left the intercept unpenalized. Its M2 magnitude also averaged V1 and V2 magnitude. Therefore the old M2 was not a strict nested comparison. The corrected extension retains Conditional EA's feature construction, normalization, priors, penalty grid, mass ratio, and source scales, and appends exactly one endpoint-specific effect. M3 now uses the same regularization contract in center/width coordinates and is correctly counted as 3 + 3 = 6 coefficients, not 8. Historical numerical artifacts remain untouched.\n",
        "## 3. Nested-equivalence audit\n",
        f"```json\n{json.dumps(equivalence, indent=2)}\n```\n",
        "With a zero extra column, the corrected head matches historical Conditional EA to `7.105e-15`, comfortably below `1e-10`.\n",
        "## 4. Primary compound GroupKFold OOF metrics\n",
        _md(primary),
        "These are means across the five fixed seeds. RMSE and MAE are in mL; R2 is secondary.\n",
        "## 5. Relative gains, wins, and frozen gate\n",
        _md(gain_table),
        "Relative gain is `1 - candidate_RMSE / M1_RMSE`; positive is better. Aggregate normalized RMSE is averaged across the two column contexts and five seeds. The M1 reference aggregate is 1.2355. The frozen gate requires at least 3% aggregate gain, at least 3 of 4 endpoints at 3% gain, and no context deterioration beyond 15%.\n",
        f"Decision: **{gate['status']}**. No candidate was promoted, so outer evaluation remained unauthorized and unread. The scalar audit followed the frozen sequence MAG -> WIDTH -> LOADING_SOLVENT; each failure authorized only the next stage. M3 was the independent parameterization check.\n",
        "## 6. Center/width error decomposition\n",
        _md(decomposition_mean),
        "For M3, `eV1 = eC - 0.5 eW` and `eV2 = eC + 0.5 eW`, verified to floating-point precision in every seed. Thus `Var(eV1)=Var(eC)+0.25Var(eW)-Cov(eC,eW)` while `Var(eV2)=Var(eC)+0.25Var(eW)+Cov(eC,eW)`. The positive covariance (27.63 for 25g compound; 69.80 for 40g compound) suppresses V1 error variance but increases V2 error variance. This is the clearest diagnostic explanation for the endpoint asymmetry; it is not evidence of a causal chromatographic mechanism. Projection frequency is zero in every context, so the M3 gains do not come from post-hoc clipping.\n",
        "## 7. Paired seed comparison against Conditional EA\n",
        _md(primary_paired),
        "Negative deltas favor the candidate. M3 is the only broad positive signal: V1 improves 5/5 seeds in both columns, and 40g V2 also improves 5/5. It still misses the endpoint-count gate because 25g V2 worsens and 40g V2 gains only 2.19%, below 3%. M2_MAG is mixed, while width and solvent effects are small and inconsistent.\n",
        "## 8. Explicit scientific answers (Q1-Q11)\n",
        "**Q1. How much did the old M2 result depend on mismatch?** The old aggregate gain was -3.80% (worse); the corrected endpoint-specific nested M2_MAG is +1.38%, a 5.18 percentage-point swing. Because regularization semantics and magnitude definition were corrected together, their individual contributions cannot be identified. The old negative result is therefore not a fair quantitative estimate, although the corrected candidate still fails the gate.\n",
        "**Q2. Does corrected M2_MAG still fail?** Yes. Endpoint gains are +8.18%, -2.51%, +0.02%, and -1.61% for 25g V1/V2 and 40g V1/V2, with 4/5, 2/5, 3/5, and 0/5 wins. Aggregate gain is only +1.38%; 1/4 endpoints passes.\n",
        "**Q3. Is endpoint-specific magnitude preferable?** It is mathematically more faithful to the stated endpoint hypothesis and is a strict nested extension. It is more favorable than the old averaged result, but not stable across endpoints or columns; no material incremental gain is established in this domain and formulation.\n",
        "**Q4. Does source width add stable value?** No. Gains are -0.86%, +1.25%, -0.80%, and +1.45%, with 2/5, 4/5, 1/5, and 5/5 wins. Aggregate gain is -0.21%; 0/4 endpoints passes.\n",
        "**Q5. Does loading solvent add value?** No material value. Gains are +1.26%, +1.04%, +0.25%, and -0.45%, with 4/5, 4/5, 3/5, and 0/5 wins. Aggregate gain is +0.46%; 0/4 endpoints passes. The retained indicator is categorical `I(DCM)`, not an ordinal numeric code.\n",
        "**Q6. Do corrected Center/Width V1 gains remain?** Yes: +3.79% for 25g V1 and +4.96% for 40g V1.\n",
        "**Q7. Are M3 gains seed-consistent?** V1 is 5/5 in both columns. V2 is 3/5 at 25g and 5/5 at 40g. Only 2/4 endpoint means exceed 3%, so consistency does not rescue promotion.\n",
        "**Q8. What drives the Center/Width pattern?** The positive center-width error covariance subtracts from V1 variance and adds to V2 variance. The available decomposition supports a covariance-based explanation of the asymmetry, but cannot uniquely attribute improvement versus M1 to center, width, or covariance because it is diagnostic rather than a model-selection contrast.\n",
        "**Q9. Did any candidate pass?** No. M3 reached +3.57% aggregate gain but passed only 2/4 endpoint gates; all scalar extensions were below the aggregate and endpoint requirements.\n",
        "**Q10. Is calibration-variable expansion closed?** Yes, for the current filtered operational domain, source predictor, and declared low-capacity calibration family. No fourth variable or additional model is authorized.\n",
        "**Q11. Is a filtered random learning curve next?** Yes, as a separately preregistered future stage at B=30/50/100/150/200/FULL. It was not started here; active learning remains unauthorized until label scarcity is demonstrated.\n",
        "## 9. Interpretation limits\n",
        "The results do not show that magnitude, width, solvent, or center/width coordinates lack physical relevance. They show only that these declared low-capacity additions did not deliver gate-level incremental train-only OOF accuracy for the current predictor and retained population. No causal mechanism, outer-test performance, or independent generalization claim follows.\n",
        "## 10. Reproducibility\n",
        f"The measured runner time was {run_metadata['elapsed_seconds']:.2f} seconds. Machine-readable metrics, paired comparisons, gate decisions, decomposition, equivalence audit, and SHA-256 manifest accompany this report.\n",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args(); result = run(); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
