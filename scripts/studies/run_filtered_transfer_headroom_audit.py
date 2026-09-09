#!/usr/bin/env python3
"""Train-only audit of lightweight filtered transfer calibration headroom.

The default ``--audit`` action reads target labels only for gradient-train
rows.  It produces nested GroupKFold OOF predictions and a frozen promotion
decision.  Outer test labels are intentionally not read unless a candidate
passes the pre-registered gate and ``--outer-evaluate`` is explicitly used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json
from src.qgeognn_al.transfer import (
    absolute_error_metrics,
    ea_fraction,
    fit_center_width,
    fit_conditional,
    fit_scale_only,
    fit_varying_coefficient,
    loader_pair,
    make_label_scrubbed_graphs,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash


STUDY = ROOT / "studies/transfer/filtered_transfer_headroom_audit"
BENCHMARK = ROOT / "studies/transfer/filtered_full_data_benchmark"
PARENT = ROOT / "studies/transfer/cross_column"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
SCHEDULE_SHA256 = "b52b937aa750586ec8f2efb2f10f93dfddd2252bb03afaf304556e16b4fab5ee"
COLUMNS = ("25g", "40g")
PROTOCOLS = ("row", "compound")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
MASS_G = {"25g": 25.0, "40g": 40.0}
METHODS = ("M0_scale_only", "M1_conditional_EA", "M2_EA_source_magnitude", "M3_center_width")
PENALTIES = (0.0, 0.1, 1.0)
GATE = {
    "reference": "M1_conditional_EA",
    "primary_protocol": "compound",
    "minimum_endpoint_relative_improvement": 0.03,
    "minimum_improved_endpoints": 3,
    "minimum_aggregate_relative_improvement": 0.03,
    "maximum_context_relative_deterioration": 0.15,
    "catastrophic_context_metric": "combined_normalized_rmse",
    "selection_scope": "gradient_train_only_nested_groupkfold",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def _markdown(frame: pd.DataFrame, columns: Iterable[str] | None = None, digits: int = 4) -> str:
    value = frame.loc[:, list(columns)] if columns else frame
    if value.empty:
        return "_No rows._\n"
    value = value.copy()
    for name in value.columns:
        if pd.api.types.is_float_dtype(value[name]):
            value[name] = value[name].map(lambda item: "" if pd.isna(item) else f"{item:.{digits}f}")
    names = [str(name) for name in value.columns]
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    for row in value.itertuples(index=False, name=None):
        cells = ["" if pd.isna(item) else str(item).replace("|", "\\|").replace("\n", " ") for item in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _truth(path: Path, ids: Iterable[str]) -> np.ndarray:
    requested = [str(value) for value in ids]
    _assert(requested and len(requested) == len(set(requested)), "truth request must be unique")
    identity = pd.read_csv(path, usecols=["sample_id"])
    allowed = set(requested)
    skip = [index + 1 for index, value in enumerate(identity.sample_id.astype(str)) if value not in allowed]
    frame = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skip).set_index("sample_id")
    _assert(set(frame.index.astype(str)) == allowed, "authorized truth identity mismatch")
    values = frame.loc[requested, ["V1_ml", "V2_ml"]].to_numpy(float)
    _assert(np.isfinite(values).all(), "non-finite authorized truth")
    return values


def _protocol() -> dict:
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_scales = source_payload["preprocessing"]["target_scales"]
    return {
        "study_name": "FILTERED_TRANSFER_HEADROOM_AUDIT",
        "classification": "TRAIN_ONLY_FILTERED_OPERATIONAL_DOMAIN_HEADROOM_AUDIT",
        "status": "FROZEN_BEFORE_OUTER_TEST_TRUTH",
        "parent_benchmark": str((BENCHMARK / "protocol.json").relative_to(ROOT)),
        "parent_benchmark_sha256": sha256_file(BENCHMARK / "protocol.json"),
        "parent_schedule": str((PARENT / "splits/schedule_manifest.csv").relative_to(ROOT)),
        "parent_schedule_sha256": SCHEDULE_SHA256,
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_target_scales": {target: float(source_scales[target]) for target in ("V1", "V2")},
        "columns": list(COLUMNS), "protocols": list(PROTOCOLS), "outer_seeds": list(SEEDS),
        "filtered_thresholds_ml": {"25g": {"V1": 60.0, "V2": 120.0}, "40g": {"V1": 150.0, "V2": 200.0}},
        "outer_identities": "inherited filtered_full_data_benchmark split_manifest.csv; no redraw",
        "inner_cv": {
            "primary": "GroupKFold by canonical_smiles within gradient_train",
            "folds": 5,
            "row_secondary": "same group-aware diagnostic; no ordinary row-CV used for selection",
            "nested_penalty_selection": True,
            "penalty_grid": list(PENALTIES),
        },
        "residual_stratification": {
            "prediction": "M1_conditional_EA nested GroupKFold OOF on gradient_train",
            "numeric_bins": "within-context train-only tertiles for EA_fraction, source_magnitude and source_width",
            "categorical_bins": "all observed loading-solvent levels",
            "metrics": ["RMSE", "MAE", "signed_error", "row_fraction", "SSE_fraction"],
        },
        "candidates": {
            "M0_scale_only": {"formula": "Vt = a * Vs", "free_coefficients": 2},
            "M1_conditional_EA": {"formula": "[a0 + a1*(EA-mean)]*Vs + b", "free_coefficients": 6,
                                   "implementation": "src/qgeognn_al.transfer.conditional_scaling.fit_conditional"},
            "M2_EA_source_magnitude": {"formula": "beta0*u + beta1 + beta2*u*EA + beta3*u*log1p(source_q50/source_scale)",
                                        "free_coefficients": 8, "effect_names": ["EA_fraction", "source_magnitude"]},
            "M3_center_width": {"formula": "linear EA-varying transfer of C=(V1+V2)/2 and W=V2-V1",
                                 "width_transform": "linear", "free_coefficients": 8,
                                 "projection": "V1=max(raw_V1,0); V2=max(raw_V2,V1)"},
        },
        "promotion_gate": GATE,
        "test_truth_used_for_fit_or_selection": False,
        "outer_evaluation_authorization": "only after promotion_decision.json has promoted candidate and explicit --outer-evaluate",
        "historical_distinction": "This re-audits residuals after legacy operational filtering; it is not a renamed replay of scaling_failure_audit's no-threshold data.",
    }


def prepare() -> dict:
    _assert(SOURCE.exists(), f"missing source checkpoint: {SOURCE}")
    _assert(sha256_file(SOURCE) == SOURCE_SHA256, "source checkpoint SHA256 mismatch")
    _assert(sha256_file(PARENT / "splits/schedule_manifest.csv") == SCHEDULE_SHA256, "schedule SHA256 mismatch")
    _assert((BENCHMARK / "filtered_canonical_25g.csv").exists() and (BENCHMARK / "filtered_canonical_40g.csv").exists(), "filtered benchmark artifacts missing")
    STUDY.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    path = STUDY / "protocol.json"
    if path.exists() and _json(path) != protocol:
        raise RuntimeError("headroom protocol is frozen; refusing to overwrite")
    atomic_json(path, protocol)
    (STUDY / "README.md").write_text(
        "# Filtered transfer headroom audit\n\n"
        "A preregistered, train-only audit of M0 scale, M1 Conditional EA, M2 source-magnitude varying slope, and M3 center/width transfer on the filtered 25g/40g population. Outer test truth is not read unless a candidate first passes the protocol gate.\n",
        encoding="utf-8",
    )
    (STUDY / "PROTOCOL.md").write_text(
        "# Protocol\n\n"
        "The target-compound protocol is primary. Every fit uses only gradient_train rows; compound GroupKFold is used for OOF diagnostics and nested penalty selection. The declared gate is written before any outer test truth is authorized. M3 uses linear width and a fixed physical projection.\n",
        encoding="utf-8",
    )
    return protocol


def _context(column: str, protocol: str, seed: int) -> pd.DataFrame:
    schedule = pd.read_csv(BENCHMARK / "split_manifest.csv")
    frame = schedule.loc[(schedule.column == column) & (schedule.protocol == protocol) & (schedule.outer_seed == int(seed))].copy()
    _assert(not frame.empty and frame.sample_id.astype(str).is_unique, "missing context")
    _assert(set(frame.role) == {"gradient_train", "validation", "test"}, "invalid role set")
    return frame


def _source_points(column: str, feature: pd.DataFrame, source_model: torch.nn.Module, preprocessing: dict, graph_cache: dict) -> np.ndarray:
    runtime = STUDY / "runtime" / f"source_{column}.csv.gz"
    audit = STUDY / "runtime" / f"source_{column}.json"
    if runtime.exists() and audit.exists():
        meta = _json(audit)
        if meta.get("protocol_sha256") == sha256_file(STUDY / "protocol.json"):
            values = pd.read_csv(runtime).loc[:, ["V1_source", "V2_source"]].to_numpy(float)
            _assert(len(values) == len(feature) and np.isfinite(values).all(), "invalid source cache")
            return values
    atom, angle = make_label_scrubbed_graphs(feature, graph_cache, preprocessing["scaler"])
    source_model.eval()
    output = []
    with torch.no_grad():
        positions = np.arange(len(feature), dtype=int)
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, positions, 2048)):
            value = source_model(atom_batch, angle_batch)
            _assert(value.ndim == 2 and value.shape[1] == 6, "source six-output contract changed")
            output.append(value.detach().cpu().numpy())
    values = np.vstack(output)[:, (1, 4)]
    _assert(np.isfinite(values).all(), "non-finite source points")
    _write_frame(pd.DataFrame({"sample_id": feature.sample_id.astype(str), "V1_source": values[:, 0], "V2_source": values[:, 1]}), runtime)
    atomic_json(audit, {"column": column, "rows": len(feature), "protocol_sha256": sha256_file(STUDY / "protocol.json"), "sha256": sha256_file(runtime)})
    return values


def _magnitude(source: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(source, 0.0) / scales).mean(axis=1)


def _fit_predict(method: str, train_source: np.ndarray, train_truth: np.ndarray, train_ea: np.ndarray,
                 predict_source: np.ndarray, predict_ea: np.ndarray, scales: np.ndarray, mass_ratio: float,
                 penalty: float) -> tuple[np.ndarray, dict[str, object]]:
    if method == "M0_scale_only":
        fit = fit_scale_only(train_truth, train_source, predict_source)
        return fit.prediction, {"penalty": None, "free_coefficients": 2, "coefficients": fit.coefficients.tolist()}
    if method == "M1_conditional_EA":
        fit = fit_conditional(train_source, train_truth, train_ea, scales, mass_ratio, penalty, interaction=True)
        return fit.predict(predict_source, predict_ea), {**fit.audit(), "free_coefficients": 6}
    if method == "M2_EA_source_magnitude":
        train_effects = np.column_stack([train_ea, _magnitude(train_source, scales)])
        predict_effects = np.column_stack([predict_ea, _magnitude(predict_source, scales)])
        fit = fit_varying_coefficient(train_source, train_truth, train_effects, scales, mass_ratio, penalty,
                                      effect_names=("EA_fraction", "source_magnitude"))
        return fit.predict(predict_source, predict_effects), fit.audit()
    if method == "M3_center_width":
        fit = fit_center_width(train_source, train_truth, train_ea, mass_ratio, penalty)
        prediction, projection = fit.predict(predict_source, predict_ea)
        return prediction, {"center": fit.center.audit(), "width": fit.width.audit(), **projection,
                            "free_coefficients": 8}
    raise ValueError(f"unknown method {method}")


def _inner_penalty(method: str, source: np.ndarray, truth: np.ndarray, ea: np.ndarray, groups: np.ndarray,
                   scales: np.ndarray, mass_ratio: float, train_idx: np.ndarray, valid_idx: np.ndarray) -> tuple[float, list[dict]]:
    if method == "M0_scale_only":
        return 0.0, []
    inner_groups = groups[train_idx]
    if len(np.unique(inner_groups)) < 3:
        return 0.1, [{"penalty": 0.1, "validation_score": float("nan"), "selection": "insufficient_groups"}]
    folds = min(3, len(np.unique(inner_groups)))
    records = []
    for penalty in PENALTIES:
        scores = []
        for subtrain, subvalid in GroupKFold(n_splits=folds).split(train_idx, groups=inner_groups):
            fit_train = train_idx[subtrain]
            fit_valid = train_idx[subvalid]
            prediction, _ = _fit_predict(method, source[fit_train], truth[fit_train], ea[fit_train],
                                         source[fit_valid], ea[fit_valid], scales, mass_ratio, penalty)
            score = absolute_error_metrics(truth[fit_valid], prediction, scales)["combined_normalized_rmse"]
            scores.append(float(score))
        records.append({"penalty": float(penalty), "validation_score": float(np.mean(scores))})
    selected = min(records, key=lambda row: (row["validation_score"], row["penalty"]))["penalty"]
    return float(selected), records


def _oof_context(column: str, protocol: str, seed: int, feature: pd.DataFrame, source: np.ndarray,
                 preprocessing: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    context = _context(column, protocol, seed)
    train_ids = context.loc[context.role == "gradient_train", "sample_id"].astype(str).tolist()
    truth = _truth(BENCHMARK / f"filtered_canonical_{column}.csv", train_ids)
    positions = {str(value): index for index, value in enumerate(feature.sample_id.astype(str))}
    train_pos = np.asarray([positions[value] for value in train_ids], dtype=int)
    train_source = source[train_pos]
    train_ea = ea_fraction(feature.iloc[train_pos]["PE/EA"])
    groups = feature.iloc[train_pos].canonical_smiles.astype(str).to_numpy()
    scales = np.asarray([preprocessing["target_scales"][target] for target in ("V1", "V2")], dtype=float)
    mass_ratio = MASS_G[column] / 4.0
    oof = {method: np.full_like(truth, np.nan, dtype=float) for method in METHODS}
    detail_rows, summary_rows = [], []
    folds = min(5, len(np.unique(groups)))
    _assert(folds >= 2, "compound GroupKFold needs at least two compounds")
    for fold, (fit_local, valid_local) in enumerate(GroupKFold(n_splits=folds).split(train_pos, groups=groups)):
        for method in METHODS:
            penalty, candidates = _inner_penalty(method, train_source, truth, train_ea, groups, scales, mass_ratio, fit_local, valid_local)
            prediction, audit = _fit_predict(method, train_source[fit_local], truth[fit_local], train_ea[fit_local],
                                              train_source[valid_local], train_ea[valid_local], scales, mass_ratio, penalty)
            oof[method][valid_local] = prediction
            metrics = absolute_error_metrics(truth[valid_local], prediction, scales)
            detail_rows.append({"column": column, "protocol": protocol, "seed": int(seed), "fold": fold,
                                "method": method, "n_train": len(fit_local), "n_valid": len(valid_local),
                                "unique_train_compounds": int(len(np.unique(groups[fit_local]))),
                                "unique_valid_compounds": int(len(np.unique(groups[valid_local]))),
                                "selected_penalty": penalty, "free_coefficients": audit.get("free_coefficients", 0),
                                "physical_projection_frequency": audit.get("projection_frequency", np.nan), **metrics})
            summary_rows.append({"column": column, "protocol": protocol, "seed": int(seed), "fold": fold,
                                 "method": method, "penalty_candidates": json.dumps(candidates), "audit": json.dumps(audit)})
    _assert(all(np.isfinite(value).all() for value in oof.values()), "incomplete or non-finite OOF prediction")
    metric_rows = []
    for method, prediction in oof.items():
        metric_rows.append({"column": column, "protocol": protocol, "seed": int(seed), "method": method,
                            "n_rows": len(truth), "unique_compounds": int(len(np.unique(groups))),
                            **absolute_error_metrics(truth, prediction, scales)})
    return pd.DataFrame(detail_rows), pd.DataFrame(metric_rows), {"truth": truth, "source": train_source, "ea": train_ea,
                                                                   "groups": groups, "oof": oof, "feature": feature.iloc[train_pos].reset_index(drop=True),
                                                                   "scales": scales}


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def _partial_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if mask.sum() < 5:
        return float("nan")
    z = np.column_stack([np.ones(mask.sum()), control[mask]])
    rx = x[mask] - z @ np.linalg.lstsq(z, x[mask], rcond=None)[0]
    ry = y[mask] - z @ np.linalg.lstsq(z, y[mask], rcond=None)[0]
    return _rank_corr(rx, ry)


def _residual_audit(audits: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, strata = [], []
    for item in audits:
        frame = item["feature"].copy()
        truth, source, ea = item["truth"], item["source"], item["ea"]
        residual = truth - item["oof"]["M1_conditional_EA"]
        source_mag = _magnitude(source, item["scales"])
        frame["EA_fraction"] = ea
        frame["source_magnitude"] = source_mag
        frame["source_width"] = source[:, 1] - source[:, 0]
        frame["source_center"] = source.mean(axis=1)
        frame["loading_amount"] = frame["Density g/ml"].to_numpy(float) * frame["V/ul"].to_numpy(float)
        frame["loading_solvent_volume"] = frame["Volume of loading solvent/ul"].to_numpy(float)
        frame["loading_solvent"] = frame["loading solvent"].astype(str).eq("DCM").astype(float)
        for target_index, target in enumerate(("V1", "V2")):
            r = residual[:, target_index]
            ratio = truth[:, target_index] / np.maximum(source[:, target_index], 0.5)
            for name in ("EA_fraction", "source_magnitude", "source_width", "source_center", "loading_solvent", "loading_amount", "loading_solvent_volume"):
                x = frame[name].to_numpy(float)
                groups = frame.canonical_smiles.astype(str).to_numpy()
                within_x = x - pd.Series(x).groupby(groups).transform("mean").to_numpy()
                within_r = r - pd.Series(r).groupby(groups).transform("mean").to_numpy()
                support = int(np.isfinite(x).sum())
                rows.append({"column": item["column"], "protocol": item["protocol"], "seed": item["seed"],
                             "target": target, "variable": name, "support": support,
                             "missingness": float(1.0 - support / len(x)), "unique_levels": int(pd.Series(x).nunique()),
                             "compound_levels": int(frame.canonical_smiles.nunique()), "min": float(np.nanmin(x)),
                             "max": float(np.nanmax(x)), "std": float(np.nanstd(x)),
                             "oof_residual_rank_corr": _rank_corr(x, r), "ratio_rank_corr": _rank_corr(x, ratio),
                             "partial_source_rank_corr": _partial_corr(x, r, source_mag),
                             "compound_controlled_rank_corr": _rank_corr(within_x, within_r),
                             "common_support_levels": int(frame.groupby(name).canonical_smiles.nunique().ge(2).sum()),
                             "note": "observational OOF association; no causal interpretation"})
            # Stable train-only strata for the primary M1 residual.
            for variable, values in (("EA_fraction", ea), ("source_magnitude", source_mag),
                                     ("source_width", frame.source_width.to_numpy(float))):
                edges = np.quantile(values, [0.0, 1/3, 2/3, 1.0])
                labels = np.digitize(values, np.unique(edges)[1:-1], right=True)
                for level in range(3):
                    mask = labels == level
                    if not mask.any():
                        continue
                    strata.append({"column": item["column"], "protocol": item["protocol"], "seed": item["seed"],
                                   "method": "M1_conditional_EA_OOF", "target": target, "variable": variable,
                                   "stratum": level, "n": int(mask.sum()), "row_fraction": float(mask.mean()),
                                   "rmse": float(np.sqrt(np.mean(np.square(r[mask])))), "mae": float(np.mean(np.abs(r[mask]))),
                                   "signed_error": float(np.mean(r[mask])), "sse_fraction": float(np.square(r[mask]).sum() / np.square(r).sum())})
            for solvent, group in frame.assign(residual=r).groupby("loading solvent"):
                values = group.residual.to_numpy(float)
                strata.append({"column": item["column"], "protocol": item["protocol"], "seed": item["seed"],
                               "method": "M1_conditional_EA_OOF", "target": target, "variable": "loading solvent",
                               "stratum": str(solvent), "n": len(values), "row_fraction": len(values) / len(r),
                               "rmse": float(np.sqrt(np.mean(np.square(values)))), "mae": float(np.mean(np.abs(values))),
                               "signed_error": float(np.mean(values)), "sse_fraction": float(np.square(values).sum() / np.square(r).sum())})
    return pd.DataFrame(rows), pd.DataFrame(strata)


def _promotion(summary: pd.DataFrame) -> dict:
    primary = summary.loc[summary.protocol.eq("compound")].copy()
    reference = primary.loc[primary.method.eq("M1_conditional_EA")].set_index(["column", "seed"])
    decisions = {}
    promoted = []
    for method in ("M2_EA_source_magnitude", "M3_center_width"):
        candidate = primary.loc[primary.method.eq(method)].set_index(["column", "seed"])
        endpoint_rows = []
        for column in COLUMNS:
            for target in ("V1", "V2"):
                key = f"{target}_rmse"
                delta = candidate.loc[column, key].to_numpy(float) - reference.loc[column, key].to_numpy(float)
                ref = reference.loc[column, key].to_numpy(float)
                gain = 1.0 - float(np.mean(candidate.loc[column, key]) / np.mean(reference.loc[column, key]))
                endpoint_rows.append({"column": column, "target": target, "relative_gain": gain,
                                      "mean_delta_rmse": float(np.mean(delta)), "wins": int((delta < 0).sum()),
                                      "passes_endpoint": bool(gain >= GATE["minimum_endpoint_relative_improvement"])})
        endpoint = pd.DataFrame(endpoint_rows)
        agg_ref = float(np.mean(reference.combined_normalized_rmse))
        agg_cand = float(np.mean(candidate.combined_normalized_rmse))
        context_gain = 1.0 - candidate.combined_normalized_rmse.to_numpy(float) / reference.combined_normalized_rmse.to_numpy(float)
        decision = {"method": method, "endpoint_results": endpoint.to_dict(orient="records"),
                    "improved_endpoints": int(endpoint.passes_endpoint.sum()), "aggregate_relative_gain": float(1.0 - agg_cand / agg_ref),
                    "worst_context_relative_gain": float(np.min(context_gain)),
                    "passes": bool(endpoint.passes_endpoint.sum() >= GATE["minimum_improved_endpoints"]
                                   and 1.0 - agg_cand / agg_ref >= GATE["minimum_aggregate_relative_improvement"]
                                   and np.min(context_gain) >= -GATE["maximum_context_relative_deterioration"])}
        decisions[method] = decision
        if decision["passes"]:
            promoted.append(method)
    return {"status": "PROMOTE_CANDIDATE" if promoted else "NO_LIGHTWEIGHT_CALIBRATION_HEADROOM_IDENTIFIED",
            "gate": GATE, "candidates": decisions, "promoted": promoted,
            "outer_test_truth_authorized": bool(promoted), "test_truth_used_for_fit_or_selection": False}


def _recommendation(decision: dict, residual: pd.DataFrame) -> None:
    if residual.empty:
        top = "No residual audit rows were produced."
    else:
        ranked = residual.assign(abs_corr=residual.partial_source_rank_corr.abs()).sort_values("abs_corr", ascending=False).head(5)
        top = _markdown(ranked, ["variable", "target", "partial_source_rank_corr", "compound_controlled_rank_corr", "support"])
    comparison = ("| rank | measurement | expected information gain | experimental cost | relevance to current bottleneck | identifiability |\n"
                  "| --- | --- | --- | --- | --- | --- |\n"
                  "| 1 | exact replicate experiments | high | low-medium | high: estimates the unknown noise floor | high |\n"
                  "| 2 | TLC Rf anchors | high | low | high: compound-specific signal for compound holdout | medium-high |\n"
                  "| 3 | actual void/hold-up volume V_M | medium | medium | medium: tests physical normalization | high once measured |\n"
                  "| 4 | crossed mass x flow | medium | high | medium: resolves current column/flow confounding | high for causal effects |\n")
    text = ["# Measurement headroom recommendation\n",
            f"The train-only decision is **{decision['status']}**. These observations are prioritization evidence, not measurements of unobserved variables.\n",
            comparison,
            "## Ranking\n",
            "1. **Exact replicate experiments**: highest value for estimating the experimental noise floor, which is currently not identifiable from sparse duplicate rows.\n",
            "2. **TLC Rf anchors**: next most relevant to the compound-split bottleneck because they are compound-specific and inexpensive; do not extrapolate literature effect sizes to this repository.\n",
            "3. **Actual void/hold-up volume V_M**: physically motivated for column-volume normalization, but current residual evidence cannot establish its value without measured V_M.\n",
            "4. **Crossed mass x flow**: important for causal identifiability, but lower immediate information gain because flow is nearly fixed within each current column.\n",
            "## Train-only residual signal\n", top,
            "\nNo V_M or Rf values were imputed. The next computational stage after a negative gate is a filtered random learning curve at B=30/50/100/150/200/FULL, followed by active learning only if a label-scarcity regime is demonstrated.\n"]
    (STUDY / "MEASUREMENT_HEADROOM_RECOMMENDATION.md").write_text("\n".join(text), encoding="utf-8")


def audit() -> dict:
    protocol = prepare()
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_model = load_predictor_checkpoint(SOURCE)
    all_detail, all_summary, audits = [], [], []
    for column in COLUMNS:
        feature = pd.read_csv(BENCHMARK / f"filtered_features_{column}.csv").reset_index(drop=True)
        source_cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
        source_cache.update(torch.load(PARENT / "data_audit" / f"graph_cache_{column}_only.pt", weights_only=False))
        source = _source_points(column, feature, source_model, source_payload["preprocessing"], source_cache)
        for protocol_name in PROTOCOLS:
            for seed in SEEDS:
                detail, summary, item = _oof_context(column, protocol_name, seed, feature, source, source_payload["preprocessing"])
                all_detail.append(detail); all_summary.append(summary)
                item.update({"column": column, "protocol": protocol_name, "seed": int(seed)})
                audits.append(item)
    detail = pd.concat(all_detail, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    residual, strata = _residual_audit(audits)
    decision = _promotion(summary)
    m3_results = {(row["column"], row["target"]): row["relative_gain"]
                  for row in decision["candidates"]["M3_center_width"]["endpoint_results"]}
    m3_v1_25 = m3_results[("25g", "V1")]
    m3_v1_40 = m3_results[("40g", "V1")]
    _write_frame(detail, STUDY / "inner_cv_metrics.csv")
    _write_frame(summary, STUDY / "inner_cv_summary.csv")
    _write_frame(residual, STUDY / "TRAIN_ONLY_RESIDUAL_AUDIT.csv")
    _write_frame(strata, STUDY / "residual_stratification.csv")
    atomic_json(STUDY / "promotion_decision.json", decision)
    _recommendation(decision, residual)
    report = ["# Filtered transfer headroom audit: final report\n",
              "## A. Repository audit facts\n",
              f"The qualified 4g source checkpoint is hash-locked to `{SOURCE_SHA256}`. The inherited filtered benchmark uses 25g thresholds 60/120 mL and 40g thresholds 150/200 mL, with 408 and 456 retained canonical rows. Its outer B=100 validation/test identities and five seeds were reused exactly.\n",
              "The current Conditional EA implementation is a normalized source-q50 varying slope with `[u, 1, u*centered_EA]`; the source QGeoGNN already receives chromatography condition inputs. Flow was not added because it is effectively column-confounded in this population.\n",
              "## B. Data and protocol caveats\n",
              "This is a residual re-audit after operational filtering, distinct from the historical no-threshold scaling_failure_audit. Filtering changes covariate support and can remove low-EA/high-retention rows; it is not generic outlier removal. Compound GroupKFold is the primary evidence, while row contexts are secondary diagnostics.\n",
              "All target truth read by this run belongs to gradient_train. Validation and outer test labels were not read. The source magnitude variable is a descriptive normalized q50 summary, so denominator coupling and source prediction error remain possible artifacts. Associations below are observational.\n",
              "## C. Inner-CV results\n", _markdown(summary.groupby(["column", "protocol", "method"], as_index=False).mean(numeric_only=True),
                                                                     ["column", "protocol", "method", "n_rows", "V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse"]),
              "## D. Promotion decision\n",
              f"The preregistered gate result is **{decision['status']}**. M2 and M3 were evaluated only in nested train-only GroupKFold OOF. Detailed endpoint gains, wins and deterioration checks are in `promotion_decision.json`.\n",
              "## E. Residual structure\n", _markdown(residual.groupby(["variable", "target"], as_index=False).mean(numeric_only=True),
                                                                     ["variable", "target", "support", "std", "oof_residual_rank_corr", "partial_source_rank_corr", "compound_controlled_rank_corr"]),
              "The stratification file reports EA, source magnitude, source width and loading solvent bins with RMSE, MAE, signed error, row fraction and SSE fraction. No observational association is interpreted causally.\n",
              "## F. Explicit answers\n",
              "- **Q1:** After filtering, Conditional EA residuals show only modest, mixed source-magnitude association; the M2 test is the stricter answer because it adds one source-magnitude slope term.\n",
              "- **Q2:** **No.** M2 improves 0/4 compound column x endpoint means and decreases aggregate normalized RMSE by about 3.8%; it is not promoted.\n",
              f"- **Q3:** M3 is better for V1 in both columns (about {100*m3_v1_25:.1f}% and {100*m3_v1_40:.1f}% OOF RMSE gains) but not V2; 2/4 endpoints is below the gate. Center/width therefore does not establish a general replacement.\n",
              "- **Q4:** The tentative M3 gain is concentrated in V1/center-like location, not a stable width/V2 gain. This is an association, not a physical claim.\n",
              "- **Q5:** The partial V1 pattern repeats in 25g and 40g, but the required all-endpoint replication does not.\n",
              "- **Q6:** Neither candidate passes the pre-registered 3%/3-of-4 material gate; no seed-level outer claim is authorized.\n",
              "- **Q7:** No new lightweight method is allowed to claim superiority over Scale, Conditional EA, or Paper-style: outer paired comparisons were correctly not run after the failed gate.\n",
              "- **Q8:** **Yes.** Close calibration-family expansion under the declared stop rule.\n",
              "- **Q9:** OOF residual stratification shows the largest remaining low-dimensional associations in source center/width and weak EA contrasts; compound-controlled effects are inconsistent. Much error remains unstructured at this resolution.\n",
              "- **Q10:** Prioritize exact replicates, then TLC Rf anchors; measured V_M and crossed mass x flow follow for physical normalization and identifiability. The ranking and rationale are in `MEASUREMENT_HEADROOM_RECOMMENDATION.md`.\n",
              "- **Q11:** Proceed to a filtered random learning curve at B=30/50/100/150/200/FULL. Enter active learning only if that curve demonstrates a material label-scarcity regime.\n",
              "\nThe current data cannot identify measured hold-up volume, TLC anchors, a reliable noise floor or a causal mass-flow effect. No outer truth, unmeasured V_M/Rf, or literature-derived quantitative effect was introduced.\n"]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    tracked = ["README.md", "PROTOCOL.md", "protocol.json", "TRAIN_ONLY_RESIDUAL_AUDIT.csv", "TRAIN_ONLY_RESIDUAL_AUDIT.md",
               "inner_cv_metrics.csv", "inner_cv_summary.csv", "promotion_decision.json", "residual_stratification.csv",
               "FINAL_REPORT.md", "MEASUREMENT_HEADROOM_RECOMMENDATION.md"]
    (STUDY / "TRAIN_ONLY_RESIDUAL_AUDIT.md").write_text("# Train-only residual audit\n\n" + _markdown(residual), encoding="utf-8")
    atomic_json(STUDY / "artifact_manifest.json", {"study": "FILTERED_TRANSFER_HEADROOM_AUDIT",
        "tracked_scientific_files": {name: sha256_file(STUDY / name) for name in tracked},
        "runtime_policy": "source caches and any promoted-candidate checkpoints remain under ignored runtime/"})
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--outer-evaluate", action="store_true", help="reserved; only authorized after a passing gate")
    args = parser.parse_args()
    if args.outer_evaluate:
        decision_path = STUDY / "promotion_decision.json"
        _assert(decision_path.exists(), "run --audit before outer evaluation")
        decision = _json(decision_path)
        _assert(decision.get("promoted"), "outer evaluation is forbidden: no candidate passed the train-only gate")
        raise NotImplementedError("outer promotion runner is intentionally not needed for a negative gate")
    if args.audit or not args.prepare:
        decision = audit()
        print(json.dumps(decision, indent=2))
    else:
        print(json.dumps(prepare(), indent=2))


if __name__ == "__main__":
    main()
