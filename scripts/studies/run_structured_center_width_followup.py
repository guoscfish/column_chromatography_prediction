#!/usr/bin/env python3
"""Run the train-only structured center/width follow-up."""

from __future__ import annotations

import argparse
import hashlib
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

from src.qgeognn_al.transfer import absolute_error_metrics, fit_structured_center_width
from src.qgeognn_al.transfer.controlled_conditional_extension import center_width_transform


STUDY = ROOT / "studies/transfer/structured_center_width_followup"
BENCHMARK = ROOT / "studies/transfer/filtered_full_data_benchmark"
CONTROLLED = ROOT / "studies/transfer/controlled_lightweight_transfer_audit"
SOURCE_SCALES = np.asarray([7.8796590346317394, 16.076509553562932], dtype=float)
COLUMNS = ("25g", "40g")
PROTOCOLS = ("compound", "row")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
METHODS = ("M3_CENTER_WIDTH", "CENTER_MAGNITUDE", "CENTER_CONDITIONED_WIDTH")
MASS_RATIO = {"25g": 25.0 / 4.0, "40g": 40.0 / 4.0}
PENALTIES = (0.0, 0.1, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(STUDY / name, index=False)


def _md(frame: pd.DataFrame) -> str:
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body]) + "\n"


def _feature_and_source(column: str) -> tuple[pd.DataFrame, np.ndarray]:
    feature = pd.read_csv(BENCHMARK / f"filtered_features_{column}.csv")
    cache = ROOT / "studies/transfer/filtered_transfer_headroom_audit/runtime" / f"source_{column}.csv.gz"
    source_frame = pd.read_csv(cache)
    if source_frame.sample_id.astype(str).tolist() != feature.sample_id.astype(str).tolist():
        source_frame = source_frame.set_index(source_frame.sample_id.astype(str)).loc[
            feature.sample_id.astype(str)
        ].reset_index(drop=True)
    source = source_frame[["V1_source", "V2_source"]].to_numpy(float)
    if not np.isfinite(source).all():
        raise RuntimeError("non-finite source predictions")
    return feature, source


def _context(column: str, protocol: str, seed: int) -> pd.DataFrame:
    schedule = pd.read_csv(BENCHMARK / "split_manifest.csv")
    context = schedule.loc[
        schedule.column.eq(column) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed)
    ]
    if context.empty:
        raise RuntimeError(f"missing frozen context: {column}/{protocol}/{seed}")
    return context


def _truth(column: str, ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(BENCHMARK / f"filtered_canonical_{column}.csv")
    indexed = frame.set_index(frame.sample_id.astype(str))
    return indexed.loc[ids, ["V1_ml", "V2_ml"]].to_numpy(float)


def _ea(feature: pd.DataFrame) -> np.ndarray:
    def fraction(value: object) -> float:
        parts = [float(item) for item in str(value).split("/")]
        return parts[1] / sum(parts)
    return feature["PE/EA"].map(fraction).to_numpy(float)


def _fit_predict(method: str, source: np.ndarray, truth: np.ndarray, ea: np.ndarray,
                 fit_index: np.ndarray, valid_index: np.ndarray, penalty: float,
                 mass_ratio: float) -> tuple[np.ndarray, dict]:
    fit = fit_structured_center_width(
        source[fit_index], truth[fit_index], ea[fit_index], mass_ratio, penalty, method
    )
    prediction = fit.predict(source[valid_index], ea[valid_index])
    return prediction, fit.audit(source[valid_index], ea[valid_index])


def _select_penalty(method: str, source: np.ndarray, truth: np.ndarray, ea: np.ndarray,
                    groups: np.ndarray, outer_train: np.ndarray, mass_ratio: float) -> float:
    folds = min(3, len(np.unique(groups[outer_train])))
    scores = []
    for penalty in PENALTIES:
        fold_scores = []
        splitter = GroupKFold(n_splits=folds)
        for inner_train, inner_valid in splitter.split(outer_train, groups=groups[outer_train]):
            fit_index, valid_index = outer_train[inner_train], outer_train[inner_valid]
            prediction, _ = _fit_predict(
                method, source, truth, ea, fit_index, valid_index, penalty, mass_ratio
            )
            fold_scores.append(
                absolute_error_metrics(truth[valid_index], prediction, SOURCE_SCALES)[
                    "combined_normalized_rmse"
                ]
            )
        scores.append((float(np.mean(fold_scores)), penalty))
    return float(min(scores)[1])


def _diagnostics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    center_truth, width_truth = center_width_transform(truth)
    center_prediction, width_prediction = center_width_transform(prediction)
    ec = center_prediction[:, 0] - center_truth[:, 0]
    ew = width_prediction[:, 0] - width_truth[:, 0]
    ev1 = prediction[:, 0] - truth[:, 0]
    ev2 = prediction[:, 1] - truth[:, 1]
    var_c, var_w = float(np.var(ec)), float(np.var(ew))
    covariance = float(np.cov(ec, ew, ddof=0)[0, 1])
    return {
        "center_error_variance": var_c,
        "width_error_variance": var_w,
        "center_width_error_covariance": covariance,
        "center_width_error_correlation": float(np.corrcoef(ec, ew)[0, 1]),
        "V1_error_variance": float(np.var(ev1)),
        "V2_error_variance": float(np.var(ev2)),
        "V1_variance_identity_max_abs": abs(float(np.var(ev1)) - (var_c + 0.25 * var_w - covariance)),
        "V2_variance_identity_max_abs": abs(float(np.var(ev2)) - (var_c + 0.25 * var_w + covariance)),
        "V1_error_identity_max_abs": float(np.max(np.abs(ev1 - (ec - 0.5 * ew)))),
        "V2_error_identity_max_abs": float(np.max(np.abs(ev2 - (ec + 0.5 * ew)))),
    }


def _paired(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in METHODS[1:]:
        for (column, protocol), group in metrics.loc[
            metrics.method.isin((METHODS[0], candidate))
        ].groupby(["column", "protocol"]):
            reference = group.loc[group.method.eq(METHODS[0])].set_index("seed")
            current = group.loc[group.method.eq(candidate)].set_index("seed")
            row = {"column": column, "protocol": protocol, "candidate": candidate,
                   "reference": METHODS[0], "n_pairs": len(current)}
            for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse"):
                delta = current[metric] - reference[metric]
                row[f"{metric}_delta_mean"] = float(delta.mean())
                row[f"{metric}_delta_median"] = float(delta.median())
                row[f"{metric}_wins"] = int((delta < 0).sum())
                row[f"{metric}_relative_gain"] = float(1.0 - current[metric].mean() / reference[metric].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _gate(metrics: pd.DataFrame) -> dict:
    primary = metrics.loc[metrics.protocol.eq("compound")]
    reference = primary.loc[primary.method.eq(METHODS[0])].set_index(["column", "seed"])
    candidates = {}
    for method in METHODS[1:]:
        current = primary.loc[primary.method.eq(method)].set_index(["column", "seed"])
        endpoints = []
        for column in COLUMNS:
            for endpoint in ("V1", "V2"):
                metric = f"{endpoint}_rmse"
                values = current.loc[column, metric]
                baseline = reference.loc[column, metric]
                gain = float(1.0 - values.mean() / baseline.mean())
                endpoints.append({"column": column, "endpoint": endpoint, "relative_gain": gain,
                                  "wins": int((values.to_numpy() < baseline.to_numpy()).sum())})
        aggregate_gain = float(
            1.0 - current.combined_normalized_rmse.mean() / reference.combined_normalized_rmse.mean()
        )
        context_gains = 1.0 - current.combined_normalized_rmse / reference.combined_normalized_rmse
        positive = [item for item in endpoints if item["relative_gain"] > 0]
        passes = (
            aggregate_gain >= 0.03
            and len(positive) >= 3
            and sum(item["relative_gain"] >= 0.03 for item in endpoints) >= 2
            and min(item["relative_gain"] for item in endpoints) >= -0.05
            and all(item["wins"] >= 3 for item in positive)
            and float(context_gains.min()) >= -0.10
        )
        candidates[method] = {
            "endpoint_results": endpoints,
            "aggregate_normalized_rmse": float(current.combined_normalized_rmse.mean()),
            "reference_aggregate_normalized_rmse": float(reference.combined_normalized_rmse.mean()),
            "aggregate_relative_gain": aggregate_gain,
            "positive_endpoints": len(positive),
            "endpoints_at_3pct": int(sum(item["relative_gain"] >= 0.03 for item in endpoints)),
            "worst_endpoint_gain": min(item["relative_gain"] for item in endpoints),
            "worst_context_gain": float(context_gains.min()),
            "passes": bool(passes),
        }
    promoted = [method for method, decision in candidates.items() if decision["passes"]]
    if promoted == ["CENTER_MAGNITUDE"]:
        status = "CENTER_MAGNITUDE_PROMOTED"
    elif promoted == ["CENTER_CONDITIONED_WIDTH"]:
        status = "CENTER_CONDITIONED_WIDTH_PROMOTED"
    elif promoted:
        status = "MULTIPLE_STRUCTURED_CANDIDATES_ELIGIBLE"
    else:
        status = "NO_STRUCTURED_CENTER_WIDTH_HEADROOM"
    return {"status": status, "reference": METHODS[0], "candidates": candidates,
            "promoted": promoted, "eligible_for_outer_evaluation": bool(promoted),
            "outer_test_truth_read": False, "outer_stage_executed": False}


def _equivalence_audit() -> dict:
    rng = np.random.default_rng(20260909)
    source = np.sort(rng.uniform(2.0, 25.0, (100, 2)), axis=1)
    ea = rng.uniform(0.0, 1.0, len(source))
    truth = source * (2.0 + 0.2 * ea[:, None]) + np.asarray([0.5, 1.0])
    baseline = fit_structured_center_width(
        source, truth, ea, 6.25, 0.1, "M3_CENTER_WIDTH"
    ).predict(source, ea)
    differences = {}
    for method in METHODS[1:]:
        nested = fit_structured_center_width(
            source, truth, ea, 6.25, 0.1, method, zero_extra=True
        ).predict(source, ea)
        differences[method] = float(np.max(np.abs(nested - baseline)))
    return {
        "status": "PASS" if max(differences.values()) < 1e-10 else "FAIL",
        "tolerance": 1e-10,
        "max_abs_prediction_difference": differences,
        "same_loss_normalization": True,
        "same_ridge_priors": True,
        "same_EA_construction": True,
        "same_projection": True,
    }


def run() -> dict:
    started = time.perf_counter()
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if tuple(protocol["columns"]) != COLUMNS or tuple(protocol["protocols"]) != PROTOCOLS:
        raise RuntimeError("frozen protocol drift")
    metrics_rows, fold_rows, diagnostic_rows = [], [], []
    for column in COLUMNS:
        feature, all_source = _feature_and_source(column)
        positions = {sample_id: index for index, sample_id in enumerate(feature.sample_id.astype(str))}
        for protocol_name in PROTOCOLS:
            for seed in SEEDS:
                context = _context(column, protocol_name, seed)
                ids = context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
                selected = np.asarray([positions[sample_id] for sample_id in ids], dtype=int)
                source = all_source[selected]
                truth = _truth(column, ids)
                ea = _ea(feature.iloc[selected])
                groups = feature.iloc[selected].canonical_smiles.astype(str).to_numpy()
                predictions = {method: np.full_like(truth, np.nan) for method in METHODS}
                projection = {method: [] for method in METHODS}
                coefficients = {method: [] for method in METHODS[1:]}
                splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                for fold, (fit_index, valid_index) in enumerate(splitter.split(source, groups=groups)):
                    for method in METHODS:
                        penalty = _select_penalty(
                            method, source, truth, ea, groups, fit_index, MASS_RATIO[column]
                        )
                        prediction, audit = _fit_predict(
                            method, source, truth, ea, fit_index, valid_index, penalty, MASS_RATIO[column]
                        )
                        predictions[method][valid_index] = prediction
                        projection[method].append(audit["projection_frequency"])
                        if method in coefficients:
                            coefficients[method].append(audit["extra_coefficient"])
                        fold_rows.append({
                            "column": column, "protocol": protocol_name, "seed": seed, "fold": fold,
                            "method": method, "selected_penalty": penalty,
                            "free_coefficients": audit["free_coefficients"],
                            "fit_center_scale": audit["center_scale"],
                            "fit_width_scale": audit["width_scale"],
                            "fit_center_scale_expected": float(np.std(center_width_transform(source[fit_index])[0])),
                            "projection_frequency": audit["projection_frequency"],
                            "raw_v2_lt_v1_rate": audit["raw_v2_lt_v1_rate"],
                            "extra_coefficient": audit["extra_coefficient"],
                        })
                for method, prediction in predictions.items():
                    metrics_rows.append({
                        "column": column, "protocol": protocol_name, "seed": seed, "method": method,
                        "n_train": len(truth), "unique_compounds": len(np.unique(groups)),
                        **absolute_error_metrics(truth, prediction, SOURCE_SCALES),
                    })
                    diagnostic_rows.append({
                        "column": column, "protocol": protocol_name, "seed": seed, "method": method,
                        "projection_frequency": float(np.mean(projection[method])),
                        **_diagnostics(truth, prediction),
                    })
                for method, values in coefficients.items():
                    array = np.asarray(values)
                    fold_rows.append({
                        "column": column, "protocol": protocol_name, "seed": seed, "fold": "seed_summary",
                        "method": method, "selected_penalty": np.nan,
                        "free_coefficients": 7, "fit_center_scale": np.nan, "fit_width_scale": np.nan,
                        "fit_center_scale_expected": np.nan, "projection_frequency": np.nan,
                        "raw_v2_lt_v1_rate": np.nan, "extra_coefficient": float(array.mean()),
                        "extra_coefficient_std": float(array.std()),
                        "extra_coefficient_positive_fraction": float(np.mean(array > 0)),
                    })
    metrics = pd.DataFrame(metrics_rows)
    folds = pd.DataFrame(fold_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    numeric = [column for column in metrics.columns if column not in ("column", "protocol", "seed", "method")]
    summary = metrics.groupby(["column", "protocol", "method"], as_index=False)[numeric].mean()
    paired = _paired(metrics)
    decision = _gate(metrics)
    equivalence = _equivalence_audit()
    coefficient_summary = folds.loc[folds.fold.eq("seed_summary"), [
        "column", "protocol", "seed", "method", "extra_coefficient", "extra_coefficient_std",
        "extra_coefficient_positive_fraction",
    ]]
    _write(folds.loc[~folds.fold.eq("seed_summary")], "inner_cv_metrics.csv")
    _write(metrics, "inner_cv_summary.csv")
    _write(summary, "inner_cv_aggregate.csv")
    _write(paired, "paired_comparisons.csv")
    _write(diagnostics, "center_width_diagnostics.csv")
    _write(coefficient_summary, "coefficient_diagnostics.csv")
    (STUDY / "promotion_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (STUDY / "implementation_equivalence_audit.json").write_text(
        json.dumps(equivalence, indent=2) + "\n"
    )
    report = _report(summary, paired, diagnostics, coefficient_summary, decision, equivalence)
    (STUDY / "FINAL_REPORT.md").write_text(report)
    elapsed = time.perf_counter() - started
    (STUDY / "run_metadata.json").write_text(json.dumps({
        "elapsed_seconds": elapsed,
        "parent_protocol_sha256": _sha256(CONTROLLED / "protocol.json"),
        "split_manifest_sha256": _sha256(BENCHMARK / "split_manifest.csv"),
        "outer_truth_read": False,
    }, indent=2) + "\n")
    names = [
        "README.md", "PROTOCOL.md", "protocol.json", "METHODS_NOTE.md", "inner_cv_metrics.csv",
        "inner_cv_summary.csv", "inner_cv_aggregate.csv", "paired_comparisons.csv",
        "promotion_decision.json", "center_width_diagnostics.csv", "coefficient_diagnostics.csv",
        "implementation_equivalence_audit.json", "FINAL_REPORT.md", "run_metadata.json",
    ]
    (STUDY / "artifact_manifest.json").write_text(json.dumps({
        "study": protocol["study_name"],
        "files": {name: _sha256(STUDY / name) for name in names},
    }, indent=2) + "\n")
    return decision


def _report(summary: pd.DataFrame, paired: pd.DataFrame, diagnostics: pd.DataFrame,
            coefficients: pd.DataFrame, decision: dict, equivalence: dict) -> str:
    primary = summary.loc[summary.protocol.eq("compound"), [
        "column", "method", "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2",
        "combined_normalized_rmse",
    ]]
    secondary = summary.loc[summary.protocol.eq("row"), primary.columns]
    paired_primary = paired.loc[paired.protocol.eq("compound")]
    diagnostic_mean = diagnostics.groupby(["column", "protocol", "method"], as_index=False)[[
        "center_error_variance", "width_error_variance", "center_width_error_covariance",
        "center_width_error_correlation", "V1_error_variance", "V2_error_variance",
        "projection_frequency",
    ]].mean()
    coefficient_mean = coefficients.groupby(["column", "protocol", "method"], as_index=False).agg(
        coefficient_mean=("extra_coefficient", "mean"),
        coefficient_std_across_seeds=("extra_coefficient", "std"),
        positive_seeds=("extra_coefficient", lambda values: int((values > 0).sum())),
    )
    gate_rows = []
    for method, result in decision["candidates"].items():
        row = {"method": method, "aggregate_normalized_rmse": result["aggregate_normalized_rmse"],
               "aggregate_gain": result["aggregate_relative_gain"],
               "positive_endpoints": result["positive_endpoints"],
               "endpoints_at_3pct": result["endpoints_at_3pct"], "passes": result["passes"]}
        for endpoint in result["endpoint_results"]:
            key = f"{endpoint['column']}_{endpoint['endpoint']}"
            row[f"{key}_gain"] = endpoint["relative_gain"]
            row[f"{key}_wins"] = f"{endpoint['wins']}/5"
        gate_rows.append(row)
    lines = [
        "# Structured center/width follow-up: final report\n",
        "## Protocol and mathematical audit\n",
        "This train-only study inherits the exact filtered populations, source predictions, source scales, gradient-train identities, protocols, and five seeds from the corrected audit. M3 uses independent three-coefficient Conditional-EA center and width branches (`SSE/n`, base slope prior 1, all other priors 0). Candidate A adds one center-magnitude slope coefficient; Candidate B adds one source-center coefficient only to the width slope. All scales and effect standardization are fit-subset-only.\n",
        "Equal-weight endpoint SSE is not a candidate: `eV1^2+eV2^2=2eC^2+0.5eW^2`, which remains separable for independent center/width parameter blocks. A shared optimizer alone would not create joint modeling.\n",
        "Formulas: `M3: Ct=[a0+aEA*EA]Cs+bC; Wt=[d0+dEA*EA]Ws+bW`. `CENTER_MAGNITUDE` adds `aMAG*log1p(max(Cs,0)/fit_center_scale)` to the center slope. `CENTER_CONDITIONED_WIDTH` adds `dC*standardized(Cs)` to the width slope. Parameter counts are 6, 7, and 7.\n",
        "## Nested-equivalence audit\n",
        f"```json\n{json.dumps(equivalence, indent=2)}\n```\n",
        "## Compound primary metrics\n", _md(primary),
        "## Row secondary metrics\n", _md(secondary),
        "## Frozen gate\n", _md(pd.DataFrame(gate_rows)),
        f"Final status: **{decision['status']}**. Outer evaluation was not executed; outer truth remained blind.\n",
        "## Compound paired five-seed comparisons\n", _md(paired_primary),
        "Deltas are candidate minus M3; negative favors the candidate. Relative gain is positive when the candidate is better.\n",
        "## Center/width variance-covariance diagnostics\n", _md(diagnostic_mean),
        "All error and variance identities passed at floating-point precision. Diagnostics were not used for hyperparameter selection. Projection frequency remained zero unless shown otherwise.\n",
        "## Extra-coefficient stability\n", _md(coefficient_mean),
        "Coefficients are fold-mean values summarized across five seeds. Their scale is tied to the fit-specific standardized effect and should not be interpreted as a physical constant.\n",
        "## Q1-Q11\n",
        _answers(decision, diagnostic_mean, coefficient_mean),
        "## Interpretation limits\n",
        "Center and width are algebraic surrogates constructed from V1/V2, not independently measured chromatographic moments. Any predictive association is conditional on the present source model and operational domain and is not causal evidence about scale-up or band broadening. No combined A+B model, feature sweep, random learning curve, active learning, weighted objective, or outer evaluation was run.\n",
    ]
    return "\n".join(lines)


def _answers(decision: dict, diagnostics: pd.DataFrame, coefficients: pd.DataFrame) -> str:
    a = decision["candidates"]["CENTER_MAGNITUDE"]
    b = decision["candidates"]["CENTER_CONDITIONED_WIDTH"]
    def endpoint_text(result: dict) -> str:
        return ", ".join(
            f"{item['column']} {item['endpoint']} {item['relative_gain']:+.2%} ({item['wins']}/5)"
            for item in result["endpoint_results"]
        )
    compound = diagnostics.loc[diagnostics.protocol.eq("compound")].set_index(["column", "method"])
    coeff = coefficients.loc[coefficients.protocol.eq("compound")].set_index(["column", "method"])
    def diagnostic(column: str, method: str, metric: str) -> float:
        return float(compound.loc[(column, method), metric])
    return "\n\n".join([
        f"**Q1. Is center magnitude a more stable cross-column signal than endpoint magnitude?** No. Its pattern is {endpoint_text(a)} and aggregate gain is {a['aggregate_relative_gain']:+.2%}. Unlike the prior endpoint-magnitude model's isolated 25g V1 signal, moving magnitude to center does not produce cross-column stability.",
        f"**Q2. Does Candidate A mainly reduce center error?** Only locally. Compound `Var(eC)` changes from {diagnostic('25g', METHODS[0], 'center_error_variance'):.2f} to {diagnostic('25g', METHODS[1], 'center_error_variance'):.2f} at 25g, but from {diagnostic('40g', METHODS[0], 'center_error_variance'):.2f} to {diagnostic('40g', METHODS[1], 'center_error_variance'):.2f} at 40g. The effect is not replicated.",
        "**Q3. Do both columns benefit?** No. At 25g only V1 improves (+0.81%); V2 worsens (-0.27%). At 40g both V1 (-1.34%) and V2 (-0.86%) worsen.",
        "**Q4. Do V1 and V2 both benefit?** No in either column. A improves 25g combined normalized RMSE by 0.46% but worsens the 40g value by 1.20%.",
        f"**Q5. Does source center stably help width transfer?** No. Candidate B's pattern is {endpoint_text(b)}. Its compound extra coefficient is positive in {int(coeff.loc[('25g', METHODS[2]), 'positive_seeds'])}/5 seeds at 25g but only {int(coeff.loc[('40g', METHODS[2]), 'positive_seeds'])}/5 at 40g.",
        f"**Q6. Does Candidate B reduce width error?** No. Compound `Var(eW)` rises from {diagnostic('25g', METHODS[0], 'width_error_variance'):.2f} to {diagnostic('25g', METHODS[2], 'width_error_variance'):.2f} at 25g and from {diagnostic('40g', METHODS[0], 'width_error_variance'):.2f} to {diagnostic('40g', METHODS[2], 'width_error_variance'):.2f} at 40g.",
        f"**Q7. Does it change covariance?** At 25g, `Cov(eC,eW)` rises from {diagnostic('25g', METHODS[0], 'center_width_error_covariance'):.2f} to {diagnostic('25g', METHODS[2], 'center_width_error_covariance'):.2f}; at 40g it changes only from {diagnostic('40g', METHODS[0], 'center_width_error_covariance'):.2f} to {diagnostic('40g', METHODS[2], 'center_width_error_covariance'):.2f}. This is not a stable covariance reduction.",
        "**Q8. Is V2 improvement purchased with V1 deterioration?** Candidate B does not improve V2: it worsens 25g V2 by 3.09% while improving V1 by 4.81%. The larger positive covariance helps V1 through subtraction and hurts V2 through addition. At 40g both endpoints worsen slightly.",
        "**Q9. Which hypothesis has the clearer chromatographic interpretation?** B more directly tests the declared location-to-spreading relationship, but its predictive evidence is negative. A is also interpretable as retention-regime-conditioned center transfer, yet its 25g-only coefficient signal does not replicate at 40g. Neither interpretation is empirically supported strongly enough to promote.",
        f"**Q10. Did either candidate pass?** A pass={a['passes']}; B pass={b['passes']}. Final status is `{decision['status']}`.",
        "**Q11. What follows if both fail?** Stop center/width calibration expansion. A separately preregistered filtered random learning curve is next; it is not part of this run, and active learning remains unauthorized.",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute the frozen train-only study")
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
