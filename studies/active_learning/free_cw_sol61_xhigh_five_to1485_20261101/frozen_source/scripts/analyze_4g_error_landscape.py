#!/usr/bin/env python3
"""Build a leakage-aware, descriptive 4 g error/uncertainty landscape.

The script consumes frozen predictor outputs and X-only canonical metadata.  It
does not fit a predictor, read a label outside an already materialized prediction
artifact, or choose an acquisition hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "studies/predictor/final_4g_qualification/results"
DEFAULT_CANONICAL = ROOT / "experiments/e0_4g_baseline/canonical_4g.csv"
DEFAULT_SPLITS = ROOT / "studies/predictor/final_4g_qualification/splits"
DEFAULT_OUTPUT = ROOT / "experiments/4g_error_landscape"
DEFAULT_UQ = ROOT / "experiments/e1_signal_qualification/uq_predictions.csv.gz"


PREDICTION_COLUMNS = [
    "V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90",
    "V1_true", "V2_true", "sample_id", "split",
]
TARGETS = ("V1", "V2")
CONTROL_KEYS = ["canonical_smiles", "loading solvent", "V/ul", "Volume of loading solvent/ul", "Flow mL/min", "column_specs"]


def _spearman(left: pd.Series, right: pd.Series) -> float:
    mask = np.isfinite(left.to_numpy(float)) & np.isfinite(right.to_numpy(float))
    if mask.sum() < 3:
        return float("nan")
    left_values, right_values = left.to_numpy(float)[mask], right.to_numpy(float)[mask]
    if np.unique(left_values).size < 2 or np.unique(right_values).size < 2:
        return float("nan")
    value = spearmanr(left_values, right_values).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _safe_qcut(values: pd.Series, q: int = 5) -> pd.Series:
    """Return stable quantile labels even when a feature has tied values."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().nunique() <= 1:
        return pd.Series(np.where(numeric.notna(), "all", "missing"), index=values.index)
    try:
        result = pd.qcut(numeric, q=q, duplicates="drop")
        return result.astype(str).replace("nan", "missing")
    except ValueError:
        return pd.Series(np.where(numeric.notna(), "all", "missing"), index=values.index)


def _parse_ea_fraction(values: pd.Series) -> pd.Series:
    parts = values.astype(str).str.split("/", expand=True)
    left = pd.to_numeric(parts[0], errors="coerce")
    right = pd.to_numeric(parts[1], errors="coerce")
    return right / (left + right)


def _descriptor_matrix(smiles: pd.Series) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    rows = []
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            rows.append([np.nan] * 5)
            continue
        rows.append([
            Descriptors.ExactMolWt(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.MolLogP(molecule),
        ])
    return np.asarray(rows, dtype=float)


def _nearest_distance(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    train_values = np.asarray(train_values, dtype=float)
    mean = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    values = np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    train_values = np.nan_to_num((train_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    result = np.empty(len(values), dtype=float)
    block = 1024
    for start in range(0, len(values), block):
        chunk = values[start:start + block]
        squared = np.square(chunk[:, None, :] - train_values[None, :, :]).sum(axis=2)
        result[start:start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
    return result


def _add_distances(frame: pd.DataFrame) -> pd.DataFrame:
    descriptor = _descriptor_matrix(frame["canonical_smiles"])
    condition = np.column_stack([
        frame["ea_fraction"].to_numpy(float),
        frame["Density g/ml"].to_numpy(float) * frame["V/ul"].to_numpy(float),
        frame["Volume of loading solvent/ul"].to_numpy(float),
        frame["Flow mL/min"].to_numpy(float),
        frame["loading_solvent_code"].to_numpy(float),
    ])
    result = frame.copy()
    for mode, seed in result[["mode", "seed"]].drop_duplicates().itertuples(index=False):
        mask = (result["mode"] == mode) & (result["seed"] == seed)
        train = mask & result["split"].eq("train")
        if not train.any():
            continue
        result.loc[mask, "descriptor_nn_distance"] = _nearest_distance(
            descriptor[mask.to_numpy()], descriptor[train.to_numpy()]
        )
        result.loc[mask, "condition_nn_distance"] = _nearest_distance(
            condition[mask.to_numpy()], condition[train.to_numpy()]
        )
    return result


def _load_uq(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, compression="infer")
    needed = {"sample_id", "split_mode", "outer_seed", "evaluation_split"}
    if not needed.issubset(frame.columns):
        return None
    frame = frame.rename(columns={"split_mode": "mode", "outer_seed": "seed", "evaluation_split": "split"})
    keep = [c for c in ["sample_id", "mode", "seed", "split", "ensemble_score", "latent_distance", "random_score"] if c in frame]
    return frame[keep].drop_duplicates(["sample_id", "mode", "seed", "split"])


def load_landscape(prediction_root: Path, canonical_path: Path, splits_path: Path, uq_path: Path) -> pd.DataFrame:
    canonical = pd.read_csv(canonical_path)
    canonical = canonical.set_index("sample_id", drop=False)
    parts = []
    for mode in ("row", "compound"):
        for seed in (42, 525, 1101):
            path = prediction_root / mode / f"seed_{seed}" / "predictions.csv.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            prediction = pd.read_csv(path, compression="infer")
            missing = set(PREDICTION_COLUMNS) - set(prediction.columns)
            if missing:
                raise ValueError(f"{path} missing prediction columns: {sorted(missing)}")
            prediction = prediction[PREDICTION_COLUMNS].copy()
            prediction["mode"], prediction["seed"] = mode, seed
            prediction = prediction.merge(canonical.reset_index(drop=True), on="sample_id", how="left", validate="one_to_one")
            if prediction["canonical_smiles"].isna().any():
                raise ValueError(f"canonical metadata missing for {path}")
            split_path = splits_path / f"{mode}_seed_{seed}.csv"
            if split_path.exists():
                split = pd.read_csv(split_path)[["sample_id", "split"]].rename(columns={"split": "manifest_split"})
                prediction = prediction.merge(split, on="sample_id", how="left", validate="one_to_one")
                if prediction["manifest_split"].notna().any() and (prediction["split"] != prediction["manifest_split"]).any():
                    raise ValueError(f"prediction/manifest split mismatch in {path}")
                prediction = prediction.drop(columns=["manifest_split"])
            parts.append(prediction)
    result = pd.concat(parts, ignore_index=True)
    result["ea_fraction"] = _parse_ea_fraction(result["PE/EA"])
    result["pe_fraction"] = 1.0 - result["ea_fraction"]
    result["loading_solvent_code"] = result["loading solvent"].map({"PE": 0, "EA": 1, "DCM": 2}).astype(float)
    result["compound_frequency"] = result.groupby(["mode", "seed", "canonical_smiles"])["sample_id"].transform("size")
    result["condition_frequency"] = result.groupby(["mode", "seed", "canonical_smiles", "PE/EA", "loading solvent", "V/ul", "Volume of loading solvent/ul", "Flow mL/min", "column_specs"])["sample_id"].transform("size")
    for target in TARGETS:
        result[f"{target}_abs_error"] = (result[f"{target}_true"] - result[f"{target}_q50"]).abs()
        result[f"{target}_squared_error"] = np.square(result[f"{target}_true"] - result[f"{target}_q50"])
        result[f"{target}_interval_width"] = result[f"{target}_q90"] - result[f"{target}_q10"]
        result[f"{target}_interval_width_clipped"] = result[f"{target}_interval_width"].clip(lower=0)
        result[f"{target}_covered_80"] = result[f"{target}_true"].between(result[f"{target}_q10"], result[f"{target}_q90"])
        result[f"{target}_crossing"] = (result[f"{target}_q10"] > result[f"{target}_q50"]) | (result[f"{target}_q50"] > result[f"{target}_q90"])
    result["combined_squared_error"] = 0.5 * (result.V1_squared_error + result.V2_squared_error)
    result["true_retention_magnitude"] = 0.5 * (result.V1_true + result.V2_true)
    result["predicted_retention_magnitude"] = 0.5 * (result.V1_q50 + result.V2_q50)
    # Scales are added per fit below; this field is populated after normalization load.
    result = _add_distances(result)
    uq = _load_uq(uq_path)
    uq_status = "not_provided"
    if uq is not None:
        join_keys = ["sample_id", "mode", "seed", "split"]
        left_keys = set(map(tuple, result[join_keys].itertuples(index=False, name=None)))
        right_keys = set(map(tuple, uq[join_keys].itertuples(index=False, name=None)))
        overlap = len(left_keys & right_keys)
        if overlap:
            if overlap != len(right_keys):
                raise ValueError("uncertainty artifact partially overlaps prediction identities")
            result = result.merge(uq, on=join_keys, how="left", validate="many_to_one")
            uq_status = f"joined_{overlap}_rows"
        else:
            uq_status = "rejected_no_shared_prediction_sample_ids"
    result.attrs["uq_alignment"] = uq_status
    return result


def add_scales(frame: pd.DataFrame, prediction_root: Path) -> pd.DataFrame:
    result = frame.copy()
    scales = {}
    for mode in ("row", "compound"):
        for seed in (42, 525, 1101):
            path = prediction_root / mode / f"seed_{seed}" / "normalization.json"
            payload = json.loads(path.read_text())
            scales[(mode, seed)] = payload["target_scales"]
    result["V1_train_scale"] = [scales[(m, s)]["V1"] for m, s in zip(result["mode"], result["seed"])]
    result["V2_train_scale"] = [scales[(m, s)]["V2"] for m, s in zip(result["mode"], result["seed"])]
    result["V1_normalized_abs_error"] = result.V1_abs_error / result.V1_train_scale
    result["V2_normalized_abs_error"] = result.V2_abs_error / result.V2_train_scale
    result["combined_normalized_error"] = np.sqrt(0.5 * ((result.V1_abs_error / result.V1_train_scale) ** 2 + (result.V2_abs_error / result.V2_train_scale) ** 2))
    return result


def _summary_rows(frame: pd.DataFrame, signal: str, value_column: str, group_column: str = "bin") -> list[dict]:
    rows = []
    for keys, group in frame.groupby(["mode", "seed", "split", group_column], dropna=False, observed=False):
        mode, seed, split, label = keys
        rows.append({
            "mode": mode, "seed": int(seed), "split": split, "signal": signal,
            "bin": str(label), "n": len(group), "mean_signal": float(group[value_column].mean()),
            "median_signal": float(group[value_column].median()), "mean_combined_error": float(group.combined_normalized_error.mean()),
            "median_combined_error": float(group.combined_normalized_error.median()),
            "mean_V1_abs_error": float(group.V1_abs_error.mean()), "mean_V2_abs_error": float(group.V2_abs_error.mean()),
            "mean_V1_width": float(group.V1_interval_width_clipped.mean()), "mean_V2_width": float(group.V2_interval_width_clipped.mean()),
            "V1_coverage_80": float(group.V1_covered_80.mean()), "V2_coverage_80": float(group.V2_covered_80.mean()),
        })
    return rows


def make_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    signals = [
        "V1_true", "V2_true", "V1_q50", "V2_q50", "true_retention_magnitude", "predicted_retention_magnitude", "ea_fraction", "compound_frequency",
        "condition_frequency", "descriptor_nn_distance", "condition_nn_distance", "ensemble_score", "latent_distance",
    ]
    bin_rows = []
    for signal in signals:
        if signal not in frame or not pd.api.types.is_numeric_dtype(frame[signal]):
            continue
        working = frame.copy()
        working["bin"] = _safe_qcut(working[signal])
        bin_rows.extend(_summary_rows(working, signal, signal))
    bins = pd.DataFrame(bin_rows)

    interactions = [("ea_fraction", "V1_q50"), ("ea_fraction", "V2_q50"), ("ea_fraction", "descriptor_nn_distance"),
                    ("descriptor_nn_distance", "condition_nn_distance"), ("compound_frequency", "condition_frequency"),
                    ("V1_q50", "V2_q50")]
    interaction_rows = []
    for left, right in interactions:
        working = frame.copy()
        working["left_bin"] = _safe_qcut(working[left])
        working["right_bin"] = _safe_qcut(working[right])
        for keys, group in working.groupby(["mode", "seed", "split", "left_bin", "right_bin"], observed=False, dropna=False):
            mode, seed, split, left_bin, right_bin = keys
            if len(group) == 0:
                continue
            interaction_rows.append({"mode": mode, "seed": int(seed), "split": split, "left": left, "right": right,
                                     "left_bin": str(left_bin), "right_bin": str(right_bin), "n": len(group),
                                     "mean_combined_error": float(group.combined_normalized_error.mean()),
                                     "mean_V1_abs_error": float(group.V1_abs_error.mean()),
                                     "mean_V2_abs_error": float(group.V2_abs_error.mean()),
                                     "mean_V1_width": float(group.V1_interval_width_clipped.mean()),
                                     "mean_V2_width": float(group.V2_interval_width_clipped.mean())})
    calibration_rows = []
    slice_columns = [None, "ea_fraction", "V1_q50", "V2_q50", "descriptor_nn_distance", "condition_nn_distance", "compound_frequency"]
    for target in TARGETS:
        for slice_column in slice_columns:
            working = frame.copy()
            if slice_column is None:
                working["slice"] = "overall"
            else:
                working["slice"] = _safe_qcut(working[slice_column])
            for keys, group in working.groupby(["mode", "seed", "split", "slice"], observed=False, dropna=False):
                mode, seed, split, label = keys
                calibration_rows.append({"mode": mode, "seed": int(seed), "split": split, "target": target,
                    "slice_signal": slice_column or "overall", "slice": str(label), "n": len(group),
                    "coverage_80": float(group[f"{target}_covered_80"].mean()),
                    "mean_width": float(group[f"{target}_interval_width_clipped"].mean()),
                    "crossing_rate": float(group[f"{target}_crossing"].mean()),
                    "mean_abs_error": float(group[f"{target}_abs_error"].mean()),
                    "width_error_spearman": _spearman(group[f"{target}_interval_width_clipped"], group[f"{target}_abs_error"])})
    correlation_rows = []
    error_columns = ["V1_abs_error", "V2_abs_error", "combined_normalized_error"]
    signal_columns = ["V1_interval_width_clipped", "V2_interval_width_clipped", "ensemble_score", "latent_distance", "descriptor_nn_distance", "condition_nn_distance", "ea_fraction", "V1_q50", "V2_q50", "true_retention_magnitude", "predicted_retention_magnitude"]
    for mode, seed, split in frame[["mode", "seed", "split"]].drop_duplicates().itertuples(index=False):
        group = frame[(frame["mode"] == mode) & (frame["seed"] == seed) & (frame["split"] == split)]
        for signal in signal_columns:
            if signal not in group or group[signal].notna().sum() < 3:
                continue
            for error in error_columns:
                correlation_rows.append({"mode": mode, "seed": int(seed), "split": split, "signal": signal, "error": error,
                                          "n": int(group[[signal, error]].dropna().shape[0]), "spearman": _spearman(group[signal], group[error])})
    return {"bin_summary": bins, "interaction_summary": pd.DataFrame(interaction_rows),
            "calibration_summary": pd.DataFrame(calibration_rows), "correlation_summary": pd.DataFrame(correlation_rows)}


def repeated_condition_summary(canonical_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(canonical_path)
    key = ["canonical_smiles", "PE/EA", "loading solvent", "V/ul", "Volume of loading solvent/ul", "Flow mL/min", "column_specs"]
    grouped = frame.groupby(key, dropna=False)
    result = grouped.agg(n_rows=("sample_id", "size"), n_compounds=("canonical_smiles", "nunique"),
                         V1_mean=("V1_ml", "mean"), V1_std=("V1_ml", "std"),
                         V2_mean=("V2_ml", "mean"), V2_std=("V2_ml", "std"),
                         V1_min=("V1_ml", "min"), V1_max=("V1_ml", "max"),
                         V2_min=("V2_ml", "min"), V2_max=("V2_ml", "max")).reset_index()
    result = result[result.n_rows >= 2].copy()
    result["V1_std"] = result["V1_std"].fillna(0.0)
    result["V2_std"] = result["V2_std"].fillna(0.0)
    return result.sort_values(["n_rows", "canonical_smiles"], ascending=[False, True])


def local_sensitivity_summary(canonical: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Estimate within-compound EA gradients/curvature where the design supports it."""
    source = canonical.copy()
    source["ea_fraction"] = _parse_ea_fraction(source["PE/EA"])
    point = source.groupby(CONTROL_KEYS + ["ea_fraction"], dropna=False).agg(
        n_rows=("sample_id", "size"), V1_mean=("V1_ml", "mean"), V2_mean=("V2_ml", "mean")
    ).reset_index()
    gradient_rows, curvature_rows = [], []
    for _, group in point.groupby(CONTROL_KEYS, dropna=False):
        group = group.sort_values("ea_fraction").reset_index(drop=True)
        if len(group) < 2:
            continue
        for index in range(len(group) - 1):
            left, right = group.iloc[index], group.iloc[index + 1]
            delta = float(right.ea_fraction - left.ea_fraction)
            if delta <= 0:
                continue
            gradient_rows.append({**{key: left[key] for key in CONTROL_KEYS},
                "left_ea_fraction": float(left.ea_fraction), "right_ea_fraction": float(right.ea_fraction),
                "delta_ea_fraction": delta, "group_unique_ea": len(group), "left_n": int(left.n_rows), "right_n": int(right.n_rows),
                "V1_abs_gradient": abs(float(right.V1_mean - left.V1_mean)) / delta,
                "V2_abs_gradient": abs(float(right.V2_mean - left.V2_mean)) / delta,
                "V1_signed_gradient": float(right.V1_mean - left.V1_mean) / delta,
                "V2_signed_gradient": float(right.V2_mean - left.V2_mean) / delta})
        if len(group) < 3:
            continue
        for index in range(1, len(group) - 1):
            previous, center, following = group.iloc[index - 1], group.iloc[index], group.iloc[index + 1]
            d_left = float(center.ea_fraction - previous.ea_fraction)
            d_right = float(following.ea_fraction - center.ea_fraction)
            if d_left <= 0 or d_right <= 0:
                continue
            d1_v1 = float(center.V1_mean - previous.V1_mean) / d_left
            d2_v1 = float(following.V1_mean - center.V1_mean) / d_right
            d1_v2 = float(center.V2_mean - previous.V2_mean) / d_left
            d2_v2 = float(following.V2_mean - center.V2_mean) / d_right
            curvature_rows.append({**{key: center[key] for key in CONTROL_KEYS},
                "previous_ea_fraction": float(previous.ea_fraction), "center_ea_fraction": float(center.ea_fraction),
                "following_ea_fraction": float(following.ea_fraction), "group_unique_ea": len(group),
                "V1_abs_curvature": abs(d2_v1 - d1_v1), "V2_abs_curvature": abs(d2_v2 - d1_v2),
                "V1_monotonicity_violation": bool(d1_v1 * d2_v1 < 0),
                "V2_monotonicity_violation": bool(d1_v2 * d2_v2 < 0)})
    gradients = pd.DataFrame(gradient_rows)
    curvatures = pd.DataFrame(curvature_rows)

    # Relate local geometry to frozen prediction errors without using it to select a model.
    error_keys = ["mode", "seed", "split"] + CONTROL_KEYS + ["ea_fraction"]
    point_error = predictions.groupby(error_keys, dropna=False).agg(
        V1_abs_error=("V1_abs_error", "mean"), V2_abs_error=("V2_abs_error", "mean"),
        combined_normalized_error=("combined_normalized_error", "mean"), n_predictions=("sample_id", "size")
    ).reset_index()
    relation_rows = []
    if not gradients.empty:
        for mode, seed, split in predictions[["mode", "seed", "split"]].drop_duplicates().itertuples(index=False):
            group = point_error[(point_error["mode"] == mode) & (point_error["seed"] == seed) & (point_error["split"] == split)]
            left = group.rename(columns={"ea_fraction": "left_ea_fraction", "V1_abs_error": "left_V1_error", "V2_abs_error": "left_V2_error", "combined_normalized_error": "left_combined_error"})
            right = group.rename(columns={"ea_fraction": "right_ea_fraction", "V1_abs_error": "right_V1_error", "V2_abs_error": "right_V2_error", "combined_normalized_error": "right_combined_error"})
            local = gradients.copy()
            local["mode"], local["seed"], local["split"] = mode, seed, split
            joined = local.merge(left[["mode", "seed", "split"] + CONTROL_KEYS + ["left_ea_fraction", "left_V1_error", "left_V2_error", "left_combined_error"]], on=["mode", "seed", "split"] + CONTROL_KEYS + ["left_ea_fraction"], how="inner")
            joined = joined.merge(right[["mode", "seed", "split"] + CONTROL_KEYS + ["right_ea_fraction", "right_V1_error", "right_V2_error", "right_combined_error"]], on=["mode", "seed", "split"] + CONTROL_KEYS + ["right_ea_fraction"], how="inner")
            if len(joined):
                joined["mean_V1_endpoint_error"] = (joined.left_V1_error + joined.right_V1_error) / 2
                joined["mean_V2_endpoint_error"] = (joined.left_V2_error + joined.right_V2_error) / 2
                joined["mean_combined_endpoint_error"] = (joined.left_combined_error + joined.right_combined_error) / 2
                for signal in ["V1_abs_gradient", "V2_abs_gradient"]:
                    for error in ["mean_V1_endpoint_error", "mean_V2_endpoint_error", "mean_combined_endpoint_error"]:
                        relation_rows.append({"mode": mode, "seed": int(seed), "split": split, "kind": "gradient", "signal": signal, "error": error,
                            "n": int(joined[[signal, error]].dropna().shape[0]), "spearman": _spearman(joined[signal], joined[error])})
    return {"local_gradient_summary": gradients, "local_curvature_summary": curvatures,
            "local_sensitivity_error_relation": pd.DataFrame(relation_rows)}


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_landscape(args.predictions, args.canonical, args.splits, args.uq)
    frame = add_scales(frame, args.predictions)
    summaries = make_summaries(frame)
    canonical = pd.read_csv(args.canonical)
    local_summaries = local_sensitivity_summary(canonical, frame)
    frame.to_csv(args.output / "row_level_metrics.csv.gz", index=False, compression="gzip")
    for name, table in summaries.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    for name, table in local_summaries.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    repeated_condition_summary(args.canonical).to_csv(args.output / "repeated_condition_summary.csv", index=False)
    metadata = {"rows": int(len(frame)), "prediction_runs": int(frame[["mode", "seed"]].drop_duplicates().shape[0]),
                "splits": sorted(frame["split"].dropna().unique().tolist()), "source": {
                    "canonical": str(args.canonical.relative_to(ROOT)), "predictions": str(args.predictions.relative_to(ROOT)),
                    "uq": str(args.uq.relative_to(ROOT)) if args.uq.exists() else None,
                    "uq_alignment": frame.attrs.get("uq_alignment", "unknown")},
                "post_hoc_descriptive": True, "test_used_for_selection": False}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--uq", type=Path, default=DEFAULT_UQ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
