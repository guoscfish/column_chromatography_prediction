#!/usr/bin/env python3
"""Run Gate 0-3 paired legacy-threshold versus no-threshold experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import sha256_file, write_artifact_manifest, write_environment
from scripts.run_e0_8g_controls import DATA_DIR, load_graph_cache
from scripts.run_e0_8g_transfer import (
    SOURCE_4G_DIR,
    build_model,
    build_model_data,
    make_loaders,
    metrics_from_arrays,
)
from scripts.run_g0_1_quantile_monotonicity import (
    install_monotonic_head,
    predict,
    run_configuration,
)
from scripts.run_g0_2_interval_calibration import (
    calibration_curve_metrics,
    finite_sample_quantile,
    normalized_residual_scores,
)


PROTOCOLS = ("legacy_threshold", "no_threshold")
TARGETS = ("V1", "V2")
SLICES = ("full", "common", "tail")


def prepare_no_threshold_data() -> pd.DataFrame:
    decisions = pd.read_csv(DATA_DIR / "sample_decisions_8g.csv")
    if len(decisions) != 574 or set(decisions["decision"]) != {"keep", "drop"}:
        raise ValueError("Unexpected 8g decision table; re-audit before G0-3")
    decisions["canonical_smiles"] = decisions["smiles"].map(
        lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(str(value)), canonical=True)
    )
    decisions["threshold_excluded"] = decisions["decision"].eq("drop")
    decisions["over_v1_threshold"] = decisions["V1_ml"].gt(60.0)
    decisions["over_v2_threshold"] = decisions["V2_ml"].gt(120.0)
    decisions = decisions.reset_index(drop=True)
    if int(decisions["threshold_excluded"].sum()) != 22:
        raise ValueError("Expected exactly 22 legacy-threshold exclusions")
    return decisions


def _assign_stratified_rows(data: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    assignments = np.empty(len(data), dtype=object)
    for excluded in (False, True):
        indices = np.flatnonzero(data["threshold_excluded"].to_numpy() == excluded)
        indices = rng.permutation(indices)
        train_end = int(0.8 * len(indices))
        valid_end = train_end + int(0.1 * len(indices))
        assignments[indices[:train_end]] = "train"
        assignments[indices[train_end:valid_end]] = "valid"
        assignments[indices[valid_end:]] = "test"
    return assignments


def _assign_compounds(data: pd.DataFrame, seed: int, candidates: int = 20000) -> np.ndarray:
    """Find a deterministic compound split balanced for rows and tail rows."""
    grouped = data.groupby("canonical_smiles", sort=True).agg(
        rows=("sample_id", "size"), tail=("threshold_excluded", "sum")
    )
    groups = grouped.index.to_numpy()
    row_sizes = grouped["rows"].to_numpy(dtype=int)
    tail_sizes = grouped["tail"].to_numpy(dtype=int)
    target_rows = np.array([0.8, 0.1, 0.1]) * len(data)
    target_tail = np.array([0.8, 0.1, 0.1]) * int(data["threshold_excluded"].sum())
    rng = np.random.RandomState(seed)
    best_score, best_assignment = float("inf"), None
    for _ in range(candidates):
        order = rng.permutation(len(groups))
        cumulative = np.cumsum(row_sizes[order])
        train_cut = int(np.argmin(np.abs(cumulative - target_rows[0]))) + 1
        valid_cut = int(np.argmin(np.abs(cumulative - target_rows[:2].sum()))) + 1
        valid_cut = max(valid_cut, train_cut + 1)
        parts = (order[:train_cut], order[train_cut:valid_cut], order[valid_cut:])
        counts = np.array([row_sizes[part].sum() for part in parts], dtype=float)
        tails = np.array([tail_sizes[part].sum() for part in parts], dtype=float)
        missing_tail_penalty = 100.0 * float(np.any(tails == 0))
        score = float(
            np.abs(counts - target_rows).sum() / len(data)
            + 5.0 * np.abs(tails - target_tail).sum() / max(target_tail.sum(), 1)
            + missing_tail_penalty
        )
        if score < best_score:
            best_score = score
            labels = np.empty(len(groups), dtype=object)
            for label, part in zip(("train", "valid", "test"), parts):
                labels[part] = label
            best_assignment = dict(zip(groups, labels))
    if best_assignment is None:
        raise RuntimeError("Unable to construct compound split")
    return data["canonical_smiles"].map(best_assignment).to_numpy(dtype=object)


def make_paired_split(
    data: pd.DataFrame, split_mode: str, seed: int, compound_candidates: int = 20000
) -> pd.DataFrame:
    assignments = (
        _assign_stratified_rows(data, seed)
        if split_mode == "row"
        else _assign_compounds(data, seed, candidates=compound_candidates)
    )
    split = data[
        ["sample_id", "source_row_1based", "canonical_smiles", "threshold_excluded"]
    ].copy()
    split["canonical_index"] = np.arange(len(data), dtype=int)
    split["split"] = assignments
    split["seed"] = seed
    split["split_mode"] = split_mode
    tail_counts = split.groupby("split")["threshold_excluded"].sum()
    if any(int(tail_counts.get(name, 0)) == 0 for name in ("train", "valid", "test")):
        raise ValueError(f"{split_mode}/seed={seed} has a split without tail records")
    if split_mode == "compound" and split.groupby("canonical_smiles")["split"].nunique().max() != 1:
        raise AssertionError("Compound leakage detected")
    return split


def load_trained_model(checkpoint_path: Path, device: torch.device):
    model = build_model(device)
    install_monotonic_head(model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def prediction_frame(
    model,
    loaders: dict,
    data: pd.DataFrame,
    protocol: str,
    split_mode: str,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for evaluation_split in ("valid", "test"):
        _, y_true, pred, indices = predict(model, loaders[evaluation_split], device)
        for row_index, true, quantiles in zip(indices, y_true, pred):
            source = data.iloc[int(row_index)]
            rows.append(
                {
                    "protocol": protocol,
                    "split_mode": split_mode,
                    "seed": seed,
                    "evaluation_split": evaluation_split,
                    "sample_id": source["sample_id"],
                    "source_row_1based": int(source["source_row_1based"]),
                    "canonical_smiles": source["canonical_smiles"],
                    "threshold_excluded": bool(source["threshold_excluded"]),
                    "over_v1_threshold": bool(source["over_v1_threshold"]),
                    "over_v2_threshold": bool(source["over_v2_threshold"]),
                    "V1_true": float(true[0]),
                    "V2_true": float(true[1]),
                    "V1_q10": float(quantiles[0]),
                    "V1_q50": float(quantiles[1]),
                    "V1_q90": float(quantiles[2]),
                    "V2_q10": float(quantiles[3]),
                    "V2_q50": float(quantiles[4]),
                    "V2_q90": float(quantiles[5]),
                }
            )
    return pd.DataFrame(rows)


def fit_and_apply_calibration(
    predictions: pd.DataFrame, protocol: str
) -> tuple[pd.DataFrame, list[dict]]:
    calibrated = predictions.copy()
    validation = calibrated[calibrated["evaluation_split"].eq("valid")]
    if protocol == "legacy_threshold":
        validation = validation[~validation["threshold_excluded"]]
    factors = []
    for target in TARGETS:
        scores = normalized_residual_scores(validation, target)
        raw_alpha = finite_sample_quantile(scores, 0.8)
        alpha = max(1.0, raw_alpha)
        median = calibrated[f"{target}_q50"].to_numpy(dtype=float)
        lower = calibrated[f"{target}_q10"].to_numpy(dtype=float)
        upper = calibrated[f"{target}_q90"].to_numpy(dtype=float)
        calibrated[f"{target}_q10_calibrated"] = median - alpha * (median - lower)
        calibrated[f"{target}_q90_calibrated"] = median + alpha * (upper - median)
        factors.append(
            {
                "target": target,
                "calibration_rows": len(validation),
                "raw_alpha_80": raw_alpha,
                "alpha_80": alpha,
                "inflation_only_constraint_active": bool(raw_alpha < 1.0),
            }
        )
    return calibrated, factors


def add_sample_scores(predictions: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    result = predictions.copy()
    error = np.zeros(len(result), dtype=float)
    raw_uq = np.zeros(len(result), dtype=float)
    calibrated_uq = np.zeros(len(result), dtype=float)
    for target in TARGETS:
        scale = max(float(scales[target]), 1e-8)
        error += np.abs(result[f"{target}_true"] - result[f"{target}_q50"]) / scale
        raw_uq += (result[f"{target}_q90"] - result[f"{target}_q10"]) / scale
        calibrated_uq += (
            result[f"{target}_q90_calibrated"] - result[f"{target}_q10_calibrated"]
        ) / scale
    result["normalized_absolute_error"] = error
    result["raw_quantile_width_score"] = raw_uq
    result["calibrated_quantile_width_score"] = calibrated_uq
    return result


def select_slice(frame: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "full":
        return frame
    if slice_name == "common":
        return frame[~frame["threshold_excluded"]]
    if slice_name == "tail":
        return frame[frame["threshold_excluded"]]
    raise ValueError(slice_name)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def slice_metrics(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    scales: dict[str, float],
    slice_name: str,
) -> dict:
    frame = select_slice(predictions, slice_name)
    if frame.empty:
        return {"rows": 0}
    y_true = frame[["V1_true", "V2_true"]].to_numpy(dtype=np.float32)
    pred = frame[["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"]].to_numpy(
        dtype=np.float32
    )
    result = {"rows": len(frame), **metrics_from_arrays(y_true, pred)}
    normalized_rmse = []
    auce_values = []
    for target in TARGETS:
        lower = frame[f"{target}_q10_calibrated"].to_numpy(dtype=float)
        upper = frame[f"{target}_q90_calibrated"].to_numpy(dtype=float)
        true = frame[f"{target}_true"].to_numpy(dtype=float)
        result[f"{target}_calibrated_coverage"] = float(
            np.mean((true >= lower) & (true <= upper))
        )
        result[f"{target}_calibrated_width"] = float(np.mean(upper - lower))
        normalized_rmse.append(result[f"{target}_rmse"] / max(scales[target], 1e-8))
        if len(frame) >= 2:
            auce, _ = calibration_curve_metrics(calibration, frame, target)
            result[f"{target}_auce"] = auce
            auce_values.append(auce)
        else:
            result[f"{target}_auce"] = float("nan")
    result["mean_normalized_rmse"] = float(np.mean(normalized_rmse))
    result["mean_normalized_absolute_error"] = float(
        frame["normalized_absolute_error"].mean()
    )
    result["raw_width_error_spearman"] = safe_spearman(
        frame["raw_quantile_width_score"], frame["normalized_absolute_error"]
    )
    result["calibrated_width_error_spearman"] = safe_spearman(
        frame["calibrated_quantile_width_score"], frame["normalized_absolute_error"]
    )
    result["mean_auce"] = float(np.mean(auce_values)) if auce_values else float("nan")
    result["calibrated_crossing_rate"] = float(
        np.mean(
            (frame["V1_q10_calibrated"] > frame["V1_q50"])
            | (frame["V1_q50"] > frame["V1_q90_calibrated"])
            | (frame["V2_q10_calibrated"] > frame["V2_q50"])
            | (frame["V2_q50"] > frame["V2_q90_calibrated"])
        )
    )
    return result


def high_score_diagnostics(predictions: pd.DataFrame, score_column: str) -> dict:
    count = max(1, int(math.ceil(0.1 * len(predictions))))
    selected = predictions.nlargest(count, score_column)
    hard = predictions.nlargest(count, "normalized_absolute_error")
    base_tail = float(predictions["threshold_excluded"].mean())
    selected_tail = float(selected["threshold_excluded"].mean())
    overlap = len(set(selected["sample_id"]) & set(hard["sample_id"])) / count
    return {
        "score": score_column,
        "rows": len(predictions),
        "top_rows": count,
        "base_tail_fraction": base_tail,
        "top_tail_fraction": selected_tail,
        "tail_enrichment": selected_tail / base_tail if base_tail else float("nan"),
        "hard_error_top10_overlap": overlap,
        "top_mean_V1": float(selected["V1_true"].mean()),
        "top_mean_V2": float(selected["V2_true"].mean()),
        "top_mean_error": float(selected["normalized_absolute_error"].mean()),
    }


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split_mode", "seed", "evaluation_split", "slice"]
    for context, group in metrics.groupby(keys, sort=True):
        if set(group["protocol"]) != set(PROTOCOLS):
            continue
        legacy = group[group["protocol"].eq("legacy_threshold")].iloc[0]
        no_threshold = group[group["protocol"].eq("no_threshold")].iloc[0]
        row = dict(zip(keys, context))
        for metric in (
            "mean_normalized_rmse",
            "mean_normalized_absolute_error",
            "V1_calibrated_coverage",
            "V2_calibrated_coverage",
            "mean_auce",
            "raw_width_error_spearman",
            "calibrated_width_error_spearman",
        ):
            row[f"delta_{metric}_no_threshold_minus_legacy"] = (
                no_threshold.get(metric, np.nan) - legacy.get(metric, np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validation_decision(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> dict:
    validation = metrics[metrics["evaluation_split"].eq("valid")]
    means = validation.groupby(["protocol", "slice"]).mean(numeric_only=True)
    legacy_common = means.loc[("legacy_threshold", "common"), "mean_normalized_rmse"]
    no_threshold_common = means.loc[("no_threshold", "common"), "mean_normalized_rmse"]
    legacy_tail = means.loc[("legacy_threshold", "tail"), "mean_normalized_absolute_error"]
    no_threshold_tail = means.loc[("no_threshold", "tail"), "mean_normalized_absolute_error"]
    common_change = float(no_threshold_common / legacy_common - 1)
    tail_change = float(no_threshold_tail / legacy_tail - 1)
    valid_diag = diagnostics[
        diagnostics["evaluation_split"].eq("valid")
        & diagnostics["score"].eq("calibrated_quantile_width_score")
    ]
    legacy_enrichment = float(
        valid_diag[valid_diag["protocol"].eq("legacy_threshold")]["tail_enrichment"].mean()
    )
    # Pre-registered conservative default: do not delete valid tail chemistry unless
    # inclusion both harms common validation >10% and fails to improve tail error.
    choose_no_threshold = bool(common_change <= 0.10 or tail_change <= -0.10)
    return {
        "selected_threshold_protocol": "no_threshold" if choose_no_threshold else "legacy_threshold",
        "selection_uses_test": False,
        "common_validation_normalized_rmse_relative_change": common_change,
        "tail_validation_normalized_absolute_error_relative_change": tail_change,
        "legacy_calibrated_width_tail_enrichment_validation": legacy_enrichment,
        "rule": "select legacy only if no-threshold harms common validation >10% and improves tail error <10%; otherwise preserve tail chemistry",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "g0_3_threshold_sensitivity"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 525, 1101])
    parser.add_argument(
        "--split-modes", nargs="+", choices=["row", "compound"], default=["row", "compound"]
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--compound-split-candidates", type=int, default=20000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if (output_dir / "slice_metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite finalized experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    data = prepare_no_threshold_data()
    graph_cache = load_graph_cache()
    missing_graphs = sorted(set(data["canonical_smiles"]) - set(graph_cache))
    if missing_graphs:
        raise ValueError(f"Missing graph cache entries: {missing_graphs}")
    data.to_csv(output_dir / "canonical_8g_no_threshold.csv", index=False)
    scaler = json.loads((SOURCE_4G_DIR / "scaler.json").read_text(encoding="utf-8"))
    source_checkpoint = SOURCE_4G_DIR / "checkpoints" / "best.pt"
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    all_training_results, all_predictions, all_factors = [], [], []
    split_manifest = {}

    for split_mode in args.split_modes:
        for seed in args.seeds:
            run_key = f"{split_mode}_seed_{seed}"
            split = make_paired_split(
                data, split_mode, seed, compound_candidates=args.compound_split_candidates
            )
            split_path = splits_dir / f"{run_key}.csv"
            split.to_csv(split_path, index=False)
            split_manifest[run_key] = {
                "path": str(split_path.relative_to(ROOT)),
                "sha256": sha256_file(split_path),
                "counts": {
                    f"{name}|tail={bool(excluded)}": int(count)
                    for (name, excluded), count in split.groupby(
                        ["split", "threshold_excluded"]
                    ).size().items()
                },
            }
            full_atom, full_angle = build_model_data(data, graph_cache, split, scaler)
            full_loaders = make_loaders(full_atom, full_angle, split, args.batch_size)
            common_train = split["split"].eq("train") & ~split["threshold_excluded"]
            common_labels = data.loc[common_train, ["V1_ml", "V2_ml"]]
            reference_scales = {
                target: float(common_labels[f"{target}_ml"].std(ddof=0)) for target in TARGETS
            }

            for protocol in PROTOCOLS:
                protocol_mask = (
                    ~data["threshold_excluded"]
                    if protocol == "legacy_threshold"
                    else pd.Series(True, index=data.index)
                )
                protocol_data = data.loc[protocol_mask].reset_index(drop=True)
                protocol_split = split.loc[protocol_mask].reset_index(drop=True)
                atom_data, angle_data = build_model_data(
                    protocol_data, graph_cache, protocol_split, scaler
                )
                loaders = make_loaders(atom_data, angle_data, protocol_split, args.batch_size)
                train_mask = protocol_split["split"].eq("train").to_numpy()
                labels = protocol_data.loc[train_mask, ["V1_ml", "V2_ml"]]
                target_variance = {
                    target: float(labels[f"{target}_ml"].var(ddof=0)) for target in TARGETS
                }
                run_dir = output_dir / "runs" / run_key / protocol
                result_path = run_dir / "monotonic_softplus.result.json"
                checkpoint_path = run_dir / "checkpoints" / "monotonic_softplus.pt"
                if result_path.exists() and checkpoint_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    print(json.dumps({"starting": run_key, "protocol": protocol}), flush=True)
                    result, _ = run_configuration(
                        config_name="monotonic_softplus",
                        loaders=loaders,
                        canonical_df=protocol_data,
                        run_dir=run_dir,
                        source_checkpoint=source_checkpoint,
                        seed=seed,
                        epochs=2 if args.smoke else args.epochs,
                        patience=2 if args.smoke else args.patience,
                        target_variance=target_variance,
                    )
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                result_row = {
                    key: value for key, value in result.items() if key not in {"valid", "test"}
                }
                result_row.update(
                    {
                        "split_mode": split_mode,
                        "seed": seed,
                        "protocol": protocol,
                        "train_rows": int(train_mask.sum()),
                        "valid_rows": int(protocol_split["split"].eq("valid").sum()),
                        "test_rows": int(protocol_split["split"].eq("test").sum()),
                        "train_tail_rows": int(
                            protocol_split.loc[train_mask, "threshold_excluded"].sum()
                        ),
                    }
                )
                all_training_results.append(result_row)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_trained_model(checkpoint_path, device)
                predictions = prediction_frame(
                    model, full_loaders, data, protocol, split_mode, seed, device
                )
                predictions, factors = fit_and_apply_calibration(predictions, protocol)
                predictions = add_sample_scores(predictions, reference_scales)
                all_predictions.append(predictions)
                for factor in factors:
                    factor.update({"split_mode": split_mode, "seed": seed, "protocol": protocol})
                    all_factors.append(factor)
                print(
                    json.dumps(
                        {
                            "finished": run_key,
                            "protocol": protocol,
                            "best_epoch": result["best_epoch"],
                            "train_tail_rows": result_row["train_tail_rows"],
                        }
                    ),
                    flush=True,
                )

    training = pd.DataFrame(all_training_results)
    predictions = pd.concat(all_predictions, ignore_index=True)
    factors = pd.DataFrame(all_factors)
    metric_rows, diagnostic_rows = [], []
    for keys, context in predictions.groupby(
        ["split_mode", "seed", "protocol", "evaluation_split"], sort=True
    ):
        split_mode, seed, protocol, evaluation_split = keys
        calibration = context[context["evaluation_split"].eq("valid")]
        # The grouped context contains one evaluation split, so retrieve validation
        # from the full predictions table for AUCE factor fitting.
        calibration = predictions[
            predictions["split_mode"].eq(split_mode)
            & predictions["seed"].eq(seed)
            & predictions["protocol"].eq(protocol)
            & predictions["evaluation_split"].eq("valid")
        ]
        if protocol == "legacy_threshold":
            calibration = calibration[~calibration["threshold_excluded"]]
        split_table = pd.read_csv(splits_dir / f"{split_mode}_seed_{seed}.csv")
        common_train = split_table["split"].eq("train") & ~split_table["threshold_excluded"]
        common_labels = data.loc[common_train, ["V1_ml", "V2_ml"]]
        scales = {
            target: float(common_labels[f"{target}_ml"].std(ddof=0)) for target in TARGETS
        }
        for slice_name in SLICES:
            row = {
                "split_mode": split_mode,
                "seed": int(seed),
                "protocol": protocol,
                "evaluation_split": evaluation_split,
                "slice": slice_name,
                **slice_metrics(context, calibration, scales, slice_name),
            }
            metric_rows.append(row)
        for score in ("raw_quantile_width_score", "calibrated_quantile_width_score"):
            diagnostic = high_score_diagnostics(context, score)
            diagnostic.update(
                {
                    "split_mode": split_mode,
                    "seed": int(seed),
                    "protocol": protocol,
                    "evaluation_split": evaluation_split,
                }
            )
            diagnostic_rows.append(diagnostic)

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    effects = paired_effects(metrics)
    decision = validation_decision(metrics, diagnostics)
    training.to_csv(output_dir / "training_comparison.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    factors.to_csv(output_dir / "calibration_factors.csv", index=False)
    metrics.to_csv(output_dir / "slice_metrics.csv", index=False)
    diagnostics.to_csv(output_dir / "high_uncertainty_diagnostics.csv", index=False)
    effects.to_csv(output_dir / "paired_effects.csv", index=False)
    (output_dir / "threshold_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    summary = metrics.groupby(["evaluation_split", "slice", "protocol"], as_index=False).agg(
        rows_mean=("rows", "mean"),
        normalized_rmse_mean=("mean_normalized_rmse", "mean"),
        normalized_mae_mean=("mean_normalized_absolute_error", "mean"),
        V1_coverage_mean=("V1_calibrated_coverage", "mean"),
        V2_coverage_mean=("V2_calibrated_coverage", "mean"),
        calibrated_width_error_spearman_mean=("calibrated_width_error_spearman", "mean"),
        auce_mean=("mean_auce", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    config = {
        "stage": "G0-3_threshold_sensitivity",
        "smoke": args.smoke,
        "data_source": "experiments/e0_8g_transfer/sample_decisions_8g.csv",
        "data_source_sha256": sha256_file(DATA_DIR / "sample_decisions_8g.csv"),
        "rows": len(data),
        "legacy_threshold_rows": int((~data["threshold_excluded"]).sum()),
        "tail_rows": int(data["threshold_excluded"].sum()),
        "tail_compounds": int(data.loc[data["threshold_excluded"], "canonical_smiles"].nunique()),
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "seeds": args.seeds,
        "split_modes": args.split_modes,
        "splits": split_manifest,
        "protocols": list(PROTOCOLS),
        "model": "monotonic_softplus, last2+head, lr=1e-4, equal V1/V2 loss",
        "checkpoint_selection": "protocol-specific validation only",
        "calibration": "protocol-specific validation-only per-target split-conformal inflation",
        "metric_scales": "shared in-threshold train standard deviations within each split/seed",
        "threshold_selection": "validation-only rule recorded in threshold_decision.json",
        "test_role": "final reporting only",
        "epochs": 2 if args.smoke else args.epochs,
        "patience": 2 if args.smoke else args.patience,
        "batch_size": args.batch_size,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    test_summary = summary[summary["evaluation_split"].eq("test")]
    common = test_summary[test_summary["slice"].eq("common")].set_index("protocol")
    tail = test_summary[test_summary["slice"].eq("tail")].set_index("protocol")
    tail_context = metrics[
        metrics["evaluation_split"].eq("test") & metrics["slice"].eq("tail")
    ].pivot(
        index=["split_mode", "seed"],
        columns="protocol",
        values="mean_normalized_absolute_error",
    )
    tail_worse_contexts = int(
        (tail_context["no_threshold"] > tail_context["legacy_threshold"]).sum()
    )
    test_diagnostics = diagnostics[
        diagnostics["evaluation_split"].eq("test")
        & diagnostics["score"].eq("calibrated_quantile_width_score")
    ].groupby("protocol").mean(numeric_only=True)
    readme = f"""# G0-3：Legacy threshold vs no-threshold

## 状态与决定

本实验只改变8g的`V1≤60 mL, V2≤120 mL`过滤。两个协议共享阈值分层的row/compound splits、4g anchor、单调头和训练预算。阈值决定只使用validation；test仅作最终报告。

Validation-only决定：**{decision['selected_threshold_protocol']}**。no-threshold相对legacy的common validation normalized RMSE变化为{decision['common_validation_normalized_rmse_relative_change']:+.1%}，tail validation normalized absolute error变化为{decision['tail_validation_normalized_absolute_error_relative_change']:+.1%}。

独立test不完全复现tail改善：no-threshold的common normalized RMSE仅变化{common.loc['no_threshold','normalized_rmse_mean']/common.loc['legacy_threshold','normalized_rmse_mean']-1:+.1%}，但tail normalized absolute error平均恶化{tail.loc['no_threshold','normalized_mae_mean']/tail.loc['legacy_threshold','normalized_mae_mean']-1:+.1%}，且在{tail_worse_contexts}/6个paired contexts中更差。由于threshold决定已预注册为validation-only，这不用于反向改选；它被登记为小tail test高方差和尾部拟合不稳定的失败模式。

## 独立test汇总

| slice | protocol | normalized RMSE | normalized absolute error | V1/V2 calibrated coverage | width-error Spearman | AUCE |
|---|---|---:|---:|---:|---:|---:|
| common | legacy | {common.loc['legacy_threshold','normalized_rmse_mean']:.3f} | {common.loc['legacy_threshold','normalized_mae_mean']:.3f} | {common.loc['legacy_threshold','V1_coverage_mean']:.3f}/{common.loc['legacy_threshold','V2_coverage_mean']:.3f} | {common.loc['legacy_threshold','calibrated_width_error_spearman_mean']:.3f} | {common.loc['legacy_threshold','auce_mean']:.3f} |
| common | no threshold | {common.loc['no_threshold','normalized_rmse_mean']:.3f} | {common.loc['no_threshold','normalized_mae_mean']:.3f} | {common.loc['no_threshold','V1_coverage_mean']:.3f}/{common.loc['no_threshold','V2_coverage_mean']:.3f} | {common.loc['no_threshold','calibrated_width_error_spearman_mean']:.3f} | {common.loc['no_threshold','auce_mean']:.3f} |
| tail | legacy | {tail.loc['legacy_threshold','normalized_rmse_mean']:.3f} | {tail.loc['legacy_threshold','normalized_mae_mean']:.3f} | {tail.loc['legacy_threshold','V1_coverage_mean']:.3f}/{tail.loc['legacy_threshold','V2_coverage_mean']:.3f} | {tail.loc['legacy_threshold','calibrated_width_error_spearman_mean']:.3f} | {tail.loc['legacy_threshold','auce_mean']:.3f} |
| tail | no threshold | {tail.loc['no_threshold','normalized_rmse_mean']:.3f} | {tail.loc['no_threshold','normalized_mae_mean']:.3f} | {tail.loc['no_threshold','V1_coverage_mean']:.3f}/{tail.loc['no_threshold','V2_coverage_mean']:.3f} | {tail.loc['no_threshold','calibrated_width_error_spearman_mean']:.3f} | {tail.loc['no_threshold','auce_mean']:.3f} |

Tail只占test的约4.3%，但在calibrated width最高10%中，legacy/no-threshold分别富集{test_diagnostics.loc['legacy_threshold','tail_enrichment']:.2f}×/{test_diagnostics.loc['no_threshold','tail_enrichment']:.2f}×，说明legacy阈值确实会预先删除主动学习最可能查询的难例。不要按单个R²解释本实验；`slice_metrics.csv`保留full/common/tail的逐seed结果。

## 产物

- `canonical_8g_no_threshold.csv`与`splits/`：574行完整数据和阈值分层paired splits。
- `training_comparison.csv`：12次配对训练。
- `predictions.csv.gz`、`calibration_factors.csv`：逐样本预测和validation-only校准。
- `slice_metrics.csv`、`paired_effects.csv`：尾部误差、校准、signal-error关系。
- `high_uncertainty_diagnostics.csv`：top-10% tail enrichment与hard-error overlap。
- `threshold_decision.json`：完全基于validation的冻结决定。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
