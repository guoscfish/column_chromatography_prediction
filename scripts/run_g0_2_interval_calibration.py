#!/usr/bin/env python3
"""Run Gate 0-2 validation-only interval calibration for the G0-1 winner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import sha256_file, write_artifact_manifest, write_environment


TARGETS = ("V1", "V2")
LEVELS = np.arange(0.1, 1.0, 0.1)


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    """Split-conformal order statistic ceil((n+1)*coverage), capped at n."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        raise ValueError("Calibration set is empty")
    rank = min(int(np.ceil((len(ordered) + 1) * coverage)), len(ordered))
    return float(ordered[rank - 1])


def normalized_residual_scores(group: pd.DataFrame, target: str) -> np.ndarray:
    y = group[f"{target}_true"].to_numpy(dtype=np.float64)
    lower = group[f"{target}_q10"].to_numpy(dtype=np.float64)
    median = group[f"{target}_q50"].to_numpy(dtype=np.float64)
    upper = group[f"{target}_q90"].to_numpy(dtype=np.float64)
    lower_width = np.maximum(median - lower, 1e-8)
    upper_width = np.maximum(upper - median, 1e-8)
    return np.where(y <= median, (median - y) / lower_width, (y - median) / upper_width)


def scaled_interval(group: pd.DataFrame, target: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lower = group[f"{target}_q10"].to_numpy(dtype=np.float64)
    median = group[f"{target}_q50"].to_numpy(dtype=np.float64)
    upper = group[f"{target}_q90"].to_numpy(dtype=np.float64)
    return median - alpha * (median - lower), median + alpha * (upper - median)


def interval_metrics(group: pd.DataFrame, target: str, alpha: float) -> dict[str, float]:
    y = group[f"{target}_true"].to_numpy(dtype=np.float64)
    lower, upper = scaled_interval(group, target, alpha)
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "crossing_rate": float(np.mean(lower > upper)),
    }


def calibration_curve_metrics(
    validation: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[float, list[dict]]:
    scores = normalized_residual_scores(validation, target)
    rows = []
    for level in LEVELS:
        alpha = finite_sample_quantile(scores, float(level))
        observed = interval_metrics(test, target, alpha)["coverage"]
        rows.append(
            {
                "target": target,
                "nominal_coverage": float(level),
                "alpha": alpha,
                "test_observed_coverage": observed,
                "absolute_calibration_error": abs(observed - float(level)),
            }
        )
    errors = np.asarray([row["absolute_calibration_error"] for row in rows])
    auce = float(np.trapz(errors, LEVELS) / (LEVELS[-1] - LEVELS[0]))
    return auce, rows


def g0_1_gate(comparison: pd.DataFrame) -> dict:
    means = comparison.groupby("config").mean(numeric_only=True)
    legacy = means.loc["legacy_independent"]
    monotonic = means.loc["monotonic_softplus"]
    relative = {
        metric: float(monotonic[metric] / legacy[metric] - 1)
        for metric in ("normalized_valid_score", "valid_V1_rmse", "valid_V2_rmse")
    }
    passed = all(value <= 0.05 for value in relative.values()) and bool(
        comparison.loc[
            comparison["config"].eq("monotonic_softplus"), "valid_quantile_crossing_rate"
        ].max()
        == 0
    )
    return {"passed": passed, "relative_validation_degradation": relative}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g0-1-dir",
        type=Path,
        default=ROOT / "experiments" / "g0_1_quantile_monotonicity",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "g0_2_interval_calibration",
    )
    args = parser.parse_args()
    g0_1_dir = args.g0_1_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (output_dir / "metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite finalized experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    comparison_path = g0_1_dir / "comparison.csv"
    predictions_path = g0_1_dir / "predictions.csv.gz"
    comparison = pd.read_csv(comparison_path)
    gate = g0_1_gate(comparison)
    if not gate["passed"]:
        raise RuntimeError(f"G0-1 did not pass the preregistered validation gate: {gate}")
    predictions = pd.read_csv(predictions_path)
    predictions = predictions[predictions["config"].eq("monotonic_softplus")].copy()

    factor_rows, curve_rows, metric_rows, calibrated_rows = [], [], [], []
    for (split_mode, seed), context in predictions.groupby(["split_mode", "seed"], sort=True):
        validation = context[context["evaluation_split"].eq("valid")].copy()
        test = context[context["evaluation_split"].eq("test")].copy()
        if validation.empty or test.empty:
            raise ValueError(f"Missing validation/test predictions for {split_mode}, seed={seed}")
        calibrated_test = test.copy()
        for target in TARGETS:
            validation_scores = normalized_residual_scores(validation, target)
            raw_alpha = finite_sample_quantile(validation_scores, 0.8)
            alpha_80 = max(1.0, raw_alpha)
            before = interval_metrics(test, target, 1.0)
            after = interval_metrics(test, target, alpha_80)
            auce, context_curve = calibration_curve_metrics(validation, test, target)
            factor_rows.append(
                {
                    "split_mode": split_mode,
                    "seed": int(seed),
                    "target": target,
                    "validation_rows": len(validation),
                    "raw_conformal_alpha_80": raw_alpha,
                    "alpha_80": alpha_80,
                    "alpha_constrained_to_inflation_only": bool(raw_alpha < 1.0),
                }
            )
            for row in context_curve:
                row.update({"split_mode": split_mode, "seed": int(seed)})
                curve_rows.append(row)
            metric_rows.append(
                {
                    "split_mode": split_mode,
                    "seed": int(seed),
                    "target": target,
                    "alpha_80": alpha_80,
                    "test_rows": len(test),
                    "test_coverage_before": before["coverage"],
                    "test_coverage_after": after["coverage"],
                    "test_width_before": before["mean_width"],
                    "test_width_after": after["mean_width"],
                    "test_crossing_before": before["crossing_rate"],
                    "test_crossing_after": after["crossing_rate"],
                    "test_auce_validation_calibrated_curve": auce,
                }
            )
            lower, upper = scaled_interval(test, target, alpha_80)
            calibrated_test[f"{target}_q10_calibrated"] = lower
            calibrated_test[f"{target}_q90_calibrated"] = upper
            calibrated_test[f"{target}_alpha_80"] = alpha_80
        calibrated_rows.append(calibrated_test)

    factors = pd.DataFrame(factor_rows)
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    calibrated = pd.concat(calibrated_rows, ignore_index=True)
    factors.to_csv(output_dir / "calibration_factors.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    curves.to_csv(output_dir / "calibration_curves.csv", index=False)
    calibrated.to_csv(
        output_dir / "test_predictions_calibrated.csv.gz", index=False, compression="gzip"
    )
    summary = metrics.groupby("target", as_index=False).agg(
        alpha_80_mean=("alpha_80", "mean"),
        alpha_80_std=("alpha_80", "std"),
        alpha_80_median=("alpha_80", "median"),
        alpha_80_min=("alpha_80", "min"),
        alpha_80_max=("alpha_80", "max"),
        test_coverage_before_mean=("test_coverage_before", "mean"),
        test_coverage_after_mean=("test_coverage_after", "mean"),
        test_width_before_mean=("test_width_before", "mean"),
        test_width_after_mean=("test_width_after", "mean"),
        test_auce_mean=("test_auce_validation_calibrated_curve", "mean"),
        test_crossing_after_max=("test_crossing_after", "max"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    config = {
        "stage": "G0-2_validation_only_interval_calibration",
        "input_comparison": str(comparison_path.relative_to(ROOT)),
        "input_comparison_sha256": sha256_file(comparison_path),
        "input_predictions": str(predictions_path.relative_to(ROOT)),
        "input_predictions_sha256": sha256_file(predictions_path),
        "selected_parameterization": "monotonic_softplus",
        "g0_1_gate": gate,
        "nominal_coverage": 0.8,
        "alpha_80_rule": "max(1, split-conformal validation order statistic)",
        "test_role": "final reporting only; never used to estimate alpha",
        "auce_levels": LEVELS.tolist(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = {row["target"]: row for row in summary.to_dict(orient="records")}
    readme = f"""# G0-2：Validation-only 区间校准

## 结论

本实验只使用每个 split/seed 的 validation 预测估计 V1/V2 缩放因子，然后将冻结因子原样应用于 test。test 未参与 alpha 选择。输入为 G0-1 通过 Gate 后的 `monotonic_softplus` 模型。

| target | alpha median [min,max] | test coverage before→after | test width before→after | test AUCE | crossing after |
|---|---:|---:|---:|---:|---:|
| V1 | {rows['V1']['alpha_80_median']:.3f} [{rows['V1']['alpha_80_min']:.3f}, {rows['V1']['alpha_80_max']:.3f}] | {rows['V1']['test_coverage_before_mean']:.3f}→{rows['V1']['test_coverage_after_mean']:.3f} | {rows['V1']['test_width_before_mean']:.3f}→{rows['V1']['test_width_after_mean']:.3f} | {rows['V1']['test_auce_mean']:.3f} | {rows['V1']['test_crossing_after_max']:.3f} |
| V2 | {rows['V2']['alpha_80_median']:.3f} [{rows['V2']['alpha_80_min']:.3f}, {rows['V2']['alpha_80_max']:.3f}] | {rows['V2']['test_coverage_before_mean']:.3f}→{rows['V2']['test_coverage_after_mean']:.3f} | {rows['V2']['test_width_before_mean']:.3f}→{rows['V2']['test_width_after_mean']:.3f} | {rows['V2']['test_auce_mean']:.3f} | {rows['V2']['test_crossing_after_max']:.3f} |

`alpha_80` 使用有限样本 split-conformal 顺序统计量，并限制为不小于1（只放大、不收缩）。AUCE 使用 validation 上各 nominal coverage 的 conformal factor 构成校准曲线，再在独立 test 上评价。

校准把平均 test coverage 拉回名义值附近，但 row/seed=1101 因原始区间塌缩，需要 V1/V2 alpha=16.11/55.91；因此不能把跨 seed 的 raw/calibrated quantile width 直接当成稳定 acquisition score。该失败模式必须带入 E1 的 signal-error qualification。

## 产物

- `calibration_factors.csv`：每个 split/seed/target 的 validation-only alpha。
- `metrics.csv`、`summary.csv`：test 最终 coverage、width、AUCE、crossing。
- `calibration_curves.csv`：0.1～0.9 nominal coverage 的独立 test 校准曲线。
- `test_predictions_calibrated.csv.gz`：应用冻结 alpha 后的逐样本 test 区间。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(summary.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
