#!/usr/bin/env python3
"""Score existing formal predictions on permanently role-gated train/validation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT))
from data_boundary import FormalLabelBoundary, QUARANTINED_IDS, SEEDS, sha256_file
from common import metrics, summary


def combined_nrmse(normalized_rmse: list[float]) -> float:
    """Match official point.py: RMS, rather than mean, of two normalized RMSEs."""
    values = np.asarray(normalized_rmse, float)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("combined NRMSE requires two finite nonnegative endpoint RMSEs")
    return float(np.sqrt(np.mean(np.square(values))))


def markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    def value(item):
        if pd.isna(item):
            return "NA"
        if isinstance(item, (float, np.floating)):
            return f"{item:.{decimals}f}"
        return str(item)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def enrich_measures(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["amplitude_ratio"] = result.predicted_abs_mean / result.observed_abs_mean
    result["mae_reduction_vs_zero_response"] = 1 - result.difference_mae / result.observed_abs_mean
    return result


def equal_model_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Average already-computed per-model measures, never pool dependent pairs."""
    measures = ["n_pairs", "n_meaningful", "molecules", "series", "observed_abs_mean", "predicted_abs_mean",
                "difference_mae", "amplitude_error_mean", "direction_accuracy", "correct_nonflat_fraction",
                "wrong_nonflat_fraction", "flat_fraction", "amplitude_ratio", "mae_reduction_vs_zero_response"]
    available = [column for column in measures if column in frame]
    outputs = []
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row.update(n_models=group.row_seed.nunique(), aggregation="equal_model_mean_of_per_model_measures")
        for column in available:
            row[column + "_mean"] = float(group[column].mean())
            row[column + "_std"] = float(group[column].std(ddof=1))
            row[column + "_models_nonmissing"] = int(group[column].notna().sum())
        row["ratio_of_equal_model_amplitude_means"] = float(group.predicted_abs_mean.mean() / group.observed_abs_mean.mean())
        outputs.append(row)
    return pd.DataFrame(outputs)


def score_pairs(pairs: pd.DataFrame, labels: pd.DataFrame, predictions: pd.DataFrame, row_seed: int) -> pd.DataFrame:
    """Require complete legal identity before joining any numerical values."""
    required = {"sample_id", "V1_q50", "V2_q50"}
    if not required <= set(predictions):
        raise ValueError("formal prediction columns missing")
    if predictions.sample_id.duplicated().any() or labels.sample_id.duplicated().any():
        raise ValueError("prediction/label sample IDs must be unique")
    if set(labels.sample_id) & QUARANTINED_IDS or set(predictions.sample_id) & QUARANTINED_IDS:
        raise PermissionError("global quarantine reached the scoring tables")
    legal = pairs[pairs.scoring_allowed].copy()
    if not legal.label_authorized.all() or legal.global_quarantine.any():
        raise PermissionError("pair legality flag inconsistent")
    endpoints = set(legal.low_id) | set(legal.high_id)
    if not endpoints <= set(labels.sample_id) or not endpoints <= set(predictions.sample_id):
        raise ValueError("a legal pair endpoint is missing authorized labels or predictions")
    truth = labels.set_index("sample_id")
    pred = predictions.set_index("sample_id")
    rows = []
    for endpoint in ("V1", "V2"):
        item = legal.copy()
        item["target"] = endpoint
        item["delta_true"] = truth.loc[item.high_id, endpoint + "_ml"].to_numpy(float) - truth.loc[item.low_id, endpoint + "_ml"].to_numpy(float)
        item["delta_pred"] = pred.loc[item.high_id, endpoint + "_q50"].to_numpy(float) - pred.loc[item.low_id, endpoint + "_q50"].to_numpy(float)
        item["series_key"] = item.series_id
        item["molecule"] = item.molecule_id
        item["row_seed"] = int(row_seed)
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def duplicate_noise(metadata_groups: pd.DataFrame, split: pd.DataFrame, labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    roles = dict(zip(split.sample_id, split.source_role))
    values = labels.set_index("sample_id")
    rows = []
    for group in metadata_groups.itertuples():
        allowed = [sid for sid in group.ids.split("|") if roles[sid] in ("train", "validation") and sid not in QUARANTINED_IDS]
        for low, high in itertools.combinations(sorted(allowed), 2):
            if low not in values.index or high not in values.index:
                raise ValueError("duplicate analysis endpoint missing authorized labels")
            exposure = "train_train" if roles[low] == roles[high] == "train" else (
                "validation_validation" if roles[low] == roles[high] == "validation" else "train_validation")
            for target in ("V1", "V2"):
                rows.append(dict(row_seed=seed, duplicate_group_id=group.duplicate_group_id, exposure=exposure,
                    canonical_smiles=group.canonical_smiles, low_id=low, high_id=high, target=target,
                    repeated_input_delta_ml=float(values.loc[high, target + "_ml"] - values.loc[low, target + "_ml"])))
    return pd.DataFrame(rows)


def write_reports(per_model: pd.DataFrame, equal: pd.DataFrame, row_metrics: pd.DataFrame, duplicate_summary: pd.DataFrame) -> None:
    main = per_model[per_model.actual_threshold_ml.eq(.5)]
    measures = ["row_seed", "axis", "target", "n_pairs", "n_meaningful", "molecules", "series", "observed_abs_mean",
                "predicted_abs_mean", "amplitude_ratio", "difference_mae", "mae_reduction_vs_zero_response", "direction_accuracy", "flat_fraction"]
    common = """体积变化以样品体积`V/ul`定义，固定另四项条件与canonical SMILES；PE/EA对固定样品体积及其余条件。V1/V2 q50与真实差均为mL。主方向门限为|Δtrue|>0.5mL；预测平坦为|Δpred|<0.1mL。`difference_mae=mean(|Δpred−Δtrue|)`，零响应基线为`mean(|Δtrue|)`；误差下降比例为1−前者/后者。幅度比为mean(|Δpred|)/mean(|Δtrue|)，不等于逐pair比值均值。0.1与1.0mL门限敏感性另列CSV。

每个模型先单独计算pair均值，三个模型再等权汇总；不把重复端点、多水平系列、共享分子或跨seed重合pair当作独立科研重复，不给伪独立p值。逐系列、逐分子及系列/分子等权补充见CSV。`amplitude_ratio_mean`是逐模型幅度比等权均值；另列`ratio_of_equal_model_amplitude_means`，不能混用。

源模型训练身份仍为3330行。预览事故3ID在所有seed永久隔离；此次实际读取授权train为3327/3328/3328行，validation为416/416/416行。test仅X可访问，官方store只按角色转换授权train/validation请求，未新增投影或使用test目标。初始文本查看例外详见`LABEL_BOUNDARY_INCIDENT.json`，不能把本轮全过程写成从未可见test目标。预测只使用本轮从严格加载的冻结模型取得的无truth预测表，未读历史predictions.csv.gz。
"""
    for scope, path, title in [("train_train", "TRAIN_FIT_RESPONSE.md", "充分训练模型在训练内是否拟合单因素响应"),
                               ("validation_validation", "VALIDATION_RESPONSE.md", "充分训练模型的source-validation响应")]:
        selected = main[main.exposure.eq(scope)]
        eq = equal[equal.exposure.eq(scope) & equal.actual_threshold_ml.eq(.5)]
        eqcols = ["axis", "target", "n_models", "n_pairs_mean", "observed_abs_mean_mean", "predicted_abs_mean_mean",
                  "amplitude_ratio_mean", "difference_mae_mean", "mae_reduction_vs_zero_response_mean", "direction_accuracy_mean", "flat_fraction_mean"]
        if scope == "train_train":
            finding = ("3330条原训练数据并未使三个正式模型稳定拟合体积响应。Row42/525/1101的V1幅度比为5.58%/51.58%/0.71%，"
                       "V2为5.13%/27.31%/0.79%；对应差分MAE相对零响应基线的改善只有2.67%/10.92%/0.40%（V1）和1.34%/1.81%/0.21%（V2）。"
                       "Row525有明显较强的预测变化，但未稳定重现真实差分；另两模型仍强烈压缩。不能选最好的seed称问题已解决，也不能仅凭这三个状态证明架构没有容量。")
        else:
            finding = ("原validation上，Row42/525/1101的V1幅度比为7.04%/37.85%/0.87%，V2为5.94%/35.26%/0.94%。"
                       "V1差分MAE比零响应基线改善3.37%/10.65%/0.36%；V2则为−0.92%/−2.16%/+0.23%，未见稳定真实差分恢复。"
                       "因此充分训练并未带来三seed一致的体积响应泛化；Row525的非平坦变化也不等于准确预测真实变化。")
        finding += (" PE/EA参照的预测幅度接近真实幅度且差分MAE明显低于零响应基线，说明问题具有因素选择性；"
                    "这不是对两个因素物理效应大小的因果比较。方向准确率即使较高，也可能来自小于0.1mL的同号微小预测，必须与平坦比例及差分误差一起看。")
        lines = ["# " + title, "", finding, "", "## 逐模型主结果", "", markdown(selected[measures]), "", "## 三模型等权描述", "", markdown(eq[eqcols]), "", common]
        if scope == "train_train":
            lines.extend(["## 完全相同输入的重复行参考", "", markdown(duplicate_summary), "",
                "同条件不一致既可能是实验变异，也可能来自未记录条件、批次或记录误差；当前数据无法区分。重复对的多个组合相关，量级只作参考，不能把该差值直接当作不可约噪声或证明条件输入无效。严格因素pair不会把同水平重复当作变化，但跨水平重复产生的多个ID组合保留，并在系列等权补充中降低多重性的影响。"])
        else:
            lines.extend(["## 原validation整体误差核验", "", markdown(row_metrics[row_metrics.source_role.eq("validation")]), "",
                "整体combined NRMSE严格沿用官方`sqrt(0.5*((RMSE_V1/scale_V1)^2+(RMSE_V2/scale_V2)^2))`，不是两个normalized RMSE的算术均值。完整416行validation与历史记录的逐指标核验见`formal_validation_metric_checks.csv`（容差1e-6）；R1公式修复审计保存在`formal_row_metric_R1_fix.json`。原validation用于best checkpoint选择，因此这里是原冻结validation的条件泛化描述，不是新独立test；不同Row seed的validation组成也不同。V strict pairs仅44–48对、35–39个分子、39–42个系列，足以作本域描述，但仍不支持把所有pair视为独立重复。没有放宽条件、借test凑样本或按结果筛选分子。",
                "## 部分已见的混合对（单列）", "", markdown(main[main.exposure.eq("train_validation")][measures]), "",
                "混合对一个端点曾用于训练，不能当作两端未见泛化；不并入上面的validation结果。"])
        lines.extend(["", "复现：`/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/predictor/full_training_volume_response_audit_20261117/analyze_formal.py`。需要先运行coverage与正式预测脚本；全部label receipts、scoring子集、预测/源码SHA写入`formal_response_inputs.json`与`formal_label_access_receipts.json`。0新训练、0LLM/API。"])
        (OUT / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    incident = json.loads((OUT / "LABEL_BOUNDARY_INCIDENT.json").read_text())
    if set(incident["quarantined_ids"]) != QUARANTINED_IDS:
        raise ValueError("quarantine incident and executable gate differ")
    pairs = pd.read_csv(OUT / "formal_pairs.csv", dtype={"low_id": str, "high_id": str})
    all_roles = pd.read_csv(OUT / "formal_split_roles.csv", dtype={"sample_id": str})
    duplicate_groups = pd.read_csv(OUT / "covariate_duplicate_groups.csv")
    identities = {int(row["seed"]): row for row in json.loads((OUT / "checkpoint_audit_manifest.json").read_text())["identities"]}
    prediction_paths = [OUT / f"formal_predictions_seed_{seed}.csv" for seed in SEEDS]
    if any(not path.is_file() for path in prediction_paths):
        raise FileNotFoundError("formal prediction tables must all exist before any target projection")
    prediction_manifest = json.loads((OUT / "FORMAL_PREDICTION_MANIFEST.json").read_text())
    if prediction_manifest["status"] != "PASS" or set(prediction_manifest["quarantined_ids"]) != QUARANTINED_IDS:
        raise ValueError("formal prediction manifest or quarantine mismatch")
    prediction_outputs = {entry["path"]: entry for entry in prediction_manifest["outputs"]}
    for path in prediction_paths:
        expected = prediction_outputs[str(path.relative_to(ROOT))]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError("formal prediction file SHA differs from the inference manifest")
    inputs = [OUT / name for name in ["LABEL_BOUNDARY_INCIDENT.json", "formal_pairs.csv", "formal_split_roles.csv",
              "covariate_duplicate_groups.csv", "checkpoint_audit_manifest.json", "data_boundary.py", "common.py", "ANALYSIS_PLAN.json",
              "data_boundary_inputs.json", "FORMAL_PREDICTION_MANIFEST.json"]]
    inputs.extend(prediction_paths)
    inputs.append(Path(__file__))
    historical_metric_paths = [ROOT / f"studies/predictor/final_4g_qualification/results/row/seed_{seed}/metrics.json"
                               for seed in SEEDS]
    inputs.extend(historical_metric_paths)
    inputs.append(ROOT / "src/qgeognn_al/evaluation/point.py")
    input_records = [dict(path=str(path.relative_to(ROOT)), sha256=sha256_file(path), bytes=path.stat().st_size)
                     for path in inputs]
    plan = json.loads((OUT / "ANALYSIS_PLAN.json").read_text())
    if plan["main_direction_threshold_ml"] != .5 or plan["prediction_flat_threshold_ml"] != .1:
        raise ValueError("frozen response definitions changed")
    results, duplicate_results, row_results, receipts = [], [], [], []
    for seed, prediction_path in zip(SEEDS, prediction_paths):
        boundary = FormalLabelBoundary(seed)
        split = all_roles[all_roles.row_seed.eq(seed)]
        if boundary.split.sample_id.tolist() != split.sample_id.tolist() or boundary.split.source_role.tolist() != split.source_role.tolist():
            raise ValueError("derived split role table differs from source identity")
        expected_ids = set(split.loc[split.label_authorized, "sample_id"])
        # The new prediction tables contain no labels. Only these columns enter scoring.
        predictions = pd.read_csv(prediction_path, usecols=["sample_id", "V1_q50", "V2_q50"], dtype={"sample_id": str})
        if set(predictions.sample_id) != expected_ids or predictions.sample_id.duplicated().any():
            raise ValueError("formal prediction IDs differ from exact legal train+validation subset")
        if not np.isfinite(predictions[["V1_q50", "V2_q50"]].to_numpy(float)).all():
            raise ValueError("nonfinite predictions")
        authorized = []
        for role in ("train", "validation"):
            ids = split.loc[split.source_role.eq(role) & split.label_authorized, "sample_id"].tolist()
            authorized.append(boundary.reveal(ids, role))
        labels = pd.concat(authorized, ignore_index=True)
        receipts.extend(boundary.audit)
        scores = score_pairs(pairs[pairs.row_seed.eq(seed)], labels, predictions, seed)
        results.append(scores)
        duplicate_results.append(duplicate_noise(duplicate_groups[duplicate_groups.row_seed.eq(seed)], split, labels, seed))
        joined = labels.merge(predictions, on="sample_id", validate="one_to_one")
        for role, group in joined.groupby("source_role"):
            normalized = []
            for target in ("V1", "V2"):
                y = group[target + "_ml"].to_numpy(float)
                p = group[target + "_q50"].to_numpy(float)
                rmse = float(np.sqrt(np.mean((p-y)**2)))
                scale = float(identities[seed]["preprocessing"]["target_scales"][target])
                normalized.append(rmse / scale)
                row_results.append(dict(row_seed=seed, source_role=role, target=target, n_rows=len(group),
                    rmse_ml=rmse, mae_ml=float(np.abs(p-y).mean()), target_scale_ml=scale,
                    normalized_rmse=rmse / scale,
                    r2=1-float(np.sum((p-y)**2)/np.sum((y-y.mean())**2))))
            row_results[-1]["combined_normalized_rmse"] = combined_nrmse(normalized)
            row_results[-2]["combined_normalized_rmse"] = combined_nrmse(normalized)
    response = pd.concat(results, ignore_index=True)
    per_model = enrich_measures(summary(response, ["row_seed", "exposure", "axis", "target"]))
    equal = equal_model_summary(per_model, ["exposure", "axis", "target", "actual_threshold_ml"])
    per_series = enrich_measures(summary(response, ["row_seed", "exposure", "axis", "target", "series_key", "molecule"]))
    per_molecule = enrich_measures(summary(response, ["row_seed", "exposure", "axis", "target", "molecule"]))
    for unit, frame in [("series", per_series), ("molecule", per_molecule)]:
        numeric = ["observed_abs_mean", "predicted_abs_mean", "difference_mae", "amplitude_error_mean",
                   "direction_accuracy", "correct_nonflat_fraction", "wrong_nonflat_fraction", "flat_fraction"]
        groups = ["row_seed", "exposure", "axis", "target", "actual_threshold_ml"]
        aggregate = frame.groupby(groups)[numeric].mean().reset_index()
        aggregate["n_units"] = frame.groupby(groups).size().to_numpy()
        aggregate = enrich_measures(aggregate)
        aggregate.to_csv(OUT / f"formal_response_{unit}_equal_per_model.csv", index=False)
    response.to_csv(OUT / "formal_response_pairs.csv", index=False)
    per_model.to_csv(OUT / "formal_response_per_model.csv", index=False)
    equal.to_csv(OUT / "formal_response_equal_models.csv", index=False)
    per_series.to_csv(OUT / "formal_response_per_series.csv", index=False)
    per_molecule.to_csv(OUT / "formal_response_per_molecule.csv", index=False)
    row_frame = pd.DataFrame(row_results)
    row_frame.to_csv(OUT / "formal_row_metrics.csv", index=False)
    validation_checks, fixes = [], []
    for seed, historical_path in zip(SEEDS, historical_metric_paths):
        # This archive contains aggregate metrics, not row targets or predictions.
        recorded = json.loads(historical_path.read_text())["validation"]
        current = row_frame[row_frame.row_seed.eq(seed) & row_frame.source_role.eq("validation")]
        if set(current.n_rows) != {416}:
            raise ValueError("validation must retain all 416 original authorized rows")
        for item in current.itertuples():
            for metric, column in [("rmse", "rmse_ml"), ("mae", "mae_ml"), ("r2", "r2")]:
                previous = float(recorded[item.target + "_" + metric])
                computed = float(getattr(item, column))
                validation_checks.append(dict(row_seed=seed, target=item.target, metric=metric,
                    recorded=previous, recomputed=computed, absolute_difference=abs(computed-previous),
                    tolerance=1e-6, passed=abs(computed-previous) <= 1e-6))
        combined = float(current.combined_normalized_rmse.iloc[0])
        prior = float(recorded["combined_normalized_rmse"])
        validation_checks.append(dict(row_seed=seed, target="combined", metric="combined_normalized_rmse",
            recorded=prior, recomputed=combined, absolute_difference=abs(combined-prior),
            tolerance=1e-6, passed=abs(combined-prior) <= 1e-6))
        fixes.append(dict(row_seed=seed, source_role="validation",
            erroneous_arithmetic_mean=float(current.normalized_rmse.mean()), corrected_official_RMS=combined,
            historical_official_value=prior, absolute_difference_from_historical=abs(combined-prior)))
    if not all(check["passed"] for check in validation_checks):
        raise ValueError("validation metrics differ from the historical frozen model beyond 1e-6")
    pd.DataFrame(validation_checks).to_csv(OUT / "formal_validation_metric_checks.csv", index=False)
    (OUT / "formal_row_metric_R1_fix.json").write_text(json.dumps(dict(
        problem="Initial new combined NRMSE used arithmetic mean of normalized endpoint RMSEs.",
        corrected_formula="sqrt(0.5*sum((rmse_target/source_train_scale_target)**2))",
        source="src/qgeognn_al/evaluation/point.py", new_inference_calls_for_fix=0,
        local_pair_response_metrics_changed=False, complete_validation_rows=416, tolerance=1e-6, checks=fixes), indent=2) + "\n")
    noise = pd.concat(duplicate_results, ignore_index=True)
    noise.to_csv(OUT / "duplicate_replicate_differences.csv", index=False)
    noise_summary = noise.assign(absolute_difference_ml=noise.repeated_input_delta_ml.abs()).groupby(
        ["row_seed", "exposure", "target"]).agg(n_replicate_pairs=("repeated_input_delta_ml", "size"),
        n_groups=("duplicate_group_id", "nunique"), mean_abs_difference_ml=("absolute_difference_ml", "mean"),
        median_abs_difference_ml=("absolute_difference_ml", "median"), max_abs_difference_ml=("absolute_difference_ml", "max")).reset_index()
    noise_summary.to_csv(OUT / "duplicate_noise_summary.csv", index=False)
    (OUT / "formal_label_access_receipts.json").write_text(json.dumps(dict(receipts=receipts,
        labels_projected=sum(receipt["requested_rows"] for receipt in receipts), test_labels_projected=0,
        quarantined_labels_projected=0, label_adapter="Frozen source-train to l0; source-validation unchanged; test permanently blocked"), indent=2) + "\n")
    for entry in input_records:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("an analysis input changed during the authorized evaluation")
    manifest = dict(status="PASS_WITH_DISCLOSED_TEXT_INSPECTION_INCIDENT", inputs=input_records,
        original_training_rows_per_model=3330, scoring_train_rows={seed: int(all_roles[all_roles.row_seed.eq(seed) & all_roles.source_role.eq("train")].label_authorized.sum()) for seed in SEEDS},
        scoring_validation_rows_per_model=416, test_labels_projected=0, quarantined_ids=sorted(QUARANTINED_IDS),
        numeric_label_requests=receipts, source_csv_sha256_verified_before_each_boundary=True,
        pair_endpoint_scoring_identity_complete=True, model_fits=0, model_forward_calls_in_this_script=0,
        llm_api_calls=0, original_predictions_with_truth_opened=False,
        per_model_then_equal_model_aggregation=True, paired_scientific_independence_claim=False)
    manifest["validation_historical_metric_check"] = dict(n_checks=len(validation_checks), tolerance=1e-6,
        all_passed=True, max_absolute_difference=max(check["absolute_difference"] for check in validation_checks),
        formula="RMS of two source-scale-normalized endpoint RMSEs; official point.py")
    (OUT / "formal_response_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_reports(per_model, equal, row_frame, noise_summary)
    print(json.dumps(dict(status=manifest["status"], scored_response_rows=len(response),
        label_request_count=len(receipts), test_numeric_projection=0, labels_projected=sum(x["requested_rows"] for x in receipts))))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
