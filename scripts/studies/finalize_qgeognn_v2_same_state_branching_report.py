#!/usr/bin/env python3
"""Build the human-readable report from frozen same-state study outputs.

This is deliberately separate from the sealed experiment implementation.  It
reads only the already materialized CSV/JSON outputs and never opens the source
dataset, a checkpoint, a prediction file, or a branch runtime label store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/qgeognn_v2_same_state_branching"
RESULTS = STUDY / "results"
KEYS = ["seed", "source_trajectory", "anchor_budget"]
STRATEGIES = ["gradient_lcmd", "kernel_ivr", "gradient_maxdet"]


def _display(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.2f}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame without pandas' optional tabulate dependency."""

    headers = [str(column) for column in frame.columns]
    rows = [[_display(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(RESULTS / name, index=False)


def _winner_change_details(winners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_rows = []
    for (seed, budget), rows in winners.groupby(["seed", "anchor_budget"], sort=True):
        mapping = rows.set_index("source_trajectory")["winner_short_AULC"].to_dict()
        source_rows.append({"seed": seed, "anchor_budget": budget,
                            "gradient_lcmd_source_winner": mapping["gradient_lcmd"],
                            "kernel_ivr_source_winner": mapping["kernel_ivr"],
                            "winner_changed": len(set(mapping.values())) > 1})
    budget_rows = []
    for (seed, source), rows in winners.groupby(["seed", "source_trajectory"], sort=True):
        mapping = rows.set_index("anchor_budget")["winner_short_AULC"].to_dict()
        budget_rows.append({"seed": seed, "source_trajectory": source,
                            "winner_at_429": mapping[429], "winner_at_653": mapping[653],
                            "winner_changed": len(set(mapping.values())) > 1})
    seed_rows = []
    for (source, budget), rows in winners.groupby(["source_trajectory", "anchor_budget"], sort=True):
        mapping = rows.set_index("seed")["winner_short_AULC"].to_dict()
        seed_rows.append({"source_trajectory": source, "anchor_budget": budget,
                          "winner_seed_157": mapping[157], "winner_seed_6101": mapping[6101],
                          "winner_changed": len(set(mapping.values())) > 1})
    return pd.DataFrame(source_rows), pd.DataFrame(budget_rows), pd.DataFrame(seed_rows)


def _preference_correlations(branches: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    wide = branches.pivot(index=KEYS, columns="strategy", values="short_AULC").reset_index()
    merged = diagnostics.merge(wide, on=KEYS, validate="one_to_one")
    features = [
        "recent_validation_nrmse",
        "recent_validation_last_step_delta",
        "recent_validation_2step_linear_slope_per_label",
        "recent_validation_3step_linear_slope_per_label",
        "gradient_norm_mean",
        "gradient_norm_std",
        "gradient_norm_p90",
        "gradient_norm_p95",
        "gradient_effective_rank_participation_ratio",
        "coverage_nearest_distance_mean",
        "coverage_nearest_distance_p90",
        "ivr_predicted_relative_reduction_B32",
        "maxdet_total_marginal_logdet_gain_B32",
    ]
    comparisons = [
        ("LCMD_minus_IVR_AULC", "gradient_lcmd", "kernel_ivr"),
        ("LCMD_minus_MaxDet_AULC", "gradient_lcmd", "gradient_maxdet"),
        ("IVR_minus_MaxDet_AULC", "kernel_ivr", "gradient_maxdet"),
    ]
    rows = []
    for outcome, left, right in comparisons:
        margin = merged[left] - merged[right]
        for feature in features:
            rho = float(merged[feature].corr(margin, method="spearman"))
            rows.append({"feature": feature, "outcome": outcome, "n_anchors": len(merged),
                         "spearman_rho": rho, "abs_spearman_rho": abs(rho),
                         "p_value": None, "interpretation": "exploratory_descriptive_only"})
    return pd.DataFrame(rows).sort_values(["outcome", "abs_spearman_rho"], ascending=[True, False])


def _top_correlations(correlations: pd.DataFrame, per_outcome: int = 3) -> pd.DataFrame:
    return (correlations.sort_values(["outcome", "abs_spearman_rho"], ascending=[True, False])
            .groupby("outcome", sort=False).head(per_outcome)
            [["outcome", "feature", "spearman_rho", "n_anchors"]])


def _manifest(paths: Iterable[Path]) -> dict[str, object]:
    files = {}
    for path in sorted(paths):
        files[str(path.relative_to(ROOT))] = _sha256(path)
    return {"kind": "postfreeze_reporting_manifest", "source_data_opened": False,
            "sealed_experiment_code_modified": False, "files": files}


def main() -> None:
    decision = json.loads((STUDY / "decision.json").read_text())
    freeze = json.loads((STUDY / "global_pre_test_freeze.json").read_text())
    branches = pd.read_csv(RESULTS / "branch_results.csv")
    winners = pd.read_csv(RESULTS / "winner_matrix.csv")
    overlaps = pd.read_csv(RESULTS / "selection_overlap.csv")
    diagnostics_all = pd.read_csv(RESULTS / "state_diagnostics.csv")
    diagnostics = diagnostics_all.loc[diagnostics_all.state_kind.eq("anchor")].copy()
    paired = pd.read_csv(RESULTS / "paired_comparison.csv")
    lineage = pd.read_csv(STUDY / "anchor_lineage_audit.csv")
    validation = pd.read_csv(RESULTS / "validation_metrics.csv")
    access = pd.read_csv(RESULTS / "test_label_access_audit.csv")

    mean_results = (branches.groupby(["source_trajectory", "anchor_budget", "strategy"], sort=True)
                    [["NRMSE_anchor", "NRMSE_plus32", "NRMSE_plus64", "short_AULC",
                      "delta_NRMSE_32", "delta_NRMSE_64"]].mean().reset_index())
    winner_counts = pd.concat([
        winners[column].value_counts().rename_axis("strategy").reset_index(name="count")
        .assign(endpoint=column.replace("winner_", ""))
        for column in ("winner_plus32", "winner_plus64", "winner_short_AULC")
    ], ignore_index=True)[["endpoint", "strategy", "count"]]
    overlap_summary = (overlaps.groupby(["left_strategy", "right_strategy"], sort=True)
                       .agg(intersection_mean=("intersection_count", "mean"),
                            intersection_min=("intersection_count", "min"),
                            intersection_max=("intersection_count", "max"),
                            jaccard_mean=("jaccard", "mean"),
                            jaccard_min=("jaccard", "min"),
                            jaccard_max=("jaccard", "max")).reset_index())
    source_changes, budget_changes, seed_changes = _winner_change_details(winners)
    preference_correlations = _preference_correlations(branches, diagnostics)
    _write_csv(mean_results, "mean_branch_results.csv")
    _write_csv(winner_counts, "winner_counts.csv")
    _write_csv(overlap_summary, "selection_overlap_summary.csv")
    _write_csv(source_changes, "source_trajectory_winner_changes.csv")
    _write_csv(budget_changes, "budget_winner_changes.csv")
    _write_csv(seed_changes, "seed_winner_changes.csv")
    _write_csv(preference_correlations, "strategy_preference_correlations.csv")

    anchor_ranges = winners.within_anchor_AULC_range
    plus32_to_plus64_changes = int((winners.winner_plus32 != winners.winner_plus64).sum())
    total_hours = decision["total_execution_seconds"] / 3600
    training_seconds = float(validation.training_seconds.sum())
    training_epochs = int(validation.epochs_run.sum())
    exact_matches = int(winners.validation_test_ranking_exact_match.sum())

    lineage_view = lineage[["seed", "source_trajectory", "anchor_budget", "source_round", "source_study",
                            "checkpoint_sha256", "ordered_L_hash", "ordered_U_hash", "gradient_bank_hash",
                            "test_truth_access_count"]].copy()
    for column in ("checkpoint_sha256", "ordered_L_hash", "ordered_U_hash", "gradient_bank_hash"):
        lineage_view[column] = lineage_view[column].str.slice(0, 12)

    state_history = diagnostics[["seed", "source_trajectory", "anchor_budget", "active_label_count",
                                 "candidate_pool_size", "recent_validation_nrmse",
                                 "recent_validation_last_step_delta",
                                 "recent_validation_2step_linear_slope_per_label",
                                 "recent_validation_3step_linear_slope_per_label"]]
    state_gradient = diagnostics[["seed", "source_trajectory", "anchor_budget", "gradient_norm_mean",
                                  "gradient_norm_std", "gradient_norm_p50", "gradient_norm_p90",
                                  "gradient_norm_p95", "gradient_norm_max",
                                  "gradient_effective_rank_participation_ratio",
                                  "coverage_nearest_distance_mean", "coverage_nearest_distance_median",
                                  "coverage_nearest_distance_p90", "coverage_nearest_distance_max"]]
    state_objectives = diagnostics[["seed", "source_trajectory", "anchor_budget",
                                    "ivr_integrated_variance_before",
                                    "ivr_predicted_total_reduction_B32",
                                    "ivr_predicted_relative_reduction_B32",
                                    "maxdet_total_marginal_logdet_gain_B32",
                                    "lcmd_largest_cluster_score_first"]]
    branch_view = branches[["seed", "source_trajectory", "anchor_budget", "strategy", "NRMSE_anchor",
                            "NRMSE_plus32", "NRMSE_plus64", "short_AULC", "delta_NRMSE_32",
                            "delta_NRMSE_64", "V1_RMSE_plus64", "V1_R2_plus64",
                            "V2_RMSE_plus64", "V2_R2_plus64"]]
    winner_view = winners[["seed", "source_trajectory", "anchor_budget", "winner_plus32", "winner_plus64",
                           "winner_short_AULC", "test_AULC_ranking", "validation_plus64_ranking",
                           "validation_test_ranking_exact_match", "within_anchor_AULC_range"]]
    overlap_view = overlaps[["seed", "source_trajectory", "anchor_budget", "left_strategy", "right_strategy",
                             "intersection_count", "intersection_fraction_of_32", "jaccard"]]
    paired_view = paired[["seed", "source_trajectory", "anchor_budget", "left_strategy", "right_strategy",
                          "plus32_left_minus_right", "plus64_left_minus_right", "short_AULC_left_minus_right"]]

    report = f"""# Same-state short-rollout branching：最终开发研究报告

## 结论摘要

本研究在 8 个严格复用的 anchor state 上完成了 24 个分支、48 次新的 scratch fit。所有 acquisition、fit 与 test-X prediction 均先冻结，随后才进行一次统一 test truth reveal。证据等级是 **development / mechanism**，不是独立确认。

核心结论是：从完全相同的 `(L_t, U_t, checkpoint)` 出发，三种 acquisition mechanism 的确会产生不同的后续收益，但 winner 不是 label budget 的稳定函数。8 个 anchor 内 short-AULC 的策略跨度为 {anchor_ranges.min():.6f}–{anchor_ranges.max():.6f}（均值 {anchor_ranges.mean():.6f}）；第一批到第二批的 winner 在 {plus32_to_plus64_changes}/8 个 anchor 中改变，说明 one-step 比较会误判相当一部分短 rollout。

- short-AULC winner：Kernel-IVR 4/8，Gradient-LCMD 4/8，Gradient-MaxDet 0/8。
- budget 从 429 变到 653 时，winner 仅在 2/4 个 seed-source 配对中改变，且方向不一致；不支持一个稳健的固定 stage switch。
- 同一 budget 改变 source trajectory 时，winner 在 2/4 个 seed-budget 配对中改变。两个变化都发生在 budget 429：Gradient-LCMD 来源 state 偏向 IVR，而 Kernel-IVR 来源 state 偏向 LCMD，呈现 trajectory/state crossover。
- 改变 seed 时，winner 在 2/4 个 source-budget 配对中改变；两个变化都发生在 budget 653。两个开发 seed 不足以学习 controller。
- validation +64 与 test +64 的完整三策略排序只在 {exact_matches}/8 个 anchor 完全一致。该 validation 已参与 checkpoint selection，只能作为开发期 state observable，不能当成独立泛化证据。

因此，更值得保留的工作假设是 `a_t = pi(s_t)`，而不是 `a_t = pi(n_t)`；但当前数据只支持继续验证这个假设，不支持训练或部署 adaptive selector。

## 执行与泄漏控制

| item | value |
| --- | --- |
| global freeze status | {freeze['status']} |
| anchor states | 8 |
| branches | {decision['branch_count']} |
| actual new fits | {decision['actual_new_fits']} |
| summed branch wall time | {decision['total_execution_seconds']:.3f} s ({total_hours:.3f} h) |
| summed model-training time | {training_seconds:.3f} s |
| summed epochs run | {training_epochs} |
| test accesses before global freeze | {freeze['test_truth_access_count']} |
| unified post-freeze test accesses | {len(access)} (one per seed) |
| controller trained | {decision['controller_trained']} |

`test_label_access_audit.csv` 的两次访问均显示 `acquisitions_frozen=True` 且 `predictions_frozen=True`。历史 source artifacts 保持只读；封印验证负责检查其哈希及所有递归 branch prediction 哈希。

## Anchor lineage audit

429 与 653 分别从历史 source trajectory 的 round 3 与 round 10 复用，没有重训 anchor。下表哈希显示前 12 位；完整值见 `anchor_lineage_audit.csv`。

{_markdown(lineage_view)}

## State diagnostics（test-blind）

### 历史与 validation observable

{_markdown(state_history)}

### Gradient geometry 与 coverage

Effective rank 使用 participation ratio `(sum lambda)^2 / sum(lambda^2)`。

{_markdown(state_gradient)}

### 三种 acquisition objective

{_markdown(state_objectives)}

三种 selector 使用同一当前 full-network q50、512D CountSketch gradient bank。历史方法定义中的归一化差异被保留：LCMD 使用 raw squared-Euclidean geometry；IVR 使用 all-R RMS normalization 和既有 prior/noise；MaxDet 使用 current-L RMS normalization 与 D-opt/logdet objective。它们是机制定义的一部分，而不是未受控的 representation 差异。

## Pairwise branch results 与 short AULC

`delta_NRMSE` 为 anchor NRMSE 减去未来 NRMSE，越大越好；NRMSE 与 AULC 越小越好。

{_markdown(branch_view)}

### 两 seed 平均

{_markdown(mean_results)}

### 成对差值

差值定义为 left 减 right；负值表示 left 更好。

{_markdown(paired_view)}

## Strategy winner matrix

`validation_test_ranking_exact_match` 比较 validation +64 与 test +64 的完整排序；`test_AULC_ranking` 单独给出 short-AULC 排序。

{_markdown(winner_view)}

### Winner counts

{_markdown(winner_counts)}

## Ranking 是否随 state 改变

### Source trajectory effect

{_markdown(source_changes)}

### Budget effect

{_markdown(budget_changes)}

### Seed effect

{_markdown(seed_changes)}

stage、trajectory 与 seed 各自都造成 2/4 个 matched comparison 的 winner 改变。关键区别是变化模式：source effect 集中在 429，而 seed effect 集中在 653；budget effect 又只出现在两个不同的 seed/source 组合且方向不统一。这不符合一个简单、可复现的 label-count switch 法则。

## 第一批 selection overlap

{_markdown(overlap_view)}

### 汇总

{_markdown(overlap_summary)}

LCMD–IVR 的平均交集为 {overlap_summary.loc[(overlap_summary.left_strategy == 'gradient_lcmd') & (overlap_summary.right_strategy == 'kernel_ivr'), 'intersection_mean'].iloc[0]:.3f}/32，LCMD–MaxDet 为 {overlap_summary.loc[(overlap_summary.left_strategy == 'gradient_lcmd') & (overlap_summary.right_strategy == 'gradient_maxdet'), 'intersection_mean'].iloc[0]:.3f}/32；IVR–MaxDet 较高，为 {overlap_summary.loc[(overlap_summary.left_strategy == 'kernel_ivr') & (overlap_summary.right_strategy == 'gradient_maxdet'), 'intersection_mean'].iloc[0]:.3f}/32，但仍远非同一批次。这既验证了 branch 确实分叉，也表明 observed performance differences 对应实际 acquisition differences。

## 哪些 test-blind feature 可能预测 preference

为避免 anchor 难度支配相关性，这里相关的是**策略间 short-AULC margin**，而不是各策略的绝对 AULC。下表每个成对 margin 仅列绝对 Spearman rho 最大的三个 feature；`left_minus_right < 0` 表示 left 更好。

{_markdown(_top_correlations(preference_correlations))}

最值得带入下一次预注册验证的是：

1. gradient dispersion（尤其 `gradient_norm_std` / `p95`），对 LCMD–IVR preference 的描述性关联最大；
2. coverage tail（尤其 `coverage_nearest_distance_p90`，其次 mean），对 IVR–MaxDet preference 的描述性关联最大；
3. recent validation slope 可作为候选补充，但 validation reuse 使它的解释更弱。

这些都是 `n=8`、多重探索性比较、无 p-value 的线索。不能据此设 threshold、训练 controller，或声称有统计显著性。完整结果见 `results/strategy_preference_correlations.csv`。

## 对七个科学问题的直接回答

1. **同一 state 下是否有不同未来收益？** 有。策略选择批次明显不同，short-AULC 跨策略差异在每个 anchor 都非零，最大达到 {anchor_ranges.max():.6f}；但差异大小随 state 变化。
2. **ranking 是否随 budget 改变？** 部分改变（2/4），但不一致，不能归纳成固定 stage law。
3. **同 budget 是否随 source trajectory 改变？** 会（2/4），且 429 上两个 seed 都发生 crossover，这是 state/history effect 的直接描述性证据。
4. **seed dependence 多强？** 2/4 matched comparisons 改变 winner；对只有两个 development seeds 的研究而言，这足以阻止 controller 训练或强泛化。
5. **fixed stage switch 是否成立？** 不成立为稳健规则。数据不支持继续围绕固定 653 switch 调参。
6. **是否更应写成 `a_t = pi(s_t)`？** 是，作为下一步待确认的机制假设；不是已经学到的 policy。
7. **最可能的 state features？** gradient dispersion、coverage tail/mean，以及较弱的 recent validation slope；现阶段只应预注册后验证。

## Adaptive strategy 是否值得继续、下一步最小实验

值得继续做一次**确认性验证**，不值得现在实现 RL、contextual bandit、神经 policy 或自动阈值搜索。最小下一步是：

- 在未用于这些开发决策的新 seeds，最好是 held-out 时间/实验 cohort 上，预注册同一套 3 个冻结 selector；
- 保留两个 anchor stages 和 two-round rollout，但预先固定主终点为 short-AULC、次终点为 +64 NRMSE；
- 只预注册 `gradient_norm_std` 与 `coverage_nearest_distance_p90` 两个 test-blind diagnostics（validation slope 仅作附录）；
- 先检验 ranking/state interaction 能否复现，再决定是否值得收集足够 anchor states 来训练简单、可解释的 policy。

Gradient-MaxDet 在本研究中没有赢得任何 short-AULC anchor，只在 1/8 的 +64 endpoint 获胜；下一次仍可作为冻结机制 comparator，但没有证据把它作为 adaptive policy 的优先动作。

## Artifact index

- 原始 per-seed test metrics：`results/test_metrics.csv`
- 24 条 branch summary：`results/branch_results.csv`
- 24 条 pairwise comparison：`results/paired_comparison.csv`
- 8 条 winner matrix：`results/winner_matrix.csv`
- 32 条 state diagnostics（8 anchor + 24 branch-after-step-1）：`results/state_diagnostics.csv`
- 24 条 first-batch overlap：`results/selection_overlap.csv`
- 48 条 validation fit metrics：`results/validation_metrics.csv`
- test access audit：`results/test_label_access_audit.csv`
- postfreeze 派生表与哈希清单：`results/mean_branch_results.csv`、`results/winner_counts.csv`、`results/strategy_preference_correlations.csv`、`postfreeze_artifact_manifest.json`

报告修复只改变冻结后呈现层；详见 `REPORTING_NOTE.md`。
"""
    (STUDY / "FINAL_REPORT.md").write_text(report)
    (STUDY / "REPORTING_NOTE.md").write_text(
        "# Post-freeze reporting note\n\n"
        "The sealed experiment completed, froze all predictions, revealed test truth once, and wrote its "
        "canonical CSV/JSON outputs successfully. Its first Markdown rendering used pandas `to_markdown`, "
        "but the optional `tabulate` dependency was unavailable and a temporary compatibility shim rendered "
        "DataFrame column names character-by-character.\n\n"
        "`scripts/studies/finalize_qgeognn_v2_same_state_branching_report.py` repairs only the presentation "
        "from frozen CSV/JSON outputs. It does not open source data, checkpoints, predictions, runtime label "
        "stores, or test labels, and it does not modify sealed experiment code or numerical results.\n"
    )

    manifest_inputs = [
        STUDY / "FINAL_REPORT.md", STUDY / "REPORTING_NOTE.md", STUDY / "PROTOCOL.md",
        STUDY / "README.md", STUDY / "decision.json", STUDY / "seal.json",
        STUDY / "global_pre_test_freeze.json", STUDY / "anchor_lineage_audit.csv",
        ROOT / "scripts/studies/run_qgeognn_v2_same_state_branching.py",
        ROOT / "scripts/studies/finalize_qgeognn_v2_same_state_branching_report.py",
        ROOT / "src/qgeognn_al/active_learning_v2/same_state_branching.py",
        ROOT / "tests/active_learning_v2/test_same_state_branching.py",
    ] + sorted(RESULTS.glob("*.csv")) + sorted((STUDY / "lineage").glob("*.json"))
    (STUDY / "postfreeze_artifact_manifest.json").write_text(
        json.dumps(_manifest(manifest_inputs), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "REPORT_FINALIZED_FROM_FROZEN_OUTPUTS",
                      "report": str((STUDY / 'FINAL_REPORT.md').relative_to(ROOT)),
                      "source_data_opened": False,
                      "sealed_experiment_code_modified": False}, indent=2))


if __name__ == "__main__":
    main()
