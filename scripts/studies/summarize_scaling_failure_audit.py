#!/usr/bin/env python3
"""Reporting only; never fits or selects a new model."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.evaluation.reporting import markdown_table
from src.qgeognn_al.transfer.scaling_audit import BIN_NAMES, DESCRIPTORS, match_key

STUDY = ROOT/"studies/transfer/scaling_failure_audit"
COLUMNS = ("8g", "25g", "40g")


def table(frame, index=False):
    return markdown_table(frame, index=index)


def save_figure(fig, name):
    fig.savefig(STUDY/"plots"/f"{name}.png", dpi=180)
    fig.savefig(STUDY/"plots"/f"{name}.pdf")
    plt.close(fig)


def make_plots(stability, tail, slices, associations, neighbors, pairs, model_aulc):
    (STUDY/"plots").mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for r, target in enumerate(("V1", "V2")):
        for c, column in enumerate(COLUMNS):
            ax = axes[r, c]
            for protocol, color in (("row", "#087e8b"), ("compound", "#b33f62")):
                group = stability.loc[stability.column.eq(column) & stability.target.eq(target) & stability.protocol.eq(protocol)].sort_values("budget")
                ax.errorbar(group.budget, group["mean"], yerr=group["std"], label=protocol, marker="o", color=color, capsize=3)
            ax.set(title=f"{column} / {target}", xlabel="Revealed target budget", ylabel="Fitted scale a")
            ax.grid(alpha=.15)
    axes[0, 0].legend()
    fig.suptitle("Scale stability under frozen training subsets (mean ± SD across five seeds)")
    fig.tight_layout(rect=(0, 0, 1, .95)); save_figure(fig, "scale_stability")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    contexts = [(c, p) for c in COLUMNS for p in ("row", "compound")]
    for ax, target in zip(axes, ("V1", "V2")):
        group = tail.loc[tail.target.eq(target)].groupby(["column", "protocol"])[["tail_row_fraction", "tail_sse_fraction"]].mean().reindex(contexts)
        x = np.arange(len(group))
        ax.bar(x-.18, 100*group.tail_row_fraction, width=.35, color="#8b99a5", label="Rows in tail")
        ax.bar(x+.18, 100*group.tail_sse_fraction, width=.35, color="#b33f62", label="Scale SSE in tail")
        ax.set_xticks(x, [f"{c}\n{p}" for c, p in contexts], fontsize=8)
        ax.set(title=target, ylabel="Share (%)", ylim=(0, 100)); ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("High-retention tail: test q50 above the training 90th percentile, budget 100")
    fig.tight_layout(rect=(0, 0, 1, .95)); save_figure(fig, "tail_error_concentration")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for r, protocol in enumerate(("row", "compound")):
        for c, column in enumerate(COLUMNS):
            ax = axes[r, c]
            for target, color in (("V1", "#087e8b"), ("V2", "#b33f62")):
                group = slices.loc[slices.column.eq(column) & slices.protocol.eq(protocol) & slices.target.eq(target) &
                                   slices.dimension.eq("source_bin")].groupby("level").ratio_median.agg(["mean", "std"]).reindex(BIN_NAMES)
                ax.errorbar(range(4), group["mean"], yerr=group["std"], marker="o", label=target, color=color, capsize=3)
            ax.set(title=f"{column} / {protocol}", ylabel="Median target/source q50 ratio")
            ax.set_xticks(range(4), ["Low", "Medium", "High", "Tail"])
            ax.grid(alpha=.15)
    axes[0, 0].legend()
    fig.suptitle("Ratio across train-defined source-magnitude bins (descriptive test, mean ± seed SD)")
    fig.tight_layout(rect=(0, 0, 1, .95)); save_figure(fig, "ratio_by_source_range")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, target in zip(axes, ("V1", "V2")):
        for i, (column, protocol) in enumerate(contexts):
            group = associations.loc[associations.column.eq(column) & associations.protocol.eq(protocol) &
                                     associations.target.eq(target) & associations.feature.eq("EA_fraction")]
            ax.scatter(i+np.linspace(-.12, .12, len(group)), group.ratio_partial_source_rho, s=25, color="#087e8b")
            ax.scatter([i], [group.ratio_partial_source_rho.mean()], s=55, marker="D", color="#111")
        ax.axhline(0, color="#777", lw=.8); ax.axhline(-.3, color="#999", lw=.8, linestyle="--")
        ax.set_xticks(range(6), [f"{c}\n{p}" for c, p in contexts], fontsize=8)
        ax.set(title=target, ylabel="EA/ratio partial rank correlation", ylim=(-1, .5)); ax.grid(alpha=.15)
    fig.suptitle("Training evidence: EA association after controlling source magnitude\nDots: five seeds; diamonds: mean; not a causal effect estimate")
    fig.tight_layout(rect=(0, 0, 1, .92)); save_figure(fig, "condition_partial_association")

    interactions = pd.read_csv(STUDY/"training_source_condition_interactions.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for r, protocol in enumerate(("row", "compound")):
        for c, column in enumerate(COLUMNS):
            ax = axes[r, c]
            group = interactions.loc[interactions.column.eq(column) & interactions.protocol.eq(protocol) &
                                     interactions.target.eq("V1") & interactions.feature.eq("EA_fraction")]
            means = group.groupby(["condition_level", "source_bin"]).ratio_median.mean().unstack().reindex(index=[0, 1], columns=BIN_NAMES)
            count = group.groupby(["condition_level", "source_bin"]).rows.mean().unstack().reindex(index=[0, 1], columns=BIN_NAMES)
            im = ax.imshow(means, cmap="viridis", aspect="auto")
            for i in range(2):
                for j in range(4):
                    value = means.iloc[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.2f}\nn≈{count.iloc[i,j]:.0f}", ha="center", va="center", fontsize=8,
                                color="white" if value < np.nanmean(means.to_numpy()) else "black")
            ax.set(title=f"{column} / {protocol}", xticks=range(4), xticklabels=["Low", "Mid", "High", "Tail"],
                   yticks=[0, 1], yticklabels=["Lower EA", "Higher EA"])
            fig.colorbar(im, ax=ax, fraction=.04)
    fig.suptitle("Training V1 ratio: source-magnitude × EA strata\nMean of per-seed medians; n is mean observed stratum count; sparse cells are descriptive")
    fig.tight_layout(rect=(0, 0, 1, .92)); save_figure(fig, "source_condition_interaction")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, target in zip(axes, ("V1", "V2")):
        for i, (column, protocol) in enumerate(contexts):
            values = neighbors.loc[neighbors.column.eq(column) & neighbors.protocol.eq(protocol) & neighbors.target.eq(target), "neighbor_rho"]
            ax.scatter(i+np.linspace(-.12, .12, len(values)), values, s=25, color="#3b5bdb")
        ax.axhline(0, color="#777", lw=.8); ax.axhline(.3, color="#999", linestyle="--", lw=.8)
        ax.set_xticks(range(6), [f"{c}\n{p}" for c, p in contexts], fontsize=8)
        ax.set(title=target, ylabel="Compound ratio / neighbor-ratio correlation", ylim=(-1, 1)); ax.grid(alpha=.15)
    fig.suptitle("Frozen descriptor neighborhoods: no stable positive training signal\nThree nearest other compounds; each compound has equal weight")
    fig.tight_layout(rect=(0, 0, 1, .92)); save_figure(fig, "molecule_neighborhood_consistency")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    coverage = pairs.set_index("column").reindex(COLUMNS)
    for i, (name, color, label) in enumerate([
        ("exact_rows", "#087e8b", "Exact / all source identities"),
        ("exact_source_train_rows", "#386cb0", "Exact / source-train labels"),
        ("relaxed_rows", "#d08700", "Relaxed / all source identities"),
        ("relaxed_source_train_rows", "#b33f62", "Relaxed / source-train labels")]):
        ax.bar(x+(i-1.5)*.18, coverage[name]/coverage.rows*100, width=.17, color=color, label=label)
    ax.set(xticks=x, xticklabels=COLUMNS, ylabel="Canonical target rows matched (%)", ylim=(0, 100),
           title="Strict and flow-relaxed pairing: identity overlap differs from label availability")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.15)
    fig.tight_layout(); save_figure(fig, "pair_coverage")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    references = ("scale_only", "affine", "local_identity_shrinkage", "additive_policy")
    for r, protocol in enumerate(("row", "compound")):
        for c, column in enumerate(COLUMNS):
            ax = axes[r, c]
            pivot = model_aulc.loc[model_aulc.column.eq(column) & model_aulc.protocol.eq(protocol)].pivot(index="seed", columns="method", values="normalized_aulc")
            for i, ref in enumerate(references):
                values = 100*(1-pivot.conditional_policy/pivot[ref])
                ax.scatter(np.full(5, i)+np.linspace(-.1, .1, 5), values, s=20, color="#087e8b")
                ax.scatter([i], [100*(1-pivot.conditional_policy.mean()/pivot[ref].mean())], marker="D", s=45, color="#111")
            ax.axhline(0, color="#777", lw=.8); ax.axhline(5, color="#999", linestyle="--", lw=.8)
            ax.set(title=f"{column} / {protocol}", ylabel="Conditional policy AULC gain (%)")
            ax.set_xticks(range(4), ["Scale", "Affine", "Shrink", "Additive"], rotation=15)
            ax.grid(alpha=.15)
    fig.suptitle("One conditional-scaling experiment: strong controls limit the incremental gain\nDots: paired seeds; diamonds: relative gain of means; dashed: 5%")
    fig.tight_layout(rect=(0, 0, 1, .92)); save_figure(fig, "conditional_model_paired_effects")


def main():
    training_decision = json.loads((STUDY/"training_direction_decision.json").read_text())
    model_decision = json.loads((STUDY/"model_decision.json").read_text())
    stability = pd.read_csv(STUDY/"scale_stability_summary.csv")
    sensitivity = pd.read_csv(STUDY/"scale_stability_by_seed.csv")
    associations = pd.read_csv(STUDY/"training_associations.csv")
    neighbors = pd.read_csv(STUDY/"training_molecule_consistency.csv")
    tail = pd.read_csv(STUDY/"test_descriptive_tail.csv")
    slices = pd.read_csv(STUDY/"test_descriptive_slices.csv")
    pairs = pd.read_csv(STUDY/"pair_coverage.csv")
    model_aulc = pd.read_csv(STUDY/"model_aulc_by_seed.csv")
    model_paired = pd.read_csv(STUDY/"model_paired_aulc.csv")
    metrics = pd.read_csv(STUDY/"model_all_metrics.csv")
    selectors = pd.read_csv(STUDY/"model_selection.csv")
    stable100 = stability.loc[stability.budget.eq(100)].copy()
    stable100["cv_percent"] = stable100.coefficient_cv*100
    stable100.to_csv(STUDY/"scale_budget100_summary.csv", index=False)
    ea = associations.loc[associations.feature.eq("EA_fraction")].groupby(["column", "protocol", "target"])[
        ["ratio_partial_source_rho", "ratio_partial_source_compound_rho", "relative_standardized_contrast"]].mean().reset_index()
    ea.to_csv(STUDY/"condition_evidence_summary.csv", index=False)
    tail_summary = tail.groupby(["column", "protocol", "target"])[
        ["tail_row_fraction", "tail_sse_fraction", "tail_rmse", "rest_rmse", "tail_signed_mean"]].mean().reset_index()
    tail_summary.to_csv(STUDY/"tail_summary.csv", index=False)
    pair_slices = slices.loc[slices.dimension.isin(["pair_status", "relaxed_pair_status"])].groupby(
        ["column", "protocol", "target", "dimension", "level"]).agg(
            seeds_with_observations=("seed", "nunique"), mean_rows_when_present=("rows", "mean"),
            mean_compounds_when_present=("compounds", "mean"), ratio_median=("ratio_median", "mean"),
            scale_rmse=("scale_rmse", "mean"), scale_mae=("scale_mae", "mean"),
            affine_rmse=("affine_rmse", "mean"), affine_mae=("affine_mae", "mean")).reset_index()
    pair_slices.to_csv(STUDY/"pair_error_summary.csv", index=False)
    repeat_rows = []
    for column in COLUMNS:
        features = pd.read_csv(STUDY/f"features_{column}.csv")
        counts = pd.Series([match_key(row) for row in features.to_dict("records")]).value_counts()
        repeat_rows.append({"column": column, "exact_replicate_groups": int((counts>1).sum()),
                            "rows_in_exact_replicate_groups": int(counts[counts>1].sum()), "max_replicates": int(counts.max()),
                            "source_V1_below_ratio_floor": int(features.source_V1.lt(.5).sum()),
                            "source_V2_below_ratio_floor": int(features.source_V2.lt(.5).sum())})
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(STUDY/"replication_and_ratio_floor_audit.csv", index=False)
    influence = sensitivity.loc[sensitivity.budget.eq(100)].groupby(["column", "protocol", "target"])[
        ["upper10_x2_leverage", "leave_compound_max_relative_change", "without_upper10_relative_change"]].mean().reset_index()
    influence.to_csv(STUDY/"scale_range_sensitivity_summary.csv", index=False)
    neighbor_summary = neighbors.groupby(["column", "protocol", "target"])[
        ["neighbor_rho", "neighbor_scale_residual_rho", "condition_halves_rho", "condition_halves_compounds", "source_error_partial_rho"]].mean().reset_index()
    neighbor_summary.to_csv(STUDY/"molecule_evidence_summary.csv", index=False)
    # Descriptive adjusted counts treat sub-micro differences as numerical ties, without changing any fit or gate.
    effect_rows = []
    for (column, protocol), group in model_aulc.groupby(["column", "protocol"]):
        pivot = group.pivot(index="seed", columns="method", values="normalized_aulc")
        for reference in ("scale_only", "affine", "local_identity_shrinkage", "additive_policy"):
            delta = pivot.conditional_policy-pivot[reference]
            effect_rows.append({"column": column, "protocol": protocol, "reference": reference,
                               "wins_beyond_1e_7_aulc": int((delta < -1e-7).sum()),
                               "numerical_ties_within_1e_7": int((np.abs(delta)<=1e-7).sum())})
    pd.DataFrame(effect_rows).to_csv(STUDY/"model_numerical_tie_audit.csv", index=False)
    make_plots(stability, tail, slices, associations, neighbors, pairs, model_aulc)

    audit_text = f"""# Scaling Failure Audit

**发现了结构化失败，但不能将强 scale-only 解释成普适物理规律。** 训练内预设筛选只支持进入 `CONDITIONAL_SCALING`：V1 的 ratio 与 EA 比例存在跨三个柱、row/compound 均复现的关系。随后完成一个条件缩放实验，其主策略未通过相对强对照的 material-gain 门槛。详见 [NEXT_MODEL_DECISION.md](NEXT_MODEL_DECISION.md)。

## A. Scale-only 参数是否稳定？

答案是有条件的稳定。budget 100 时，25g/40g 的跨 seed scale CV 为 1.9%–6.4%；8g row 为 2.2%–2.4%，8g compound V1 则为 11.1%。budget 30 的 CV 达 7.4%–24.8%。因此不能用一个“不稳定/稳定”概括所有预算和协议。

更关键的是 `a = sum(x*y)/sum(x²) = sum(w_i * ratio_i)`，其中 `w_i=x_i²/sum(x²)`。这是一种 **source-q50 平方加权的 ratio 平均**，不是平均样本的 ratio。训练上端 10% q50 样本占 46%–70% 的平均 x² 权重；删除这部分训练范围后，25g/40g 的 scale 均值改变约 -7% 到 -13%。在固定采样分布下跨 seed 较稳定，并不等于对 source 范围或条件分布稳定。

{table(stable100[['column','protocol','target','mean','std','median','min','max','cv_percent']])}

[全部预算/seed 系数](scale_stability_by_seed.csv)保留 train compound 数、source 范围、尾部权重、leave-one-compound scale 与截去尾部后的变化；这只是训练子集敏感性分析，不用于挑模型。

![Scale stability](plots/scale_stability.png)

## B. 残差/ratio 主要依赖什么？

**最可复现的条件信号是 EA fraction / V1；绝对误差还明显依赖 source magnitude 和柱规格。** 不能从这些观察性统计做唯一因果方差分解。

1. **Source magnitude / high-retention tail。** 使用各 seed 的 train q33/q67/q90 定义 low/medium/high/extreme tail，test 不重估边界。8g/25g 上约 11%–14% 的尾部行承担约 42%–63% 的 scale SSE；40g 尾部占约 8%–11% 行、约 20%–28% SSE，尾部 RMSE 虽大，但主体样本也有很大误差。不能把 40g 的问题归结为少数尾部点。尾部 signed error 方向也不是所有柱/协议一致。40g 的 ratio 与 source q50 的训练 Spearman 均值约 0.38–0.48；8g 接近零。ratio 的分母/source predictor error 可能产生耦合，不能直接解释成真实曲率。

{table(tail_summary)}

![Tail concentration](plots/tail_error_concentration.png)
![Ratio by source range](plots/ratio_by_source_range.png)

2. **Conditions。** 控制 source magnitude 后，EA fraction 与 V1 ratio 的平均 partial rank rho 在六个场景为 -0.41 至 -0.61；再控制 compound 后约 -0.57 至 -0.79。共同 source-bin 支持下，较高 EA 的相对 ratio contrast 约 -20% 至 -42%。V2 信号较弱。每个条件都同时有原类别分箱、source×condition 交互分箱和 common-support 标准化汇总，绝非只算相关系数。loading solvent、上样体积/实际量、加载溶剂体积没有通过同样的跨 seed 门槛；其中不少特征在部分柱内近乎常量、与 molecule 混杂或缺少共同支持，这不等于无效。

{table(ea)}

标准化汇总仅在同一 source bin 中两组各至少 3 行、2 个 compound 且至少两个 bin 可比较时计算。它是观察数据的 support-restricted summary，不是替换实验条件后的 counterfactual partial dependence。rho 调整也不能消除所有非线性混杂。

![Condition association](plots/condition_partial_association.png)
![Source-condition interaction](plots/source_condition_interaction.png)

3. **Molecule。** 已记录每个 compound 的 mean/median ratio、ratio variance、OOF residual variance、不同 EA 条件数及 residual sign consistency。七个可审计 RDKit descriptor 的 compound-level 关联和 3-nearest-neighbor 检查没有稳定复现的正邻域信号；low/high-EA 两半的 compound scaling 一致性也较弱。部分 compound 可表现系统偏高/偏低，但没有足够证据证明这在所测 descriptor 空间可泛化。row 低标签训练中每个 compound 往往不足四行，条件两半比较常为不可估计；报告 NaN，不填零。这里使用 descriptor-space，未把它冒称 QGeoGNN latent space，也未排除后者可能有结构。

{table(neighbor_summary)}

![Molecule neighborhoods](plots/molecule_neighborhood_consistency.png)

4. **Pair status。** 8g 的配对/条件不同两组误差有差异，但方向随 output/protocol 变化。例如 V1 的 row 配对 RMSE 较小，compound 配对 RMSE 反而较大。两组同时有 molecule 和 source-range 组成差异，不能把误差差异归因于配对本身。source-absent 仅 8g 一个 compound/7 行、25g 一个 compound/19 行、40g 零行；分组表记录非空 seed 数，不能把这些小组宣布为 source-unseen OOD 性能。

## 严格配对与 relaxed pairing

exact 只允许柱规格不同，其余匹配字段为 canonical molecule、精确有理数 PE/EA 组成、loading solvent、density 和 sample volume、loading-solvent volume、flow。density 和 volume 分别相等比仅比较它们的乘积更严格。relaxed 只再忽略 flow，未使用容差、最近邻或过宽 matching。

{table(pairs)}

历史 8g exact overlap 483 行没有检查 density；本轮严格口径为 412 行（71.8%），其中 324 行（56.4%）有原 source-train 标签可用。不能把 source validation/test 标签当成免费模型输入。25g/40g 严格 exact 为零，因为 flow 不同；relaxed 为 297/490（60.6%）、422/529（79.8%），source-train 可用分别为 240、358 行。完整匹配源 IDs 与匹配数在 [pair_identity_audit.csv](pair_identity_audit.csv)，source-train 标签特征只来自源训练集合。

8g 的 source-error anchor 与 OOF scale residual 在训练中的平均 partial rho 约 0.18–0.30，跨 seed 波动大；配对数量足够，但信号未在 row+compound 各达到 4/5 seeds，故本轮不运行 paired/delta 模型。25g/40g relaxed pairs 正式保留为 backlog，不能直接当作 flow 的因果对照。

{table(pair_slices.loc[pair_slices.column.eq('8g') & pair_slices.dimension.eq('pair_status')])}

表中 rows/compounds 是**有观测的 seed 内均值**；请同时看 seeds_with_observations。完整 25g/40g relaxed 分组 RMSE/MAE/ratio 见 [pair_error_summary.csv](pair_error_summary.csv)。它包含每个 column、row/compound、V1/V2 的三类分组；无样本组不伪造零误差。

![Pair coverage](plots/pair_coverage.png)

## C. Additive condition Ridge 的负结果意味着什么？

不能解释成 conditions 不重要。训练内 EA/V1 信号足以排除这种过强解释；但条件作用已部分被 source q50 压缩，ratio 分母耦合、拟合方差和条件/分子结构也会混入 residual。原结果仍应表述为 `ADDITIVE_LINEAR_CONDITION_RESIDUAL_NOT_MATERIALLY_SUPPORTED`。

本轮真正让 condition 进入 slope，并用同容量 additive-EA control 比较。conditional 在少数场景更好，但跨场景没有稳定超过强 additive/shrinkage controls 的 material 改善。因此证据**不能进一步证明 additive formulation 就是主要瓶颈，或 multiplicative formulation 已解决问题**。

## D/E. 是否进入模型方向，是否停止？

训练证据选择了一个方向：`CONDITIONAL_SCALING`，并已完成最小实验。B `MOLECULE_DEPENDENT_SCALING` 和 C `PAIRED_DELTA_LEARNING` 未过预设筛选，不执行。最终是 **`{model_decision['decision']}`**；不能写 `NO_STRUCTURED_SCALING_FAILURE_IDENTIFIED`，因为 EA/V1 的稳定失败结构确实被发现。主策略既未达到跨场景 `LABEL_EFFICIENCY_GAIN`，也未达到 `ACCURACY_GAIN` 门槛。见 [模型决策](NEXT_MODEL_DECISION.md) 中 standalone 40g 信号与 validation 回退的完整区别。

## 证据边界、复现和数据需求

训练内方向选择使用各 frozen context 的 budget-100 gradient_train，并按 compound GroupKFold 得到 OOF residual。没有合并跨 seed 标签来拟合模型；这些 seeds 的样本仍重叠，不能当五个独立外部数据集。随后冻结 model protocol、全部 120 组预测，再读取本轮 target test。test 表仅用于描述/冻结门槛，未追加方法。所有 qualified predictor、原 splits、旧结果与 checkpoint 保留不变。

当前仅能称 `EMPIRICAL CROSS-COLUMN STRUCTURE`，不能称 `PHYSICAL LAW`。目标分子高度重叠 source；8g 条件大量匹配；25g/40g flow/条件范围不同；mass/flow/spec 混杂。相同条件重复行很少，且没有足够独立重复批次，无法估计 irreducible experimental noise：

{table(repeats)}

更有价值的数据是：独立 compound/实验批次的确认集、同条件重复实验、高 retention 尾部覆盖、source-unseen 分子，以及 crossed mass×flow。关于共享 backbone、column heads/context/task embedding、多任务、多保真、adaptive readout、paired/delta 等方向，继续以 [FUTURE_HYPOTHESES / EXPERIMENT_BACKLOG](../../../docs/research/CROSS_COLUMN_TRANSFER_STATUS.md) 保存。

按顺序执行（各阶段检查/写入同一协议文件，不并发运行）：

```bash
conda run --no-capture-output -n fish python scripts/studies/run_scaling_failure_audit.py --train-audit
conda run --no-capture-output -n fish python scripts/studies/run_conditional_scaling_audit_model.py --fit
conda run --no-capture-output -n fish python scripts/studies/run_conditional_scaling_audit_model.py --evaluate
conda run --no-capture-output -n fish python scripts/studies/run_scaling_failure_audit.py --test-descriptive
MPLCONFIGDIR=/tmp/scaling-failure-mpl conda run --no-capture-output -n fish python scripts/studies/summarize_scaling_failure_audit.py
```

依赖原 qualified checkpoint、source prediction cache（缺失时仅重新推理，不训练）及固定 conda fish 环境。输入/代码/设计哈希漂移会停止复用。详细 train/test bins、interaction、molecule、pairing、scaling stability、逐 seed/预算性能和 label ledgers 均在本目录 CSV/JSON 中。
"""
    (STUDY/"SCALING_FAILURE_AUDIT.md").write_text(audit_text)
    primary = model_paired.loc[model_paired.method.eq("conditional_policy")].copy()
    primary["relative_gain_percent"] = primary.relative_gain*100
    primary = primary.merge(pd.DataFrame(effect_rows), on=["column", "protocol", "reference"], how="left")
    b100 = metrics.loc[metrics.budget.eq(100)].groupby(["column", "protocol", "method"])[
        ["V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae", "normalized_rmse"]].mean().reset_index()
    aulc_summary = pd.read_csv(STUDY/"model_aulc_summary.csv")
    text = f"""# Next Model Decision

**`{model_decision['decision']}`**。识别了可复现的 EA/V1 条件结构，并完成一个 Conditional Scaling 实验；目前没有达到预先定义的跨场景 material gain，停止本轮模型扩展。

## 选择的模型与对照

模型：`target/(mass_ratio*S) = a*u+b+gamma*standardize_train(u*(EA-mean_train(EA)))`，u=source_q50/S。第三项真正改变 EA-dependent slope。对照只把第三项改成标准化 EA，即同参数数量的 additive control。两者每个输出三个参数、相同 lambda={{0,0.1,1}}、相同 train/validation、相同数值目标。

同时报告 standalone conditional/additive 和两个对称 validation policies；policy 可回退到 scale、affine、原 local identity shrinkage。两输出共用 validation 选择的 penalty，避免隐藏 V2 回归。Head-only 复用原 frozen neural reference。每个模型仍只消耗原单柱 actual_budget（train+validation），没有借用其他柱 target 标签，也没有使用配对 source 真值作为预测输入。

## 结果与停止依据

- 25g conditional policy 的 AULC 相对 scale 改善 row 5.32%、compound 7.68%，相对 affine 6.59%/11.43%。但相对 local shrinkage 仅 3.48%/2.62%，相对 additive policy 仅 1.04%/2.56%。有稳定方向性收益，但增量未达 5%。
- 40g row 相对 scale 改善 14.09%，相对 local shrinkage 4.97%、additive policy 3.47%。40g compound 的 validation policy 在全部 20 个 seed/budget contexts 都回退到 affine/shrinkage，实际与 local/additive policy 数值等价。保留原始配对值，另外给出 1e-7 AULC 数值 ties，避免把浮点差当成 seed wins。
- 8g 相对 scale/head/shrinkage 没有稳定优势。

**不能隐藏 standalone 的局部正信号：** 40g compound 的 standalone conditional AULC 为 3.4170，相对 scale 改善 5.12%，相对 local/additive policy 改善 7.76%，均赢 5/5 seeds。可是 40g row 相对 additive policy 只有 4.50%；25g 相对强对照也未达门槛；8g 没有收益。因而未达到预定的“两个 compound columns，或一个 column 的 row+compound 均超过所有强对照”的门槛。不能事后依据 test 撤销 policy fallback、选择 standalone 为主策略或调整 validation。该分歧记录为低标签 validation 可靠性的未来问题。

**High-budget absolute accuracy 没有实质解决。** budget 100，25g conditional policy 的 compound RMSE 为 13.10/19.67 mL，相对原 affine 14.14/20.13 有限改善；row 为 17.89/25.43。40g compound policy 仍为 33.57/40.97；standalone 40g compound 的 NRMSE 仅比 affine 改善 0.95%。这不是已经解决 25g/40g absolute error。完整 R²/RMSE/MAE 及各 seed 见下表/CSV。

最终主策略门槛：label efficiency={model_decision['label_efficiency_material']}；high-budget accuracy={model_decision['high_budget_accuracy_material']}。局部 AULC 改善与跨场景 `LABEL_EFFICIENCY_GAIN`、`ACCURACY_GAIN` 必须分开说，不能用单个 column/seed/budget 推动继续复杂化。

## Paired AULC：主策略 versus 全部必要参考

{table(primary[['column','protocol','reference','relative_gain_percent','median_delta','std_delta','wins','wins_beyond_1e_7_aulc','numerical_ties_within_1e_7','material']])}

![Model paired effects](plots/conditional_model_paired_effects.png)

## 全方法 seed stability

{table(aulc_summary)}

## Budget 100 点指标（五 seeds 均值）

{table(b100)}

## Validation policy choices

{table(selectors.groupby(['column','protocol','conditional_policy']).size().rename('contexts').reset_index())}

## 可回答和不可回答的问题

训练审计足以进入 `CONDITIONAL_SCALING` 的一次受控试验；它不支持 `conditions 无用`。本轮对照又说明不能把主要瓶颈确定为 additive formulation：改变为 varying slope 的增益仍受估计方差、验证选择和强正则化对照限制。数据不支持这时再追加 MLP、molecule-dependent、paired/delta 或 readout 模型。

项目级历史结论保持 provenance，当前解释继续是 `NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS`。本轮识别了结构，所以不写 `NO_STRUCTURED_SCALING_FAILURE_IDENTIFIED`；也不宣称现有数据关闭了所有复杂模型研究空间。

B/C、multi-column / multi-condition、adaptive readout、noise floor、真正 source-unseen OOD、crossed mass×flow 继续放入 [科研 backlog](../../../docs/research/CROSS_COLUMN_TRANSFER_STATUS.md)。完整 paired source IDs 为将来设计留存；本轮没有后验放宽匹配规则或增加新方法。

## Artifacts and checks

- [Training-only evidence](training_direction_evidence.csv) / [frozen choice](training_direction_decision.json)
- [Model preregistration](MODEL_PREREGISTRATION.md) / [frozen protocol](model_protocol.json)
- [All seed/budget R², RMSE, MAE, NRMSE](model_all_metrics.csv)
- [Point metric mean/std/median/min/max](model_aggregate_metrics.csv)
- [Planned/actual-budget AULC](model_aulc_by_seed.csv)
- [All paired AULC comparisons](model_paired_aulc.csv) / [budget-100 comparisons](model_paired_budget100.csv)
- [Label usage](model_label_usage.csv) / [validation choices](model_selection.csv)
- [Execution audit](model_execution_audit.json): 120 contexts, 960 metric records, 0 unresolved model failures; all predictions frozen before test.

历史 test 已被使用过，因此本轮是开发性证据。source/q50 和 label roles 复用原协议；未训练 QGeoGNN，未改变 qualified baseline。描述性 test 汇总不能重开训练筛选或改变预定方法。
"""
    (STUDY/"NEXT_MODEL_DECISION.md").write_text(text)


if __name__ == "__main__":
    main()
