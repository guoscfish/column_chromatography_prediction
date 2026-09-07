#!/usr/bin/env python3
"""Scientific report and a small set of figures from the frozen evaluation."""
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
from scripts.studies import run_source_anchored_transfer as run
from scripts.studies import evaluate_source_anchored_transfer as evaluate
from src.qgeognn_al.evaluation.reporting import markdown_table

STUDY = run.STUDY
ALIASES = {"scale_only": "Scale", "local_identity_shrinkage": "Shrinkage", "conditional_EA": "Conditional EA",
           "conditional_policy": "Conditional policy", "target_head_only": "N0 head",
           "standard_shallow_finetune": "N1 shallow", "standard_full_finetune": "N2 full",
           "source_anchored_shallow": "M1 anchored shallow", "source_anchored_full": "M2 anchored full"}
COLORS = dict(zip(ALIASES, ["#333333", "#888888", "#cc79a7", "#9467bd", "#d4a017", "#0072b2", "#56b4e9", "#d55e00", "#009e73"]))


def table(frame):
    return markdown_table(frame, index=False)


def generate():
    evaluate.lock_evaluation()
    evaluate.validate_global_freeze()
    audit = json.loads((STUDY / "execution_audit.json").read_text())
    if audit["new_fits"] != 480 or audit["failed_fits"] or not audit["all_predictions_frozen_before_test"]:
        raise RuntimeError("incomplete scientific experiment")
    result = json.loads((STUDY / "decision.json").read_text())
    metrics = pd.read_csv(STUDY / "all_metrics.csv")
    aulc = pd.read_csv(STUDY / "aulc_by_seed.csv")
    pairs = pd.read_csv(STUDY / "paired_comparisons.csv")
    drift = pd.read_csv(STUDY / "source_drift_metrics.csv")
    training = pd.read_csv(STUDY / "training_audit.csv")
    strata = pd.read_csv(STUDY / "error_stratification.csv")
    stability = pd.read_csv(STUDY / "stability_comparisons.csv")
    b100 = metrics.loc[metrics.budget.eq(100)]
    aulc_summary = aulc.groupby(["column", "protocol", "method"]).normalized_aulc.agg(["mean", "std", "median", "min", "max"]).reset_index()
    aulc_summary.to_csv(STUDY / "aulc_summary.csv", index=False)
    accuracy = b100.groupby(["column", "protocol", "method"])[["V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae", "combined_normalized_rmse"]].mean().reset_index()
    accuracy.to_csv(STUDY / "budget100_mean_metrics.csv", index=False)
    source = drift.groupby(["column", "protocol", "method"])[["before_V1_rmse", "after_V1_rmse", "before_V2_rmse", "after_V2_rmse",
               "before_V1_r2", "after_V1_r2", "before_V2_r2", "after_V2_r2", "source_combined_nrmse_drift", "backbone_l2_drift", "target_head_l2_drift"]].mean().reset_index()
    source.to_csv(STUDY / "source_drift_summary.csv", index=False)
    training_summary = training.groupby(["column", "protocol", "method"]).agg(
        best_epoch_mean=("best_epoch", "mean"), best_epoch_min=("best_epoch", "min"), best_epoch_max=("best_epoch", "max"),
        epochs_run_mean=("epochs_run", "mean"), wall_seconds_mean=("wall_seconds", "mean"),
        replay_draws_mean=("source_replay_draws", "mean"), replay_unique_mean=("source_replay_unique_rows", "mean"),
        cap_hits=("epochs_run", lambda x: int((x == 500).sum())),
        first_5_epoch_selections=("best_epoch", lambda x: int((x <= 5).sum()))).reset_index()
    training_summary.to_csv(STUDY / "training_summary.csv", index=False)
    robustness_names = [f"{t}_{m}" for t in ("V1", "V2") for m in ("macro_compound_mae", "macro_compound_rmse", "median_relative_absolute_error")]
    robustness = b100.groupby(["column", "protocol", "method"])[robustness_names].mean().reset_index()
    robustness.to_csv(STUDY / "budget100_robustness_summary.csv", index=False)
    strata_summary = strata.loc[strata.budget.eq(100)].groupby(["column", "protocol", "method", "target", "dimension", "level"]).agg(
        mean_rows=("n", "mean"), nonempty_seeds=("n", lambda x: int((x > 0).sum())),
        rmse=("rmse", "mean"), mae=("mae", "mean"), median_absolute_error=("median_absolute_error", "mean")).reset_index()
    strata_summary.to_csv(STUDY / "budget100_stratification_summary.csv", index=False)
    selected_pairs = pairs.loc[pairs.reference.isin(["scale_only", "local_identity_shrinkage", "conditional_EA", "conditional_policy"])
                              | ((pairs.method.eq("source_anchored_shallow") & pairs.reference.eq("standard_shallow_finetune"))
                                 | (pairs.method.eq("source_anchored_full") & pairs.reference.eq("standard_full_finetune")))].copy()
    selected_pairs["gain_percent"] = 100*selected_pairs.relative_gain
    gate_table = selected_pairs[["column", "protocol", "method", "reference", "endpoint", "gain_percent", "wins", "median_delta", "material", "stronger_10pct"]]
    gate_table.to_csv(STUDY / "primary_gate_summary.csv", index=False)
    plots(metrics, accuracy, strata_summary, source)
    decision_text = result["decision"]
    report = f"""# Source-anchored shared representation transfer

本轮结论：**`{decision_text}`**。完整执行 3 柱 × 2 协议 × 5 seeds × 4 budgets，共 120 contexts、480 次新 neural fits；加上五个冻结参考方法，得到 1080 组点指标。全部预测先冻结，再执行本轮 test 评价。历史 test 已使用，因此这是 **developmental evidence**。

## 1. 为什么从 scalar 转向 representation

source q50 把 molecule 与 chromatography condition representation 压缩成每个输出的一个数。之前的 conditional、affine、shrinkage 和 spline 研究已接近收益递减。本轮直接迁移 128D latent representation，分别检验 H1（表示迁移的价值）与 H2（固定 source function 的 replay 约束是否有额外价值）。这不预设 information bottleneck 一定是主要误差来源。

这里比较的是有限标签与冻结训练配方下能学到的性能，不是单纯的函数表达能力：对正的 scale coefficient，原 head 的相应 weight/bias 同乘该系数就能表达 scale-only。因此即使 neural learner 较差，也不能据此证明 128D 中没有所需信息；若 fine-tuning 较好，也不能把所有收益唯一归因于原始 latent 中已有的信息。本轮没有据此追加 scaled-head 初始化或优化器实验。

历史结论 `NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS` 保留。本轮属于 representation-level transfer，不是新的 scalar correction。

## 2. 实际实现及 matched controls

| 方法 | 可训练部分 | 参数数 | source replay |
| --- | --- | ---: | --- |
| N1 standard_shallow_finetune | backbone.convs.4 + condition_branch + target_head | 36,387 | 无 |
| N2 standard_full_finetune | full backbone + condition_branch + target_head | 458,952 | 无 |
| M1 source_anchored_shallow | 与 N1 相同 | 36,387 | L_target + L_source |
| M2 source_anchored_full | 与 N2 相同 | 458,952 | L_target + L_source |

shared backbone 和两个独立 heads 均从 qualified final 4g row-seed-42 checkpoint 初始化。source head 固定，target head 不随机初始化；零步六输出与 source predictor 完全一致。head、loss、sum pooling、输入条件与 normalization 均未改。所有 wrapper 总参数 459,726（包含独立冻结的 774 个 source-head 参数）。N0 复用历史 frozen-backbone head-only，非本轮重新拟合。

原始 label IDs、row/compound splits 和嵌套 budgets 全部复用。每个模型只使用 4g source-train 与本柱购买标签。row 预算包含 8 个 validation labels；compound 按完整 compound 购买，实际预算见逐 context ledger。4g replay 不计入 target acquisition cost，但 draws、unique source IDs 单独报告。

Adam lr=1e-4、weight_decay=1e-5、最大 500 epochs、patience=100，checkpoint 只按 target-validation combined RMS NRMSE 选择。两个 task 使用原 raw-mL batch-mean loss、lambda=1。target pass 更新可训练 BN 的统计；source replay 固定 BN 统计并保留梯度，使 matched arms 不因额外 running-stat 更新而不同。

## 3. 完整结果

四种新方法都没有在任一场景同时超过全部强 calibration 的 material gate。25g/40g compound 中，四种新方法对最强冻结 calibration（conditional EA）的 AULC 均为 0/5 seed wins；最好的新 neural arm N1 仍分别差 39.81%/15.86%。N1 明显改善了历史 N0，但这不等于超过 calibration。8g N0 仍具有竞争力，不能把大柱否定结论扩张成所有 neural methods 在所有场景均无效。

高预算也没有解决问题：25g compound N1 的 V1/V2 RMSE 为 19.55/26.28 mL，M1 为 27.82/44.20，而 shrinkage 为 13.95/19.76；40g 分别为 37.46/48.27、54.52/75.14、33.57/40.97。N1 在 row 场景更接近 calibration，但未形成 >=5% 的全参考、双输出、跨场景优势。完整科研决策与紧凑比较见 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。

AULC 使用原 source-SD-normalized arithmetic mean RMSE，在 planned budgets 30–100 上取归一化梯形面积，越低越好；budget100 combined NRMSE 则是两输出 normalized RMSE 的 RMS，二者不能混用。下面 mean/std 来自五个 seeds，非五个独立外部数据集。

### AULC（全方法、全柱、全协议）

{table(aulc_summary)}

### Budget100 absolute accuracy（mL，五 seeds 均值）

{table(accuracy)}

![Budget curves](plots/budget_nrmse.png)

![Large-column absolute error](plots/budget100_rmse.png)

### 预注册门槛与 seed stability

material gain 要求 >=5% mean gain、negative median paired delta、>=4/5 seed wins，并在两个 compound columns 或同柱 row+compound 复现。高预算还要求两输出均不恶化；>=10% stronger gain 单独记录。必须超过每个 eligible calibration reference；M1/M2 还必须超过 matched N1/N2，不能只挑 affine 比较。完整门槛见 [primary_gate_summary.csv](primary_gate_summary.csv) 和 [decision.json](decision.json)。

{table(stability)}

## 4. source anchoring 的机制证据

M1/M2 在所有六个场景均以 5/5 seeds 降低 matched source drift，但目标误差没有同步改善。25g compound N2→M2 的 source combined NRMSE drift 从 3.674 降至 0.317，目标 AULC 却从 3.172 升至 4.402；40g 从 6.329 降至 0.342，目标 AULC 从 7.236 升至 9.118。40g compound M2 的 AULC SD 虽从 N2 的 2.076 降至 0.463，却以更差均值为代价。所有 joint stabilization gates 均失败，因此不能归入 Outcome C。source function preservation 有效，不等于得到了更好的 transfer learner。

使用同一 416-row source-validation probe，仅作诊断，从不 replay 或选择 target checkpoint。下面是各场景五 seeds、四 budgets 的 source probe 指标均值；逐 fit 的 before/after R²、RMSE、combined drift 和参数 drift 在 [source_drift_metrics.csv](source_drift_metrics.csv)。source-head drift 全部必须为零。source performance 与 backbone L2 drift 是不同诊断，不能相互代替。

{table(source)}

![Source drift](plots/source_drift.png)

source loss 与 target loss 的单位相同，不代表梯度强度相同。尤其大柱 target 初始残差更大，lambda=1 可能并不平衡两个 task；本轮不根据结果重新调 lambda。较小 source drift 只能说明 source function 保留较好，只有同时改善 matched target error 和跨 seed 稳定性，才支持其成为更好的 transfer learner。

跨 seed 的 AULC 离散度同时包含训练标签和 test 划分变化，不是固定 test 集上的纯估计方差。参数 L2 drift 应在相同 capacity 的 N1/M1 或 N2/M2 内解释，不能忽略参数维数直接跨 shallow/full 比大小。source performance drift 衡量原 source function 的保留，不能单独证明全部分子表征信息被遗忘。

### 训练稳定性与计算成本

{table(training_summary)}

epoch 到达 500 不自动等于拟合失败；表中明确报告 cap hits 和极早 checkpoint selection。逐 fit 的初始、最佳、最后一个 epoch losses、replay counts、参数数、耗时和成功状态见 [training_audit.csv](training_audit.csv)。source replay 每个 epoch 独立采样，逐 epoch source loss 不是同一批样本的收敛曲线。

## 5. Metric sensitivity 与 Scale-only

新 neural 方法没有更全面地战胜 calibration。25g/40g compound V1 top10 source-q50 RMSE，Scale 为 30.08/45.84 mL，N1 为 48.01/56.88，M1 为 70.90/119.89。40g N1 确实改善 low/mid q50 MAE 和 high-EA RMSE，但 high-q50、low-EA 退化；不是在所有样本范围均失败，也不是解决了原 EA-dependent failure。40g compound macro V1/V2 MAE，N1 为 22.66/29.70 mL，shrinkage 为 20.48/23.94；25g N1 为 13.31/18.27，shrinkage 为 9.27/13.62。Scale-only 不是每个指标/分层的绝对赢家，强 calibration 家族仍未被稳定超过。

source-q50 分层边界只来自该 context 的 gradient-train q33/q67/q90；top10 为重叠子集，test 实际占比不强制等于 10%。target-magnitude 分层使用冻结后 test truth，仅用于 characterization。EA bins 固定为 <=0.1、(0.1,0.5]、>0.5。所有分层同时报告 RMSE、MAE、median absolute error；空组为 NA。

compound macro metrics 先在每个 compound 内算 MAE/RMSE，再等权平均；relative absolute error 只对 |truth|>=1 mL 计算，中位数与排除行数均保留。这些指标没有用于选择方法。

{table(robustness.loc[robustness.column.isin(["25g", "40g"]) & robustness.protocol.eq("compound")])}

![Source q50 strata](plots/source_q50_errors.png)

完整分层结果：[error_stratification.csv](error_stratification.csv)，budget100 汇总：[budget100_stratification_summary.csv](budget100_stratification_summary.csv)。不能仅因 overall RMSE 较小就宣称在 MAE、compound macro 或所有 q50 范围全面优于强 calibration。Scale-only 仍是 empirical baseline，不是普适物理 scaling law。

## 6. 决策边界与下一步

机器可复核的冻结决策为 `{decision_text}`。详见 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。保留 scale-only / local identity shrinkage 为主要 point-transfer baseline，不提升 standard FT 或 source-anchored FT 为 AL 主模型；8g N0 和冻结 conditional 保留作对照。下一项优先研究建议是单独预注册 column-conditioned shared representation，并配置独立 compound/batch 验证；adaptive readout 是另一个独立受控备选。本轮不自动启动任何下一阶段，不追加模型、LR/feature/lambda sweep、adaptive readout、column embedding、UQ 或 Active Learning。

target-compound holdout != source-unseen molecular OOD；大多数 target compounds 已在 source train 中出现。mass/flow/column specification 混杂，因此本轮不能声称学到了 mass effect 或 flow effect。当前未控制的数据噪声、验证集小样本波动和 source/target 冲突，仍不能被单一模型结果唯一归因。
"""
    (STUDY / "RESULT_INTERPRETATION.md").write_text(report)


def plots(metrics, accuracy, strata, source):
    directory = STUDY / "plots"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    methods = ["scale_only", "local_identity_shrinkage", "conditional_EA", *run.METHODS]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for i, protocol in enumerate(("row", "compound")):
        for j, column in enumerate(run.COLUMNS):
            ax = axes[i, j]
            subset = metrics.loc[metrics.column.eq(column) & metrics.protocol.eq(protocol)]
            for method in methods:
                g = subset.loc[subset.method.eq(method)].groupby("budget").normalized_rmse.mean()
                ax.plot(g.index, g.values, marker="o", ms=3, lw=1.3, label=ALIASES[method], color=COLORS[method])
            ax.set(title=f"{column} / {protocol}", xlabel="Purchased target budget (planned)", ylabel="Arithmetic mean NRMSE")
            ax.set_xticks(run.BUDGETS)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(directory / "budget_nrmse.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, column in enumerate(("25g", "40g")):
        for j, protocol in enumerate(("row", "compound")):
            ax = axes[i, j]
            g = accuracy.loc[accuracy.column.eq(column) & accuracy.protocol.eq(protocol)].set_index("method").loc[methods]
            x = np.arange(len(methods))
            ax.bar(x-.18, g.V1_rmse, .36, label="V1", color="#0072b2")
            ax.bar(x+.18, g.V2_rmse, .36, label="V2", color="#d55e00")
            ax.set_xticks(x, [ALIASES[m] for m in methods], rotation=35, ha="right")
            ax.set(title=f"{column} / {protocol} / budget 100", ylabel="RMSE (mL), five-seed mean")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "budget100_rmse.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    levels = ["low", "mid", "high", "top10"]
    for i, column in enumerate(("25g", "40g")):
        for j, target in enumerate(("V1", "V2")):
            ax = axes[i, j]
            for method in methods:
                g = strata.loc[strata.column.eq(column) & strata.protocol.eq("compound") & strata.dimension.eq("source_q50")
                               & strata.target.eq(target) & strata.method.eq(method)].set_index("level").loc[levels]
                ax.plot(levels, g.mae, marker="o", ms=3, label=ALIASES[method], color=COLORS[method])
            ax.set(title=f"{column} compound / {target} / budget 100", ylabel="MAE (mL)", xlabel="Source q50 (train-defined cutoffs)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(directory / "source_q50_errors.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    contexts = [(c, p) for c in run.COLUMNS for p in ("row", "compound")]
    for k, method in enumerate(run.METHODS):
        g = source.loc[source.method.eq(method)].set_index(["column", "protocol"])
        values = [g.loc[context, "source_combined_nrmse_drift"] for context in contexts]
        ax.bar(np.arange(6)+(k-1.5)*.19, values, .19, label=ALIASES[method], color=COLORS[method])
    ax.set_xticks(np.arange(6), [f"{c}\n{p}" for c, p in contexts])
    ax.set(ylabel="Source-probe combined NRMSE change", title="Frozen source head: drift after adaptation (all budgets/seeds)")
    ax.axhline(0, color="#333333", lw=.7)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "source_drift.png")
    plt.close(fig)


if __name__ == "__main__":
    generate()
