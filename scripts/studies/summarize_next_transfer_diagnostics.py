#!/usr/bin/env python3
"""Descriptive reporting only: no fitting, selection or additional experiments."""
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

STUDY = ROOT / "studies/transfer/residual_diagnostics"


def main():
    decision = json.loads((STUDY/"decision.json").read_text())
    audit = json.loads((STUDY/"execution_audit.json").read_text())
    if audit["contexts"] != 120 or audit["failed"]:
        raise RuntimeError("incomplete diagnostics")
    metrics = pd.read_csv(STUDY/"all_metrics.csv")
    aulc = pd.read_csv(STUDY/"aulc_by_seed.csv")
    paired = pd.read_csv(STUDY/"paired_aulc.csv")
    portfolio = pd.read_csv(STUDY/"portfolio_aulc.csv")
    comp = aulc.loc[aulc.protocol.eq("compound")].groupby(["column", "method"]).normalized_aulc.mean().unstack()
    methods = ["scale_only", "affine", "target_head_only", "monotone_spline", "nonlinear_policy",
               "linear_policy", "shared_column_affine", "local_identity_shrinkage"]
    mean = portfolio.loc[portfolio.protocol.eq("compound")].groupby("method").normalized_aulc.mean()
    gain = lambda method, reference: 100 * (1 - mean[method]/mean[reference])
    row = portfolio.loc[portfolio.protocol.eq("row")].groupby("method").normalized_aulc.mean()
    point = metrics.loc[metrics.planned_budget.eq(100) & metrics.protocol.eq("compound") &
                        metrics.column.isin(["25g", "40g"]) &
                        metrics.method.isin(["affine", "monotone_spline", "shared_column_affine"])].groupby(
                            ["column", "method"])[["V1_rmse", "V2_rmse", "V1_mae", "V2_mae", "V1_r2", "V2_r2"]].mean()
    comparisons = paired.loc[paired.protocol.eq("compound") & paired.method.eq("shared_column_affine") &
                             paired.reference.isin(["affine", "scale_only", "local_identity_shrinkage"]),
                             ["column", "reference", "relative_mean_gain", "wins", "seeds", "stable_material"]].copy()
    comparisons["relative_mean_gain_percent"] = comparisons.pop("relative_mean_gain") * 100
    # Plot each seed paired against both affine and scale; do not hide the stronger control.
    plots = STUDY/"plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    candidates = [("monotone_spline", "affine", "Spline / affine"),
                  ("nonlinear_policy", "scale_only", "Nonlinear policy / scale"),
                  ("shared_column_affine", "affine", "Shared / affine"),
                  ("shared_column_affine", "scale_only", "Shared / scale"),
                  ("shared_column_affine", "local_identity_shrinkage", "Shared / local shrink")]
    colors = ["#72777f", "#3b5bdb", "#087e8b", "#d08700", "#b33f62"]
    for r, protocol in enumerate(["row", "compound"]):
        for c, column in enumerate(["8g", "25g", "40g"]):
            ax = axes[r, c]
            pivot = aulc.loc[aulc.column.eq(column) & aulc.protocol.eq(protocol)].pivot(
                index="seed", columns="method", values="normalized_aulc")
            for i, ((method, reference, label), color) in enumerate(zip(candidates, colors)):
                values = 100*(1-pivot[method]/pivot[reference])
                ax.scatter(values, np.full(5, i)+np.linspace(-.1, .1, 5), color=color, alpha=.75, s=25)
                aggregate = 100*(1-pivot[method].mean()/pivot[reference].mean())
                ax.scatter([aggregate], [i], color=color, marker="D", s=55, edgecolor="black", linewidth=.5)
            ax.axvline(0, color="#444", linewidth=.8)
            ax.axvline(5, color="#888", linestyle="--", linewidth=.8)
            ax.set(title=f"{column} / {protocol}", xlabel="Relative AULC gain (%) → better")
            ax.set_yticks(range(len(candidates)), [x[2] for x in candidates])
            ax.grid(axis="x", alpha=.15)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Paired seed effects: improvement over affine does not establish added value\nDots: five seeds; diamonds: relative gain of means; dashed line: 5% reference", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(plots/"paired_aulc_effects.png", dpi=180)
    fig.savefig(plots/"paired_aulc_effects.pdf")
    plt.close(fig)

    text = f"""# 结果解释与停止决定

**`{decision['decision']}`**。两个预注册诊断完成 120/120 contexts、960 条方法评估记录，未追加模型。新分支基于 `61f20c9`。

当前证据不支持把更复杂映射或共享模型作为下一阶段主线。尤其不能只引用 shared 相对 affine 的正结果：较强的 scale-only 和独立 shrinkage 对照解释了大部分表面收益。

## 非线性映射：没有足够证据支持

单独 monotone spline 的 compound 三柱平均 AULC 比 affine 恶化 {-gain('monotone_spline', 'affine'):.2f}%，三个柱都没有达到相对 affine 的稳定/material 门槛。validation 允许回退到 scale/affine 后，nonlinear policy 相对 affine 改善 {gain('nonlinear_policy', 'affine'):.2f}%，但比 scale-only 恶化 {-gain('nonlinear_policy', 'scale_only'):.2f}%，比同样使用 validation 的 linear policy 恶化 {-gain('nonlinear_policy', 'linear_policy'):.2f}%。因此不能把回退到简单模型带来的改善归因于非线性。

## 跨柱共享：相对 affine 有收益，但未超过简单对照的实际意义门槛

compound 总预算 portfolio AULC：affine {mean['affine']:.4f}、scale-only {mean['scale_only']:.4f}、独立 identity shrinkage {mean['local_identity_shrinkage']:.4f}、shared {mean['shared_column_affine']:.4f}。shared 相对 affine 改善 {gain('shared_column_affine', 'affine'):.2f}%，但相对 scale-only 只有 {gain('shared_column_affine', 'scale_only'):.2f}%，相对独立 shrinkage 只有 {gain('shared_column_affine', 'local_identity_shrinkage'):.2f}%。row portfolio 相对 affine 的改善也只有 {100*(1-row['shared_column_affine']/row['affine']):.2f}%。

{markdown_table(comparisons, index=False)}

25g/40g compound 的 shared 相对 affine 赢 5/5、4/5 seeds；但相对 scale 只改善 3.93%、1.84%，相对 local shrinkage 分别恶化 1.34%、改善 4.58%（仅 1/5、3/5 wins）。8g shared 还弱于 scale/head。没有柱通过全部预注册对照，更谈不上跨两个柱稳定保留。此结果允许说“partial pooling 能缓解部分 affine 拟合不稳定”，不允许说“已经发现额外的跨柱可迁移结构”。

预算比较单位是三柱 portfolio，每柱 B=30/50/70/100，总计划 90/150/210/300。所有 portfolio 方法使用相同购买清单并计入 validation；compound 的实际总预算为 88–299，按具体 seed/budget 原样保留。共享模型的逐柱性能不能被称为仅消耗 B 标签。focal compound validation/test 在所有供体训练中隔离；未新增标签、未重划 split。

## Compound AULC（五 seed 均值；越低越好）

{markdown_table(comp[methods])}

## 剩余 absolute error 没有实质解决

预算 100 的 compound 指标如下（RMSE/MAE 单位 mL）。25g 的 V1/V2 RMSE 仍约 14/20，40g 仍约 34/41；共享的 AULC 改善没有转化成高预算下的大幅 absolute-error 降低。

{markdown_table(point)}

这两项诊断并不能唯一地区分“sum readout 限制”和“信息量/噪声限制”，也不能证明一切 nonlinear mapping 都无效。当前拒绝的是所测两个低容量机制作为下一条复杂模型主线；**不是证明 residual 是不可约噪声，也不是自动获得 readout 改造依据**。Adaptive readout 在历史中未真正测试，本轮亦未训练；不输出 `ADAPTIVE_READOUT_WARRANTED`。

## 数据需要与边界

执行本轮有限诊断不需要额外数据。继续做机制辨别或确认增益则优先需要：独立实验批次/compound 的确认集；相同条件下的重复实验来估计测量误差；同 mass 多 flow、同 flow 多 mass 的交叉设计；高 source-q50 尾部覆盖。当前仅三个规格且 flow 各自恒定，mass/flow 效应无法分离。target-compound holdout 不代表 source-unseen OOD；后者需要更多真正未见分子。

所有 test 均是历史已使用的冻结开发性评估。knots/scalers/coefficients 仅由 train 拟合，validation 选固定候选，全部预测冻结后执行本轮 test 评估；未用 test 调参或追加方法。

## 复现与证据

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/transfer-diagnostics-mpl conda run --no-capture-output -n fish python scripts/studies/run_next_transfer_diagnostics.py --run
MPLCONFIGDIR=/tmp/transfer-diagnostics-mpl conda run --no-capture-output -n fish python scripts/studies/summarize_next_transfer_diagnostics.py
```

复现依赖原 qualified source runtime checkpoint，其哈希由 protocol 锁定；不允许用重新训练的随机 checkpoint 静默替代。已有 predictions/fit audits 全部可跟踪，源码与协议变化会停止复用。

- [预实验审计](../../../NEXT_TRANSFER_MODEL_AUDIT.md)
- [完整指标报告](NEXT_TRANSFER_DIAGNOSTICS_REPORT.md)：R²、RMSE、MAE、arithmetic/RMS NRMSE、AULC 与 seed stability。
- [逐 seed/预算指标](all_metrics.csv)、[全部配对对照](paired_aulc.csv)、[actual-budget AULC](portfolio_aulc.csv)。
- [标签预算与供体排除](label_usage_summary.csv)、[validation 选择](selection_summary.csv)、[baseline 数值复现](baseline_reproduction.csv)。
- [执行审计](execution_audit.json)：0 failures；source/baseline/splits hash 不变；float32/float64 复现最大差 {audit['maximum_baseline_reproduction_error_ml']:.8f} mL。

![Paired AULC effects](plots/paired_aulc_effects.png)
"""
    (STUDY/"RESULT_INTERPRETATION.md").write_text(text)


if __name__ == "__main__":
    main()
