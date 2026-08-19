# E2 4g Active Learning — Row 3-seed Pilot

## 状态

本目录是正式E2 row协议结果，不与`e2_random_smoke`混淆。3个paired outer seeds、4种策略、K=3、L0=375、B=25、8轮均已完成；每个预算点都从seeded random source-free初始化重新训练，固定L0-train scaler与validation。

平均normalized AULC（越低越好）：

```text
strategy
hybrid      0.542938
coverage    0.562489
ensemble    0.626849
random      0.644749
```

- 最低平均AULC：**hybrid**。
- 平均AULC优于Random的策略：**coverage, ensemble, hybrid**。
- Hybrid平均AULC同时优于Coverage和Ensemble：**True**。
- best_epoch>=490比例：**0.3%**；是否触发单独convergence decision：**False**。

## 解释边界

Round-0离线signal诊断只回答E1信号能否迁移到source-free regime，不是主动学习结论。正式科学比较来自3-seed完整learning curve与paired AULC。4g canonical数据历史上已删除60/120 mL tail，因此E2不解释tail acquisition；该机制留到E4 no-threshold 8g。Quantile Width保留在离线诊断与后续E4 Protocol B legacy baseline中，但不是E2第五种策略。

完整数值见`round_metrics.csv`、`aulc_summary.csv`、`paired_effects.csv`、`label_efficiency.csv`、`round0_signal_diagnostics.csv`、`signal_agreement.csv`、`queried_batch_diagnostics.csv`与`convergence_audit.csv`。
