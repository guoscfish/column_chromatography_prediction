# E2 4g Active Learning — Compound 3-seed Pilot

## 状态

本目录是正式E2 compound协议结果，不与smoke/preflight混淆。3个paired outer seeds、4种策略、K=3、L0=375、B=25、8轮均已完成；每个预算点都从seeded random source-free初始化重新训练，固定L0-train scaler与validation。

平均normalized AULC（越低越好）：

```text
strategy
hybrid      0.761203
coverage    0.777107
random      0.788732
ensemble    0.802149
```

- 最低平均AULC：**hybrid**。
- 平均AULC优于Random的策略：**coverage, hybrid**。
- Hybrid平均AULC同时优于Coverage和Ensemble：**True**。
- best_epoch>=490比例：**5.4%**；是否触发单独convergence decision：**False**。
- Compound Gate：**suggestive**；3/3稳定优于Random：**无**。

## Secondary diagnostics

- Compound-macro AULC均值：hybrid 0.587, coverage 0.605, ensemble 0.637, random 0.641；与row-weighted primary方向一致。
- Round-0 Quantile/Latent/Ensemble mean Spearman为0.591/0.523/0.511，均3/3为正。Quantile只保留为future post-hoc control，本阶段未运行第五条AL曲线。
- Ensemble每批平均8.50个unique compounds、max-per-compound 6.00、HHI 0.158；Hybrid分别为21.96、2.17、0.051。
- Fixed-reference Hybrid−Ensemble mean pairwise distance差为+2.444（3/3 seeds为正）；Coverage−Ensemble为+2.621（3/3）。Native与fixed-reference方向一致，但不是因果证明。
- 297个独立fits中16个命中`best_epoch>=490 OR hit_max_epoch`（5.39%），无convergence problem。

## 解释边界

Round-0离线signal诊断只回答E1信号能否迁移到source-free regime，不是主动学习结论。正式科学比较来自3-seed完整learning curve与paired AULC。4g canonical数据历史上已删除60/120 mL tail，因此E2不解释tail acquisition；该机制留到E4 no-threshold 8g。Quantile Width保留在离线诊断与后续E4 Protocol B legacy baseline中，但不是E2第五种策略。

完整数值见`round_metrics.csv`、`aulc_summary.csv`、`paired_effects.csv`、`label_efficiency.csv`、`compound_macro_metrics.csv`、`compound_macro_aulc.csv`、`common_reference_batch_diversity.csv`、`compute_cost_summary.csv`与`convergence_audit.csv`。
