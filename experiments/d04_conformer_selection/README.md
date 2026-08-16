# D04：QGeoGNN构象选择对照

## 问题与控制条件

原代码实际流程是ETKDGv3嵌入并优化10个构象后，用默认第一个构象构图；论文方法文字要求选择最低能构象。本实验只改变构象选择，其他条件固定：

- 当前仓库4g/8g canonical数据；原代码60/120 mL阈值；重复实验保留；
- 相同RDKit seed、10个ETKDGv3构象、MMFF94，缺参数的1个结构按原规则UFF回退；
- 4g使用相同row seed=42 split、legacy source损失、训练预算和scaler规则重新训练；
- 8g使用E0-3b冻结的3 seeds × row/compound splits；
- 迁移范围固定为末两层+预测头、学习率1e-4，损失固定为E0-3c暂定的V1/V2等权；
- checkpoint只由validation选择，test不参与训练或早停。

## 构象与图审计

- 4g共217个唯一结构，217/217最低能构象构图成功；8g的88个结构中87个复用4g最低能图，1个单独生成。
- 默认第一构象只有19/217恰好是最低能构象。
- 198/217个结构的键长或键角相对legacy发生变化；另外19个本来就是最低能构象。
- 2D/静态descriptor 217/217完全不变，说明实验没有混入descriptor变化。
- 216个结构使用MMFF94，1个沿用UFF fallback。
- 21个结构中共有50/2170次构象优化返回非收敛状态；原代码仍会保留这些构象，论文未说明如何处理，因此登记为后续方法边界。

共享图模块用5个结构回归测试：`first_embedded`生成的所有图数组与原缓存完全相同。

## 4g source结果

| 构象 | best epoch | validation score | test R² V1/V2 | test MAE V1/V2 | test coverage V1/V2 | crossing |
|---|---:|---:|---:|---:|---:|---:|
| 默认第一构象 | 91 | 23.562 | 0.867 / 0.903 | 1.604 / 2.868 | 0.832 / 0.616 | 5.04% |
| 最低能构象 | 72 | **22.013** | 0.853 / 0.893 | 1.690 / 2.932 | 0.827 / **0.655** | **0.96%** |

最低能构象改善了4g validation、V2 coverage和crossing，但4g test点预测略差，不能称为全面提升。

## 4g→8g配对结果

跨6个paired运行的平均结果：

| 构象 | validation score | test score | test R² V1/V2 | test MAE V1/V2 | coverage V1/V2 | crossing |
|---|---:|---:|---:|---:|---:|---:|
| 默认第一构象 | 0.604 | **0.498** | **0.692 / 0.766** | **2.438 / 4.421** | 0.741 / 0.505 | **17.7%** |
| 最低能构象 | **0.592** | 0.582 | 0.608 / 0.723 | 2.897 / 4.700 | **0.786 / 0.510** | 20.6% |

最低能减去第一构象的平均配对差：

- validation normalized score：-0.012；
- test normalized score：+0.084；
- test MAE：V1 +0.460 mL，V2 +0.279 mL；
- test crossing：+2.9个百分点。

最低能validation在6组中赢4组，但按两个split先在seed内平均后的95% paired t区间为[-0.063, 0.039]，跨0。test只有row seed=1101的综合分数改善；compound三个seed均变差。row crossing从13.7%降到6.0%，但compound crossing从21.7%升到35.2%，说明UQ影响也不稳定。

## 决定

1. 最低能构象没有形成跨split的稳定收益，当前不替换原代码的第一构象主口径。
2. 后续主实验继续使用`first_embedded`以维持代码口径；最低能构象作为论文口径敏感性结果保留。
3. 不能把论文与本实验指标直接混为同一复现：论文没有提供固定构象seed、非收敛处理或对应split。
4. 在主动学习前仍需解决分位数单调性；构象切换本身不能稳定解决crossing。

## 文件说明

- `conformer_audit_4g.csv`、`conformer_audit_8g.csv`：能量、构象选择、力场、几何变化和失败状态。
- `graph_cache_4g_lowest_energy.pt`、`graph_cache_8g_only_lowest_energy.pt`：独立最低能缓存，不覆盖legacy缓存。
- 最低能4g checkpoint只在`.work`中用于本轮迁移，结论不采用该source，因此不长期保留；descriptor和split未变，scaler直接引用`../e0_4g_baseline/scaler.json`。
- `metrics_4g_lowest_energy.json`、`history_4g_lowest_energy.csv`：4g指标和训练轨迹。
- `comparison_4g.csv`：两种构象的4g结果。
- `comparison_8g.csv`：两种构象、6个paired上下文的12行完整结果。
- `paired_effects_8g.csv`、`summary_8g.csv`：逐上下文差值和分split汇总。
- `predictions_8g_lowest_energy.csv.gz`、`histories_8g_lowest_energy.csv.gz`：最低能8g逐行预测和训练轨迹。
- `config.json`、`environment.json`、`artifact_manifest.json`：合并后的统一协议、环境与校验信息。
