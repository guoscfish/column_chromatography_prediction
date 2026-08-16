# E0-3c：4g→8g 损失尺度对照

## 目的

在 E0-3b 已确定的“4g pretrained + 末两层和预测头微调 + learning rate 1e-4”框架内，只改变 V1/V2 训练损失的尺度处理，判断原代码的 V2 权重0.5是否需要替换。

本阶段不是主动学习实验。test 不参与早停、checkpoint选择或配置排名。

## 三种损失

每个目标均使用原QGeoGNN的 q10 pinball、q50 MSE、q90 pinball和两项quantile crossing惩罚。

1. `legacy_0.5`：`loss_V1 + 0.5 × loss_V2`，对应原代码。
2. `equal_1.0`：`loss_V1 + loss_V2`。
3. `train-SD normalized`：先分别用当前8g train split的V1/V2标准差缩放真值和预测，再计算两个等权目标损失。valid/test不参与标准差估计。

其他条件完全相同：当前仓库8g阈值过滤后552行、重复实验保留、4g source scaler、相同图缓存、相同split、相同初始化seed、最多500 epoch、patience=100、batch size=2048。

矩阵为3种损失 × 3 seeds（42/525/1101）× 2 splits（row/compound），共18组。

## 结果

以下为跨6个paired运行的均值。normalized score越低越好。

| 损失 | validation score | test score | test R² V1 | test R² V2 | test MAE V1 | test MAE V2 | V1 coverage | V2 coverage | crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy 0.5 | 0.619 | 0.502 | 0.687 | 0.765 | **2.366** | 4.347 | 0.722 | 0.519 | 0.180 |
| equal 1.0 | **0.604** | **0.498** | **0.692** | **0.766** | 2.438 | 4.421 | **0.741** | 0.505 | 0.177 |
| train-SD normalized | 0.611 | 0.500 | 0.684 | 0.764 | 2.380 | **4.258** | 0.709 | **0.549** | **0.080** |

`equal 1.0`的平均validation改善为-0.015，相对legacy在6个split/seed中赢4次；`train-SD normalized`改善为-0.008，也赢4次。把两个split先在每个seed内平均后，paired 95% t区间分别为[-0.060, 0.030]和[-0.024, 0.009]，均跨0。只有3个独立seeds，不能声称统计显著优于legacy。

## 分split解读

- row：standardized validation最低（0.421），equal为0.427，legacy为0.438。
- compound：equal validation最低（0.781），legacy为0.800，standardized为0.802。
- compound点预测三者几乎相同；standardized的平均crossing从legacy的0.229降到0.023。
- row/seed=1101三种配置都接近500 epoch，并且coverage较差、crossing较高，说明该split仍是训练稳定性和UQ难例。

源行224继续保留在正式结果中。排除它只用于敏感性诊断，不改变上述模型选择。

## 决定

- 按预先写入配置的“跨paired运行平均normalized validation最低”规则，`equal 1.0`成为下一阶段的暂定损失候选。
- 它相对legacy的效应很小且区间跨0，因此不表述为性能显著提升；legacy 0.5继续作为原代码对照。
- standardized loss没有稳定改善点预测，但显著减少平均quantile crossing并略改善V2 MAE/coverage，保留到E0-4不确定性与单调分位数实验中，不作为当前点预测主损失。

## 可复现性与文件

legacy 6组从头重跑与E0-3b的validation、test、R²和best epoch逐项完全相同。E0-3c直接引用E0-3b的6个冻结split，不再保存重复副本。

- `comparison.csv`：18组逐配置结果。
- `paired_summary.csv`：按split汇总的3-seed均值和标准差。
- `paired_effects_vs_reference.csv`：相对legacy的逐seed、逐split配对差值。
- `test_predictions.csv.gz`：逐行test预测。
- `training_histories.csv.gz`：逐epoch轨迹。
- `sensitivity_excluding_source_row_224.csv`：非正式的数据敏感性结果。
- `config.json`、`environment.json`、`artifact_manifest.json`：冻结协议、环境和校验和。

