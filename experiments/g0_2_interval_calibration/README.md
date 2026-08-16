# G0-2：Validation-only 区间校准

## 结论

本实验只使用每个 split/seed 的 validation 预测估计 V1/V2 缩放因子，然后将冻结因子原样应用于 test。test 未参与 alpha 选择。输入为 G0-1 通过 Gate 后的 `monotonic_softplus` 模型。

| target | alpha median [min,max] | test coverage before→after | test width before→after | test AUCE | crossing after |
|---|---:|---:|---:|---:|---:|
| V1 | 1.396 [1.000, 16.106] | 0.709→0.824 | 7.372→11.517 | 0.112 | 0.000 |
| V2 | 1.709 [1.474, 55.909] | 0.519→0.854 | 6.063→18.453 | 0.048 | 0.000 |

`alpha_80` 使用有限样本 split-conformal 顺序统计量，并限制为不小于1（只放大、不收缩）。AUCE 使用 validation 上各 nominal coverage 的 conformal factor 构成校准曲线，再在独立 test 上评价。

校准把平均 test coverage 拉回名义值附近，但 row/seed=1101 因原始区间塌缩，需要 V1/V2 alpha=16.11/55.91；因此不能把跨 seed 的 raw/calibrated quantile width 直接当成稳定 acquisition score。该失败模式必须带入 E1 的 signal-error qualification。

## 产物

- `calibration_factors.csv`：每个 split/seed/target 的 validation-only alpha。
- `metrics.csv`、`summary.csv`：test 最终 coverage、width、AUCE、crossing。
- `calibration_curves.csv`：0.1～0.9 nominal coverage 的独立 test 校准曲线。
- `test_predictions_calibrated.csv.gz`：应用冻结 alpha 后的逐样本 test 区间。
