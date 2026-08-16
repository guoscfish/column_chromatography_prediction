# G0-3：Legacy threshold vs no-threshold

## 状态与决定

本实验只改变8g的`V1≤60 mL, V2≤120 mL`过滤。两个协议共享阈值分层的row/compound splits、4g anchor、单调头和训练预算。阈值决定只使用validation；test仅作最终报告。

Validation-only决定：**no_threshold**。no-threshold相对legacy的common validation normalized RMSE变化为-3.4%，tail validation normalized absolute error变化为-23.4%。

独立test不完全复现tail改善：no-threshold的common normalized RMSE仅变化+1.3%，但tail normalized absolute error平均恶化+40.1%，且在5/6个paired contexts中更差。由于threshold决定已预注册为validation-only，这不用于反向改选；它被登记为小tail test高方差和尾部拟合不稳定的失败模式。

## 独立test汇总

| slice | protocol | normalized RMSE | normalized absolute error | V1/V2 calibrated coverage | width-error Spearman | AUCE |
|---|---|---:|---:|---:|---:|---:|
| common | legacy | 0.541 | 0.605 | 0.768/0.756 | 0.382 | 0.071 |
| common | no threshold | 0.548 | 0.635 | 0.839/0.762 | 0.413 | 0.068 |
| tail | legacy | 1.980 | 3.367 | 0.583/0.583 | 0.333 | 0.322 |
| tail | no threshold | 2.599 | 4.716 | 0.583/0.583 | 0.500 | 0.352 |

Tail只占test的约4.3%，但在calibrated width最高10%中，legacy/no-threshold分别富集8.66×/8.11×，说明legacy阈值确实会预先删除主动学习最可能查询的难例。不要按单个R²解释本实验；`slice_metrics.csv`保留full/common/tail的逐seed结果。

## 产物

- `canonical_8g_no_threshold.csv`与`splits/`：574行完整数据和阈值分层paired splits。
- `training_comparison.csv`：12次配对训练。
- `predictions.csv.gz`、`calibration_factors.csv`：逐样本预测和validation-only校准。
- `slice_metrics.csv`、`paired_effects.csv`：尾部误差、校准、signal-error关系。
- `high_uncertainty_diagnostics.csv`：top-10% tail enrichment与hard-error overlap。
- `threshold_decision.json`：完全基于validation的冻结决定。
