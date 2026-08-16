# G0-1：分位数单调性对照

## 状态

本目录是正式配对实验。比较只改变分位数输出形式；数据、split、4g anchor、迁移范围、学习率、损失权重和 checkpoint 选择规则保持一致。test 不参与模型选择。

## 协议

- Legacy：独立 q10/q50/q90 输出，并保留 crossing penalty。
- Monotonic：`q50=m`、`q10=m-softplus(d_low)`、`q90=m+softplus(d_high)`。
- 迁移：4g pretrained，last2+head，`lr=1e-4`，V1/V2 等权。
- 选择：仅按 validation 的 train-variance normalized q50 score 选择 checkpoint。
- 正式冻结标准：validation normalized score 与 V1/V2 RMSE 均不比 legacy 平均恶化超过 5%，且 crossing 为 0。

## 当前汇总

| 参数化 | valid normalized | test normalized | valid crossing | test crossing | test V1/V2 coverage |
|---|---:|---:|---:|---:|---:|
| Legacy | 0.6040 | 0.4981 | 0.1361 | 0.1767 | 0.741 / 0.505 |
| Monotonic | 0.6082 | 0.5029 | 0.0000 | 0.0000 | 0.709 / 0.519 |

## Gate 判定

G0-1：**通过**。Monotonic 相对 legacy 的 validation 变化为 normalized score +0.70%、V1 RMSE +0.57%、V2 RMSE +0.11%，均低于预注册的5%恶化容忍线；结构性单调头在全部 validation/test 上 crossing 为0。模型选择只使用 validation，test 仅作最终报告。

## 产物

- `comparison.csv`：逐 split/seed/config 指标。
- `summary.csv`：跨运行均值和标准差。
- `paired_effects_vs_legacy.csv`：配对差值。
- `predictions.csv.gz`：validation/test 的逐样本预测，为 G0-2 validation-only calibration 保留。
- 各运行目录中的 checkpoint 与 history：可复核 checkpoint 选择和继续 G0-2。
