# G0-4：Paper-style transfer qualification

> **已被修正版取代。** 本目录只保留“新增柱RBF适配器随机初始化会破坏4g source初始函数”的实现诊断，不用于G0-4科学结构选择。最终零初始化版本与validation-only决定位于`experiments/g0_4_paper_style_transfer/`。

## 状态与决定

本实验在G0-3冻结的574行no-threshold 8g数据、3 seeds × row/compound paired splits上，只比较`last2+head`、`full fine-tune`和`paper-style`。三者共享source scaler、单调分位数头、V1/V2等权loss、`lr=1e-4`和validation-best checkpoint；test不参与结构选择。

Validation-only决定：**last2_head**。paper-style相对last2的平均validation normalized score变化为+160.9%，赢0/6个paired contexts。预注册规则为：平均至少改善2%、至少赢4/6且任一split-mode均值不恶化超过5%；否则保留last2。

## 三种实现

- `last2_head`：复用G0-3完全同协议的冻结checkpoint。
- `full_finetune`：加载4g权重与单调头后更新全部参数。
- `paper_style`：把仓库预留的`column_dia/column_len/column_den`追加到Graph G每条edge；加载所有形状兼容的4g参数；从source q50与分位差初始化新的单调输出头；更新新增column RBF adapters、末两层GNN和head。

8g柱规格沿用仓库原始实现中的`(1.5, 13.2, 0.4458)`；4g参照值为`(1.5, 6.6, 0.4458)`。论文明确要求新柱规格输入、输出层更新及`1e-4`微调，但没有给出逐层冻结图，因此本实验的adapter+last2范围是克制且显式记录的实现选择，而非声称逐行复刻论文。

## 汇总

平均normalized score（越低越好）：

| config | validation | test |
|---|---:|---:|
| last2_head | 0.2559 | 0.4506 |
| full_finetune | 0.3326 | 0.4947 |
| paper_style | 0.6678 | 1.0469 |

Calibration只在各run的validation估计per-target inflation；完整full/common/tail结果见`summary.csv`和`slice_metrics.csv`。
