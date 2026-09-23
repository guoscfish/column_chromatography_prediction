# 4g row active learning master report

本报告只整理已有 study 的冻结结果，不启动训练、不修改任何原始 study 文件。主图和主表统一使用两个共同 seed：`157`、`6101`。这样可以让 sequential、maxdet、kernel-IVR、center-width LCMD 以及探索性 study 在同一条可比的 seed 口径下汇报。

## 1. 本次整合使用的 study

| Study | 纳入方法 | 可用轨迹 | 用途 |
|---|---|---:|---|
| `qgeognn_v2_row_sequential_b32` | random、LCMD、hybrid | 333→1005 | 主图 |
| `qgeognn_v2_row_maxdet_b32` | gradient_maxdet | 333→1005 | 主图 |
| `qgeognn_v2_row_kernel_ivr_b32` | kernel_ivr | 333→1005 | 主图 |
| `qgeognn_v2_row_cw_lcmd_to_1005` | center_width_lcmd | 333→1005 | 主图 |
| `qgeognn_v2_row_fused_maxdet_b32` | gradient_latent_fusion_maxdet | 333→1005（本报告探索图截到 653） | appendix / exploratory |
| `qgeognn_v2_row_fused_maxdet_a080_b32_screen` | gradient_latent_fusion_a080_maxdet | 333→653 | appendix / exploratory |
| `qgeognn_v2_row_cw_ivr_portfolio_b32` | cw_ivr_portfolio | 333→653 | appendix / exploratory |

脚本对字段做了兼容：所有 study 都从 `results/learning_curve_metrics.csv` 读取，并把 `combined_normalized_RMSE`、`V1_RMSE`、`V2_RMSE` 和 `active_label_count` 统一到同一内部字段。不同 study 的附加列（如 `active_label_fraction_outer_train`）不参与跨 study 计算。`gradient_latent_fusion_maxdet` 在完整 fused study 中有 1005 结果，但按任务要求只在 exploratory 图中展示到 653；a=0.80 fusion 和 portfolio 没有 653 之后的轨迹，因此相应后段 AULC 和 1005 endpoint 在汇总表中留空。

## 2. 主图纳入方法及理由

主图 `figures/master_full_333_1005.png` 只放六种同时满足以下条件的方法：

1. 有完整的 333→1005 learning curve；
2. 至少有两个共同 seed（本报告固定为 157、6101）；
3. 指标定义、标签预算节点和 test 评估口径可直接对齐。

图中粗线是两个 seed 的均值，细线是对应 seed 的原始轨迹；虚线标记 429、525、653、1005 四个汇报节点。AULC 使用 active-label 作为横坐标的归一化梯形积分，数值越低越好。

## 3. 主图结论：完整轨迹

- `center_width_lcmd` 是完整 333→1005 区间的整体最佳方法，平均 `AULC_333_1005=0.589`，1005 点 `NRMSE=0.492`。
- `hybrid` 的整体 AULC 次优（0.606），并且在 525→653、653→1005 两个后期阶段都仅次于 center-width LCMD（分别为 0.612、0.532）。
- `kernel_ivr` 的整体 AULC 为 0.614，1005 点 NRMSE 为 0.512；它在 429→525 阶段优于 gradient_maxdet，在后段也保持较低水平。
- `gradient_maxdet` 在最早的 333→429 阶段最强（AULC=0.765），说明它能较快利用最初一批标签；但整体 333→1005 AULC（0.621）和后段 AULC（0.563）不如 center-width LCMD、hybrid、kernel-IVR。
- `LCMD` 的早期下降较慢，但 1005 点仍达到 NRMSE 0.519；其 V1 endpoint 很有竞争力（3.537）。
- `random` 在所有完整区间都明显落后，整体 AULC=0.781、1005 NRMSE=0.708，可作为非主动选择基线而不是候选主策略。

## 4. 分阶段结论

以下为两个共同 seed 的均值，AULC 越低越好：

| 区间 | 最佳 | AULC | 观察 |
|---|---|---:|---|
| 333–429 | Gradient MaxDet | 0.765 | 最快的初始收益 |
| 429–525 | Center-width LCMD | 0.650 | 从中早期开始取得稳定优势 |
| 525–653 | Center-width LCMD | 0.583 | 中期优势最清晰 |
| 653–1005 | Center-width LCMD | 0.516 | 后期继续保持第一 |
| 333–1005 | Center-width LCMD | 0.589 | 全程综合最佳 |

因此，汇报时应把结果描述为“gradient MaxDet 擅长最早阶段，center-width LCMD 在中后期和全程最稳定”，而不是只依据单个早期节点判断最终方法。

## 5. endpoint 结论（1005 labels）

主方法两 seed 均值如下：

| 方法 | NRMSE@1005 | V1_RMSE@1005 | V2_RMSE@1005 |
|---|---:|---:|---:|
| Center-width LCMD | **0.492** | **3.355** | 5.192 |
| Gradient MaxDet | 0.510 | 3.567 | **5.089** |
| Kernel IVR | 0.512 | 3.540 | 5.234 |
| LCMD | 0.519 | 3.537 | 5.505 |
| Hybrid | 0.527 | 3.698 | 5.324 |
| Random | 0.708 | 4.610 | 8.003 |

主结论指标 NRMSE@1005 由 center-width LCMD 最低。分量指标上，center-width LCMD 同时给出最低 V1 RMSE；gradient MaxDet 给出最低 V2 RMSE。因此如果汇报只需要一个 endpoint 主指标，使用 NRMSE@1005；如果需要解释分量差异，可补充“MaxDet 对 V2 更强，center-width LCMD 对总体和 V1 更强”。

## 6. exploratory methods 的定位

`gradient_latent_fusion_maxdet`、`gradient_latent_fusion_a080_maxdet` 和 `cw_ivr_portfolio` 放在 `figures/master_exploratory_333_653.png`，只用于解释 fusion / portfolio 在 333→653 的早中期行为，不作为完整主结论的替代。

- latent-fusion MaxDet 在 333→653 的均值轨迹可用于观察表示融合是否改变早期排序；由于它与主方法的完整主图定位不同，不在主图中竞争最终 1005 结论。
- a=0.80 fusion 和 CW-IVR portfolio 仅有 333→653 结果，不能外推 653→1005，也不能据此声称 1005 endpoint 优势。
- `master_summary.csv` 保留这些方法在可用区间的 per-seed AULC；不可用区间和 1005 endpoint 使用空值，避免把截断实验误读成完整轨迹。

## 7. 汇报时可直接口述的精炼总结

- 这次汇报整合的是已有 4g row active-learning 结果，没有新增训练。
- 为保证跨 study 可比，主结论统一使用共同 seed 157 和 6101。
- 主图覆盖 333 到 1005 labels，包含 random、LCMD、hybrid、gradient MaxDet、kernel IVR 和 center-width LCMD。
- 最早 333–429 阶段，gradient MaxDet 的 AULC 最低，初始收益最快。
- 从 429 labels 以后，center-width LCMD 的轨迹开始稳定领先。
- center-width LCMD 在 525–653、653–1005 和完整 333–1005 区间均为最佳。
- 完整区间平均 AULC 以 center-width LCMD 最低，为 0.589。
- 1005 labels 时，center-width LCMD 的 NRMSE 最低，为 0.492。
- V1 endpoint 也是 center-width LCMD 最低；V2 endpoint 则由 gradient MaxDet 最低。
- random 在所有阶段明显落后，适合作为基线而不是候选主策略。
- fusion 和 portfolio 目前只到 653，应作为 exploratory / appendix 证据，不外推 1005。
- 因此主图和主表已经可以直接用于汇报，不需要任何额外训练。

## 输出文件

- `figures/master_full_333_1005.png`
- `figures/master_stage_aulc.png`
- `figures/master_endpoint_1005.png`
- `figures/master_exploratory_333_653.png`
- `results/master_summary.csv`
- `scripts/studies/build_qgeognn_v2_row_master_report.py`

