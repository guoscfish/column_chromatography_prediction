# F01–F03 与 F08 性能数值复核

## 比较口径

4g Row split，QGeoGNN-V2，combined validation NRMSE。每个 seed 的两个 RMSE 以本 seed 初始 L333 的固定尺度归一化，再取均方根。每个 seed 有 416 个开发验证行，训练预算只计训练行；不同 seed 的验证行可能重叠。各方法在同 seed 共用初始 L/U、checkpoint、验证口径，选择后的训练状态与历史路径各自发展。数值越低越好。

以保存的每个原始检查点计算梯形积分，再除以预算差，最后在 seed 间等权平均。每轮新增 32 行。五 seed 长窗每方法 185 个检查点（37 × 5）；两 seed Hybrid 窗每方法 72 个（36 × 2）；五 seed 短窗每方法 110 个（22 × 5）。图中样本标准差反映 seed 间离散度。

## 长窗 L333–1485（已完成的历史记录）

| 方法 | raw AULC | L1485 NRMSE | AULC 比 CW-LCMD 更低的 seed | 终点比 CW-LCMD 更低的 seed |
| --- | ---: | ---: | ---: | ---: |
| CW-LCMD32 | 0.406501 | 0.339061 | — | — |
| Historical CW-Free32 | 0.439159 | 0.334085 | 0/5 | 3/5 |
| New CW-Free32 | 0.454195 | 0.322703 | 0/5 | 4/5 |
| NoCW-Free32 | 0.445809 | 0.349299 | 0/5 | 1/5 |

Historical 的 157/6101 含 Astra/high 前缀，自选前 L525/L653 起 Sol/xhigh；157 的 r5 接受 ID 抄写修正，服务端模型身份未核实。其余三 seed 从 L333 使用 Sol/xhigh。New 与 NoCW 也为 Sol/xhigh，但历史、纠错及日期不同。NoCW 去掉显式 CW 字段和说明，保留点预测、分位宽度和其他数值条件。CW-LCMD32 使用中心/宽度几何，raw Gradient-LCMD32 使用原始 V1/V2 q50 梯度几何。五 seed 同向的 AULC 差与 New 的终点优势是不同统计量。此表提供整条历史轨迹的部分可比证据，无法识别 CW 字段的单独效应。

## 两 seed L333–1453（探索性）

| seed | CW16+LLM16 raw AULC | CW-LCMD32 raw AULC | CW16+LLM16 终点 | CW-LCMD32 终点 |
| --- | ---: | ---: | ---: | ---: |
| 157 | 0.389976 | 0.404365 | 0.311397 | 0.312361 |
| 6101 | 0.462717 | 0.482090 | 0.364172 | 0.383001 |
| 等权均值 | 0.426346 | 0.443228 | 0.337784 | 0.347681 |

Hybrid 在两个 seed 的全窗 AULC 较低，157 的终点接近持平。157 的 L333–525 阶段 AULC 差为 +0.043569，体现早期反例。Hybrid 的两条轨迹均有 Astra/high 前缀，选前 L813 起使用 Sol/xhigh；从该点继续已有训练历史。157 有 L1485，6101 截止 L1453。共同窗口固定为 L1453，后续相同状态的配对重训及统一 LLM 配置的五 seed 效果尚无此项证据。

## 短窗 L333–1005（已完成的历史记录）

| 方法 | raw AULC | L1005 NRMSE | 五 seed 中首次达到 NRMSE≤0.40 |
| --- | ---: | ---: | ---: |
| Numerical Hybrid32 | 0.438420 | 0.351918 | 4 |
| raw Gradient-LCMD32 | 0.462549 | 0.366328 | 5 |
| Historical CW-Free32 | 0.508802 | 0.349855 | 5 |
| New CW-Free32 | 0.532876 | 0.378752 | 3 |
| NoCW-Free32 | 0.497741 | 0.405025 | 3 |
| Random32 | 0.614742 | 0.513940 | 1 |

统一采用 20261112 综合诊断的共同记录窗口。Random32、raw Gradient-LCMD32 和 Numerical Hybrid32 对应历史 sequential_b32 的三个数值策略；New CW-Free 是独立 Sol/xhigh 重跑的已保存曲线。Numerical Hybrid 使用三初始化 q50 分歧筛选当轮最高 25% 后按 latent 覆盖取样；CW16+LLM16 为另一项历史策略。相同预算和已验证的初态对齐支持描述性性能比较，数据来源与各策略协议差异限制归因。

## 达到阈值需要多少训练标签

沿用 20261112 综合诊断定义中的 NRMSE≤0.40 作为描述性目标。首达取第一个实际达标检查点。连续三个检查点达标同时记录起点和第三个检查点的确认预算。跨阈值的单个观察点可能回退。未观察到达标保留至窗口末尾的右删失，避免把末位预算当作达标。新增采集标签数为总训练行数 L 减去初始 333 行。

| L333–1485 的方法 | 首达中位数（总训练行） | 已首次达标 | 已确认连续三个检查点 |
| --- | ---: | ---: | ---: |
| CW-LCMD32 | 717 | 5/5 | 5/5 |
| Historical CW-Free32 | 845 | 5/5 | 5/5 |
| New CW-Free32 | 909 | 5/5 | 5/5 |
| NoCW-Free32 | 941 | 5/5 | 4/5 |
| raw Gradient-LCMD32（历史参考） | 813 | 5/5 | 5/5 |

NoCW 6101 首达 L1453，截至 L1485 无法确认连续三点。New 2357 首达 L909，连续三点从 L1005 开始，于 L1069 确认。New 对 raw Gradient-LCMD32 的首达预算五 seed 均更晚。短窗仍右删失的 seed 见逐 seed CSV。

## 输入与检查

- 长窗与双 seed 采用 `studies/active_learning/llm_decision_behavior_audit_20261116/row_curves.csv`，逐点核对其引用的三个已封存 `validation_curves.csv`。裁决来源为该目录 `REPORT.md`、`PROTOCOL_AUDIT.md` 与 `studies/active_learning/decision_mechanism_followup_20261117/PHASE_PERFORMANCE.md`。
- 短窗读取 `studies/active_learning/comprehensive_history_diagnostics_20261112/tables/round_metrics.csv`，对照同目录 `tables/stage_metrics.csv` 和 `tables/threshold_crossings.csv`。阈值原定义见 `PROTOCOL_AND_FAIRNESS_AUDIT.md`。源表及封存清单 SHA256 见 [performance_qa.json](performance_qa.json)。
- 884 个原始点与各自归档 CSV 完全一致，1,046 个重叠点与早期审计表误差最大 3.18 × 10⁻¹⁴。54 个区间逐 seed 重新积分，以独立等距点权公式复核，最大差 1.11 × 10⁻¹⁶。132 个归档汇总检查最大差 4.77 × 10⁻¹⁵。54 条当前窗内的 method-seed 轨迹均保留至少一次原始误差上升，未做平滑或累计最优。
- 本次性能复算只读归档验证汇总和封存路径/哈希；使用的 CSV 不含 test 逐行数据或候选真实响应。

[原始曲线](performance_curves.csv) · [逐 seed 指标](performance_seed_metrics.csv) · [等权汇总](performance_summary.csv) · [成对差](performance_paired_differences.csv) · [阈值](performance_thresholds.csv) · [复算脚本](../scripts/audit_performance.py)
