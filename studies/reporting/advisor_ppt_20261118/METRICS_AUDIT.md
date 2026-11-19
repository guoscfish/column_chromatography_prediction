# 导师汇报指标核验

2026-11-18 · 4g 柱色谱主动学习 · 历史开发证据

逐项数值、方法版本、样本数、seed、预算、行级/条件对口径、评价集合、冻结状态、可比性和限制见 [metrics_source.csv](metrics_source.csv)。逐图数据及 SHA256 见 [FIGURE_MANIFEST.md](FIGURE_MANIFEST.md)。**已完成**指归档试验或指定的历史核验已完成；**探索性**指可审计的机制或初步轨迹；**尚未完成**指该效果问题没有完整合格的验证结果。

## 共同指标和公平范围

- 4g Row split 每 seed 有 3330 个外层训练行、416 个开发验证行。标签预算 L 只计训练行，L333 是共同起点。新增采集数为 **L−333**。不同 seed 的数据划分重叠。
- 合并 NRMSE 为 `sqrt(((RMSE_V1/s1)^2 + (RMSE_V2/s2)^2) / 2)`，`s1/s2` 是各 seed 初始 L333 的固定尺度。V1/V2 的保留体积误差用 mL，NRMSE 无量纲。验证集参与各轮 checkpoint 选择。
- raw AULC 是原始 NRMSE 相邻点的梯形面积之和除以标签跨度。每 seed 先积分，再等权平均。含所有有记录的非单调点，每轮 32 行。每条轨迹都有误差回升。长窗 37 点、融合共同窗 36 点、短窗 22 点。可比性来自共同起点、匹配 seed 和共同窗口，选择后训练集与 checkpoint 分叉。
- CW-LCMD 在 center/width 梯度几何上选点。raw Gradient-LCMD 使用 V1/V2 的 q50 梯度几何。数值 Hybrid32 是三初始化预测分歧筛选后做 latent 覆盖。CW16+LLM16 先锁定 16 个 CW-LCMD 点，再由 LLM 补 16 个。
- 历史 Free 的 157/6101 先使用 Astra/high，分别从选前 L525/L653 用 Sol/xhigh。157 r5 接受一次已有授权的 ID 抄写修正，最终服务模型身份无法核验。New CW-Free 和 NoCW-Free 为历史独立轨迹，日期、上下文、纠错及状态有差异。NoCW 仍有预测、分位宽度、结构条件与历史反馈。

## F01、F02、F08｜五 seed 性能，已完成

每项为同窗五 seed 等权均值。长窗四法采用 11 月 16 日的完整归档，短窗采用综合诊断统一的 L333–1005。后者把独立运行的新 CW-Free 与历史数值轨迹合并为开发比较。

| 共同窗 | 方法 | raw AULC | 终点 NRMSE |
| --- | --- | ---: | ---: |
| L333–1485 | CW-LCMD32 | 0.406501 | 0.339061 |
| L333–1485 | Historical CW-Free32 | 0.439159 | 0.334085 |
| L333–1485 | New CW-Free32 | 0.454195 | 0.322703 |
| L333–1485 | NoCW-Free32 | 0.445809 | 0.349299 |
| L333–1005 | Numerical Hybrid32 | 0.438420 | 0.351918 |
| L333–1005 | raw Gradient-LCMD32 | 0.462549 | 0.366328 |
| L333–1005 | Historical CW-Free32 | 0.508802 | 0.349855 |
| L333–1005 | New CW-Free32 | 0.532876 | 0.378752 |
| L333–1005 | NoCW-Free32 | 0.497741 | 0.405025 |
| L333–1005 | Random32 | 0.614742 | 0.513940 |

**成对方向。** 长窗 CW-LCMD 的 AULC 对三种 Free 均 5/5 更低。New CW-Free 的 L1485 终点对 CW-LCMD 4/5 更低，2357 高 0.001842。综合诊断所述 New CW-Free 终点 5/5 更低对应 raw Gradient-LCMD（长窗对照 raw LCMD 0.417049 / 0.347582）。比较对象不同。 [raw LCMD 长窗单列复算](sources/performance_raw_long_reference.csv)

**标签预算。** 沿用综合诊断的描述性 NRMSE≤0.40 门槛。首次达到取实际观测的 L。持续达到同时报告连续 3 个达标点的首点和第 3 点才可确认的预算。未达到保留右删失。

| 长窗方法 | 5 seeds 首达中位数 L | 新增标签中位数 | 截至 L1485 首达 | 连续 3 点已确认 |
| --- | ---: | ---: | ---: | ---: |
| CW-LCMD32 | 717 | 384 | 5/5 | 5/5 |
| Historical CW-Free32 | 845 | 512 | 5/5 | 5/5 |
| New CW-Free32 | 909 | 576 | 5/5 | 5/5 |
| NoCW-Free32 | 941 | 608 | 5/5 | 4/5 |

NoCW seed 6101 首达 L1453，持续确认在 L1485 右删失。短窗至 L1005，Numerical Hybrid、raw Gradient-LCMD、New CW-Free、Random 首达分别为 4、5、3、1 个 seed。全部门槛及逐 seed 的右删失状态见 [阈值明细](sources/performance_thresholds.csv)。原始点、两种积分复核、历史记录交叉对照及其他比较见 [性能证据](sources/PERFORMANCE_NOTES.md)。

## F03、F04｜融合，探索性

| L333–1453，两个 seed | raw AULC | L1453 NRMSE |
| --- | ---: | ---: |
| CW16+LLM16 | 0.426346 | 0.337784 |
| 同 seed CW-LCMD32 | 0.443228 | 0.347681 |

融合两条轨迹的 AULC 均较低。157 的终点差只有 −0.000964，且 L333–525 的阶段面积略高于配对数值轨迹（+0.043569）。历史融合两 seed 都从选前 L813 切换 Astra/high 到 Sol/xhigh。只有 157 到 L1485，共同窗止于 L1453。

F04 核验 71 个已保存历史 Hybrid 状态的 LCMD32 重放。真实 pending 前 16 与重放顺序 71/71 严格相同，实际 LLM16 与接受选择一致。以下比较每状态固定前 16，只更换后 16。体积对照占比的比较基准为当时历史 H 加相同前 16。CW 边际覆盖为后 16 带来的固定参考到最近中心平均距离下降，占当时 H 基准距离的比例。先算每状态比例，再在 seed 内平均。

| seed | 历史状态数 | LLM 后16 上样体积对照 | LCMD 后16 上样体积对照 | LLM 后16 CW 边际覆盖 | LCMD 后16 CW 边际覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 157 | 36 | 262/576（45.49%） | 0/576 | 1.52% | 5.01% |
| 6101 | 35 | 293/560（52.32%） | 0/560 | 1.17% | 5.33% |

上样体积对照 67 状态 LLM 更多、4 个相同。CW 增量为 LLM 较低 52、相同 1、较高 18。参照仅为 H 时，LLM 上样体积对照分子相同且仅该条件变化的名额为 253/576、282/560，LCMD 仍为零；数值后 16 也可以形成 PE/EA 对照。71 个状态包含相邻轮次，分母名额不能当独立实验重复。此重放只识别选样时的几何和结构分工。完整 LLM 融合效应、后 16 的反事实表现与独立五 seed 确认尚未完成。 [逐状态](sources/f04_replay_states.csv) · [逐 seed](sources/f04_replay_seed_summary.csv)

## F05｜纠错与方案保真

**已完成历史审计。** 540 个接受批次含 1 个已有授权的非前缀人工修正。127/540（23.52%）接受批为候选表前 32 行集合，严格原答子集为 127/539（23.56%）。Historical、New、NoCW 分别 25/180、40/180、62/180。127 批首份可解析方案均非前缀，全部先触发校验拒绝，首错为 123 个候选身份问题、4 个 packet 绑定问题。短路校验的首错只说明最先触发的错误。历史接受前缀保留首案合法唯一 ID 事件 440/3,634（12.11%）。

**探索性离线修复。** 对同 127 批，局部草稿冻结原合法唯一 ID 3,634 个，占 4,064 个目标名额的 89.42%。余 430 个空额，27 个批次仍需证据成员资格审查，完整合法 **0/127**。保留率分母与完整性分母不同。该记录只支持原选择保真和人工可审查性。没有自动 fallback 的证据。真实补位成功率、学习收益与服务商因果归因尚未获得。 [逐批口径](sources/f05_prefix_batches.csv) · [复核汇总](sources/f05_prefix_repair_summary.csv)

## F06、F07｜三套数据与各自边界

1. **既有已选反馈（探索性）。** New CW-Free 历史同批上样体积单因素对 165 个、93 个分子。选前旧 checkpoint 的预测对照选后已测反馈。按五 seed 等权，V1 真实/预测绝对变化 2.696/0.263 mL，V2 为 2.287/0.154 mL。整体以对为权重时为 V1 2.525/0.393，V2 2.370/0.235 mL。它只描述历史选中分布。 [权重口径](sources/selected_165_volume_pairs_summary.csv)
2. **原冻结训练 H（探索性，F06）。** L525 和 L1005 各有五个旧 checkpoint。只在同预算、同因素、同端点内配真值和预测。每端点 L525 上样体积/PE/EA 对 464/1,277，L1005 分别 1,393/4,054。L1005 真实/预测上样幅度 V1 为 3.367/0.220、V2 为 3.558/0.107 mL，PE/EA 对照为 10.224/9.819、22.172/21.259 mL。两预算 H 和对组成不同。 [F06 汇总](sources/f06_frozen_response_equal_seed.csv)
3. **15 次固定历史标签拟合（已完成，F07）。** 五 seed × 原模型、Amount Skip、Volume Skip。每 seed 从历史 H1005 隔离 10 个上样系列后从头拟合。每臂共 68 个隔离对，Row validation 每 seed 416 行。检查系列与原验证条件仍有部分关联，两 seed 各有一行精确同条件，移除相关系列的固定预测敏感性保留原方向。

| 旧模型 | 局部差分 MAE（mL） | 4g Row validation NRMSE |
| --- | ---: | ---: |
| 原模型 | 3.32233 | 0.39613 |
| Amount Skip | 3.33457 | 0.39453 |
| Volume Skip | 3.35214 | 0.39529 |

两种 skip 的局部平均误差分别上升 0.37% 和 0.90%。Row 平均误差降 0.40% 和 0.21%，各仅 3/5 非劣。均未通过原机制门槛（局部至少改善 20%、至少 4/5 改善、平均 Row 恶化不超过 1%）或 Row 资格（平均至少改善 1%，且至少 4/5 非劣）。支持“该两条简单 skip 不具备资格”，上样量通路压缩的唯一成因及更广泛架构的效果仍待检验。 [原始对照复算](sources/f07_control_seed.csv) · [门槛](sources/f07_control_gates.csv)

## F09、P12｜数值信号资格与状态依赖

- **F09 探索性。** New CW-Free 180 批、5,760 个选中行事件，选前 q10–q90 对照选后反馈，名义 80% 的实际覆盖 V1 63.21%、V2 57.29%。后段 L1005–1485 为 62.08% 和 55.54%。源分位宽度与实际覆盖只能描述这批选择分布，不能称校准的 epistemic 或联合后验。 [分阶段复算](sources/f09_selected_interval_coverage.csv)
- **Same-state branching 探索性。** 8 个真实 anchor、24 个分支、48 次已归档拟合由验证记录表计数。原报告公开的全局冻结后 test 汇总 short-AULC 胜数为 IVR 4、LCMD 4、MaxDet 0，本次只核对公开计数及验证侧元数据，未访问 test 真值或逐行预测。状态、来源和 seed 都会改变赢家，固定预算切换/controller 尚无资格。 [原报告](../../active_learning/qgeognn_v2_same_state_branching/FINAL_REPORT.md)
- **GP/GOLLuM，历史阶段 No-Go。** 已完成离线开发，概率/稳定性等门槛没有支持正式 AL 效能。 [GOLLuM 裁决](../../active_learning/gollum_adaptation_fusion_20261106/NEXT_STAGE_DECISION.md) · [表示 GP 裁决](../../active_learning/representation_gp_benchmark_20261104/NEXT_STAGE_DECISION.md) · [GP-guided LLM 裁决](../../active_learning/gp_guided_llm_stability_20261103/NEXT_STAGE_DECISION.md)
- **EPIG v1，非 main 归档。** 本地核验 stage 1 的 60 次资格辅助拟合和独立封存 stage 2 的 240 次嵌套辅助拟合均已完成。每阶段只有 3/5 seeds 有合格后验，887/6101 未达门槛，正式五 seed L333–1005 EPIG/IVR 主配对 0 次 QGeoGNN 尝试。真实效能尚未评价。另有 EPIG v2 已持久化探索性进度，版本和状态见 [MISSING_INPUTS.md](MISSING_INPUTS.md)。 [v1 决议](../../active_learning/epig_prediction_oriented_20261117/NEXT_STAGE_DECISION.md) · [stage 2 状态](../../active_learning/epig_prediction_oriented_20261117/stage2_nested_calibration/formal_status.json)

## 权限与复算质量

本轮读取保存的验证汇总和固定历史检查数据、已选择的历史反馈、候选身份及条件、冻结重放轨迹、确定性校验记录。只核验历史训练/验证/已选角色。F04 依据候选几何、条件和原 ID，F05 依据展示顺序和 validator 记录。F06 已观测 H，F07 历史隔离组加原 validation。没有新训练、rollout、LLM 调用、GP 拟合、标签揭示、未来 U 响应或 test 真值访问。旧综合诊断披露过一次搜索越界，其记录未作为本轮数值输入。

逐点来源和积分复算见 [性能 QA](sources/performance_qa.json)，F04/F05 的锁定材料见 [行为输入 SHA](sources/f04_f05_input_hashes.csv)，F06/F07 的 492 个交叉检查与全部输入 SHA 见 [模型 QA](sources/model_input_provenance.json)。图形验证记录见 [visual_qa.json](sources/visual_qa.json)，在 [rendered_previews](rendered_previews/) 留存 PDF 原比例预览。
