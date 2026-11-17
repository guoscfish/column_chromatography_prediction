# 历史运行与可比性审计

本目录是只读历史审计。仓库 HEAD 为 `44be2c0070219d71b2560604f8234a390cc14304`；开始时工作树已有大量修改和未跟踪文件。未覆盖旧文件、改变策略、重训、重跑选择或请求新标签。旧综合诊断 `scripts/finalize.py --check` 实际返回 `SEALED_INPUTS_AND_OUTPUTS_MATCH`，4587 个输入及 122 个输出全部匹配。本次独立建目录，旧封存不重写。

## 1. 五组的真实身份与完成范围

逐运行路径、首尾 checkpoint SHA、初始化/训练行/验证行 hash 见 [RUN_INVENTORY.csv](RUN_INVENTORY.csv)。该表由 [inventory.py](inventory.py) 从既有曲线与 fit 元数据生成，输入 SHA 见 [protocol_inputs.json](protocol_inputs.json)。共 **22 条 campaign**，不能把后续目录续跑误计为独立重复。

| 组别 | 实际标识 | seed | 真实已拟合标签范围 | 接受批次 |
|---|---|---|---|---:|
| Historical CW-Free32 | `free_llm32_scientist_v2` | 157/887/2357/6101/12203 | 均 L333–1485 | 180 |
| New CW-Free32 | `free_cw_sol61_xhigh_rerun32` | 同五 seed | 均 L333–1485 | 180 |
| NoCW-Free32 | `llm32_nocw` | 同五 seed | 均 L333–1485 | 180 |
| CW-gradient LCMD32 | `center_width_lcmd` | 同五 seed | 五 seed 至1485；157/6101另至1549 | 180＋两 seed 各2扩展批 |
| CW-Free16＋CW-LCMD16 | `cw16_llm16_scientist_v2` | 157/6101 | 157至1485；6101至1453 | 36＋35=71 |

NoCW 与 CW-LCMD 的最新完整证据是 `studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112/reports/final_five_seed_L1485_20261116/`。2026-11-16 完成回执记录十条曲线共370个真实拟合评价点、每条36批/1152个新增唯一行。旧11月1日和11月12日 NoCW 未完成端点只代表当时快照，不再代表当前可用 horizon。157/6101 的 CW-LCMD 至1005复用既有轨迹并延长；其1485数值与旧综合诊断一致。新增三个 CW-LCMD seed 不能遗漏，也不能把重读旧157/6101的后段当成新的独立重复。

`raw Gradient-LCMD32` 的 `lcmd` 是原始 V1/V2 梯度几何；`Numerical Hybrid32` 是三初始化 q50 分歧筛前25%再做 member0 latent farthest-first。两者均不是本表的 CW-gradient LCMD 或 16＋16；没有纳入五组主结果。

## 2. 共享实验基础与评价方向

全部五组属于4g、Row split、QGeoGNN-V2。4163行中 outer train3330，初始L333、U2997，validation416、test417。标签预算L只数训练行，416个共用开发验证标签不计入L。不同seed的行划分可重叠，五seed不等于五个独立化学数据集。

各seed的五组首个 fit 的 `initialization_hash`、`validation_ids_hash`、`train_ids_hash`、`checkpoint_sha256` 在 inventory 脚本中逐组核对一致。该证据支持**同seed的相同初始状态**；后续选择一旦分叉，就不再是相同训练状态。

训练为458952参数的QGeoGNN-V2，每轮scratch，Adam学习率0.001、weight decay0，训练batch2048、至多1000epoch、patience100、两目标loss等权，按validation combined normalized RMSE选择checkpoint。初始化seed=`outer_seed+2000003`；训练RNG=`outer_seed+3000017`；CPU两线程。L≤1549每epoch只有一个训练batch，曲线波动不能简单归咎于minibatch随机性。代码依据为 `benchmark_protocol.py` 及旧协议封存审计。

沿用每seed初始L333固定尺度：

\[
 e(L)=\sqrt{\tfrac12[(RMSE_{V1}/s_1)^2+(RMSE_{V2}/s_2)^2]}.
\]

NRMSE与AULC均**越低越好**。归一化AULC使用真实非单调点的梯形面积除以区间宽度，不累计最优、不替换尖峰、不外推缺轮。这里 `best_validation` 是该轮训练的checkpoint选择规则，不是跨标签预算取best-so-far。下一轮改善定义为 `gain_t=e(L_t)-e(L_t+32)`；其正值代表真实重训后的验证损失下降，仍不是当前batch的独立因果贡献。

## 3. LLM配置与Prompt混杂

| 轨迹/阶段 | 实际配置与输入 |
|---|---|
| Historical seed157 | r0–4接受Astra/high；r5为既有授权ID抄写修正，最终served identity未知；L525起Sol/xhigh |
| Historical seed6101 | r0–9 Astra/high；L653起Sol/xhigh |
| Historical seeds887/2357/12203 | L333起Sol/xhigh |
| New CW五seed | 全程请求/served lock均为 `gpt-6.1-sol`、xhigh；新轨迹，初始科学记忆为空 |
| NoCW五seed | `gpt-6.1-sol`、xhigh及served lock；原轨迹延长，没有从L333重新实验 |
| 16＋16两seed | Astra/high前缀，L813起Sol/xhigh续跑；保留不同模型前缀的混杂 |

新CW从实际五seed NoCW接口恢复 `cw_distance_percentile` 及原两行定义，保持历史Free的候选盐值、排序、lossless full catalog编码和可见method/version。CW Prompt SHA为 `992df34be8c3822d2a2f0449e0911181c66cf4661bd52842221a72ddebd38089`，NoCW Prompt SHA为 `61d1f3cbd6b1b0b94c3df0a55282c91890cf14e35924448577c5618fea1adf4e`。差异另见新CW既有 `SCHEMA_DIFF.md`，不是仅凭方法名判断可见性。

NoCW移除显式CW列和CW说明，并进行相关可见内容负向扫描；仍保留V1/V2点预测、q10/q50/q90、q90−q10宽度、化学描述符、五条件、分子/条件新颖性、Morgan相似度、当时L中的已测记录、历史获取时预测误差与自身解释记忆。**NoCW不是没有数值信息。** 三条LLM轨迹均是完整候选交付，无tools、无数值shortlist。

新CW预设有限8次答案×4次传输；NoCW历史包含无上限纠错授权、上游服务/认证/SSE恢复和2026-11-15失败账本重放修复。修复回执记录151个已冻批次可按原始选择严格重放；这些工程修复不应写成新的独立选点或重训。有限/无限重试、日期、有效上下文长度、纠错路径和不可见服务端差异仍可能影响最终接受的LLM结果。模型字符串相同不能证明跨日期后台权重或服务路由完全恒定。

## 4. CW、梯度、CountSketch、LCMD分别是什么

依据 `gradient_transforms.py`、`gradient_features.py`、`scientist_lifecycle.py` 与 `lcmd.py`：

1. CW是输出坐标变换。`C=(V1+V2)/2`、`W=V2−V1`；`s_C`、`s_W`为各seed **L333** 标签的总体标准差（ddof=0），之后固定。实际变换为 `[(V1+V2)/(2s_C),(V2−V1)/s_W]`。本次只审阅已保存变换和源码，不重新读取L333标签。
2. 在**当前轮checkpoint**上，对变换后的两个q50输出分别计算全网络梯度，拼接两块，经512维CountSketch映射。映射seed为 `outer_seed+4000037`，bucket与随机±1符号固定；不是latent、不是原始V1/V2的简单标量梯度范数。
3. LLM可见的 `cw_distance_percentile` 是当前U候选到当前L在CW梯度空间的最小欧氏距离，在当轮U内的 pandas `rank(pct=True)`。它是相对几何排名，既不是预期信息增益、也不是后续行级误差改善预测。它也**不是LCMD完整集合选择分数**。
4. LCMD是批量集合算法：先把全部L装作中心，以平方欧氏距离将剩余U分配到最近中心；每一步选“最近中心平方距离总和最大”的簇，再选簇内最远点，将其加作中心并更新。并列按稳定输入顺序；批内顺序会改变后续分数。
5. 梯度表示随重训checkpoint变化。原始欧氏距离不能跨轮/策略当作同一固定坐标量直接平均比较。跨方法主比较采用同轮自身合法池排名/随机校正，或使用固定分子指纹与固定条件归一化表示；后者才可直接横向比较。初始bank复用须同时满足checkpoint、L/U身份与顺序、变换和映射一致。

## 5. 硬约束与随机基准

`scientist_selector.SYSTEM_PROMPT` 明确无分子上限、新分子配额、固定探索比例、不确定性配额或数值策略配额；允许围绕同一分子作有意义的局部实验，但不是强制单因素对照。`scientist_full_delivery.validate_direct` 强制唯一、真实合法且已供应的候选ID、规定名额、packet绑定、假设与证据ID/文本schema；不存在最低科学对照名额。

因此Free/NewCW/NoCW的状态随机基准可从**对应轮真实U_t**无放回均匀抽32个，并排除所有已测L与已占pending行。它复制的是行身份/预算/排除规则，不能宣称复现随机文本的科学合法性。不能以统一全局候选池代替各自U_t，也不能将局部对照建议强行转换成随机配额。

Hybrid先产生CW-LCMD16，作为可见pending；LLM再补16，不能重复选择pending。其联合32个实际接受集合与来源必须从freeze/selection核对，不能把文件中的前16行自动当LLM。固定真实数值半批、随机抽另16是“条件于实际数值半批的随机LLM补充”基准；把32个全部随机化则是联合行合法性基准，均不等于另一次LCMD或LLM rollout。覆盖边际的两种加入顺序只支持几何归因，不支持训练性能归因。

## 6. 预先固定的比较层次

| 比较 | 核实后的主窗口 | 等级与解释 |
|---|---|---|
| A：五组 | seeds157/6101，L333–1005主窗口；可用共同延长至1453另列 | **部分可比、需要控制**：共享初始/Row目标/预算，但LLM模型前缀、prompt及后续状态不同；双seed仅描述 |
| B：三LLM | 五seed L333–1485；887/2357/12203从头Sol子组单列 | 配置一致子组比混合前缀更可比，但仍是**部分可比**，不是同状态CW因果效应 |
| C：LLM与CW-LCMD | 五seed L333–1485；与Hybrid比较时降为共同两seed | 同预算、同初始；各自状态历史分叉，属**部分可比** |
| D：16＋16与纯策略 | seeds157/6101，L333–1005；共同延长至1453另列 | **部分可比/描述性归因**；无配对重训，无法拆分学习收益 |
| 相同初始L333状态 | 同seed L/U、checkpoint hash均一致的r0 | **严格可比**的选前状态/输入几何；仍需记录LLM配置和接口差异 |
| Astra/high对Sol/xhigh的“CW效应” | 不适用 | **不可比较**为单因素效应；只能描述整条轨迹 |
| raw-LCMD/数值Hybrid32旧图 | 各自既有范围 | **仅描述性参考**，不可更名替代本任务目标方法 |

L333–1005是便于与已封存分析核对的主分析窗口；L333–1453是当前五组真实共同两seed可用范围，不能遗漏，也不把后者追称预注册。所有更长范围均标明真实seed数与horizon。每次比较以完整曲线的共同预算为准；不以目标预算、运行目录名或pending batch当成完成。

## 7. 条件与身份边界

真实五条件只有PE/EA配比、Density g/mL、样品上样体积V/µL、loading solvent类别、加载溶剂体积/µL。PE/EA转为EA体积分数；两个体积分别报告，density经常随分子固定，不冒称独立操控因素。没有可用流速、pH等字段，不能添加假想因素。连续条件固定归一化/分箱，不能把任何非零差都等价解释为同样大的物理变动。

实验身份用稳定row ID，分子身份用canonical SMILES；lossless packet的m0/m1等短键会随轮次重编码，不能跨轮用作固定分子ID。相同分子不同条件不自动是冗余，完全相同记录条件不同row另列。最近“高价值”历史样本若无事先定义的合法价值预测，不从事后性能中反向挑选。

本次未读取test truth、未打开test预测行，也未使用候选未来反馈。旧封存报告披露过一次此前会话的标签搜索越界；本次复用其不含响应值的行为表不等于重新发生该访问，亦不把旧报告的全会话声明冒充本次核验。
