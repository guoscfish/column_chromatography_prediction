# 实现、配置与条件语义核对附录

本附录对应主分析固定快照 **2026-11-01 16:35:02.564895 +08:00**，快照存储为`包含原始JSON的snapshot_inputs.json.xz`，截止值为解压内容的 `cutoff_ns = 1793522102564895000`。新增的 `protocol_input_manifest.json` 固定了46个关键源码、配置、历史修订和既有报告的 SHA256、字节数、mtime及实际读取时间。全部来源的mtime严格早于截止值，双读字节一致、设备/inode/size/mtime/ctime均未变化；JSON均成功解析，跳过项为0。

此次附录制作只读取这些静态材料，不执行其中的模块，不读取原始响应表，不运行训练、选点、LLM接口或任何重试入口。主分析的批次、拟合、元数据输入另由 `input_manifest.json` 和 `snapshot_inputs.json.xz` 固定。以下行号均为仓库根目录下实际文件的行号。

## 1. 主比较对象的实际身份

| 报告名称 | 实际实现与证据 |
|---|---|
| NoCW LLM32 | `llm32_nocw`，32个LLM自由选点、pending为0；实际五seed wrapper沿用历史Free的catalog编码和排序salt，移除CW列及CW说明。`scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py:52–85`；原配置 `studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029/config.json:11–31`。 |
| Free LLM32 | `free_llm32_scientist_v2`，全候选完整交付、每批自由选32个，含显式CW分位。`src/qgeognn_al/active_learning_v2/scientist_full_delivery.py:6–22,36–54`；`studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol61_xhigh_free_five_seeds_to1005_20261023/protocol.json:55–97`。 |
| 纯数值 Hybrid32 | `qgeognn_v2_row_sequential_b32` 中的 `hybrid`：当前三个独立初始化模型的V1/V2 q50分歧，经原L333两个目标尺度归一化并取范数；保留当前U_t最高25%候选，再在当前member0的128维latent中做farthest-first，以全部当前L_t作中心，选32个。`src/qgeognn_al/active_learning_v2/uncertainty.py:6–17`、`sequential_acquisition.py:59–103`、`sequential_runner.py:420–485`。 |
| Gradient-LCMD32 | `lcmd`：当轮全网络q50的512维CountSketch、corrected LCMD-TP与平方欧氏几何；续跑显式使用原始`8081e86`梯度源码。`src/qgeognn_al/active_learning_v2/sequential_acquisition.py:127–138`；`reports/free_lcmd_five_seeds_to1485_20261026/lcmd/extension_manifest.json:77–101`。 |

当前NoCW对比图的来源确实指向`qgeognn_v2_row_sequential_b32`，见 `reports/nocw_progress_to1389_20261101/plot_latest.py:19–28`。`studies/active_learning/verifiable_scientific_agent_20261030/REPORT.md:11–15` 中名为“Hybrid”的历史LLM审计对象却是 **CW16＋LLM16**，只有16个LLM名额。本次不将那条轨迹当成纯数值Hybrid32；该报告仅用于理解历史合法性、重试和实际validator规则。

## 2. NoCW仍有哪些预测和历史信息

NoCW消融针对显式`cw_distance_percentile`字段和相应CW文字说明，不能解释成“没有模型预测信息”或“没有几何代理”。允许字段和移除规则见 `src/qgeognn_al/active_learning_v2/nocw_inputs.py:60–69,125–162`。实际wrapper改用历史Free盐值和packet版本，见 `scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py:60–71`。

保留信息包括：

- 点预测及V1/V2各自q10、q50、q90；q90−q10宽度。
- canonical SMILES、分子描述符、全部五个实验条件。
- 分子新颖性、完整分子加条件tuple新颖性、到已观测分子的Morgan相似性和上样质量。
- 当前L_t已测响应；历史获取时的预测与实测误差、假设与解释记忆。

原始`catalog.json`卡片没有全部派生列；`ReadOnlyCatalog`在编码前补入它们，见 `src/qgeognn_al/active_learning_v2/scientist_selector.py:155–185`。因此不能仅按原始卡片键判断LLM最终看到了哪些列。完整交付表由 `scientist_full_delivery.py:36–54` 编码。本次先前已抽查NoCW与Free seed157 r0的实际`input.json`，确认候选列仅差CW；本附录未重新读取响应表。

q90−q10是单预测器的分位宽度，不是三模型ensemble epistemic uncertainty，也不保证校准；历史误差是获取时预测减实测，不是当前模型重算残差，见 `scientist_selector.py:91–97`。

实际五seed wrapper还覆盖了paired模块的validator：沿用Free的证据ID与prose规则，未追加未运行paired协议的prose-ID禁令，见 `run_qgeognn_v2_row_llm_nocw_five.py:74–85`；既有审计亦记录此区别，见 `verifiable_scientific_agent_20261030/REPORT.md:15`。

## 3. 共同协议、初始状态与因果边界

共同确认seed为 **157、887、2357、6101、12203**。数据为4g、Row split，共4163行；outer train为3330行，其中L0=333、U0=2997；validation=416、test=417；主动学习批量为32。证据：`src/qgeognn_al/active_learning_v2/sequential_protocol.py:31–47,153–172`、`benchmark_protocol.py:21–30`。NoCW实际采用这五个baseline split，见 `run_qgeognn_v2_row_llm_nocw_five.py:92–123`。

训练采用QGeoGNN-V2、Adam、学习率0.001、weight decay 0、训练batch 2048、最大1000 epoch、patience 100、每epoch确定性shuffle、V1/V2 loss权重各1、validation combined normalized RMSE选checkpoint。每次从相同初始化scratch拟合；member0初始化seed为`outer_seed+2000003`，训练RNG为`outer_seed+3000017`，CPU两线程。证据：`benchmark_protocol.py:25–43`；NoCW原配置`:32–52`；Free五seed协议`:71–89`；LCMD延长manifest`:58–76,99–101`。

前期只读核对中，五seed的NoCW和Free r0完整有序L0、U0均一致，初始checkpoint SHA256均相同；L0规范化集合哈希与纯数值Hybrid r0 contract一致。主分析快照的初始状态、split与fit核对记录是本次后续复查入口。

**相同seed只有初始状态可以据这些记录对齐；后续L_t、U_t、checkpoint和反馈历史随实际选择分叉。** 相同seed和标签预算并不恢复同一状态。本次是历史轨迹的行为比较，不能称为严格同状态CW因果消融。三个从r0统一Sol配置的Free seed也仍有后续状态混杂。

L1389延长保留L1005科学协议，扩展目标预算而已，见 `scripts/studies/run_qgeognn_v2_row_llm_nocw_to1389.py:54–89` 和 `studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1389_20261101/extension_to1389_amendment.json:81`；实际共同已完成预算由主分析本次快照重新推导，不使用旧报告终点。

## 4. 历史Free的LLM配置混杂

原Free两seed保留Astra/high前缀；转换manifest明确157从L525、6101从L653使用`gpt-6.1-sol/xhigh`。证据：

- `studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/free_sol_xhigh_continuation_20261012/transition_manifest.json:4–28`。
- 新增887、2357、12203从r0使用Sol/xhigh：`studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol61_xhigh_free_five_seeds_to1005_20261023/protocol.json:11–14,56–69,96–97`。
- 后续至L1485仍为Sol/xhigh：`reports/free_llm_five_seeds_to1485_20261024/extension_manifest.json:69–83`。

| Free seed | 接受批次的历史配置/served-model证据 | 预算解释 |
|---|---|---|
| 157 | r0–4接受响应为Astra；r5最终为历史授权ID抄写修正，accepted served-model保留unknown；r6起Sol | r0–5的L_before为333–493；Sol从L_before=525开始 |
| 6101 | r0–9接受响应为Astra；r10起Sol | Astra最后一批L621→653；Sol从L_before=653开始 |
| 887、2357、12203 | r0起Sol | 从L333开始同配置 |

以上分段来自前期对旧逐批索引与少量原始turn的只读核对；本附录只重新固定静态配置与已发布诊断报告。`studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/REPORT.md:27`保留了按最终接受响应统计的164批Sol、15批Astra和1批unknown结论，不将r5失败Astra响应误标成最终已执行回答。建议在全部五seed结果之外，单列887/2357/12203三seed配置一致子组；不以换模型后的157/6101后段冒充从头同配置历史。

NoCW原始LLM配置是`gpt-6.1-sol/xhigh`、完整候选交付、无tools、无自动fallback、输出上限32768tokens，见 `studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029/config.json:53–73`。**原配置里的逻辑3次/传输3次并非当前整条轨迹的实际总上限**：后续合法修订包括额外纠错、persistent retry及无上限重试，最终`unbounded_correction_amendment_20261031_v3.json:15–17`明确纠错/传输上限为null；L1389延长amendment`:7`继续保留授权。本附录仅记录这种配置与工程历史混杂，不以配置值反推实际失败次数。

Free的历史授权额外纠正/ID抄写及887 r33、12203 r24例外也存在，见已固定的`reports/llm_eight_to1549_20261018/historical_protocol_deviations.json`与`reports/free_lcmd_five_seeds_to1485_20261026/free/continuation_amendment.json`。无论纠错多少次，分析单位始终是最终接受且冻结的批次；失败、纠错请求不是新增实际执行选点。

## 5. 条件含义、单位、精度与分子键

只分析实际存在的五字段，定义来自 `src/qgeognn_al/active_learning_v2/scientist_selector.py:18,32–40`；单位亦见 `docs/model/INPUT_SCHEMA.md:19–22`。该文档含较新的Clean模型架构描述，本附录仅复用其单位解释；当前研究V2确有density×V与加载溶剂体积输入，独立源码证据是 `src/qgeognn_al/schemas/conditions.py:78–79`。

| 字段 | 解释与匹配处理 |
|---|---|
| PE/EA | 石油醚/乙酸乙酯的记录配比a/b；EA体积分数组成为b/(a+b)，无单位。保留原配比语义，不将不同配比随意舍入合并。 |
| Density g/ml | 密度，g/mL；与V相乘得到mg上样质量。该字段通常随分子固定，本次不声称它是独立可操控因素。 |
| V/ul | 样品上样体积，µL。只有在其余条件包括Density一致时，改变V才可作为这里的严格上样量对照。 |
| loading solvent | 分类字段PE、EA或DCM；相等按类别精确判断，不按数值编码远近解释。 |
| Volume of loading solvent/ul | 加载溶剂体积，µL，与样品上样体积V为不同字段。 |

先前只读抽查seed157的L333允许训练候选加已标注集合（3330行），PE/EA有10档，Density有151个值，V有8个值，加载溶剂体积有11个整数档。V中包含`34.01360544`、`68.02721088`、`102.0408163`等值；Density也有`1.0477`、`1.0491`等4位小数。这些只是精度与字段盘点，不是额外响应信息。匹配使用原记录数值的严格等值、零容差；不以任意四舍五入制造相同条件。PE/EA如转换为分数，应按精确比例归一，不能用低精度小数并档。

没有流速、pH、额外浓度或可变柱规格字段；不能把缺失字段引入“其他条件相同”的推断。严格匹配表明记录的五条件满足规则，不保证未记录的实验变量相同，也不自动证明物理因果。

稳定row ID用于实验身份，分子身份沿用仓库canonical SMILES。`scientist_full_delivery.py:36–54`按每轮SMILES集合重新分配packet的`m0/m1/...`：这些短键是当轮编码，不能跨轮当真实compound ID；row ID也不能当分子ID。同分子不同条件可构成有目的的条件对照，不能单因分子重复就称冗余。完整分子加五条件tuple也相同时，应另列重复条件行。

## 6. 本附录的复查边界

本附录固定实现、信息输入、配置历史与条件单位，不能单独证明某种选点行为带来了误差下降。CW/latent由方法各自当轮checkpoint产生，跨方法原始距离不能直接横比；覆盖/距离是代理指标，实际比较应使用本次主分析的同轮状态内随机参照或明确标记缺项。

该目录主报告和CSV负责共同预算、逐批规模/ID/阶段核对、逐seed方向、权重与代表案例；旧报告只提供实现线索和已完成历史审计。整个诊断完成后停止，保留当前正在运行的NoCW实验。
