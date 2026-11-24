# 4g 数据字段、样本身份与模型可表达性

正式 4g Row 数据有 4,163 行、217 个 canonical SMILES、220 个 CAS。历史候选和观测卡都缺少原始表已有的 **Flow mL/min、Purity、CAS、name**。V3 在完整信息配置中向候选和观测提供这四项，并向模型说明 **Silica-CS 4g 正相硅胶与 PE/EA** 的全局实验条件。Flow 和 Purity 都没有进入当前 QGeoGNN-V2 的独立输入。其中 Flow 用于 E0 的标签换算，Purity 在 E0 保留于记录。

## 核对数据链

| 环节 | 核查结论 | 代码位置 |
|---|---|---|
| CSV → E0 | 原始 4,243 行。冻结 E0 以原文件 SHA256 和从 2 开始的 CSV 记录序号生成 20 字符 `sample_id`。以 `V1/V2=t_raw×flow/1200` 换算体积。冻结过滤后的 4,163 行保留全部十六个原字段与重复实验。 | [E0 sample_id](../../../scripts/run_e0_4g_baseline.py#L104) · [E0 标签换算与过滤](../../../scripts/run_e0_4g_baseline.py#L151) · [E0 canonical 写入](../../../scripts/run_e0_4g_baseline.py#L248) |
| E0 → Row | 正式 5 seed 各有 L333、U2997、validation 416、test 417。审计额外读取同一 manifest 中 5 个 development seed 的角色表。核对 split 文件散列、角色 ID 散列、canonical 顺序、结构和 seed。 | [冻结 Row 规则](../../../src/qgeognn_al/active_learning_v2/protocol.py#L36) · [Row split manifest](../qgeognn_v2_row_small_batch_benchmark/splits/split_manifest.json) |
| Row → feature loader → graph | 加载 `sample_id`、canonical SMILES 与原有五项条件。使用结构图、构象、分子描述符、加权洗脱剂描述符、加载溶剂类别、名义加载质量和加载溶剂体积。 | [feature loader](../../../src/qgeognn_al/active_learning_v2/benchmark_protocol.py#L23) · [resources](../../../src/qgeognn_al/resources.py#L10) · [图特征](../../../src/qgeognn_al/data.py#L193) · [typed 条件分支](../../../src/qgeognn_al/schemas/conditions.py#L173) |
| graph/checkpoint → candidate 卡 | 历史条件为 PE/EA、密度、样品体积、上样溶剂与上样溶剂体积。每个候选有 6 个预测输出及区间宽度。NoCW 从旧目录投影时排除外部距离，历史后续轮可能仍在宿主侧计算距离。 | [历史候选卡](../../../src/qgeognn_al/active_learning_v2/dialog_study.py#L227) · [历史候选预测与几何](../../../src/qgeognn_al/active_learning_v2/scientist_lifecycle.py#L102) · [NoCW 字段投影](../../../src/qgeognn_al/active_learning_v2/nocw_inputs.py#L125) |
| observed 卡与反馈 | 仅当前已测行有实测体积。后续获得的行还带获取时的预测和当时误差。计算误差的符号为预测减观测。候选卡的预测来自当前模型。 | [历史已测卡](../../../src/qgeognn_al/active_learning_v2/dialog_study.py#L240) · [冻结后反馈](../../../src/qgeognn_al/active_learning_v2/scientist_lifecycle.py#L128) |
| encoded packet → 请求 | 历史 full delivery 共享分子字典、按表行展开条件列表。NoCW 以 `request_*.json` 归档实际序列化的 messages 与配置。V3 由白名单 metadata 投影补充卡片条件，经完整上下文和角色隔离验证生成请求。 | [历史表编码](../../../src/qgeognn_al/active_learning_v2/scientist_full_delivery.py#L36) · [历史实际请求](../../../src/qgeognn_al/active_learning_v2/nocw_study.py#L262) · [V3 metadata](pure_llm_v3/metadata_view.py#L133) |

逐字段的模型、历史请求、反馈和 V3 去向见 [字段可用性矩阵](field_availability_matrix.csv)。[数据集 README](../../../dataset/README.md) 说明柱规格和洗脱体系的目录组织。审计读取了 `dataset/` 下九张 CSV 的表头。4g、8g、25g、40g、DCM 的各表头均有 `name/Purity`。CN、NH2、C18 省略这两列。 `Compounds.csv` 为各体系的 CAS、英文名与 SMILES 对照表。只有 4g 原始和 canonical 各自的 metadata 列被投影。其他体系只有表头被读取，其实验和响应未进入 V3。

## 元数据对齐与隔离

独立 sidecar 核对原文件与 E0 来源清单 SHA256、canonical 文件与冻结 Row protocol SHA256，逐行重算 `sample_id`，核对原始与 canonical 的 metadata 和结构记录。对样品体积和密度等字段，允许 E0 写盘引入 `50` → `50.0` 这种等值文本变化。V3 保留原文，实数比较使用精确的十进制值。PE/EA 的等值比使用精确的有理数比较，`EA_fraction=b/(a+b)` 的数字值用于读表，原比例原文仍在场。该份额是记录比例中的 EA 份额。Density g/mL × sample µL 在量纲上给出名义毫克，既不乘以纯度，也不等同于上样溶剂体积。

读取 CSV 使用显式 `usecols`。 `Ending time`、原始 SMILES 拼写和 source row 仅供后台核对。源文件路径进入清单。 `t1/t2`、验证时间、V1/V2、标签来源质量警告和排除原因均未进入 sidecar。留出的 validation/test 只以 split 角色身份参与核对。冻结状态审计使用 ID、散列和既有 label-access 收据，不读取获得这些标签的响应值。本次没有打开新的标签存储渠道。

[字段统计](raw_canonical_metadata_summary.csv) · [无标签 sidecar](label_free_metadata.csv) · [SHA 与数据来源清单](label_free_metadata_manifest.json)

## Flow 与 Purity 分布

| 范围 | 行数 | 4 | 5 | 6 | 8 | 10 | AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 4g | 4,243 | 8 | 371 | 12 | 11 | 3,841 | 21 |
| canonical 4g | 4,163 | 8 | 371 | 12 | 11 | 3,761 | 21 |
| 157 outer train | 3,330 | 6 | 302 | 9 | 6 | 3,007 | 16 |
| 887 outer train | 3,330 | 7 | 298 | 10 | 11 | 3,004 | 18 |
| 2357 outer train | 3,330 | 6 | 305 | 11 | 9 | 2,999 | 15 |
| 6101 outer train | 3,330 | 7 | 300 | 10 | 9 | 3,004 | 19 |
| 12203 outer train | 3,330 | 5 | 284 | 8 | 9 | 3,024 | 16 |

流速列单位为 mL/min。4g raw 和 canonical 均无空白纯度。已明确的比例派生为数字，原文照存。 `AR` 保存为等级，派生比例为 `null`。未知或空白记录标明类型，不填补。两种 raw CAS 出现多种纯度记录，同样保留于 canonical。 `120-46-7` 有 `0.98` 与 `AR`， `611-70-1` 有 `0.97` 与 `0.98`。五个 canonical 结构出现多种纯度。

[逐角色 Flow 和全部纯度分布](flow_purity_distribution.csv) 单列各 split 的 outer train、L0、U0、validation、test，以及 1,525 个已核验的独立 seed/L/U 状态。相同 seed 和 L/U ID 集去重，来源路径、SHA 和凭据逻辑保留在清单。1,199 份 batch freeze 提供状态证据，另有顺序 benchmark 的已核验索引及 round-state 文件。NoCW、Free、CW-Free 和匹配 CW-LCMD 五个 L1485 状态均在表中。25 份 batch freeze 不进入覆盖统计，其中 13 份属于 mock 或 dry-run，12 份来自旧 LLM16 屏幕实验，只有 128 条候选卡而无完整冻结 L/U 凭据。

## 记录身份和历史对照重分类

以 canonical SMILES 加 CAS、柱规格、纯度原文、密度、样品体积、上样溶剂、上样溶剂体积、PE/EA 和 Flow 作为完整记录条件。CAS 与名称独立保留。三个结构对应多个 CAS，四个结构对应多个记录名称，两个 CAS 对应多个名称。原始数据甚至将 `Ethyl (2Z)-3-aminobut-2-enoate` 同时用于两个 CAS，其中 `103-36-6` 的其他记录又使用 `Ethyl cinnamate`。这些记录维持各自的行身份，名称无需参与合并，也不据此确定试剂纠正结果。

旧结构加五条件键有 309 个看似重复的组。加入完整记录条件后，287 组被拆开，其中 266 组由 Flow 拆开，21 组由 Purity 拆开。完整键下仍有 31 个两行组，共 62 行。它们是可能的复测，保留不同 ID。是否为真正复测还取决于批次或实验台账等未记录信息。

[条件对照重分类](historical_contrast_reclassification.csv) 区分既有 NoCW 审阅样例、诊断 pair-events、独立研究的 legacy pair-events 和历史理由文字筛查。377 个去重后的 NoCW 诊断 pair-events 中，有 30 个旧单因素配对变为多因素，29 个新增 Flow 差异，1 个新增 Purity 差异。独立研究的 165 个 legacy pairs 中有 21 个被重新分类，其中 20 个涉及 Flow、1 个涉及 Purity。不同证据集可重叠，不合并为单一比例。继续保留同一行、跨结构、多因素和可能复测等客观区别。关键词和字面引用只用来定位可能的对照。有些引用指向理由中的另一对历史行，整个筛查的计数不能作为科学错误率。

- NoCW seed 6101 R07 的已审阅 PE/EA 对照同时改变 Flow 5 → 10 mL/min。来源行 `0ccbd28a4414f97dfad8`、`1a55639d058c08a1a06e`。 [既有审阅](../nocw_failure_attribution_20261112/tables/nocw_scientific_chain_adjudications.csv#L3)
- NoCW seed 157 R01 第 9 个选择的 5/1 与实测 20/1 记录确实同样品加载量，同时也有 5 与 10 mL/min 的 Flow 差异。这限制了将两者的差异归因于洗脱比例的结论。 [冻结选择](../nocw_vs_cw_lcmd_five_to1485_20261112/nocw/selections/seed_157/llm32_nocw/round_01/selection.json)
- Free seed 12203 R20 第 16 个选择明确称 DCM 上样溶剂体积是唯一变化。所引实测行的 Purity 是 `AR`，候选记录为 `0.98`。这是记录条件上的多因素对照。 [冻结选择](../qgeognn_v2_row_llm_scientist_v2/revisions/sol61_xhigh_free_five_seeds_to1005_20261023/selections/seed_12203/free_llm32_scientist_v2/round_20/selection.json)
- Hybrid seed 157 R03 第 9 个选择把样品体积 50 与 150 µL 的比较视为配对加载对照。两个记录的 Flow 是 5 与 10 mL/min。 [冻结选择](../qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001/selections/seed_157/cw16_llm16_scientist_v2/round_03/selection.json)

[记录身份核查](metadata_identity_audit.json) · [11 月 23 日单因素研究](../4g_single_factor_response_20261123/DATA_AUDIT.md)

## QGeoGNN-V2 可表达边界

E0 使用 Flow 换算真实洗脱体积。当前 QGeoGNN-V2 的 X 未包含 Flow、Purity、CAS 或 Name。若其余模型输入相同，Flow 或 Purity 不同的记录对同一模型产生相同输入。数据量增加可能改变模型学到的平均分布，也可能减少分布中常见条件的误差。单靠给这类输入增加标签，无法保证模型分辨被省略的条件轴。密度与样品体积以乘积进入当前模型，因此同样可能产生输入碰撞。后续将 Flow 或 Purity 加入 predictor 需要独立的模型研究。

`q50` 历史字段是以均方误差训练的中心预测，同时受交叉惩罚项约束。 `q90−q10` 描述模型输出宽度，`V2−V1` 描述洗脱区间宽度。当前训练目标与损失实现没有给这些值提供校准后的中位数或 epistemic uncertainty 保证。

[冻结条件分支](../../../src/qgeognn_al/schemas/conditions.py#L93) · [实际训练损失](../../../src/qgeognn_al/training/predictor.py#L75) · [单因素模型响应](../4g_single_factor_response_20261123/MODEL_RESPONSE.md)

## 重建与测试

```sh
python studies/active_learning/pure_llm_v3_20261124/scripts/audit_metadata.py
python -m pytest -q studies/active_learning/pure_llm_v3_20261124/tests/test_metadata_view.py
```

四项 metadata 测试通过。数据审计真实 API 请求、标签揭示与正式拟合均为 0。
