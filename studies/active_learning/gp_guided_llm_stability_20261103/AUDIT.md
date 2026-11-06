# 第一阶段离线审计

审计日期：2026-11-03，Asia/Shanghai。工作基线为 `main` / `HEAD` 的 `32fdc85cb8411f9e93c2e3db56f6648c14d620b2`；本任务已通过只读远端引用核实最新 main。没有执行 pull、checkout、reset、清理工作区或修改历史实验。工作区原有已修改文件、大量未跟踪选择记录与本地 runtime 均保留。

本审计读取源码、协议、既有报告和白名单状态/拟合元数据；没有读取全量响应标签表、含真值的候选/测试预测文件，没有揭示标签、调用付费 LLM、训练 QGeoGNN 或发出进程控制指令。历史报告中已有的测试评估只用于识别证据等级与历史曝光事实，未读取测试逐行真值，也不用于本研究选状态、定阈值或优化 GP。进程枚举 `ps` 被当前沙箱拒绝，因此下文将持久化状态与进程存活分开表述。

## 1. 最新实验状态：必须区别新旧快照与陈旧进度

### 1.1 CW-Free 重跑

权威研究目录为 `studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/`。`README.md` 指明实际入口为 `immediate_retry_20261102/controller_overlay.py`；不能仅根据最初的 `PROTOCOL.md` 或 `config.json` 推断当前调度与重试。有限重试、并行执行、持久化恢复、HTTP/2、HTTP403 和立即重试授权存在按时间叠加的修订。模型/科学协议变化不属于传输恢复。

已提交的固定报告 `reports/free_cw_progress_20261103/REPORT.md` 截于 **11-03 10:39:22 +08:00**，为 **84 个接受批次、84 次完成拟合**，五 seed 分别到 L813/L941/L909/L845/L845。它不是 L1485 终局。

本次 **11-03 12:08:39 +08:00** 只读本地 `progress/completed_seed_*_round_*.json`、冻结批次文件名及 `fit_audit.json` 元数据，得到更新的持久化状态：

| Seed | 冻结批次 | 完成回执/拟合数 | 已完成 L | 当时持久化阶段 |
|---|---:|---:|---:|---|
| 157 | 15 | 15 | 813 | SELECTING，round 15 |
| 887 | 20 | 20 | 973 | SELECTING，round 20 |
| 2357 | 19 | 19 | 941 | SELECTING，round 19 |
| 6101 | 17 | 17 | 877 | WAITING_COMPUTE_SLOT |
| 12203 | 16 | 16 | 845 | SELECTING，round 16 |
| 合计 | 87 | 87 | 五 seed 共同终点 L813 | 仍有进行中选择记录 |

以上为顺序读取时的持久化观察，不是全局原子快照，也不是重新核验全部大 checkpoint 与预测文件。进行中 response、未冻结候选和未完成拟合均未计入。现有模型与记录只读；本任务没有验证 worker 进程即时存活，也没有尝试恢复、停止或重启它们。

### 1.2 NoCW

`five_seed_nocw_to1005_20261029` 五 seed 均完成 L1005。其后延长到 L1389 的目录是 `qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1389_20261101/`，但 **11-01 19:24:35 +08:00 用户已明确要求停止**。

`progress/user_stop_20261101T112435Z.json` 明确记录 `USER_REQUESTED_STOP`、所有目标进程已停止、历史/进行中记录保留及恢复前须核对未决请求。该文件比旧 `seed_*.json`、`stream_seed_*.json` 的 SELECTING/IN_PROGRESS 更晚；不能把这些旧字段当成目前仍在运行。该停止记录 SHA256 为 `d12e085bbadfc0156ed194e5045c908f33d6d225ab229e0d046dd03d55cfbd54`。

| Seed | 最后冻结并完成批次 | 最后已完成 L |
|---|---:|---:|
| 157 | 23 | 1069 |
| 887 | 21 | 1005 |
| 2357 | 25 | 1133 |
| 6101 | 26 | 1165 |
| 12203 | 28 | 1229 |

本次只读枚举的 `fit_audit.json` 数与该停止回执逐 seed 一致。延长目录继承原批次，不能把原目录与延长目录相加成新的独立实验。当前本地证据支持“有明确停止回执，未见后续恢复凭据”，不把历史停止回执升级成今日进程枚举结论。本任务不恢复 NoCW。

### 1.3 严格可比性结论

CW-Free 重跑和 NoCW 对应五 seed、Row split、L333 初始集合、初始 checkpoint、原 opaque ID/排序/无损精度、QGeoGNN-V2 架构、同 scratch 初始化与训练 RNG、归一化、optimizer、epoch/patience、validation 选 checkpoint 规则均已有冻结协议与初始校验。它们足以作配对的历史开发比较，但**不是完全同时实施、除 CW 字段外所有执行因素相同的随机化因果对照**：

- 两者请求时期、供应商后端、纠错/传输恢复政策和实际尝试轨迹不同；未传 temperature/top_p 不意味着响应确定。served model 身份锁不能证明服务器参数与后端完全未变。
- 初始共状态之后，每条轨迹的 L/U、checkpoint、预测、残差、记忆会分叉；后期选点差异不能当作同状态的信息消融。
- 历史原 Free 的 seed 157 与 6101 有 Astra/high 前缀，另三 seed 从初始即 Sol/xhigh；原 Free 整体必须保留“混合模型历史轨迹”标签。CW-Free 新重跑固定请求 `gpt-6.1-sol/xhigh`，不能将旧 Free 当成另一份同模型独立重复。
- 本次更新状态下 CW-Free/NoCW 五 seed 共同预算仍为 L813；不能用不同完成端点构造公平排名。以 raw、非单调预算点的归一化梯形 AULC 为准，不平滑、不外推、不取 best-so-far。
- 这些 seed 已多次参与开发，不能称为全新外部确认。历史部分数值研究已曝光测试结果；本任务不利用这些测试结果进行新开发。

## 2. Scientist Selector 的真实输入、Prompt、工具与记忆

主要来源为 `src/qgeognn_al/active_learning_v2/scientist_selector.py`、`scientist_full_delivery.py`、`nocw_inputs.py` 以及实际五 seed runner；CW-Free 使用自身 `frozen_source/` 副本和 process-local adapter。

### 2.1 真实 Prompt 语义

Prompt 明确：目标是重训 QGeoGNN 后在 Row 分布上 V1/V2 的预测精度；不优化候选自身色谱表现。允许化学先验、响应曲线、假设区分、探索/利用、覆盖、不确定性和历史残差；数值信号是证据，没有固定 acquisition formula、分子上限、探索比例或数值配额。可用有明确科学意义的同分子/同家族局部实验，不能为多样性而追求多样性。

以下数字信息已经存在，故统一将 Arm A 称为 **LLM-led / unconstrained numerical decision baseline**，不称“完全无数值信息的纯 LLM”：

| 输入内容 | CW-Free | NoCW | 解释边界 |
|---|---|---|---|
| canonical SMILES、MW/LogP/TPSA/HBD/HBA、五个实测条件 | 有 | 有 | 条件只有 PE/EA、density、样品体积 V、loading solvent、加载溶剂体积；没有流速等额外条件 |
| QGeoGNN V1/V2 point prediction 与 q10/q50/q90 | 有 | 有 | q90−q10 是单预测器分位宽度；并非已校准 epistemic uncertainty；V2−V1 是洗脱区间宽度 |
| molecule/condition novelty、Morgan radius-2 Tanimoto、density×V 推导 loading amount | 有 | 有 | 属于现成数值/结构信息，NoCW 并非无数字 |
| `cw_distance_percentile` 与定义 | 有 | 删除 | 当前 center/width 梯度空间到 L 的距离分位；不是 GP 风险或校准概率 |
| 已观测 V1/V2、采集前预测与真实误差、科学反馈 | 有 | 有 | 初始 L333 没有合法的 out-of-sample error；历史误差是采集时 prediction−measurement |

真实五 seed runner 保留了旧 DIRECT_PROMPT 的 hybrid/free 说明，通过 `packet.selection_count=32`、pending=0 表示自由 32 点；CW-Free schema diff 有意不顺手改这段措辞。准确入口是冻结 `run_free_cw_sol61_five_20261101.py::prompt("cw_aware")`：仅从 DIRECT_PROMPT 删除 `Pending-CW outcomes are identified separately.` 这一句，`activate()` 覆盖 `nocw_study.prompt`，不使用 `nocw_inputs.COMMON_PROMPT`。新研究的 Arm A 应冻结这个实际 Prompt，不悄悄换成未运行 paired 版本。

### 2.2 工具与交付方式

历史 query-based Scientist 有只读 JSON relay（最多24查询、480 distinct views、每次24条等）；**当前 Free/NoCW 是完整候选池一次性无损交付**：`DIRECT_PROMPT` 切去 relay 说明，明确没有 retrieval、query、paging 或额外工具，transport `tools=[]`、`store=false`。不可把代码中仍存在的 query API 误写成当前 LLM 实际可调用工具。

`encode_catalog()` 把重复分子元数据分解成 molecule 字典，候选/observed/pending 以 columns+rows 无损编码，conditions 按固定五字段序列；全池没有 numerical shortlist、舍入或删行。排序使用固定 seed/round salt 的 ID hash。新 Arm B/C 应追加独立 GP 侧车并保留原表、ID、顺序与精度，不能将 GP Top-32 当作候选池。

### 2.3 科学记忆与响应协议

`scientific_memory()` 仅接受本 seed、method 的历史，保存上一轮原文 hypotheses、feedback_interpretation、batch_rationale；全部历史已选 ID；最近32条真实结果；按两个 target 最大绝对误差排序的8条精确高误差例子。旧观测完整保留在 observed 表，不是从其他方法/seed 蒸馏的科学总结。无 validation reward、测试误差或报告回流。

最终对象含 `type=selection`、`packet_sha256`、`batch_strategy`、32个 `{id,reason,hypothesis_id}`、0–8个 hypotheses、`feedback_interpretation`、`batch_rationale`。hypothesis 的 basis、confidence、status 为枚举；支持/反证 ID 列表只能引用 observed。`validate_direct()` 已检查包绑定、数量、候选合法与唯一、hypothesis 唯一和引用、必填科学说明、30,000字符限制。NoCW 有输入/记忆/输出/纠错消息的禁止 CW 通道扫描与正向白名单。

**运行版本差异很重要：** `nocw_inputs.validate_plan()` 的未执行 paired 版本额外禁止所有自由文本 row-ID 引用；实际 `run_qgeognn_v2_row_llm_nocw_five.py::validate_plan()` 和 CW-Free runner 覆盖了它，保留历史 Free 的 prose-ID 规则与 typed evidence 列表合法性。不得把未执行协议的更严格禁令追加给历史，也不得让新 C 独占这种工程修复。A/B/C 的合法性与证据校验必须完全相同。

## 3. 已有数值方法与同状态研究：不可换名重复实现

| 已有内容 | 实现/研究 | 可复用内容与限制 |
|---|---|---|
| Raw gradient LCMD | `active_learning_v2/lcmd.py`、`qgeognn_v2_row_sequential_b32` | full-network q50 的 CountSketch 梯度与 cluster coverage；不是概率模型 |
| CW gradient LCMD | `gradient_transforms.py::center_width_transform()`、CW extension/factorial 研究 | L0 固定中心/宽度输出坐标与尺度、独立 raw/CW bank；CW 距离不等同 GP variance |
| 固定核 IVR | `ivr.py::conditional_batch_ivr()`、`qgeognn_v2_row_kernel_ivr_b32` | 已有条件化协方差与逐点方差缩减，不能重新命名为新 GP 方法 |
| CW–IVR 数值 portfolio | `cw_ivr_portfolio.py` | 两个专家对共享已选集合逐步 condition；与 LLM CW16+LLM16 是不同方法，不应混称 Hybrid |
| Numerical Hybrid32 | `short_sequential_runner.py` 等既有研究与诊断报告 | 既有 latent/uncertainty 数值组合强基线；必须按 source method/representation 明确标注 |
| Free16+CW-LCMD16 | Scientist lifecycle 与历史报告 | LLM 仅选16、另16为预定 pending CW；不是自由32，也不是本研究 Arm C |
| 化学专家+ML专家→Planner | `dual_expert_selector.py`、`qgeognn_v2_row_llm_dual_expert_v1` | 专家独立 memo、planner 最终自由32；无固定配额/投票；增加调用和上下文，存在历史模型/格式修订，未证明稳定优越 |
| 两套同状态短 rollout | `same_state_branching.py`、`qgeognn_v2_same_state_branching`、`…_cw_hybrid` | 同 L/U/checkpoint 的 anchor lineage、两批短程与预测先冻结流程可复用；它们是纯数值 selector 分支，不是本次 A/B/C LLM 决策实验 |

已有同状态研究采用两个已用开发 seed、每套8 anchor/24分支/48 fits；winner 随 source trajectory、seed 和一步/两步视窗变化。历史报告明确不支持仅按标签预算学固定 stage switch，也不支持据两个开发 seed 训练/部署 controller。本次不复用其 test 结果调阈值。

`qgeognn_v2_row_lcmd_to_ivr_b32/DEVELOPMENT_REPORT.md` 记录两个开发 seed 从同 LCMD-L653 延长，晚期 AULC 方向不一致，已决定停止余下三 seed。`qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md` 的25个冻结状态稳定性门控结论是 `STOP_BEFORE_GATE_C`，无候选通过进入独立训练研究；高 rank correlation 不自动保证 Top-32 稳定，也不能声称已有 IVR 可靠性已充分验证。

## 4. 固定 IVR 的数学性质与 GP 缺口

`ivr.py` 已使用 float64、Cholesky、整个合法 outer-training reference universe 的 RMS 归一化；不包含 test X。若归一化特征为 x，单位权重 prior 与单位观测噪声下，权重协方差为 `S_L=(I+X_LᵀX_L)⁻¹`，条件核为 `C_L(R,R)=X_R S_L X_Rᵀ`。令 `M=X_RᵀX_R/|R|`，greedy IVR 单步效用是 `x_iᵀ S_L M S_L x_i/(1+x_iᵀS_Lx_i)`，每次选点后 Sherman–Morrison 更新并校验实际均方差下降。

这可以解释为固定线性高斯 surrogate 的条件协方差，但源码明确 **无 labels、无 residuals、unit prior、unit noise**，没有拟合 V1/V2 均值、学习残差风险、概率校准或噪声可靠性验证。其单位噪声不能自动解释成真实测量噪声或经校准的不确定性。

对 `src/`、`scripts/`、`tests/` 与研究 Python/Markdown 做了 GaussianProcess/gpytorch/残差GP/posterior_variance 相关检索，排除冻结源码重复、runtime、selections 和本新目录后，**没有发现已有可用的 GP 响应/风险学习模块**。这是所检查范围内的否定发现，不是证明机器其他位置不存在代码。已有 IVR、MaxDet、uncertainty/quantiles 与静态 Numerical Critic 不填补该概率专家缺口。

合法历史 acquisition-time error 可用于未来 prequential 风险分析，但每条必须携带 source checkpoint、L规模、采集轮次、预测冻结时间和 reveal 时间。同一份历史中残差对应不同 QGeoGNN，不能合并后称为某个固定模型的同分布风险。当前 checkpoint 的训练内误差不能作为真实泛化风险标签。

另一个限制是：当前梯度 bank 的 QGeoGNN 已在整个当前 L 上训练。即使仅对 GP 在 L 内做分割/OOF，表示本身仍可能依赖留出点标签。因此它是“给定该表示的 GP 诊断”，不能冒充端到端 QGeoGNN 无泄漏外样本风险验证。真正 prequential 路径应取标签揭示前的预测与表示，或者取得合法冻结、模型未用相应标签的表示；本次不得新训模型来填补缺口。

## 5. 已有行为、覆盖与科学校验审计

下列两个目录是审计时**本地工作区已有但未跟踪**的研究，不能写成该 main 提交已经发布的功能；本次只读复用。

`nocw_free_selection_diagnostics_20261101/` 已冻结584批、历史几何来源与明确共同预算（当时 L973），含 molecule identity、严格条件匹配、raw/CW/latent 的状态内几何、随机批次参考、metadata 来源核验。其核心提醒应直接继承：

- 同分子仅一项条件变化才是严格单因素对照；PE/EA 用 Fraction 等比例，数值字段用 Decimal 原精度，不四舍五入制造匹配。批内对照与对历史 L 的对照分开。
- 不同分子/批、新分子累计覆盖、表示空间距离和实际信息收益是不同量。NoCW 后期可同时“分子更多”且“更靠近已有区域”。
- LLM 可选择样品体积/加载溶剂等局部对照；低 raw/CW 距离和同分子集中不能自动判为低价值。
- 不跨 checkpoint 比绝对梯度距离，不把不同表示的随机分位当共同真实效用，不把参考覆盖下降当学习收益。

`verifiable_scientific_agent_20261030/` 已做310冻结批、462完整响应审计。首次完整答案合法率分别 Free 62.22%、Hybrid 71.83%（LLM16）、NoCW 55.93%；153个非法完整响应首错中114个为候选身份、28个为未观测证据。最终310/310合法仅针对已冻结批次，不能当所有启动轮成功率；未观察到真实同批重复 ID，不能据错误文案捏造重复频率。模型和数量/轨迹不同，这些比例不是方法效果因果估计。

该研究已有：细粒度合法性诊断、稳定候选短索引、保留合法 slot 的局部纠错 mock、原始 validator 判定不变核验。152个追加完整答案有明确 token 代价，但未知价格未填0；还发现整批纠错经常改变已合法选择。故工程纠错代价与数值质量纠错须单独登记，不能将既有 ID 防错包装为 C 的研究贡献。

尤其值得复用 `numerical_critic.py::critique_frozen_plan()`：

| 已有接口能力 | 本次使用方式 |
|---|---|
| 输入块绑定 state/checkpoint/hash/时间；白名单拒绝未来反馈与额外 label 字段 | 上游真实文件 provenance 适配器继续验证；自报 hash 本身不证明来源真实 |
| 全 U/剩余 U 覆盖、到 L 距离、批内近邻、有效秩和逐点边际 | 作为诊断复用；不再实现任意加权 coverage 总分 |
| 32点全部496对、严格单因素、同 tuple、历史反馈邻域 | 用作局部假设豁免的客观结构凭据，不声称验证化学命题 |
| 分位 crossing 与负 width 诊断 | 保留异常，不将负宽度当低不确定性，也不拒绝合法计划 |
| 11项合成 mock 与29项既有几何指标对齐 | 已证明接口与数学计算可运行；没有真实 LLM 纠错或学习收益证据 |

这个 Critic **只有静态诊断，没有 GP、通过/驳回规则、LLM 反馈循环或正式阈值**。本研究真正新增内容应是经过可靠性检验的 GP 信号、正确联合条件信息增益，以及有冻结阈值、可失败/不可判定、保留提案变化轨迹的控制协议。

## 6. 最强可复用组件与优先级

1. 冻结 CW-Free `DIRECT_PROMPT`、全池无损 encoding、`scientific_memory()`、实际五 seed direct validator；用独立 adapter 追加 GP 字段，A/B/C 共用同一合法性屏障。不要导入会执行 Context/reveal/prepare 的生产入口作为离线便利函数。
2. 既有 raw/CW gradient bank、checkpoint/canonical-index receipt 与 L0-only transform；只读、核对 ID 顺序，不重提梯度或挪用不同状态 bank。
3. `numerical_critic.py` 与精确条件对照模块；新 controller 只增量加入数学可检验的 GP covariance/信息收益约束。局部假设豁免也要记录实际 ID 对、轴和原假设绑定，不能凭一句 rationale 自动免除所有约束。
4. `RestrictedLabelStore`、batch freeze、prediction freeze、同初始化 scratch fit 与 report/selector 隔离规则；本阶段不实例化标签库，不访问原始真值文件。
5. 历史 same-state lineage、哈希/diff、已提交 raw validation AULC 汇总口径；Phase 1 只做决策变化，不推断学习收益。Phase 2 必须另行授权和预注册后真实同状态训练。

优先可验证的新增约束为**正确条件协方差下整批信息增益的低尾触发，以及有局部对照结构豁免的高度相关性审查**；阈值未在开发状态冻结、GP 可靠性证据不足或数值不稳定时应返回不可判定，不自动拒绝。盲区覆盖可以报告，但在 GP 风险和表示有效性未验证前不设硬 quota。禁止任意效用权重搜索、GP16+LLM16、CW16+LLM16、GP Top-32 替换或由 controller 自动补点。

## 7. 审计裁决

现有证据足以支持“值得研究数值信息与决策约束是否改变 LLM 决策并降低波动”，**不足以支持 GP-guided LLM 已能提高 QGeoGNN 泛化或稳定性**。现有 Free 已有数值信息，当前 IVR 的噪声假设未经概率校准，合法历史反馈跨 checkpoint 非同分布，已有 Critic 只有静态 mock。因此必须先补可信的不确定性/风险可靠性诊断，并允许 GP 不通过门槛。

Phase 0 的最小增量是：对安全冻结的已观测数据与同状态 raw/CW 表示，报告 GP 均值/校准/风险信号与固定核 IVR 的差异及其标签依赖限制；用 mock 验证联合信息增益、冗余和局部对照豁免、A/B/C 输入差异白名单以及原提案→反馈→修订→最终方案的完整记录。下一项最小真实实验须在新的明确模型/参数/预算授权后，从少量预先冻结状态开始同状态 A/B/C 决策重复；若 GP 不可靠，先补离线证据，不能靠改名把固定 IVR 当作已验证 GP。

## 8. 主要可追溯来源

- 已提交源码：`src/qgeognn_al/active_learning_v2/{scientist_selector,scientist_full_delivery,nocw_inputs,ivr,gradient_transforms,cw_ivr_portfolio,dual_expert_selector,same_state_branching}.py`。
- 实际 adapter：`scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py`、`run_free_cw_sol61_five_20261101.py`；CW-Free 运行冻结副本见该研究 `frozen_source/`。
- 冻结可比性：CW-Free 的 `PROTOCOL.md`、`SCHEMA_DIFF.md`、`CODE_AUDIT.md`、`README.md`、初始 offline checks 与后续授权修订；NoCW 原 paired 协议与 five-seed 实际 runner 必须分别读。
- 发布状态：`reports/free_cw_progress_20261103/REPORT.md`、`reports/five_methods_cw_gradient_20261103/README.md`；本地最新状态来自各研究 progress/receipt 与 fit audit，未修改发布快照。
- 已有负/有限证据：`qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md`、`qgeognn_v2_row_lcmd_to_ivr_b32/DEVELOPMENT_REPORT.md`、两套 `qgeognn_v2_same_state_branching*/FINAL_REPORT.md`。
- 本地未跟踪研究：`nocw_free_selection_diagnostics_20261101/{REPORT,PROTOCOL_AUDIT}.md`；`verifiable_scientific_agent_20261030/{REPORT,NUMERICAL_CRITIC_INTERFACE}.md` 及 `numerical_critic.py`。

关键源码字节 SHA256：`ivr.py=e225af0a79a6e5bf3073e1cf57803be6c11fa9dedfe1139d382d32978ed08204`；`scientist_full_delivery.py=ae5b74d5cff50bb3142986812a94f495761ade0318b395232224ccf1d7f28eb4`；`scientist_selector.py=7a7df9382345add4fbf1a76a50ed8bb85f5726d7fe0ae8f2ea41503261509dc1`。这些是审计引用身份，不替代未来正式实验的全输入 freeze。
