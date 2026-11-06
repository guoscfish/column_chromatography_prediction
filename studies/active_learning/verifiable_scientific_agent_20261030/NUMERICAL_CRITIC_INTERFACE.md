# Numerical Critic：冻结计划的离线诊断接口

2026-10-30。实现为独立的 `numerical_critic.py::critique_frozen_plan(snapshot)`，只接受显式传入的 JSON 兼容数据并返回字典。它不访问文件、不加载 checkpoint、不重新计算预测或梯度、不调用 LLM、不请求标签，也不导入生产 worker/selector。当前只在合成 mock 上执行；没有对正在运行的 NoCW 接入适配器。

本接口准备 Level 2 的可检验诊断。它不评定真实学习收益，不验证 Level 3，也不替代 Phase B 的输出 schema/证据 validator。输入计划必须已经是 32 个合法唯一候选。

## 1. 冻结输入与信任边界

顶层字段严格为 `schema_version="numerical-critic-v1"`、`state`、`metadata`、`predictions`、`gradient_spaces`、`plan`、`feedback`。未知字段会被拒绝，不能携带候选实测响应或 test 标签。完整可运行的构造例子见 `test_numerical_critic.py::mock_snapshot()`。

除 schema_version 外，各对象是冻结块 `{source, payload}`；gradient_spaces 是此种冻结块的非空列表。每个 source 恰有以下字段：

| 字段 | 含义与检查 |
| --- | --- |
| `source_id` | 上游冻结来源的非空身份字符串；仅记入回执，不作为路径打开 |
| `payload_sha256` | canonical JSON 的 SHA256；用 `payload_hash(payload)` 重算，禁止 NaN/Infinity |
| `state_sha256` | 当前 state.payload 的 canonical JSON SHA256；每块必须一致 |
| `checkpoint_sha256` | 当前已冻结模型身份；每块必须与 state 一致 |
| `frozen_at` | 带时区 ISO 时间，必须不晚于 state.selection_at |

canonical JSON 使用 Python `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` 的 UTF-8 字节。**这是接口载荷摘要，不是历史文件原始字节摘要。** 上游应先依据已有 batch_freeze、contract、bank receipt 核验原文件身份、已有 bank 的 checkpoint/索引顺序，再导出白名单字段；本次不实现这个真实数据适配器。

这些检查能够发现载荷篡改、混入另一状态、顺序错位和时间越界，不能由自报哈希证明来源真实。`pool_partition="train_pool"` 同样是上游可信冻结声明：接口没有全数据集或 test ID 表，不能独立认证这些 ID 的原始划分。源真实性与 train pool 归属必须由既有冻结协议保障。不要把重新计算一套相互一致的伪造哈希当作来源证明。

| payload | 严格字段与约束 |
| --- | --- |
| state | seed、round、selection_at、checkpoint_sha256、pool_partition、synthetic、L_t_ids、U_t_ids；round 为非负整数，L/U 非空且各自唯一、互斥；pool_partition 仅允许 train_pool |
| plan | `ids`：恰好 32 个唯一 ID，全部在当前 U；不得属于 L；保留原始计划顺序 |
| metadata | `ids` 与 `rows`，顺序严格为 L+U；每行仅 id、molecule、conditions，molecule 为冻结 SMILES/分子身份字符串 |
| predictions | `ids`、`q50`，可选成对 q10/q90；顺序严格 L+U，形状 N×2，对应 V1/V2，有限数值；既有模型的 quantile crossing 被接受并单列诊断，不作合法性失败 |
| gradient space | name、kind、ids、features、preprocessing；name 唯一，顺序严格 L+U，features 为 N×d 的有限数值、d>0；preprocessing 记录原冻结变换身份，不在此重新拟合 |
| feedback | 列表；每项 evidence_id、id、observed_at、acquisition_round、error_v1_ml、error_v2_ml；ID 只能来自 L，evidence_id 唯一，observed_at 严格早于选点且不晚于反馈块冻结，acquisition_round < 当前 round；残差可同时为 null |

conditions 恰有五个原始条件：`PE/EA`、`Density g/ml`、`V/ul`、`loading solvent`、`Volume of loading solvent/ul`。数值条件须为有限数值，溶剂为字符串，PE/EA 为合法比例字符串。匹配采用冻结值完全相等，不用浮点容差创造新的“近似匹配”，也不将 density×V 再算作一个独立轴。

kind 是本接口的语义类型，生产数据可用 `current_full_network_q50_gradient_CountSketch` 或 `current_center_width_linear_output_gradient_CountSketch`；这不表示二者在所有历史 bank 中具有同名原始字段。上游必须从真实 bank receipt 解释该类型，禁止仅改名称混用 raw/CW。`mock_frozen_gradient` 仅在 synthetic=true 时允许。缺少的另一空间直接缺省，不重提梯度，也不从其他方法借 bank。不同空间独立输出，不合并成总分。

反馈仅携带**选择前已有**的 acquisition-time prediction−measurement 残差，不携带 U 的测量响应。初始观测没有这类残差时留 null。没有反馈时传空列表，诊断显示 NA；本轮批后反馈只能成为下一轮输入。多条合法回执指向同一已测样本时，top8 汇总按 observed_at 取最新一条，每个样本只计一次。

## 2. 输出与现有指标对齐

返回 `numerical-critic-result-v1`、状态摘要、原顺序 selected_ids、所有来源回执与六类诊断：

| 输出 | 内容与解释 |
| --- | --- |
| geometry[space].metrics | 选点到 L 的距离及池内平均并列 rank、top-decile 比例、U 覆盖变化、排除整批后的剩余 U 覆盖、bank centered RMS 定标、批内最近邻、精确特征重复、centered effective/participation rank |
| candidate_pair_relations | 32 点的全部 496 对关系：同分子、改变哪些原始条件、完整分子条件 tuple 重复、每个已有表示中的直接差分距离 |
| candidate_diagnostics | 每个候选的距离/rank/近邻、QGeoGNN q50 与可用的 q90−q10 宽度、与 L 的严格匹配轴、batch 严格匹配轴、完整 tuple 重复、真正新条件点补全 |
| scientific_controls | 同分子且仅一个条件不同、其余四条件完全一致的实际 ID 对；区分 prior L 与同批 peer，给出固定轴、预测差与冻结梯度距离；只证明对照结构 |
| review_items | 可复核的重复 tuple、精确特征重复、批内最近邻比到 L 距离少于一半；同时附严格科学对照关系，避免把几何接近自动解释成无用实验 |
| preselection_feedback_context | 已有反馈数量、历史 top8 误差证据 ID、所选/整个 U 的同分子比例及富集差；无历史残差时 null |

既有 QGeoGNN 的 q10/q50/q90 可能交叉。`prediction_diagnostics` 输出候选池 V1/V2 各自 crossing 数量和负宽度数量，每个已选点也保留 crossing flag 与原始有符号 q90−q10。它是模型质量/区间解释问题；接口不会修正分位数、将负宽度解释成低不确定性，或因此驳回合法计划。

核心覆盖定义直接对齐 `free_lcmd_mechanism_diagnostics_20261029/geometry_analysis.py::geometry()` 与 Hybrid 审计。令 d(u,L)=min_l ||g(u)−g(l)||，b(u)=d(u,L)，a(u)=d(u,L∪B)。full-U 相对下降为 `(mean_U b−mean_U a)/mean_U b`；remaining-U 同时在前后排除最终完整 B，避免选中点自归零直接构成增益；RMS 来自同一冻结 L+U bank 的中心化均方范数。分母为零或 remaining U 为空时为 null。绝对距离只解释当前状态和当前表示，不能跨轮/方法空间直接作尺度比较。

新增逐点 prefix 输出只是将 Hybrid 已用的同状态边际覆盖分解扩展到原计划顺序：每加入一个点，记录前缀增量的 full-U 平均距离下降，以及固定 `U\B` 上的下降/RMS。所有前缀使用同一个最终排除集，增量之和等于整批对应覆盖变化。它依赖原计划顺序，**不是候选质量排序**，也不是可加的真实学习收益。

最近邻 `nn < 0.5*d_to_L` 采用原审计已有诊断口径；没有根据 validation 结果调阈值。rank 采用 average tie percentile，与 pandas rank(method="average", pct=True) 一致。有效秩用中心化奇异值的熵，协方差 participation rank 单独输出。严格 matched contrast 和 curve extension 沿用原始条件完全匹配，且 extension 排除在 L 已见完整 tuple。

## 3. 可复核结果的使用协议

接口只返回事实与审查理由，没有 weighted utility、通过/驳回决定、自动替换、32 点重排序、CW 配额或 loading cap。低 CW/raw 距离不触发“无效”判定；同分子集中可以同时构成科学条件对照。mock 专门放入一个完全相同梯度表示的 loading 对照，验证它同时出现在几何复核和科学对照中，且计划保持原样。

若将来研究 Numerical Critic 反馈，需另行冻结同一个 L/U、checkpoint、预测、bank、信息可见性与合法原计划，在发新请求前规定反馈文本、允许局部调整的范围与停机规则。首先比较反馈前后 ID/科学对照/覆盖的变化；只有未来严格同状态、相同标签预算和训练规则的对照才能检验 AULC/学习效率。本次没有新 LLM、训练或闭环运行，不能据 mock 宣称学习效率提高。

## 4. 测试与已知局限

运行：

```bash
/Users/fish/miniforge3/envs/chromatography/bin/python -m unittest discover \
  -s studies/active_learning/verifiable_scientific_agent_20261030 \
  -p test_numerical_critic.py -v
```

测试仅构造 8 个 L、40 个 U（另测恰好 32 个 U）、5 维合成特征。覆盖合法计划不变、精确特征重复仍保留 loading 科学对照、单轴与多轴区分、共同分母的边际可加性、零 bank/空剩余集/空历史/缺分位、哈希/checkpoint/冻结时间/ID 顺序、已测/未知/重复/数量不足计划、当前轮与未来反馈、额外标签字段、数组形状、非数值/NaN；另验证分位交叉与负宽度只作为模型诊断且计划保持原样。29 项几何指标在同一个合成状态与已有 Free 审计逐项对齐，容差 1e−11。

NumPy 是接口唯一第三方依赖；metric-parity 测试导入已有只读几何脚本，额外需要 pandas，不执行其 main。测试不读取真实历史 bank、candidate/test 预测 CSV 或反馈标签。验证结果与 mock 耗时在 `numerical_critic_verification.json`，只是小规模离线开销，不代表真实候选池运行成本。

尚未验证：真实冻结文件导出适配器、上游来源认证、全池运行开销、化学解释质量与真正科学意图、QGeoGNN 区间校准、任何 learning gain。严格条件匹配可能漏掉语义等价但编码不同的条件；本阶段宁可保留明确口径，不自动归一化扩大科学对照。反馈 top8 只是历史误差邻域上下文，不是当前候选真实误差。未计算 Morgan/骨架相似性，避免为最小接口加载化学依赖；以后可在冻结元数据白名单中扩展。
