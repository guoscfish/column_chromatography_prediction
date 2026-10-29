# CW-aware / NoCW 的完整输入差异

同一 seed/round/冻结状态下，CW-aware 与 NoCW 输入仅允许下列变化：

| 位置 | CW-aware | NoCW |
|---|---|---|
| 原始 `catalog.json` candidates | `cw_distance_percentile`，保留旧5位小数表示 | 删除整个键 |
| `input.json` / 实际请求的 candidates.columns 与每行 | 相应 CW 列与值 | 删除该列和值，其他列与精度原样 |
| system prompt | CW 距离 percentile 的原定义 | 只删这段定义 |
| `packet_sha256` | 含 CW 输入的绑定 hash | 删除后输入的绑定 hash |

observed/pending 中原本没有 CW 信息；pending 始终空。通用差异审计也检查这些表的 CW列，避免后续接口漂移。host contract 的 method、variant、目录和 input/catalog hash 必然不同，但这些 metadata 不传给 LLM。两个 Arm 的 messages 使用完全相同 transport config。

候选/观测 ID与排列、molecule_id、SMILES、conditions数组和列顺序、descriptors、`pred_V1_ml/pred_V2_ml`、V1/V2 q10/q50/q90、分位宽度、Morgan相似度、loading_amount_mg、新颖性、record_kind、已测 V1/V2、premeasurement预测及误差、historical_abs_error_ml 均保持一致。相同状态下 scientific_memory 和top8误差排序也相同；NoCW 的初始记忆为空，后续只读自己的叙事/测量。两个轨迹分叉后不再要求其内容相同。

保留这些数字是因为问题只针对 CW 数值先验；预测、误差、不确定性代理、分子描述符与结构/条件覆盖线索都是原自主 LLM32的信息接口。不得称为 chemistry-only 或 completely pure LLM。

catalog 的候选正向白名单：id、smiles、五个原始条件、重算描述符、q50预测别名、六分位预测、CW（仅aware）。观测正向白名单另含原有 OBSERVED_FIELDS。未知字段、嵌套描述符键、非有限数字、条件注释通道和候选真值一律拒绝。raw NoCW catalog也扫描，不能只在编码时隐藏。描述符使用旧实现重算，两个输入不改变其舍入规则。

新实验共同修改（不构成 Arm间差异）：Free32-only prompt、共同 salt、host-only方法名、typed evidence引用、原 memory_policy 的“旧记录可查询”改为“已包含在观测表”。历史 runner 与旧 prompt 完全不变。

记忆不是对历史 Free 记忆做字符串清洗：新实验从空记忆开始，自己的后续输出经 typed schema、证据验证和 NoCW 序列化审计后才允许冻结；污染即失败。禁用显式 CW距离/排名/分数、常见梯度覆盖等价词、LCMD来源和其中文描述；无法允许任意新增 numeric-summary 字段。实际 outgoing messages 的全部回合，包括纠错中前一答案，逐次检查。

机器差异输出：`dry_run/seed_{157,6101}/input_diff.json` 列出每条被删除的候选CW值；`dry_run_audit.json` 存计数、包hash与检查结果。任意其他值、候选排列、数字精度、分子表或记忆变化都使 `pair_diff` 抛错。
