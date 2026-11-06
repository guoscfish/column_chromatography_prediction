# 失败类型与判定边界

本表区分“执行前可判定的硬错误”和“需要科学解释的行为”。一个响应可有多个细粒度问题；按事件统计时，主类别只计一次，多标签诊断另列，不能相加当作请求数。

| 层级 | 子类 | 可验证证据 | 处理 |
| --- | --- | --- | --- |
| Transport | 连接失败 | 已落盘 attempt 的 exception / error_type | 同一逻辑答案的传输重试；没有完整 answer 时不分析选择 |
| Transport | timeout | 明确 timeout 异常或错误记录 | 同上；耗时只用记录值 |
| Transport | 流中断 / 不完整响应 | 明确响应不完整或 stream 错误 | 不能把部分 SSE 文本当作候选计划 |
| Transport | API / 限流 | 明确 HTTP/API/rate-limit 记录 | 状态不明确时保留 unknown，不由长耗时反推 |
| Syntactic | JSON 解析 | 完整 answer 不能由原解析器解析 | 唯一完整 JSON fence 可尝试无语义格式处理；其他歧义需重新规划 |
| Syntactic | schema | 根、choices、hypotheses、必要字段或字段类型不合规 | 返回字段路径；结构不可靠时不强行提取确定批次 |
| Syntactic | 数量 | choices 数量不同于 packet.selection_count | 可靠结构可请求补全/删减；删哪些须由 LLM 决定 |
| Constraint | 批内重复 ID | 同一个真实候选被多次选择 | 明示所有重复位置，保留首项仅作为待确认局部修复基础，不直接冻结 |
| Constraint | 非当前 U_t ID | 不在完整冻结候选集或实际 supplied 集 | 列出 ID 与位置；禁止近似匹配、猜哈希或随机替代 |
| Constraint | 已观测 / pending ID | 在 L_t 或 pending 集 | 单独诊断，不能把已测量 ID 当新增实验 |
| Constraint | 假设引用 | choice 指向不存在的 hypothesis；hypothesis ID 重复 | 修复引用关系；不能静默删除科学假设 |
| Constraint | 证据污染 | typed supporting/contradicting 列表引用非 L_t ID | 要求 LLM 修改证据关系与相关解释，宿主不伪造证据 |
| Constraint | 冻结协议 | packet binding、大小上限、NoCW allowlist / CW 内容规则等 | 使用对应实际运行方法的冻结规则；不以未运行协议重写历史合法性 |
| Scientific decision issue | 分子集中、条件覆盖、几何冗余、与 L 相同条件 | 冻结选择与事前元数据 / 已有梯度表示 | 描述性指标和复核线索，不构成自动拒绝或无效实验判定 |
| Scientific decision issue | 理由与实际条件不符 | 原文声明与确切条件对照逐项比较 | 先确认对照含义；多轴对照不能冒充严格单轴对照 |
| Scientific decision issue | 反馈与后续假设不一致 | 该轨迹事前已观测反馈、后续明确命题及其引用 | 需语义审查；“未再选择该分子”不等于没有吸收反馈 |
| Scientific decision issue | 潜在低数值边际收益 | 同状态下已有覆盖 / 冗余指标 | 不等于学习无收益，不直接决定替换 |

`validate_direct()` 是 fail-fast 校验器，因此历史 rejected 文案往往只呈现第一个问题。“distinct legal supplied”不能仅凭字符串分成重复或池外；细分类必须用该轮完整冻结 U/L 和完整 response 重新诊断。JSON/choices 无法可靠解析时，候选数量、保留率与意图比较标为不可分析。

必须区分同名校验函数：`nocw_inputs.validate_plan()` 的未运行 paired 协议禁止 prose 中出现已知行 ID；实际五 seed runner `scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py` 覆盖该函数，明确保留 Free 的 prose ID 规则，仅追加字段 allowlist 与 CW 内容限制。审计采用实际 runner 的版本。旧 Free/Hybrid 允许某些 prose 引用；不能用未运行 paired 协议给历史方案追加失败。未知 prose ID 即使未被现有程序拒绝，也只是一条需解释的语义线索，不能未经审查断言为伪造实测证据。

“最终冻结合法”与“首次答案合法”是不同指标。人工授权增加纠错额度、历史抄写修正和模型版本切换单列；最终成功不能抹去这些代价。没有新 LLM 对照时，也不能声称局部修复降低了真实请求数。
