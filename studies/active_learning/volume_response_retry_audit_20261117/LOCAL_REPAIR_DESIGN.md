# 离线局部纠错：保留方案，分开处理ID与科学证据

本轮实际dry-run了127个最终严格选择展示表前32行的历史批次。每例都从已保存首份可解析的32项proposal读取原始科学选择，核对turn SHA、freeze、packet和当轮H/U/pending，未调用API、未更改selector或历史selection、未训练。

## 实测保留能力

| method | cases | preserved | vacant | historical_retained | science_review | structural_ready |
|---|---|---|---|---|---|---|
| historical_free_mixed_model | 25 | 719 | 81 | 101 | 9 | 0 |
| new_free_cw_sol61_xhigh | 40 | 1151 | 129 | 107 | 9 | 0 |
| nocw | 62 | 1764 | 220 | 232 | 9 | 0 |

共4064个原选择项，原合法唯一ID为3634，原型保留3634，即所有原本合法唯一选择均保留，保留预算比例89.42%。历史最终前缀方案仅保留其中440个。原型留下430个明确空额，不用表前32或其他候选自动补位。

以上都是每case先唯一化、再跨case累计的选择事件，不是全仓库不同候选ID的总数。特别是首方案已有31个合法ID的38例，原型保留1178个，历史最终只保留119个；只留下38个ID空额。不过其中10例仍需科学证据复核，不能把它们都称为“只补一个ID即可”。

必须区分：**ID保留不等于科学证据已被接受**。27例仍有可确定定位的未观测证据引用问题；历史validator在本地草稿后完整通过0例。本次确定性字段规范化发生4次，全部是有严格原请求证明的根packet_sha恢复，证据内容保持不变；没有为了提高“修复率”擅自删除证据、截短方案或添加理由。两个原本32个ID都合法的案例在恢复binding后仍有未观测证据问题，故不能宣称完全自动恢复。

127例的首份可解析方案均有32项，未出现整个choice容器无法恢复的案例；这不等于所有首次API响应可恢复：本样本以“历史首份可解析32项proposal”为条件，不能外推到JSON损坏、截断或缺日志的人群。非法ID的原始科学身份依然不能凭空恢复。125例需要补ID，另2例ID完整但科学证据仍需复核。

## 原型算法与不变式

1. 当轮可保留集合为实际U与展示候选的交集，再排除pending和已测H。候选顺序完全不用于补位。
2. 按原proposal顺序保留合法ID的首次出现，重复项单独标注。每个保留choice的reason、hypothesis引用和全部其他字段逐值不变；全部hypotheses、批次解释逐值不变。
3. 支持两种确定性工程规范化：ID首尾空白剥除且精确匹配唯一当轮候选；或原保存request的messages/config哈希与该turn精确匹配、user packet逐值等于当轮input、packet内部绑定自校验均通过时，复制已知正确的根packet_sha。后一操作保留原wrong值和全部证明哈希；不能据此证明模型内部使用了正确上下文。缺任何一项证明就不重绑定。不会编辑近似ID、纠正拼写、猜分子、删除候选型证据、把basis改成prior，或生成新的实验假设。
4. 逐个列出缺额、无效ID、错误证据归属、hypothesis引用、缺少文本、超长方案和NoCW字段边界。需要新科学判断的项保持阻断。完整草稿是审查材料，不是可执行selection。
5. 使用C1–C3共用的真实函数纯AST接口，执行Historical direct、New frozen CW wrapper及NoCW五seed wrapper；没有引入未运行paired协议额外prose-ID禁令。最终已接受的127份历史selection作为validator兼容性对照，失败数0。精确运行版本映射还应与C1–C3的transport审计一起读。

## 有效性层级

- **ID层**：合法、唯一、在当轮未标注/展示集合、非pending，可确定保留。
- **schema与证据成员资格层**：原字段、hypothesis引用、观测证据ID集合等可以自动检查；正确引用一个已观测ID仍不能证明其支持了该假设。
- **科学内容层**：理由是否与分子/条件对应、假设是否真正被证据支持、冲突证据如何解释，需要人工或未来授权LLM重新判断。当前不执行Judge，也不以通过schema冒充科学正确。

## 能减少什么，尚不能说明什么

dry-run说明无需丢弃大量原本合法的候选即可定位待修项；若未来只允许填空与定点证据修订，可以在协议上避免整批重选。然而这不是已观察的模型纠错成功率：本轮没有调用LLM，不能证明它会遵守局部编辑指令、费用必然下降、前缀率必然降低或Row性能改善。旧历史中“首方案→前缀最终”与本原型是确定性保真对照，不是学习收益反事实。

草稿choices为保留项按原相对顺序压紧的审查列表，`open_slots`表示还差多少个列表位置；`unresolved_original_choices`另存每个被排除项的原位置、完整reason/hypothesis引用与错误原因，防止把未知ID背后的科学意图一并丢弃。候选级`science_gate_status`仅标出是否发现关联的schema/证据成员资格问题，未发现问题仍不等于科学语义已证实。若新的科学复核确需撤销某个合法ID，应作为显式科学修订记录，不能伪装成确定性格式修复。

复现：`python -B studies/active_learning/volume_response_retry_audit_20261117/local_repair.py`。候选级操作见`local_repair_candidates.csv`；待补槽位及不变科学文本见`local_repair_drafts.jsonl`；问题定位见`local_repair_unresolved.csv`。所有输出只在本研究目录。
