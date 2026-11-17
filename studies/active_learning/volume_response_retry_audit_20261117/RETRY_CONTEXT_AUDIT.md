# 客户端纠错上下文审计

## 结论与分母

**在本次可核验的客户端请求层，没有发现纠错时丢掉原始候选池、旧对话或上一份完整assistant回答。** 共审计540个Free32轨迹的已接受batch、1120个保存答复/逻辑请求节点。850份有原始`request_XX.json`；Historical的270个节点由实际冻结packet、revision提示、配置、原答及纠错记录重建，全部与当时`request_sha256`精确相等。不能由客户端证据进一步保证中转服务内部没有裁剪、修改上下文或改变后端模型。

540不是540份未经人工修改的LLM最终方案：539批与唯一原始最终回答完全一致；Historical seed157 r5（L493→525）曾有已授权的单ID抄写修正，全部其他字段与理由保持不变。它不属于127个前缀案例。本文的540是完整历史记录分母，严格原始最终答子集为539；127个前缀在两口径下分别为23.52%和23.56%。详见`retry_cohort_scope.csv`、`retry_manual_selection_exception.json`。

## 1. 从实际入口到传输层，而不是只读一份源码

| 运行层 | 实际链条和适用边界 | 证据 |
|---|---|---|
| Historical | `scientist_study.select`→`scientist_transport.run_selector/run_full_selector`→`_call_with_retries`→`responses_call`；额外授权轮另有记录 | 各真实owner的`protocol.json`、`selector_prompt.txt`；8个owner×3个关键源码hash全部匹配；270个逻辑请求hash重建 |
| New CW初始 | frozen adapter激活独立`nocw_study`，替换CW输入/validator；底层用冻结`stream.body_for`与`live_stream` | `source_snapshot.json`、`frozen_source/`、实际request/attempt/stream记录；3个关键冻结源码hash匹配 |
| New CW后续 | `concurrency_v2/frozen_runner.py`及`persistent_20261102_v2/frozen_runner.py`覆盖controller与retry；仍调用原冻结`live_stream/body_for` | 实际persistent amendment标记、授权JSON、保存的完整messages及请求hash |
| NoCW | 实际five-seed adapter覆盖paired默认规则，经过stream、授权额外纠错、unbounded与后续恢复/重放controller；不能把初始3×3当全程限制 | five-seed `protocol.lock.json`中3个关键源码hash匹配；各实际request/attempt及恢复源码；NoCW420个请求hash全部吻合 |

具体函数路径、行号与SHA在`retry_controller_sources.csv`；源码绑定在`retry_source_contract_checks.csv`。Historical与New冻结副本的`scientist_full_delivery.py`相同。本次只执行其中纯validator AST和纯`body_for`构造函数；没有导入或执行任何runner的训练、凭据读取或网络函数。

Historical中三个第4次回答最初无法用常规8消息模板匹配：seed157 r5、6101 r0、6101 r19。进一步读到各自`extra_correction_request.json`与`authorized_extra_correction.json`，实际是**保留常规assistant＋错误消息后，再追加一条授权user反馈**，合计9条；不是替换旧错误。按这个已保存反馈重建后，三个请求与原始receipt及额外请求记录SHA全部一致。不能把最初模板不匹配误报为实际上下文丢失。重建过程及前后派生表hash见`retry_historical_extra_contexts.csv`、`retry_reconciliation_record.json`。

## 2. 实际发出的消息内容与顺序

初次请求为：

1. `system`：对应版本完整系统说明。
2. `user`：完整packet的无损紧凑JSON，包含全部当轮合法候选、当前已观测记录和已有科学记忆。

常规逻辑纠错保留全部旧messages，再追加：

3. `assistant`：上一轮完整的原始answer字符串，没有只取ID或只取摘要。
4. `user`：真实validator错误及要求返回完整最终JSON的纠错指令。

1120/1120个节点的原始packet解析对象与`input.json`完全相同；580/580次续接完整保留前次messages，且完整assistant原答和原错误对象都可逐条核对。Historical有3条额外授权反馈；NoCW887 r9、6101 r11、6101 r20的turn3另有保存的packet-hash格式提示。这6次各增加第三条user消息，没有删去前文。实际roles、内容长度、UTF-8字节数和内容SHA保存在`retry_message_inventory.csv`；不复制带历史观测值的全文到交付目录。

配置和载荷没有提供`previous_response_id`、`conversation`或`conversation_id`来依赖服务端续聊。Legacy SDK调用显式`input=messages`；New/NoCW的冻结`body_for`也把完整messages直接置于`input`，附model、reasoning effort、tools空列表、store=false、max_output_tokens和stream。所有实际sampling参数的键也已检查，没有偷偷加入continuation参数。每次新的逻辑请求都重新携带初始pool及整个已保存对话，不能将“同一HTTP接口”误认为自动记得上一次请求。

## 3. 逻辑纠错与传输重试是不同事件

逻辑纠错获得一个新的完整回答，校验失败后追加assistant/error并增加turn；传输重试则重发**同一turn、同一messages/config hash**，或从已持久化完成记录恢复，不能把网络故障当成新的科学提案。

| 运行 | 保存答复/逻辑节点 | 传输attempt/失败事件记录 | 其中记录成功 |
|---|---:|---:|---:|
|Historical|270|305|旧ledger只记失败，不从0推断没有成功调用|
|New CW|430|173511|430|
|NoCW|420|1907|420|

175723条传输记录与其逻辑请求hash均一致。大量记录是连接、HTTP状态或协议故障，不等于175723次付费LLM推理，也不能由失败次数估算费用或模型质量。Ledger包含已恢复本地完成状态，旧ledger还只记失败；真实账单与服务端执行结果并不完整可见。传输错误只输出类别/状态及文本hash，没有输出可能带敏感内容的原始错误正文。

**对前一轮报告的实质修正：New CW不是全程有限8×4。** 初始config确实为8个逻辑回答×4次传输；随后`persistent_20261102_v2/authorization.json`明确将answer/transport/restart limit置null，冻结persistent runner的两个循环均使用`itertools.count()`。178/180个New批次有实际persistent标记；三个批次的逻辑回答数达到11、33、82，82次对应164条messages。NoCW初始config的3×3也被额外授权和持续恢复扩展，16批超过名义3个回答。后期某些“每次dispatch最多新增请求数”也不能当成整批历史累计turn上限。字段中保留`nominal_*`与实际观察，不能拿初始config替代真实controller历史。

## 4. 模型、预算、端点与服务端可观测边界

实际四类配置如下，完整hash见`retry_transport_configs.csv`。端点和凭据值不输出；全部观察请求属于同一个主机摘要，故本数据没有官方API与另一服务商的并行对照。

| 配置hash前缀 | 请求模型/effort | 客户端生成预算 | 解释 |
|---|---|---|---|
|`9de220891739`|gpt-6-astra / high|客户端未显式设置max_output_tokens|Historical首请求16批，包括后来人工修正的一批|
|`ff5da8f292de`|gpt-6.1-sol / xhigh|客户端未显式设置max_output_tokens|Historical首请求164批|
|`eb3977a296d3`|gpt-6.1-sol / xhigh|32768|New；8×4仅为初始名义限制|
|`24e3daae2f49`|gpt-6.1-sol / xhigh|32768|NoCW；3×3仅为初始名义限制|

原始模型字段和usage保留于逐proposal表；返回模型字段不等于独立认证了服务端权重身份。Historical未显式设置预算不表示服务端没有限制。没有读任何credential文件；只读取已有配置并输出端点SHA、主机SHA、模型/effort及非敏感预算字段。

430份保存的server response是客户端主动筛选后的白名单（id/model/status/usage/output），**没有保留truncation、previous_response_id、conversation等原始服务端字段**。因此机器表中的UNRECORDED或非空计数0仅是日志缺失，不能解释为服务端没有裁剪或没有隐藏上下文。本次能够确认客户端未提供这些续聊参数、消息层未发现删减；不能证明中转站内部处理完整，更没有证据宣称其降级了模型。

## 5. 校验、文件与复现

581个保存失败的validator错误均按对应真实三方法规则精确复现；其中包含人工修正前最后一个仍非法的原答。各分母不能互换：539是能唯一匹配原始最终回答的已接受batch，1120是原始答复节点，540是完整已接受batch记录，127是完全同序前缀事件。

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/retry_context_audit.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/volume_response_retry_audit_20261117/finalize_retry_audit.py
/Users/fish/miniforge3/envs/chromatography/bin/python -B -m unittest discover -s studies/active_learning/volume_response_retry_audit_20261117/tests -p 'test_retry_context_audit.py' -v
```

必须先运行主审计，再运行finalize；后者纳入三个额外授权上下文、人工修正证明及最终汇总。`retry_audit_inputs.json`保留全部实际输入路径/hash；`retry_audit_verification.json`记录精确覆盖、语义校验与人工例外。7项单元测试覆盖请求hash重建、完整原答和错误消息顺序、额外user提示、ID合法性、首错分类、拒绝从坏JSON补造候选与端点脱敏。没有API调用或模型拟合。
