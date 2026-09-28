# 4G LLM Scientist Active Learning V2

## 研究问题与解释范围

研究问题是：在明确化学场景、预测目标、当前模型输出和本轨迹实验反馈后，LLM 能否自主设计下一批实验，提高 QGeoGNN 在 Row 分布上的 V1/V2 样本效率？数值信号是证据，不是必须遵循的 acquisition formula。

本实验比较的是完整的 LLM scientist 策略，不能单独归因于化学知识或 ML 知识。即使两枚 seed 上优于 Random/CW，也只能提供开发阶段证据。后续要检验知识的独立增益，需要保持模型、调用预算、检索界面不变，增加去化学信息、去领域提示或去反馈等消融，并使用未被历史研究反复查看的独立测试设计。本轮不新增这些 arm。

## 代码基线与历史保护

已 fetch origin 并执行 fast-forward pull。最新 LLM 基线是 `codex/llm-al-full-pool-feedback` 的 `647271d`，包含 dialog feedback V1。`main` 在 `d9b4518`；另一条 CW performance 分支的独立提交是 `6ac0f9e`，不包含更晚的 LLM 实现。

新分支为 `codex/llm-scientist-v2`。只新增 V2 模块、入口、测试、设计文档和独立 study；不改 V1 源代码，不覆盖旧 selection、result、freeze。旧 LLM study 文件逐一哈希记录在 `historical_artifacts.json`。`validate` 同时校验旧文件清单、旧文件哈希、控制组源文件哈希和 V2 代码/提示词。

独立目录为 `studies/active_learning/qgeognn_v2_row_llm_scientist_v2/`，不复用其他 seed/method 的科学记忆或选点结果作为 LLM 上下文。

## V1 到 V2

| 项目 | V1 | V2 |
| --- | --- | --- |
| 模型角色 | full-pool supplement selector | scientific experimental planner |
| 科学场景 | 简短的 V1/V2 预测描述 | 4 g 正常相 silica、PE/EA、结构及载样语义 |
| 初始展示 | 数值 shortlist 前 20 个 | 与行顺序无关的轮次盐值 ID 哈希抽样 |
| 默认搜索 | 过滤后按底层顺序截断 | 稳定哈希顺序、显式排序、分页、总匹配数 |
| 参照 | candidate/observed | 增加 pending 参照及直接查询 |
| 数值信息 | q50、CW 距离 | 增加既有 q10/q90、量化宽度、描述符、新颖性、相似性和已观测误差 |
| 反馈 | 全部历史重复进入 prompt | 最近一批、最多 8 个原样高误差记录、上一轮 LLM 自写解释与假设 |
| 选点 | CW16 + LLM16 | 保留 hybrid，并增加真正无 pending 的 Free-LLM32 |
| 输出 | 单点理由与假设 | 增加 batch strategy/rationale、假设 basis/confidence/status |
| 调用 | 持续的独立 Codex task | 每次无历史的 CLI 或 Responses 调用，通过 JSON relay 查询 |

V2 不向模型传入 IVR/MaxDet/CW recommended 标签，也不传入 128 点数值候选名单。没有发现可直接复用且绑定当前新轨迹 checkpoint 的 IVR/MaxDet 分数，所以本版不额外计算这些分数。CW 梯度距离仍作为中性、可选证据。没有新增 ensemble 或 predictor。

## 每轮可见信息

1. 本 seed、method、round、当前标注数、要求选 16 或 32 个点、候选池概况。
2. 20 个哈希顺序候选卡片和 12 个哈希顺序已观测示例；所有当前合法候选与 L_t 记录仍可查询。
3. 每个候选的真实存在的 X：canonical SMILES、PE/EA、Density g/ml、V/ul、loading solvent、Volume of loading solvent/ul。
4. 由结构计算的 MW、LogP、TPSA、HBD、HBA；按现有 predictor 输入语义计算的 density × volume 载样量（mg）。
5. 当前单模型的 V1/V2 q10、q50、q90，q90−q10，CW 梯度距离百分位，和 L_t/pending 的分子相似性、新颖性。
6. hybrid 的 pending CW16 特征及预测，且完全没有它们的标签；free arm 的 pending 为空。
7. 当前 L_t 的真实 V1/V2。L333 没有伪造的 out-of-sample error；之后获得的记录附带当时冻结的 prediction-minus-truth error。
8. 最近一批的真实反馈、LLM 单点理由/假设引用、上一轮假设及其状态、LLM 自己的反馈解释和 batch rationale、历次 selected IDs、最多 8 条按最大绝对端点误差排序的原始记录。

候选卡片通过字段白名单构造；candidate/pending 中出现真实标签字段会直接报错。观察记录只来自本轨迹 L_t。memory 通过 `(seed, method)` 严格检查，不用第二个 LLM，也不生成算法写的化学总结。重要趋势和反证可由 LLM 保留在假设或上一轮解释中，原记录可重新查询。

## 不可见信息与泄漏边界

LLM 看不到 validation/test 的标签、样本卡片、误差或成绩，看不到其他 seed 或其他 method 的实验响应，也看不到旧研究的测试结果。训练器沿用原验证集做 checkpoint selection；这与把验证标签交给 selector 是两件事。dry-run 连验证标签也不读取。

四个 LLM trajectory 均从对应 seed 相同的 Row split、L333、QGeoGNN checkpoint 和固定 target scales 开始。所有新标签必须经过 `batch_freeze.json` 后才能 reveal。freeze 绑定 input、catalog、contract、逐次请求/回答、query results、selection，以及 32 个点的预实验预测。重放会校验查询结果、selected IDs 和预测值；修改任一绑定文件都会失败。

`select` 只冻结选点。`advance` 单独读取这批标签，记录标签访问凭证与反馈，然后按原协议从相同初始化训练一个下一预算模型。没有自动开始下一次 selection。`report` 要求两 seed × 两 LLM arm × 七预算的 28 个预测点全部冻结，通过全局审计后才允许读取 test truth。Random32/CW32 精确复用 28 个历史预测点，不重训，不在选点阶段读取历史指标。

历史 Row test 已经被以前研究查看过；V2 的技术屏障不能消除研究者层面的历史 test exposure，也不能保证第三方模型训练语料完全没有相关数据。所有结论都按 development 解释。

## 自由度与检索协议

`cw16_llm16_scientist_v2` 回答“在固定 CW16 基础上，LLM 科学规划能否补充更多学习价值”。`free_llm32_scientist_v2` 回答“让 LLM 自主规划整个 batch 的整体收益如何”。free arm 不运行 CW 选点，不预留 CW 名额；CW distance 只是可以查看或忽略的模型信号。

无 molecule 上限，无新分子数量要求，无 uncertainty/residual/CW/IVR/MaxDet quota，无固定 exploration/exploitation 比例，也不强制 diversity。选择同一分子的多个条件完全允许。需要 32 个合法不同实验 ID，并通过查询实际看过所选卡片；这是执行和审计约束，不是 acquisition strategy。

支持 `get/search_candidates`、`get/search_observed`、`get/search_pending`。支持 ID、SMILES 子串、same_molecule_as、similar_to、Tanimoto 下限、条件精确匹配、描述符范围、signal 范围。sort_by/order 可按预测值、分位宽度、CW 距离、描述符、相似性、已观测绝对历史误差、新颖性和载样量排序。历史误差只允许用于 observed 查询。缺失信号排最后。`offset`/`next_offset`/`total_matches` 支持审计分页。pending 可以作为 same_molecule_as/similar_to 的参照。

每轮 24 次查询、每次最多 24 条、最多 480 个不同候选视图、最多 28 次模型回答；两个 arm 使用同一限制。限制是计算预算，会影响可观察的策略，不能将其解释为无限信息的自由选择。显式 similar_to 默认相似度降序，普通搜索默认稳定哈希排序；哈希展示仍是一种有限抽样，不能宣称完全消除所有 presentation effects。

分位宽度来自原有单模型，不等价于 epistemic uncertainty。`historical_abs_error_ml` 是两个端点绝对误差的最大值，单位 mL，没有归一化，因此可能更受 V2 尺度影响；它只是可选检索信号，不是强制选点依据。历史误差是采样当时的误差，不能解释成当前重训后仍然失败。

## 完整 Prompt

完整、逐字的系统提示词保存在同一 study 下的 `selector_prompt.txt`，由 `scientist_selector.SYSTEM_PROMPT` 导出并与 protocol 哈希绑定。该文件包含 scientific setting、prediction target、active-learning objective、化学及 ML prior、自由度、反馈闭环、JSON 查询协议和完整 selection/hypothesis schema；它是正式运行使用的原文，不是设计摘要。

化学先验和 ML 先验都被明确允许使用，但必须与 observed evidence、model prediction、untested hypothesis 区分。允许不同意数值 heuristic，并解释科学理由。模型可以输出 0 到 8 条有意义的假设，不必为了格式制造假设。整个 plan 限长 30,000 字符，控制下一轮记忆的上限。

## Codex CLI 配置及边界

可以采用 Codex CLI：后端通过 `codex exec --json` 运行，模型与 reasoning effort 在本 study transport 中冻结。2026-09-27 的 model-switch revision 指定 `gpt-6-sol`、`high`，读取当前 `~/.codex/config.toml` provider 和 Codex 自己的认证。子进程使用临时 cwd、忽略 project/user exec rules，关闭 shell、web、MCP、multi-agent、goals 和 memories，并审计 CLI JSON event 与完整 rollout。因为使用 Codex 当前 CODEX_HOME，CLI session 会按本机 Codex 配置保存在本机 history；不复制或打印 auth 文件。完整模型请求仍由 packet 和本轮 JSON query/result transcript 组成，不恢复此前 Codex task。

你提供的 `model_provider = "OpenAI"` 和 `[model_providers.OpenAI]` 名字对应即可；名字本身不表示请求发往 OpenAI。`base_url` 仍然是第三方 token4research，使用该配置会把实验 prompt 发往该服务。`model`/`review_model`/`wire_api = "responses"` 均有官方配置支持；`review_model` 不参与本实验选点。

`requires_openai_auth = true` 是有效字段。当前 CLI 后端沿用个人 Codex provider/auth 配置，已通过第三方端点真实请求；不需要另外设置 `SCIENTIST_API_KEY`。只有独立 Responses 后端使用该环境变量。`env_key` 应填写环境变量名，不能填写密钥值。

`model_catalog_json` 是可选字段；只有确实存在有效、与 CLI 版本兼容的模型 catalog 文件时才应配置，建议明确绝对路径。本实验没有读取或改动你个人的 catalog。当前官方 reference 没有列出 `disable_response_storage`，不能凭这个旧/未文档化字段保证不存储。Responses 后端明确发送 `store=False`；CLI 的服务端存储行为由当前 CLI/provider 实现决定，且任何客户端字段都不能保证第三方服务内部的日志政策。

顶层 `network_access = "enabled"` 不是本次官方 reference 中的通用配置写法。文档化的 `sandbox_workspace_write.network_access = true` 控制 workspace-write 下的工具网络，不是选择模型 provider 所必需的开关。CLI 调用模型端点和允许模型调用任意网络工具需要区分。Windows acknowledgement 在 macOS 不需要。

`[features]` 下 `goals = true` 是支持的，但科学选点需要有界调用与明确停止点，实验显式设为 false。TOML 的表头与键值应分行书写。可检查 `codex_cli.example.toml`；该文件供审阅，不会覆盖个人配置。

CLI 配置关闭 shell、web、multi-agent、memory、goals，并设 read-only/never approval；对完整本地 rollout 的 response item 类型做白名单审计，任何 function/custom/web/native call 或未知项都拒绝接受。保存回答、模型/effort、CLI version、usage、审计摘要、事件/rollout 哈希；study 不复制私人推理文本。临时工作目录会清理，原始 CLI rollout 保留在个人 CODEX_HOME，供中断恢复核验。冻结 transport 中的 `cli_isolation` 文字沿用了早期设计；实际实现使用个人 CODEX_HOME 和临时 cwd，以本段及调用源码为准，并非独立 HOME。

**CLI read-only 不是硬文件读取隔离。** 工具调用审计可以拒绝污染后的结果，但不应被称为 OS 级隔离。模型接口可能随 CLI 版本改变，未知响应类型会失败关闭。要求更强工具边界时应在正式冻结前选择 `responses`（`tools=[]`，只有宿主 JSON relay），或者给 CLI 配置独立 OS/container 的 deny-read 隔离。两种后端不能在同一冻结实验中临时切换；它们具有不同系统提示及传输语义。

参考：

- https://developers.openai.com/codex/config-reference/
- https://developers.openai.com/codex/noninteractive/

用户后续请求将内部模型切换为 gpt-6-sol/high。第一次 gpt-5.5 调用已返回查询，但在 V2 catalog 的 measured endpoint sort 漏项上失败；gpt-5.5 的全部输入、查询、回答和失败标记归档于 `revisions/internal_gpt6sol_20260927/`，没有 batch freeze 或标签揭示。查询接口补上了只允许 observed 的 V1/V2 排序/范围筛选。切换登记为新的 preselection protocol revision，保留旧 prompt 及首轮 input/catalog，gpt-6-sol 从最初 packet 独立开始，不接收 gpt-5.5 的查询结果。

真实诊断确认：CLI stdout 用 `item.completed`/`agent_message`，rollout 用 `item_completed`/`AgentMessage`。真正导致 stdout 审计失败的是 CLI 将 `disable_response_storage` 和 `mcp_servers.computer-use.type` 两个已忽略配置项的警告编码成 `error` item。适配器严格匹配并记录这两类配置警告，其他错误及未知/tool item 仍拒绝。中断恢复要求 started marker 请求哈希、精确输入、model/effort、final item ID 和 task_complete 一致，不重新抽取已完成的实验回答。查询接口同时兼容平铺字段及等价的嵌套 name/bounds 对象；冲突边界报错，observed-only 限制保持生效。全部修订均发生在 batch freeze 前，原始请求和回答保留在 revisions 和活动目录。

## 评价指标及正式命令

primary：固定 L333 target scales 的 combined normalized RMSE，按标签预算 333 到 525 做梯形积分并除以 192，即平均 label-AULC。secondary：各 seed AULC、NRMSE@525、V1/V2 RMSE/MAE/R2，逐调用 usage/query log 可审查成本。每个 LLM arm 分别与 Random32/CW32 比较，两 seed 均改进且均值降低才视为扩展信号；不声称显著性。free 与 hybrid 的差异同时包括 batch 自由度和 CW 约束，不能解释成单一机制消融。

CLI 使用当前 Codex 认证，模型、端点和 effort 读取 study 中已冻结的 protocol。以下命令只执行指定轮次的 selection；已冻结时仅重放审计：

```sh
/Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_scientist.py select --seed 157 --method free_llm32_scientist_v2 --round 0
```

该命令最多完成本轮 JSON 查询并冻结 32 个点，不读取新标签、不启动 QGeoGNN 训练。2026-09-27 已完成 seed 157 / free arm / round 0 的选点与训练：365 条训练标签，191 epoch，最佳 epoch 91。第 1 轮选点尚未启动。后续只有显式运行以下命令才揭示下一批并训练下一预算；本次未执行：

```sh
/Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_scientist.py advance --seed 157 --method free_llm32_scientist_v2 --round 0
```

hybrid 使用 `--method cw16_llm16_scientist_v2`；第二 seed 使用 `--seed 6101`；逐轮使用 `--round 0` 到 `5`。不得自动扩展 seed 数量。全部完成后 `report` 才进行新的 test evaluation。没有循环式全实验启动入口。

## 正式运行前仍需注意

1. 第三方端点已通过真实请求；CLI 兼容性以当前已验证版本为准，后续升级仍可能改变输出格式。
2. CLI 是审计边界。如果研究要求硬隔离，应在正式 selection 前注册使用 Responses 后端或外部 OS 隔离，不能在已有冻结协议内悄悄修改。
3. 四 arm 不隔离知识的因果贡献，两个 development seeds 也不足以确认一般性收益。
4. 相同查询预算不等于相同每个 LLM-selected point 的证据预算；hybrid 选 16、free 选 32。需共同报告模型开销及浏览量。
5. 过去的测试暴露无法撤销。后续确认需要独立 holdout/seed 设计；不能以本轮 test 结果反复改 prompt 后仍称 preregistered confirmatory。
6. 同一预测模型的 quantile width 可能未校准；新颖性与结构近邻也不能代替实际学习收益。允许模型判断这些信号是否有意义，不能要求事后理由必然是真实机制。

## 本次验证结果

- 41 项测试通过：34 项 V2 测试及 7 项原有 full-pool/dialog 回归测试。既有依赖发出 deprecation/requests dependency warnings，无测试失败。
- 新模块和入口通过 `py_compile`，`git diff --check` 通过。
- seeds 157/6101 的 hybrid/free 四个真实 round-0 数据包均已生成。hybrid 为 2,981 个 eligible + 16 个 pending；free 为 2,997 个 eligible + 0 pending。
- dry-run 对 RestrictedLabelStore 加额外运行时拦截：只有 `initial_fit` 且全部 ID 属于 L0 才允许 reveal。四次运行只读取各自 333 条初始标签，没有新标签、验证/测试真值读取、LLM 调用或训练。
- 943 个历史 LLM 文件哈希完全相同；132 个控制/输入文件的复用哈希通过。旧路径没有 Git diff。
- seed 157 / `free_llm32_scientist_v2` / round 0 已完成真实 CLI selection 和一轮训练，32 个合法且不同的 ID 已冻结，pending 为 0。共 3 次模型回答、13 次查询、129 个不同候选视图、6 条科学假设；前两次回答由原始 rollout 恢复，最后一次由修复后的 CLI stdout 审计直接接收。另有 1 次独立最小格式探测，不参与选点。
- 三次选点回答的 provenance 均为 `gpt-6-sol` / `high`，native tool calls 为 0；`audit_batch` 重放通过，`validate` 为 VALID。最终 protocol SHA256 为 `6f400ec3ab7ed6a8c9aae41605d8b6f0a997a7f88cf5b05ef2f1f6addec1b09e`。
- batch freeze SHA256 为 `7f910e227fae1bfe7709f76c39840917a65a9fe93ac19f3d4b896163173466bc`。随后按请求揭示这 32 个标签并完成 round 1 模型训练；新增活跃标签共 365，耗时约 90 秒，共运行 191 epoch，最佳 epoch 91。checkpoint SHA256 为 `d3ac605a19dc9016a24a9bbc1cf434bc883e7d5d0280b25d160ccbf4551e9c16`，预测 SHA256 为 `f23083f997ab741f62f6e4e76bf4193fc482703b8ac629c9d791bd92393c45fa`。没有使用 test labels；下一轮 selection 尚未启动。
- 本机 CLI 为 `codex-cli 0.155.0-alpha.16.4`，真实请求及事件/rollout 审计已通过。最后一次请求记录 input 104,510 / output 6,469 tokens，其中 reasoning output 3,541；恢复的两次回答未在 turn receipt 中保存 usage，这不是完整总成本。
- CLI 调用使用个人 Codex 配置及现有认证；当前后端无需 `SCIENTIST_API_KEY`。未修改个人认证配置。
- 机器可读 dry-run 记录在 study 的 `dry_run_audit.json`。

## 文件职责

- `scientist_selector.py`：完整 prompt、V2 catalog、selection validation、trajectory-local memory。
- `scientist_transport.py`：Responses/CLI、隔离调用、原生工具审计、逐步持久化和确定性 replay。
- `scientist_study.py`：独立 protocol、精确控制复用、stage/select/advance/report、freeze/label barrier。
- `run_qgeognn_v2_row_llm_scientist.py`：命令入口，每次 action 的副作用明确。
- `test_scientist_v2.py`：合成协议/审计/检索/恢复/标签边界测试，禁用真实模型服务。
- 本 study 的 protocol、control_reuse、historical_artifacts、selector_prompt 和四份 round-0 packet：可审计的准备产物。
