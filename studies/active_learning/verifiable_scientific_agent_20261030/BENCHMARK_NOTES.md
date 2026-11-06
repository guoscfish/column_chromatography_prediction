# 独立校验、局部修复与离线基准

本原型完成 Level 1 的离线合法性回放与 mock 协议测试。没有新的 LLM 请求、标签揭示、模型训练或生产接入；没有证明闭环学习效率改善。

## 固定输入与原规则

基准只读取 `response_manifest.jsonl` 的 462 个已冻结完整 turn：Free 270、Hybrid 93、NoCW 99。批次清单固定于 **2026-10-30 16:17:05.764164+08:00**，对应 310 个冻结批次；之后新冻结的 NoCW 轮次不追加。Free seed157 round5 有授权抄写的最终 selection，缺少完全对应的最终 raw answer，因此其四个完整错误回答进入响应基准，但不凭空补造一个合法原始回答。响应单位与最终批次单位不同。

`offline_benchmark.py` 对使用到的 **1,392** 个 `response/input/contract/batch_freeze` 原文件逐一核验固定 snapshot 的 SHA256，并另外核验 response manifest 的 response SHA 与 batch-freeze SHA。主输出记录 snapshot、response manifest、validator 源码哈希及输入核验文件数。没有扫描正在增长的目录来更新分母。

判定执行现有生产源码中纯函数的 AST，避免导入控制器、RDKit、模型或 label-store。`validate_direct()` 与 `decode_table()` 保持原函数逻辑；判定只依赖候选 ID、已观测 ID、packet 绑定、数量、schema 与提案文本，因此使用完全等价的 ID-only catalog/table，未引用实验标签值。

**NoCW 使用实际五 seed runner 的覆盖函数**：`scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py::validate_plan()`，而不是 `nocw_inputs.py` 中未运行 paired 协议的额外 prose-ID 禁令。AST 只把 wrapper 内部对 `validate_direct` 的 import 解析为同一已提取函数；其余函数语句不变。runner 的字段 allowlist、typed evidence 检查与 `audit_no_cw()` 均执行。测试同时区分这两个协议，避免把历史允许的 prose 引用误判为历史失败。

## A / B / C 结果

| 历史方法 | 完整响应 | A 原校验合法 | B 细诊断合法 | C 安全格式处理合法 | C 新挽救 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free LLM32 | 270 | 179（66.30%） | 179 | 179 | 0 |
| Hybrid LLM16 + pending CW16 | 93 | 71（76.34%） | 71 | 71 | 0 |
| NoCW LLM32 | 99 | 59（59.60%） | 59 | 59 | 0 |
| 合计 | 462 | 309（66.88%） | 309 | 309 | 0 |

462/462 的 A 与 B 判定一致，且 A 与独立历史审计的原 validator 回放一致。B 的 acceptance 直接继承对应原规则，不把新诊断当成更高合法率。这里是混合了首答与纠错答案的**响应合法率**；首次合法率、最终冻结率、网络成本请使用 `reliability_metrics.csv`，不能拿本表替代。

153 个原本不合法的完整响应都不能靠本次允许的确定性格式处理获救。不存在“局部修复已减少若干真实请求”的结果。当前可定位到可请求的局部范围有 **145/153**（Free86、Hybrid22、NoCW37）；另外 **8/153** 在绑定、超额数量、结构等方面要求完整重规划。145 个只是语法与范围可定位，仍需新的科学回答和明确的意图审查；不能理解为 145 个已经修好，或另外 8 个就是全部实际重规划需求。

本 snapshot 所有 462 个完整响应都可以严格解析 JSON，其中 **460** 个有完整、可靠的 choices 必需字段结构。不能可靠恢复一个实验批次的两个响应仍可报告单独字段诊断，但不会当成可执行计划：尤其 NoCW seed2357 round10 turn02 的 `choices` 长度 37，末尾 5 个对象实际具有 hypothesis 形状。程序不把这 5 个 H 字符串认作真实实验，也不截取前 32 个充当原意。它被明确标记 `candidate_choice_structure_reliable=False` / `individually_identifiable_fields_only`，要求完整重规划。

细粒度 code 以“出现该原始字段异常的响应数”计数，可多标签：`out_of_pool_id`117、`already_measured_id`63、`unobserved_evidence`53、`invalid_hypothesis_reference`8、数量5、hypotheses 结构2、evidence 结构1、reason1、NoCW 未授权字段1、packet 绑定2。117 包含结构不可靠响应中 raw ID 字段异常，不等于 117 个可恢复的科学计划。**精确的同批次重复 ID 为 0**；生产泛化错误消息“distinct legal”不能直接记为实际重复。

## C 的严格范围

`format_only_repair()` 只允许：原始完整严格 JSON 原样返回，或去掉包住全文的唯一完整 JSON code fence。围栏内部必须是完整、无歧义 JSON。没有前后自由文本、多个对象、重复 JSON 键、NaN/Infinity、截断补全、逗号修补、正则找 ID、ID 近似映射、证据删除或科学选择替换。

原 A 使用生产 `json.loads()` 的历史行为；若未来遇到生产 parser 接受重复键 / 非有限值的情况，B 仍保留 A 判定，同时严格子分析与 patch 边界拒绝建立确定计划。这个兼容性差异有独立 mock 测试；当前冻结样本中未出现该情况。

零次 C 新挽救说明这里的完整失败回答主要不是本规则能处理的外层围栏问题。不能把围栏模拟测试通过解释为历史收益，或推断更复杂的自动“修 JSON”会安全修复真实错误。

## 局部 patch 的可检查不变式

`prepare_repair_from_response()` 是严格原响应边界。可靠解析后，`build_repair_request()` 生成未发送的请求；调用者必须明确判断原科学意图是否清楚。意图不明、结构不可靠或冻结绑定失效时返回完整重规划。测试没有真实 provider 调用。

- 原提案 SHA、冻结 context SHA、允许位置与完整 locked proposal 都绑定。应用 patch 时宿主重新生成请求并逐字段比较，拒绝篡改允许范围和旧状态 patch。
- **合法 choice 与理由逐字保留**；原本合法的候选身份保持。缺少的选择只能显式补齐期望数量，不能超额补入。
- 非法身份的 slot 必须由 mock planner **显式重写整个 choice（ID、reason、hypothesis_id）**。禁止只把非法 ID 换成另一个 ID、机械移植原理由后宣称科学意图已验证。
- 非法证据允许显式修改**整个受影响 hypothesis**，保留其唯一 hypothesis ID，并保留原合法观测引用在 support/contradict 两表 union 中；可明确调整证据角色、content、confidence、status。程序不把“旧命题锁死、随便换证据”当成科学修复。
- 其他位置全部锁定；修改后重新执行对应完整原 validator。错误 ID、重复、未知引用或未观测证据不能借 patch 绕过。
- patch 的意图声明只是 mock planner 的明确声明。字符串保留、引用集合保留、通过 validator 都**不能证明科学意图或科学解释仍然正确**；若修改牵涉其他假设 / 选择的含义，应重新规划。

没有随机 / CW 备选项，没有自动删除多余合法选择，没有 ID 编辑距离或“看起来相似”的映射。mock 成功只证明协议范围与验后不变式，未证明真实 LLM 会按协议修好。

## 全池局部索引

`StableCandidateIndex` 按原 packet 候选表顺序建立 1-based 双向唯一表，不排序、不截断、不采集打分。presentation 保留全部原候选行、列、条件、SMILES、预测与精度以及原已观测信息；仅添加局部编号及独立 presentation 哈希。源 packet 的哈希不会被冒充为修改后 presentation 的哈希。映射绑定完整冻结 context、source packet 和原 ID 顺序；跨轮次旧表、bool / 非整数 / 越界编号、重复 ID、双重 id+local_index 身份均拒绝。解码后仍走原完整 validator。

编号映射本身仅携带身份，不增加未测量标签或数值信息，也不改变 NoCW 可见字段。保持顺序排除了代码引入的排序偏好，但**不能排除 LLM 对短编号、较小数字或表前部的行为偏好**。降低真实定位错误及保持科学设计的效应需要未来同状态对照；本次没有把任何历史坏哈希事后猜成合法编号。

## 计算开销与复现

本次最后一次单进程回放中，A 每响应均值约 5.5–7.5 ms；包含原 validator 的 B 约 5.4–10.3 ms。NoCW 的 B−A 中位约 2.6 ms。Free / Hybrid 的配对差接近零且受计时噪声影响可为负，不能据此宣称细诊断免费或更快。这里没有测量网络成本；AST 加载、文件读取、上下文构造不含在 A/B/C 单臂计时中。所有逐响应实际时间保留在 CSV，耗时受同时运行的其他进程影响，不是性能资格基准。

```bash
python3 -m unittest discover -s studies/active_learning/verifiable_scientific_agent_20261030 -p test_verifiable_planner.py -v
python3 studies/active_learning/verifiable_scientific_agent_20261030/offline_benchmark.py
```

25 项 mock 测试通过。原始测试输出见 `verifiable_planner_test_output.txt`，命令与退出码见 `verifiable_planner_test_result.json`。逐响应 / 分方法结果见 `offline_benchmark_responses.csv`、`offline_benchmark_summary.csv`，详细诊断与来源哈希见 `offline_benchmark_results.json`。

本原型支持先做明确的约束定位、可验证身份接口与有限范围的显式科学纠错。当前结果不支持升级为复杂新 Agent，也没有证据表明自动格式处理能解决本批历史失败。Numerical Critic 属于后续合法计划评价接口，其数值诊断与真实学习收益仍需分开验证。
