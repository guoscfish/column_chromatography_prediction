# 纠错后有序选择展示前32行的独立原记录抽核

结论为 **PASS_FOR_15_RAW_SAMPLES**。该现象能在原始候选顺序、最终selection、freeze及保存的接受响应中直接核验，不是只出现在后续统计CSV中的排序结果。它是可见的选择/纠错路径现象，不能单凭这些记录归因于LLM内部思考、供应商具体层级或Row学习损失的因果来源。

## 1. 抽样规则与读取范围

使用主诊断表仅定位三个LLM方法×五seed各自最低round的prefix状态，共15例；不重新扫描540轮，不导入或调用主实现的`prefix_identity`函数。独立解析每例：

1. 原 `input.json` 仅解码顶层`candidates`表，按其实际`id`列读取行序；不解码observed、历史反馈或标签字段。
2. 以freeze中的artifact SHA校验input、selection及所有相关完成turn。
3. 逐位置核对selection的32个唯一ID、freeze的batch ID、原展示行序前32个ID三者完全相同。
4. 自行解析完成turn的answer JSON，以完整plan与selection的深度相等寻找唯一接受响应；response ID仅保存hash。
5. 对接受前的可解析plan独立计数合法ID、与最终batch重叠；读取对应rejected记录的实际instruction。

原文件整体字节用于hash完整性检查，但没有解码observed表、打开feedback/标签文件或凭据。详见 [prefix_independent_verification.json](prefix_independent_verification.json) 的逐例路径和source manifest。

## 2. 15例逐项结果

所有第一可解析plan均有32个不同ID，其中29–31个属于该轮真实合法候选池；它们均不是展示前32。最终plan则全部有序等于展示前32，且均有唯一的原始接受receipt完整匹配。

| 方法 | seed | 首个prefix round | 第一plan合法ID数/32 | 第一plan与最终重叠 | 接受前可解析plan数 |
|---|---:|---:|---:|---:|---:|
| Historical CW-Free | 157 | 1 | 31 | 0 | 1 |
| Historical CW-Free | 887 | 3 | 31 | 1 | 2 |
| Historical CW-Free | 2357 | 0 | 31 | 3 | 1 |
| Historical CW-Free | 6101 | 18 | 29 | 6 | 1 |
| Historical CW-Free | 12203 | 0 | 31 | 1 | 1 |
| New CW-Free | 157 | 5 | 30 | 0 | 4 |
| New CW-Free | 887 | 8 | 31 | 4 | 1 |
| New CW-Free | 2357 | 6 | 31 | 4 | 1 |
| New CW-Free | 6101 | 5 | 31 | 1 | 5 |
| New CW-Free | 12203 | 0 | 31 | 2 | 2 |
| NoCW-Free | 157 | 5 | 29 | 0 | 1 |
| NoCW-Free | 887 | 7 | 31 | 2 | 1 |
| NoCW-Free | 2357 | 0 | 31 | 1 | 1 |
| NoCW-Free | 6101 | 3 | 31 | 1 | 1 |
| NoCW-Free | 12203 | 0 | 31 | 1 | 1 |

合计原计划480个不同ID名额中460个合法，最终仅保留27个，逐例重叠0–6。可见变化远大于只替换一个非法ID；但不能据合法ID数量断言该plan其余科学证据或文本schema已经完全合格。

15例的accepted request hash各不相同，response ID hash也各不相同。原receipt plan已经与最终选择完整相等，所以仅靠后续CSV导出排序无法解释该现象；这并不能独立验证外部服务的内部生成过程。

## 3. 纠错指令没有要求prefix

样本共24个rejected记录，只有两种实际instruction：

> Return a corrected final JSON selection from the SAME complete catalog. No queries. Only observed-table IDs may be cited as measured evidence; use empty evidence lists for prediction-only hypotheses.

> Return corrected final JSON from the SAME complete pool. Exactly 32 distinct candidate IDs. Measured evidence belongs only in typed observed-ID lists. No queries or tools.

前者出现6次，后者18次。二者要求在同一池修正合法ID及观测证据引用，没有要求“取前32”“top32”或按展示顺序选。主传输源码的相应逻辑保存原answer、校验失败后追加错误与instruction，合法plan才写selection；本次抽样没有发现selection在接受receipt之后才被单独改成prefix的证据。

## 4. 与主全量统计的关系

主诊断报告540轮中127例有序prefix：Historical25/180、New40/180、NoCW62/180；全部127例都有保存的拒绝记录。277个没有拒绝记录的状态中没有prefix，另外136个有拒绝但未变prefix。

这些**全量数字来自主诊断**，本独立审阅没有重复其540轮普查；15例原记录验证支持其核心定义和“原提案非prefix→纠错后接受prefix”的解释路径。由于抽样本身以prefix为条件，不能用15/15估计总体发生率，也不能由“全部prefix曾被拒绝”倒推出“拒绝必定导致prefix”。

## 5. 能支持和不能支持的机制判断

支持：历史LLM策略包包含一个此前整体行为均值难以揭示的工程/上下文环节——少量候选ID合法性问题后的纠错，可能伴随大规模候选替换，并在部分接受结果中表现为展示前缀选择。它会影响对“模型原始科学偏好”的解释；最终已接受batch仍是实际训练历史，不能删掉prefix轮后重写主性能。

不支持：尚不能确定是注意分配、修复压力、供应商上下文处理、提示交互或其他因素造成；不能宣称恢复了内部思维，也不能认定prefix一定降低Row收益。单凭无prefix指令不能排除其他上游机制；保存receipt的响应身份也不是供应商内部过程的独立证实。

因此，在再次完整NoCW长跑前，应先明确研究的是原始提案随机性、合法性修复稳定性，还是CW信息效应。窄范围冻结状态的纠错轨迹研究比再复制一条混合了这些因素的长历史更直接；本轮没有改写Prompt、重发请求或实施任何修复干预。
