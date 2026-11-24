# Responses 传输、故障与恢复

每轮配置固定 provider、请求模型、预期服务模型、推理等级、超时、输出预算和 SDK 版本。采用 Responses 流式输出及本地严格 JSON/schema/事实 validator。官方 OpenAI API 文档说明了 Structured Outputs 能力，历史 token4research 代理是否对应支持及限制仍未做真实能力验证。实验路径使用普通文本并做本地协议检查，保留完成事件、身份和可得 usage。比较供应方强制 schema 需要独立能力测试与成本记录。 [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) · [OpenAI Streaming](https://developers.openai.com/api/docs/guides/streaming-responses)

一份逻辑请求只能来自冻结合同和明确 kind。网络物理尝试、已完成响应、逻辑修复轮与冻结批次分别落盘。默认逻辑回复最多三份，每份物理尝试最多三次，按响应完成事件和回执计数。SDK 自动重试关闭。429 和尚未建立连接的明确 ConnectError 可以按指数退避加 jitter 重试。Retry-After 支持秒数与 HTTP 日期，超出等待上限直接停止；连续两次 429 熔断。401、403、永久错误、模型不一致、空完成和明显不完整响应停止。读超时、响应中断和 SSE frame 在不能确定服务端状态时保留为歧义，禁止自动重发。

物理调用前写 started 回执，随后写 terminal 回执。若 started 有而 terminal 无，必须用绑定 request_id 和 attempt_id 的人工授权与服务商证据对账。 `reconcile()` 可导入确认为该请求完整返回的响应，或记载已证实未处理而继续消耗原有限预算；错误回执与历史文件不被覆盖。保存的 completed 回执及冻结批次重放时直接复用。原子写入和每轮文件锁防止本地重复工作。磁盘应支持可靠的 advisory lock 和硬链接发布，GPU 闭环的运行目录应放在持久盘上。

不根据异常切模型、降推理、裁池或补点。配置文件仅记录历史对照的候选设置。真实服务、上游有效上下文上限、同名模型权重和精确费用均待核验。日志按进程阶段显示，供后续以独立监督器稳定运行；远端运行不得自动启动下一科学动作。证据不足以核算账单时按未知记录，不能用失败回执推算完整付费回答数。
