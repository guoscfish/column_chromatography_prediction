# 持续传输重试修订

2026-10-29，用户明确要求：“一直失败就一直重试”。这项授权取代原每答案最多3次物理传输的运行限制，其他科学协议不变。原config和protocol.lock保留，运行时有效的重试策略单独记录在persistent_retry_amendment.json，绑定原协议hash、修订入口及测试源码hash。既有请求hash和每次失败记录均保留，编号从已有最大连续编号向后增加，不清零、不覆盖。

连接、超时、未完成服务响应等传输失败持续重试，间隔30/60/120/240/300秒，之后保持300秒；失败请求也可能计费。每个新的尝试记录修订hash及脱敏错误摘要。模型gpt-6.1-sol/xhigh、endpoint、独立key映射、完整池、输入精度、模型训练和标签冻结规则均不改。格式/证据/批次合法性验证及3个答案的纠错上限仍保持，绝不以自动补点或换ID绕过验证。

现有仍持有seed worker锁的请求继续执行，不中断、不同时再开同seed请求。原worker因3次传输失败退出后，后台supervisor才恢复该seed，复用原目录和ledger。已完成批次和模型通过原审计重放/缓存复用，不重新采集或重复训练。持续重试worker在内部处理网络失败；协议/数据损坏及started无结果的异常中断需要核对，不能删除标记后盲目重发。

入口 scripts/studies/run_qgeognn_v2_row_llm_nocw_persistent.py，默认只读status。prepare写入授权修订；launch需同时显式allow-paid-llm、allow-training、allow-label-reveal。supervisor以独占锁保证唯一，按seed锁接管，所有有效进程和重试等待信息位于progress/persistent_supervisor.json及progress/retry_seed_*.json。

验证：49项相关离线测试通过，含延续既有3次失败至第4次、超过旧上限后成功、指数退避上限、成功响应重放不请求、原始参数不变、请求hash漂移拒绝、模糊started不重发、未授权不请求。没有为测试发送真实API请求或训练模型。

监控继续每5分钟检查。对于正常的持续网络重试不逐次通知，只在首批/预算里程碑、完成、恢复器退出、协议失败、身份/数据边界问题或需要人工核对时通知。全部任务仍以L1005为终点，不运行新CW-aware。
