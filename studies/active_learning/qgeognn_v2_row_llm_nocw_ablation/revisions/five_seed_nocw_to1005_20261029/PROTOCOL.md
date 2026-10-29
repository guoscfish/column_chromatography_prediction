# Five-seed NoCW-only execution amendment — 2026-10-29

用户授权：“开始运行吧，每个seed用一个单独的key，五个并行跑”；随后明确“跑NoCW就行”，并指定“这些任务统一都用6.1 sol xhigh”。本修订仅执行 NoCW；不启动任何新的 CW-aware 采集或拟合。保留原两seed配对协议、源码及历史结果，单独冻结本目录。

Seeds固定为历史五seed：157、887、2357、6101、12203。每seed一个进程、一个不同key，同一进程逐轮完成NoCW selection→freeze→reveal→same-initialization fit。五个seed并行。L333初始状态和逐步预算/训练/归一化与原协议相同：batch32、L525工程节点、L1005硬终点，共105次主采集和105次新拟合；最多315个答案、945次物理请求（含有界纠错/失败重试）。不得因早期效果差自动停止，也不得自动超过1005。

统一实际请求配置：Responses `https://token4research.cn/v1`，model=`gpt-6.1-sol`，reasoning=`xhigh`，其他调用限制继承原config。五个key的文件映射在config，启动前校验非空且互不相同，密钥值不进入协议、参数、日志或LLM输入。历史映射复用157/6101专用key、887/2357旧实验key、12203专用key，不共享默认key。

候选表示/精度及排序与原Free对齐：`llm_scientist_v2:seed:free_llm32_scientist_v2:round` salt，opaque ID不变。LLM包保留Free的study_version/method字符串，host方法仍为llm32_nocw。prompt沿用当前Free的DIRECT_PROMPT，只移除CW数值定义及pending-CW说明；不加入未执行配对协议中新的自由文本ID引用限制。沿用Free validate_direct的32个唯一合法候选/typed observed evidence规则，并叠加NoCW白名单与序列化泄漏检查。不会因为历史对照而继承其后续选择、反馈、解释或标签。

每seed L333都通过其baseline member0 checkpoint、split、训练初始hash、L333/U333、source/graph checksum及初始完整catalog绑定验证。新增三seed使用旧五seedFree的round0 catalog；157/6101仍用旧完整池round0 catalog。Context仅初始化本seed的L333真值；validation只用于fit checkpoint选择/报告，test标签始终不可见。模型和状态目录按seed独立。

原全局长锁改为每seed操作锁和每seed worker租约，允许五个进程真正并行。全局served-model锁仅在比较/写入模型标识时短暂持有，防止竞态；不同模型版本立即fail-closed。失败不会无限重启或自动增加纠错预算；未完成请求的started标记需要核对，不自动重发。

用户已授权发送真实候选SMILES/条件、本seed L333实测记录和模型预测到该endpoint，并授权冻结后新标签推进和训练。远端模型/上下文支持不作虚假预先认证：以首次正式完整采集请求实际验收；不另发送付费探针。拒绝、截断、非completed、错误模型或工具输出都不接受，不缩减候选池或切换模型。

主要指标仍为validation AULC333–1005，固定L333 target scales，次要指标沿用原协议。报告从各best checkpoint对应epoch的冻结validation history读指标，不给LLM实时validation奖励，不reveal test。历史CW-aware Free32结果为参考；因旧轨迹可能有不同模型/参数/服务版本，这是有混杂的历史对照，不能宣称为新配对的严格因果消融。五seed也都曾用于历史开发，不称为独立新seed确认。

本修订通过运行入口在独立进程配置适配器实现，不修改原冻结src模块。代码、测试、数据文件、环境、本协议及schema/code审计均由新的protocol.lock.json冻结；原两seedlock仍可独立验证。
