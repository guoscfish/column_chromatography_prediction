# Five-seed execution adapter audit

复用原nocw_study的initial_state完整checksum检查、write_stage、run_selection有限纠错、freeze_batch、audit_batch、state、advance与生命周期；复用RestrictedLabelStore、原训练函数和checkpoint选择。原src文件不修改。

适配器只增加五seed白名单、baseline split路径、新三seed round0 catalog路径、支持五seed的Context初始化、每seed锁/worker、五个独立key文件、原Free排序与prompt对齐。Context与WorkflowContext相同的L333-only标签初始化、零标签scrubbed图断言、冻结normalization/preprocessing；继承其new_store和训练方法。

真实调用仍走原live_call：gpt-6.1-sol/xhigh、tools=[]、完整池、相同输出/重试预算，served-model原子锁。新seed worker仅调用llm32_nocw，从未调用CW-aware路径。启动worker须同时显式允许API、训练、标签reveal，且具备绑定本协议的用户授权文件；默认入口仅status。

离线dry-run逐seed比较实际编码与历史Free编码删除CW列后的精确相等，验证初始32选择与freeze/replay。无真实API、拟合或新标签。单元测试覆盖五份key唯一性、不同seed锁可并行/相同seed互斥、全局model标识一致、worker仅NoCW及五seed初始状态绑定。
