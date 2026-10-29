# NoCW-only versus historical Free interface

候选catalog仅删除cw_distance_percentile。原Free的候选ID、顺序、SMILES、条件、预测、分位数及精度、描述符、相似度、新颖性和全部L333观测保持不变。相同seed和round使用原Free salt，不使用NoCW方法名排序。

实际prompt为当前Free DIRECT_PROMPT删除CW距离定义和pending-CW说明；不新增配对开发版中的typed-only自由文本引用规则。选点仍须32个不同U_t，hypotheses的实测证据ID只能来自本轨迹observed。NoCW继续审计所有显式/改名CW字段、嵌套反馈和实际序列化请求；污染即停止，无LCMD补点或静默替换。

host目录方法为llm32_nocw；LLM-visible method沿用free_llm32_scientist_v2以避免额外方法名提示，study_version也沿用Free。packet_sha256自然不同。初始记忆为空；后续记忆来自自己的新NoCW轨迹。

统一gpt-6.1-sol/xhigh是用户本次指定，不能据此声称历史Free全部轮次都已使用该配置。历史结果不修改、不重新拟合。
