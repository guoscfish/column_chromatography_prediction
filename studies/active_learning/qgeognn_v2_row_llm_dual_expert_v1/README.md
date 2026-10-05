# 双专家实验草案（尚未开展效果评估）

2026-10-05 按用户授权纳入主仓库。该方案由 Chemistry Scientist 与 ML Scientist 分别提供建议，再由 Planner 选样。当前实现只到 round-0 selection，不包含标签揭示、再训练或测试评估路径；现有协议、预备候选和历史准备文件不是已完成的学习曲线。

本次只整理归档，未启动模型调用或训练。未来运行前需要检查环境、provider 鉴权及冻结协议与当前代码 hash 的一致性，并完成独立预注册。

- 方案说明：[设计文档](../../../docs/research/4G_LLM_DUAL_EXPERT_ACTIVE_LEARNING_V1_2026-09-27.md)
- 入口：`scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py`
- 原目录未提交文件 hash：`draft_provenance/source_manifest.json`
- 共用 Scientist 文件多数是历史旧版本，保留主线较新的实现。旧 transport 的唯一差异是移除查询错误反馈；该差异保存为 patch，不回退主线修复。
- runtime 不入 Git；准备阶段的 banks 与冻结材料保留以供审计，不作为独立实验结果。
