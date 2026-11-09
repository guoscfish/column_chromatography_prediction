# 柱色谱 QGeoGNN 与主动学习

本工作区是隔离重构候选，分支 `codex/repository-architecture-20261109`，基线 `16eca74769f664bc378a58ccbf380a806723f587`。原 `column_chromatography_prediction` 的 main 继续运行 CW-Free。不要从此副本恢复旧冻结实验或自动合并 main。

新公共接口拆分表示、选样、工作流与代理计划验证。完整职责、迁移映射与退出门槛见 [架构](docs/ARCHITECTURE.md)；逐文件决策见 [CODE_INVENTORY.csv](docs/repository/CODE_INVENTORY.csv)，[重构报告](docs/repository/REFACTOR_REPORT.md)，[运行与冻结依赖](docs/repository/RUNTIME_AND_FROZEN_DEPENDENCIES.md)。

在具有 NumPy 的环境运行合成离线示例：

```sh
python -m scripts.run_offline_selection --config configs/offline_selection.synthetic.json --output /tmp/column_selection_demo.json
```

它完成 config → 授权表示 → LCMD → selected_ids/trace → 磁盘冻结。实际模型的 latent 提取接口及冻结数值对照见 `representations/latent.py` 和 `tests/refactor/`。CLI示例没有训练、真值揭示或LLM请求。

候选只展开源码、测试与文档。数据/历史结果仍在 Git 对象中可追溯，运行缓存和keys没有复制。读取真实训练数据或检查点前，须准备独立、只读、有SHA的授权快照。已有冻结入口保留原字节；历史文档索引仍由 `docs/repository/RESEARCH_SCOPE.json` 与 `RETIREMENTS.json` 维护，本次真实删减另见 `RETIREMENT_MANIFEST.json`。

新方法在独立的 `../column_chromatography_lab` 先孵化。其 lock 固定候选代码快照，科学收益/复用价值与用户确认通过后才提取最小核心实现晋升。分位头、损失、训练协议与现有实验选择均未在工程重构中更改。
