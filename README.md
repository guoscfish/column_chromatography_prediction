# 柱色谱 QGeoGNN 与主动学习

本项目研究 4g 柱色谱的 QGeoGNN 点预测与主动学习。重构和文档清理都在 `codex/repository-architecture-20261109` 的独立 linked worktree 继续；原工作目录的 `main` 承载 CW-Free，不接受本轮代码替换或自动合并。

从 [文档导航](docs/README.md) 找资料，查看 [当前研究状态](docs/NEXT_STAGE_DECISION.md) 区分确认结果、假设与运行中工作；实现职责统一在 [系统架构](docs/ARCHITECTURE.md)。科学证据从 [研究总索引](studies/README.md)、[历史 experiments 索引](experiments/INDEX.md) 和 [报告索引](reports/README.md) 进入。

稳定源码只有三层：`predictor/` 负责模型与训练输入，`al/` 负责表示、选样、回合和评价，`infra/` 负责身份、资源与工件。候选 CLI 目前开放合成生命周期演示：

```sh
python -m scripts.run_al --config configs/offline_synthetic.json --output /private/tmp/column_al_demo
```

首次运行使用新目录；同代码身份和配置可恢复已有回合。该命令不训练真实模型、不揭示真实标签、不调用 LLM。真实后端默认关闭训练，完整生产重训练、真实 K3 与历史 LLM 协议仍有验证边界，见 [二期测试报告](docs/repository/phase2/TEST_REPORT.md)。依赖分别列在 [核心依赖](requirements-core.txt)、[模型依赖](requirements-predictor.txt)、[测试依赖](requirements-test.txt)；在独立环境准备，不更改运行中的实验环境。

分位数与损失复盘按用户要求保留在 main 的独立提交：[2026-11-09 科学诊断](https://github.com/guoscfish/column_chromatography_prediction/blob/fcb51852c8bb6e38403fe95a302287a0a1779d47/studies/diagnostics/quantile_loss_audit_20261109/SCIENCE_AUDIT.md)。这里仅维护来源链接，不复制 CSV 或图。

[工程审计导航](docs/repository/README.md) 区分历次清理的固定版本、逐文件来源和恢复证明。[保护产物登记](docs/PROTECTED_ARTIFACTS.json)、[数据消费登记](docs/data_consumption_register.json)、[历史协议恢复锁](docs/repository/phase2/HISTORICAL_PROTOCOLS.json) 保持原路径。Git 不自动备份被忽略的模型、原始 runtime 或凭据；复现实验仍须按协议准备授权资源。
