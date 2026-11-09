# 柱色谱 QGeoGNN 与主动学习

这里是原仓库的重构分支 `codex/repository-architecture-20261109`。新代码在独立 linked worktree 中开发；运行 CW-Free 的原目录继续使用 `main`。当前重构没有合并进 `main`。

日常开发入口只有三层：

| 目录 | 负责什么 |
| --- | --- |
| `src/qgeognn_al/predictor/` | 模型、图与条件输入、预处理、训练、原始六输出推理 |
| `src/qgeognn_al/al/` | 表示、选样、代理计划、回合状态机、评价 |
| `src/qgeognn_al/infra/` | 身份、哈希、授权资源、工件 |

先从 [NEW_ARCHITECTURE.md](docs/repository/phase2/NEW_ARCHITECTURE.md) 了解调用关系与方法边界，再查看 [TEST_REPORT.md](docs/repository/phase2/TEST_REPORT.md) 的实测范围。旧 `active_learning_v2/v3/v3_2`、transfer 和历史专用 runner 已从候选稳定源码退出；历史实验通过固定 Git 来源恢复，不能直接用新代码替换其冻结运行树。

在此工作树中运行一个完整的合成离线回合示例：

```sh
python -m scripts.run_al --config configs/offline_synthetic.json --output /private/tmp/column_al_demo
```

首次运行时输出目录须为新目录；相同代码身份和配置可以恢复已有回合。示例执行合成 fit → 验证 → 反馈 → 选样 → 持久冻结 → 合成标签揭示 → 下一轮 fit，预算从 2 到 6。`requirements-core.txt` 列出所需 NumPy/SciPy 版本；模型和测试依赖另见 `requirements-predictor.txt`、`requirements-test.txt`。请在独立环境准备依赖。此次工作没有安装包或更改正在运行实验的环境。

CLI 目前只开放 `offline_synthetic`，不加载真实标签、训练科学模型或调用 LLM。生产模型后端已有独立实现，但默认关闭训练；完整科学重训练尚未验证。Free/CW/NoCW/V3 的旧提示词、重试和协议仍是历史来源，没有冒充兼容的新 LLM adapter。

分位数与损失复盘按用户要求单独提交到 `main`：`fcb51852c8bb6e38403fe95a302287a0a1779d47`，目录 `studies/diagnostics/quantile_loss_audit_20261109/`。本分支仅保留 [科学审计索引](docs/repository/phase2/SCIENCE_AUDIT.md)，不复制分析结果。

清理的逐文件依据见 [MIGRATION_MAP.csv](docs/repository/phase2/MIGRATION_MAP.csv)、[INVENTORY.csv](docs/repository/phase2/INVENTORY.csv)、[DELETION_PROOF.csv](docs/repository/phase2/DELETION_PROOF.csv)。历史恢复方法见架构文档。当前实验结束并得到正式切换授权前，继续保留 `main` 的旧源码和运行目录。
