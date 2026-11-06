# 数值引导与稳定化 LLM 主动学习：Phase 0

已完成离线审计、真实历史数据GP验证、Controller与三臂adapter原型、mock和输入差异检查。**正式GP引导及付费Phase1为NO-GO**：响应GP显著欠覆盖，当前checkpoint风险未获验证。详情见 [OFFLINE_REPORT.md](OFFLINE_REPORT.md) 与 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。本目录不提供生产LLM、标签揭示或训练入口。

| 交付内容 | 文件 |
|---|---|
| 最新仓库、实验与复用审计 | [AUDIT.md](AUDIT.md) |
| 可证伪问题与证据层级 | [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) |
| GP公式、概率目标、IVR区别 | [GP_METHOD.md](GP_METHOD.md) |
| 数值规则、触发、局部豁免、局限 | [CONTROLLER_DESIGN.md](CONTROLLER_DESIGN.md) |
| A/B/C、稳定性、成本及分阶段协议 | [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) |
| 实测离线结果 | [OFFLINE_REPORT.md](OFFLINE_REPORT.md) |
| 下一阶段结论与最小实验 | [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md) |
| GP与只读历史adapter | [gp.py](gp.py)、[offline.py](offline.py) |
| 控制器与冻结adapter | [controller.py](controller.py)、[arm_adapter.py](arm_adapter.py) |
| 冻结输入与来源 | [input_manifest.json](input_manifest.json)、[configs/offline_plan.json](configs/offline_plan.json) |
| mock审计和输入差异 | [results/mock_controller/mock_trace.json](results/mock_controller/mock_trace.json)、[results/real_frozen_input_check.json](results/real_frozen_input_check.json) |
| 测试与重放 | [results/test_results.txt](results/test_results.txt)、[results/reproduction_verification.json](results/reproduction_verification.json) |

从仓库根目录运行，使用已有chromatography环境，无需安装或凭据：

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/gp_guided_llm_stability_20261103/verify_study.py
```

单独复算历史GP：

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/gp_guided_llm_stability_20261103/offline.py
```

`offline.py --capture` 仅用于首次建立来源锁；存在锁时拒绝覆盖。默认按45个原始来源的SHA256严格重放，文件变化或缺失即失败，不寻找新标签补齐。真实输入差异脚本使用独立来源锁，不改变GP锁。

原始cache/catalog/input/feedback属于已有本地runtime策略，部分gitignored；新克隆可能没有。不能在缺本地依赖时声称重放成功。合成单元测试和mock不依赖历史文件，可单独运行：

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B -m unittest discover -s studies/active_learning/gp_guided_llm_stability_20261103 -p 'test_*.py' -v
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/gp_guided_llm_stability_20261103/mock_controller_demo.py
```

`results/predictions.csv` 仅含已观测训练池点的评分，不含test真值；`residual_provenance.csv`区分固定r00与原生采集checkpoint。数值结果是开发证据；默认Controller阈值全null、始终不可部署。mock toy阈值、未来12响应建议和任何后续门槛均不构成执行授权。
