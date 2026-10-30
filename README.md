# 柱层析预测与主动学习

用分子几何和实验条件预测柱层析保留体积。当前预测器为 **QGeoGNN-V2**，`main` 集中维护 **4g 主动学习与 LLM 选样**。

## 从这里开始

| 想了解什么 | 阅读入口 |
| --- | --- |
| 正在研究什么、哪些结论已成立 | [当前研究状态](docs/NEXT_STAGE_DECISION.md) |
| 五种子 Free / LCMD 最终结果与曲线 | [最终自动生成报告](reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md) |
| NoCW 五种子最新进度及与其他策略的对照曲线 | [NoCW 结果快照](reports/nocw_progress_20261030/REPORT.md) |
| 各项研究的协议与结果 | [研究索引](studies/README.md) |
| 目录用途、文档为何这么多、以后怎么维护 | [仓库与文档指南](docs/README.md) |
| 模型和选样代码 | [代码地图](src/qgeognn_al/README.md) |
| 脚本怎样运行 | [脚本索引](scripts/README.md) |

## 当前范围

- **预测器**：458,952 个有效参数，完成三个种子的 row / compound 4g 点预测验证。分位数输出尚不代表不确定性已合格。
- **主动学习**：已完成的连续 B32 基准中，LCMD / Hybrid 相对随机选样的平均学习曲线面积改善约 23%，均为五个种子获胜。不同研究的过滤、划分和预算须分别解释。
- **LLM 选样**：重点研究 Free LLM 与 LCMD 的标签效率、反馈利用和计算成本。五种子延长实验已完成并通过最终审计，数值与历史授权例外统一见自动报告。
- **NoCW 消融**：五种子目标 L1005，仍在运行；已完成拟合、续跑授权和历史对照限制统一见上方 NoCW 快照。

主动学习与 LLM 主实验直接在 `main` 维护和提交，不另开工作分支；只有用户明确要求时才例外。迁移学习的冻结归档分支继续保留。

## 迁移学习已归档

冻结分支：`codex/archive-transfer-20261028`，归档日期：2026-10-28。迁移学习的完整代码、专用入口、实验、报告与测试保存在该分支，**停止维护，不再跟随 main 更新**。

main 完整保留 Free LLM 的实现、协议、记录和续跑入口，并保留其冻结校验必需的 28 个源文件，以及主动学习依赖的历史公共函数和数据锚点。归档范围、校验值与兼容依赖见[分离清单](docs/repository/RESEARCH_SCOPE.json)，阅读和复现方式见[仓库指南](docs/README.md#迁移学习归档)。

## 验证

从仓库根目录使用已有 Conda `fish` 环境：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
python3 scripts/audit_repository_hygiene.py
python scripts/report_free_lcmd.py --from-snapshot reports/free_llm_five_seeds_to1485_20261024/figures --output /tmp/free_lcmd_redraw
```

原始数据、冻结划分和有效检查点按保留规则维护；本地运行缓存不随 Git 分支发布。新克隆所需的预测器检查点说明见[4g 资格验证](studies/predictor/final_4g_qualification/README.md)。
