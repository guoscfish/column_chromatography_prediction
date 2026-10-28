# 迁移学习归档（停止维护）

归档日期：2026-10-28。分支：`codex/archive-transfer-20261028`。

该分支保存分离前仓库的完整已提交快照，迁移学习的代码、实验协议、划分、指标、报告和测试均保留原路径与原始内容。此分支停止研究更新，不跟随 main 同步或合并后续主动学习进展。main 从此集中于 4g 主动学习、LLM 选样及其预测器。

基础提交：`2bb086dbd2c23a82485f27f3463b1800298a4f27`。仅本首页增加归档说明；下方是基础提交的项目介绍，描述的是归档时的历史状态。正在运行的主动学习任务和未提交进度未纳入本归档。被忽略的本地 runtime、检查点及请求缓存不属于 Git 快照，原有本地文件继续保留。

查看迁移内容：

- `studies/transfer/`：跨柱迁移的完整研究记录。
- `studies/track_b_transfer/`、`studies/track_c_active_transfer/`、`studies/active_learning/transfer/`：历史迁移与主动迁移设计。
- `src/qgeognn_al/transfer/`、`scripts/studies/`：迁移算法与执行入口。
- `docs/research/CROSS_COLUMN_TRANSFER_STATUS.md`：归档前的迁移研究总结。

需要阅读或复现时，请建立单独工作副本，不要切换正在运行主动学习实验的 main 工作目录。

---

# Column Chromatography Prediction

用分子几何和实验条件预测柱层析保留体积。当前模型是 **QGeoGNN-V2**；研究分为 4g 主动学习和跨柱迁移两条线。

## 从这里开始

| 要了解什么 | 入口 |
| --- | --- |
| 当前结论、正在做什么 | [研究状态](docs/NEXT_STAGE_DECISION.md) |
| 实验结果与证据 | [研究索引](studies/README.md) |
| 模型、训练和采样代码 | [代码地图](src/qgeognn_al/README.md) |
| 可运行的入口 | [脚本索引](scripts/README.md) |
| 目录、分支和维护规则 | [仓库结构](docs/repository/STRUCTURE.md) |

主工作副本为 `/Users/fish/Documents/GitHub/column_chromatography_prediction`；其他实验 worktree 可保留独立工作上下文。已验证的代码和研究记录统一合并回 `main`，未完成实验明确标为草案。

- [当前五seed Free / LCMD结果、进度与曲线](reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md)
- [2026-10-13两seed历史快照](reports/active_learning_latest/REPORT.md)
- [2026-10-05历史汇总](reports/active_learning_status_20261005/REPORT.md)（日期固定，不代表当前进度）
- [2026-10-05 仓库整理与合并记录](docs/repository/CLEANUP_20261005.md)
- [双专家主动学习（seed157到1005；seed6101到525）](studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/README.md)

## 当前结论

- **预测器已资格验证**：458,952 个有效参数，完成三个种子的 row / compound 4g 验证。Clean 是历史负结果，不是当前模型。
- **4g 主动学习有效**：sequential B32 中 LCMD / Hybrid 相对 Random 的平均 AULC 改善约 23%，均为 5/5 种子获胜；两种主动方法之间没有稳定排名。固定预算Static / Adaptive对照已完成；当前Free / LCMD五seed延长到1485；最新完成点以当前报告为准。
- **跨柱迁移仍有限制**：低标签、无阈值基准中简单校准仍有竞争力。后续 readout、FiLM 和 PCGrad 没有建立稳健的跨场景收益；Row-CV五seed完整评估也未建立稳定双柱收益；主动迁移尚未启动。

指标的适用范围和原始报告统一见[研究状态](docs/NEXT_STAGE_DECISION.md)，不能将不同数据过滤、标签预算或划分下的数值混合排名。

## 目录

```text
src/qgeognn_al/  模型与可复用算法
scripts/        执行入口、维护工具、仍有依赖的历史入口
tests/          实现、数据边界和结果完整性检查
studies/        predictor / active_learning / transfer 的协议与结果
experiments/    早期实验记录及仍被使用的数据锚点
docs/           当前状态、契约、维护规则
dataset/        原始数据
application/    旧应用及 Legacy 模型
automation/     旧数据采集工具
```

已完成且无现存调用依赖的一次性脚本及多余展示备份已退役，结果仍在原目录；具体路径、校验值和恢复提交见[清理记录](docs/repository/RETIREMENTS.json)。需要复现退役实验时使用历史提交的完整环境和代码。

## 验证

本机已验证的环境为 Conda `fish`，从仓库根目录执行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
python3 scripts/audit_repository_hygiene.py
python scripts/report_free_lcmd.py --from-snapshot reports/free_llm_five_seeds_to1485_20261024/figures
```

检查点和训练缓存位于被忽略的 `runtime/`。新克隆不会自动包含这些文件；迁移所需的精确源检查点及恢复说明见[4g 资格验证](studies/predictor/final_4g_qualification/README.md)。根目录 `NEXT_TRANSFER_MODEL_AUDIT.md` 是被哈希引用的历史协议，保留原文与路径。
