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

## 当前结论

- **预测器已资格验证**：458,952 个有效参数，完成三个种子的 row / compound 4g 验证。Clean 是历史负结果，不是当前模型。
- **4g 主动学习有效**：sequential B32 中 LCMD / Hybrid 相对 Random 的平均 AULC 改善约 23%，均为 5/5 种子获胜；两种主动方法之间没有稳定排名。当前在做固定总标签预算的 Static / Adaptive 对照。
- **跨柱迁移仍有限制**：低标签、无阈值基准中简单校准仍有竞争力。后续 readout、FiLM 和 PCGrad 没有建立稳健的跨场景收益；主动迁移尚未启动。

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

已完成且无现存调用依赖的 34 个一次性脚本已退役，结果仍在原目录；具体路径、校验值和恢复提交见[清理记录](docs/repository/RETIREMENTS.json)。需要复现退役实验时使用历史提交的完整环境和代码。

## 验证

本机已验证的环境为 Conda `fish`，从仓库根目录执行：

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
python3 scripts/audit_repository_hygiene.py
```

检查点和训练缓存位于被忽略的 `runtime/`。新克隆不会自动包含这些文件；迁移所需的精确源检查点及恢复说明见[4g 资格验证](studies/predictor/final_4g_qualification/README.md)。根目录 `NEXT_TRANSFER_MODEL_AUDIT.md` 是被哈希引用的历史协议，保留原文与路径。
