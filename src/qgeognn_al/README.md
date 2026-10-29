# 科学代码地图

先看当前预测器，再看数据、训练和选样。研究编排位于 `scripts/studies/`，科学包不导入运行脚本。

## 当前预测器

| 模块 | 职责 |
| --- | --- |
| [models/qgeognn_v2.py](models/qgeognn_v2.py) | 模型构建、检查点加载、预测与表示提取 |
| [schemas/conditions.py](schemas/conditions.py) | 实验条件类型约定 |
| [data.py](data.py)、[input_schema.py](input_schema.py) | 规范数据、图构建与输入语义 |
| [training/predictor.py](training/predictor.py) | 训练、预处理与检查点选择 |
| [evaluation/point.py](evaluation/point.py)、[evaluation/reporting.py](evaluation/reporting.py) | 点预测指标和资格验证报告 |

## 主动学习

按以下顺序阅读 `active_learning_v2/`：

1. `protocol.py`、`sequential_protocol.py`：样本身份、标签预算、划分与冻结来源。
2. `gradient_features.py`、`gradient_bank.py`、`lcmd.py`：梯度表示与 LCMD 选样。
3. `sequential_acquisition.py`、`sequential_runner.py`：每轮选择、标签揭示、重训、冻结与评价。
4. `batch_adaptivity.py`：固定预算的静态/自适应对照。
5. `sequential_reporting.py`、`efficiency_reporting.py`：学习曲线、目标标签数与成本。

`benchmark*`、`phase1.py` 属于早期一步基准，后者不是 9 月的批次自适应研究。`ivr*`、`block_ivr.py` 保留 IVR 实现及诊断；`gradient_transforms.py`、`innovation_screen.py` 在冻结初始状态上做只选样的机制筛选，不重训、不读测试标签。

## LLM 选样与反馈

- `scientist_full_delivery.py`、`scientist_selector.py`：全池编码、可见标签边界与选择方案验证。
- `scientist_transport.py`、`scientist_study.py`：Scientist V2 的冻结传输和轨迹编排。
- `scientist_lifecycle.py`：新研究可参数化的冻结、标签揭示、拟合与反馈流程；复用原训练规则。
- `dual_expert_selector.py`、`dual_expert_transport.py`、`dual_expert_study.py`：化学与机器学习专家建议、自由选样规划、版本继承和修复凭据。
- `active_learning_v3/`、`active_learning_v3_2/`：独立版本的上下文、记忆、查询、传输与执行约定。

`scientist_lifecycle.py` 与冻结的 `scientist_study.py` 存在相似流程，是为了保留旧实验哈希。活跃实验期间不合并这些科学源文件。当前五种子展示统一由 `scripts/report_free_lcmd.py` 维护。

## 公共与历史模块

`partitions.py`、`metrics.py`、`resources.py`、`artifacts.py` 提供公共基础设施。`model.py`、`engine.py`、`acquisition.py` 保留早期主动学习实现；`condition_complete_v2*` 与 `historical/` 保留等价性、复现及失败架构对照。`diagnostics/`、`uncertainty/` 存在不代表不确定性已资格验证。

迁移研究的运行入口、实验结果与专用测试已归档到 `codex/archive-transfer-20261028`。`transfer/`、`t1_formal.py`、`t1b1.py` 共 28 个源文件仍被现有 Free LLM 协议按哈希冻结，因此作为当前课题的必要复现依赖继续在 main 跟踪，保留原路径与原字节，停止迁移研究更新。main 还保留公共流程引用的旧数据锚点和兼容脚本，见[分离清单](../../docs/repository/RESEARCH_SCOPE.json)。
