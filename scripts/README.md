# 脚本索引

从仓库根目录使用 Conda `fish` 环境。入口存在并不意味着需要重新启动实验；复现先核对冻结协议和输出目录。

## 当前报告与续跑

- NoCW 当前控制器为 `studies/run_qgeognn_v2_row_llm_nocw_resume_2357.py`。只读预检：`KMP_DUPLICATE_LIB_OK=TRUE python scripts/studies/run_qgeognn_v2_row_llm_nocw_resume_2357.py validate`。监督器已运行，勿重复启动；它同时监督原 stream、resume 和新 resume_2357 worker。
- [NoCW 最新报告](../reports/nocw_progress_20261030/REPORT.md)集中给出进度、修订哈希和曲线。`python reports/nocw_progress_20261030/plot_comparison.py` 只读取本地冻结拟合与监控快照，无训练或 API 调用；新克隆可直接读取已提交曲线 JSON/图。
- NoCW 全部到 L1005 后须通过 resume 适配器 `activate(); validate(); retry.report(1005)` 审计，保留三轮六答案例外；不能用旧三答案入口代替。

- `report_free_lcmd.py`：五种子 Free / LCMD 的最终验证结果与三组图。`--from-snapshot reports/free_llm_five_seeds_to1485_20261024/figures` 可离线重绘；`--output` 指定独立输出；`--require-complete` 拒绝不完整轨迹。
- 正在运行的监督器和冻结续跑入口见[当前报告](../reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md)及其清单。保持单一进程所有权，避免重复启动。
- `report_active_learning_snapshot.py`：10 月 13 日两种子和双专家的历史快照读取器，不维护当前五种子进度。

## 研究入口

以下文件均位于 `scripts/studies/`，细节以[研究索引](../studies/active_learning/README.md)和对应协议为准。

| 研究 | 入口 |
| --- | --- |
| 预测器工程与资格验证 | `run_final_v2_engineering.py`、`run_final_4g_qualification.py`、`summarize_final_4g_qualification.py` |
| 连续主动学习与对照 | `run_qgeognn_v2_row_sequential_b32.py`、`run_qgeognn_v2_4g_row_small_batch_benchmark.py`、`run_qgeognn_v2_row_hybrid_extension.py`、`run_qgeognn_v2_4g_row_lcmd_pilot.py` |
| 固定预算与效率复核 | `run_qgeognn_v2_batch_adaptivity.py`、`summarize_qgeognn_v2_adaptivity.py`、`report_qgeognn_v2_efficiency.py` |
| 机制筛选与 IVR | `run_qgeognn_v2_4g_row_innovation_screen.py`、`run_qgeognn_v2_row_kernel_ivr_b32.py`、`run_ivr_mechanism_audit.py`、`report_ivr_mechanism_diagnostics.py` |
| CW 因子对照与诊断 | `run_cw_geometry_feedback.py`、`report_cw_geometry_feedback.py`、`report_cw_selection_diagnostics.py`、`audit_cw_factorial_delivery.py` |
| CW 延长与核验 | `run_cw_factorial_extension.py`、`run_cw_factorial_to1005.py`、`continue_cw_factorial_to1005.py`、`audit_cw_factorial_extension.py`、`audit_cw_factorial_to1005.py` |
| LLM Scientist | `run_qgeognn_v2_row_llm_scientist.py` 及 V3 / V3.2 的独立冻结入口 |
| 双专家选样 | `run_qgeognn_v2_row_llm_dual_expert.py`、`report_dual_expert_continuation.py` |

`prepare_cw_confirmation.py` 只做候选资格和未用种子预检，不训练或调用 LLM。`build_qgeognn_v2_row_master_report.py` 等历史报告入口保留原研究范围，不能代替当前报告。

## 历史兼容依赖

预测器历史资格验证和等价性对照仍保留。顶层 `run_e0_4g_baseline.py`、`run_e1_signal_qualification.py`、`run_e2_4g_active_learning.py` 等是早期数据/模型依赖；`al_engine.py`、`al_acquisition.py`、`qgeognn_graphs.py` 为兼容入口。

`run_e0_8g_controls.py` 与 `run_e0_8g_transfer.py` 因仍被公共图缓存加载路径引用而保留原字节。迁移专用入口已移入 `codex/archive-transfer-20261028`。新逻辑写入[科学包](../src/qgeognn_al/README.md)，不继续把历史运行器当公共库扩展。

## 维护与恢复

- `audit_repository_hygiene.py`：只读核验分支、归档、恢复、保护锚点与导入边界。
- `audit_datasets.py`：数据检查，会读取标签值，不是盲预检。
- [退役登记](../docs/repository/RETIREMENTS.json)：一次性完成脚本的原路径、哈希、恢复提交与证据入口。
- [分离清单](../docs/repository/RESEARCH_SCOPE.json)：此次迁移归档和文档合并去向。

复现旧研究使用原提交的独立工作副本和独立输出；不要把旧脚本直接复制进活跃流程。目录约定与产物保留规则统一见[仓库指南](../docs/README.md)。
