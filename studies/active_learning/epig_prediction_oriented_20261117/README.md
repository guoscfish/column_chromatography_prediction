# Prediction-Oriented Bayesian Active Learning / EPIG 迁移研究

独立分支：`codex/epig-prediction-oriented-20261117`。基线：重新 fetch 确认的 `b1f54b5132f671ab1c6068127fe8cc313df5b3ac`。历史运行、checkpoint、LLM记录和已有未提交修改均保留。

本目录包含实际实现、数学验证、五seed训练侧OOF、合法输入目标分布敏感性、实际执行的第二阶段nested calibration，以及可恢复原QGeoGNN训练入口。**用户要求的充分真实AL目标尚未完成：stage1与stage2均未让887/6101取得预注册后验资格，未启动新的五seed EPIG/IVR QGeoGNN正式拟合。不存在可报告的新EPIG AULC/终点结果；这不是EPIG NO-GO。**

stage1完整执行60次辅助拟合；stage2另行封存后完整执行240次nested辅助拟合、60个校准候选，没有放宽stage1门槛。实际总体结果见[stage1结果](RESULTS.md)、[stage2结果](stage2_nested_calibration/STAGE2_RESULTS.md)及[stage2决策](stage2_nested_calibration/STAGE2_NEXT_STAGE_DECISION.md)。计算出EPIG分数或改善中央覆盖都不等于提高Row-level AULC。

当前实际状态：第一阶段60次OOF及第二阶段240次nested校准拟合均已完成；两阶段均在seed887/6101缺少达到冻结门槛的后验。预定义NLL/central80尺度并未解除该限制，没有事后放宽门槛、删除seed或无限增加模型。**用户要求的完整五seed、210次后续QGeoGNN拟合尚未完成；本研究尚不能判定EPIG的真实Row-level效能。** 这不是资源不足，也不是短轨迹负面的NO-GO。

## 文件入口

| 内容 | 文件 |
| --- | --- |
| 当前六问与证据范围 | [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md) |
| 五seed结果与未执行项 | [RESULTS.md](RESULTS.md) |
| 代码/历史证据与可比性 | [CODE_AND_EVIDENCE_AUDIT.md](CODE_AND_EVIDENCE_AUDIT.md) |
| Gaussian公式、独立dense复核 | [THEORY_AND_IMPLEMENTATION.md](THEORY_AND_IMPLEMENTATION.md) |
| 概率资格与实际失败项 | [PROBABILISTIC_MODEL_QUALIFICATION.md](PROBABILISTIC_MODEL_QUALIFICATION.md) |
| 已实际执行的有限第二阶段 | [第二阶段结果](stage2_nested_calibration/STAGE2_RESULTS.md)、[第二阶段协议](stage2_nested_calibration/STAGE2_PROTOCOL.md) |
| 冻结协议与预算 | [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)、[config.json](config.json)、`freeze.json` |
| 机制诊断 | [ABLATION.md](ABLATION.md)、[PREDICTOR_MATCH_DIAGNOSTICS.md](PREDICTOR_MATCH_DIAGNOSTICS.md) |
| 解释边界 | [LIMITATIONS.md](LIMITATIONS.md)、[QUALIFICATION_FAILURE_ANALYSIS.md](QUALIFICATION_FAILURE_ANALYSIS.md) |
| 原验证基线 | `baseline_validation_curves.csv`、`baseline_seed_metrics.csv`、`baseline_summary.csv` |
| 逐seed/模型/OOF原数值 | `qualification/seed_metrics.csv`、`qualification/seed_*/oof_predictions.csv` |
| 逐候选与B32选择 | `acquisition/seed_*/scores.csv`、`batches.json`、`summary.json` |
| M与分布敏感性 | `acquisition/reference_sensitivity.csv`、`acquisition/summary.csv` |
| 图片与验证回执 | `figures/`、`preflight_tests.xml`、`test_receipt.json`、`initial_state_preflight.json` |
| 第二阶段独立协议、实际60候选资格与失败维度 | [STAGE2_PROTOCOL.md](stage2_nested_calibration/STAGE2_PROTOCOL.md)、[STAGE2_RESULTS.md](stage2_nested_calibration/STAGE2_RESULTS.md) |
| 单端点conditional B32补充消融（零新fit/标签） | `supplementary_output_ablation/summary.csv`、`analysis_seal.json` |

## 复现

在仓库根目录执行。已有 Conda `fish` 与原协议 scientific packages 相同；不安装依赖、不联网下载模型、不调用付费LLM。原环境需要沿用README中已有的OpenMP兼容设置。原数据、图缓存、五seed L333与历史运行是local-only依赖；缺失时明确失败。

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=src
EPIG_PYTHON=/Users/fish/miniforge3/envs/fish/bin/python
EPIG_STUDY=studies/active_learning/epig_prediction_oriented_20261117
"$EPIG_PYTHON" -B -m pytest -q tests/active_learning_v2/test_epig.py tests/active_learning_v2/test_epig_models.py tests/active_learning_v2/test_epig_study.py
"$EPIG_PYTHON" -B "$EPIG_STUDY/audit_baselines.py"
"$EPIG_PYTHON" -B "$EPIG_STUDY/run_study.py" freeze
"$EPIG_PYTHON" -B "$EPIG_STUDY/run_study.py" qualify
"$EPIG_PYTHON" -B "$EPIG_STUDY/run_study.py" acquisition
"$EPIG_PYTHON" -B "$EPIG_STUDY/predictor_response_diagnostics.py"
"$EPIG_PYTHON" -B "$EPIG_STUDY/report_study.py"
```

已有完成fold/状态会核验hash后复用。`freeze`拒绝更改已冻结代码；`qualify`逐fold保存输入/输出hash和每次拟合尝试；`acquisition`只读当时已授权L333标签拟合代理，不揭示它挑选的U标签。

```bash
"$EPIG_PYTHON" -B "$EPIG_STUDY/run_study.py" formal
```

此入口只有在五seed各自使用本seed L333 OOF通过预注册概率资格时才运行。它保留原1000 epoch上限、patience100和每轮scratch，单worker/2线程，累计210次后续拟合尝试、40小时硬预算；未通过资格不会自动换成简化预测器或删除失败seed。任何中断保留partial、标签访问审计和完成checkpoint，并可用同一命令恢复。未执行的正式学习曲线不生成虚构行。

本次科学代码和配置在任何资格结果前封存；后写的报告/诊断程序另外封存，其身份及输出在各自manifest中，不能改变已冻模型或选择规则。五个划分已经多次开发暴露，只能提供development evidence。后续独立确认应使用未暴露数据并重新预注册。

第二阶段由stage1结果触发，明确不是原stage1预注册的一部分。其identity/NLL-width/central80-width候选使用outer-train内部的honest inner OOF估计D_width，对所有covariance blocks一致DΣD。固定family的EPIG在这一变换下不变，IVR端点风险权重可变化；Gaussian温度不能修复尖峰重尾、均值或相关结构。重建第二阶段报告：

```bash
"$EPIG_PYTHON" -B "$EPIG_STUDY/stage2_nested_calibration/report_stage2.py"
```

第二阶段是查看第一阶段资格结果后、任何AL结果前**另立封存并已实际执行**的有限后续，不是原第一阶段预注册。入口见其README；阶段2的identity、NLL温度、central80温度共60个候选状态共享240次base fits，不能按三个温度重复计拟合数。DΣD保持EPIG信息排序不变，不能把通过校准门槛当成新增EPIG信息；所有候选均完整保留。两个阶段没有读取未来U、validation监督或test标签，没有调用新的付费LLM API。已有GNN只用于合法输入的冻结推理，0新增训练。
