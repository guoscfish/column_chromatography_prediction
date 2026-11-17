# GOLLuM 方法复核、GP训练诊断与加性核融合

2026-11-06。研究结论从 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md) 阅读；运行范围和预设门槛见 [PROTOCOL.md](PROTOCOL.md)。本目录独立新增，旧研究只读，不自动进入正式主动学习。

| 内容 | 入口 |
| --- | --- |
| 旧720拟合、梯度、数据身份审计 | [AUDIT.md](AUDIT.md) |
| Phase1机制诊断 | [PHASE1_GP_DIAGNOSIS.md](PHASE1_GP_DIAGNOSIS.md) |
| 分组fit/cal/check真实消融 | [PHASE1_RESULTS.md](PHASE1_RESULTS.md) |
| GOLLuM真实源码与版本对齐 | [GOLLUM_METHOD_ALIGNMENT.md](GOLLUM_METHOD_ALIGNMENT.md) |
| 模型、资源和授权边界 | [PHASE2_MODEL_AND_RESOURCE_PLAN.md](PHASE2_MODEL_AND_RESOURCE_PLAN.md) |
| T5/LoRA执行状态与工程测试 | [PHASE2_RESULTS.md](PHASE2_RESULTS.md) |
| 加性核、残差、分解式数学设计 | [PHASE3_FUSION_DESIGN.md](PHASE3_FUSION_DESIGN.md) |
| 加性核真实开发结果 | [PHASE3_RESULTS.md](PHASE3_RESULTS.md) |

`phase1/` 保留所有初始化的fit、观测/潜变量预测、校准、三种采集和模板结果；`full_l/` 是全L_t原样复跑而非独立核验；`summary/` 为逐seed/预算/端点CSV与图。`phase2/` 的toy结果全都不是T5效果；`phase3/` 为真实冻结特征加性核结果。

模块边界：`data_boundary.py`校验历史状态及权限；`gp_training.py`实现受控Exact GP和校准；`text_embeddings.py`本地离线文本编码；`lora_adaptation.py`全批可微MLL与rsLoRA；`kernel_fusion.py`加性核；`summarize_phase1.py`与`write_phase1_reports.py`只汇总结果。GNN隐藏/梯度提取和采集代数直接复用旧研究的`representations.py`、`gp_models.py`，未重复训练或重写一套GNN/采集器。`run_phase1.py`中的采集包装只增加覆盖统计，底层MaxVar/conditionalIVR/BatchIG来自已审计实现。

项目根目录下使用现有环境，或将 `PYTHON_BIN` 改为具备相同依赖的已有环境。以下命令不会下载模型/安装依赖。完成条目复用，不覆盖旧研究；主配置/代码哈希漂移会拒绝继续。

```bash
export PYTHON_BIN=/Users/fish/miniforge3/envs/chromatography/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/run_phase1.py
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/supplement_frozen_fusion.py
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/diagnose_full_l.py
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/repair_metadata.py
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/summarize_phase1.py
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/write_phase1_reports.py
"$PYTHON_BIN" -m pytest -q studies/active_learning/gollum_adaptation_fusion_20261106
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/verify_study.py
```

`phase1_metadata_repair.json`记录一处结果来源文字修正：Structured是响应无关协变量，不能标为GNN已见过L_t。先写失败回归测试再修复，只改元数据，并保留逐文件前后哈希及其余数值字段摘要。冻结训练源未改，复跑后的修正脚本可幂等重用。Phase3另保存初始化seed覆盖outer seed的元数据修复回执；不影响训练/预测/指标/采集数值。

Phase1训练源与配置在评分前冻结。Phase3配置与输入身份预先固定，但源码哈希是交付时最终哈希，不能冒称额外的评分前源码冻结。所有研究仍是开发证据；最终哈希、完整矩阵和测试回执不等于独立科学确认。
