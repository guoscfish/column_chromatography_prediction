# 充分训练QGeoGNN的样品体积响应审计

三个正式Row模型各训练3330行后，体积响应仍未稳定恢复：Row525部分响应，Row42明显压缩，Row1101几乎平坦；原validation亦然。与AL相同pair比较没有显示充分训练普遍修复。先读 [FINAL_DECISION.md](FINAL_DECISION.md)，其中开头直接回答五个科研问题；三张主图位于`figures/`。

本轮0新模型训练、0LLM/API、0新实验标签揭示、0AL重跑。存在一次初始目标文本预览事故，涉及3ID随后全局隔离，另有一次未输出数值的文件误开；不能称严格全过程0test目标可见。完整例外和补救在`LABEL_BOUNDARY_INCIDENT.json`，数值评分只使用对应正式模型授权train/validation并剔除隔离ID，VERIFICATION明确保留事故状态。

| 文件 | 内容 |
|---|---|
| CHECKPOINT_AUDIT.md | 正式Row42/525/1101路径、SHA、3330训练身份及官方原生V2严格加载 |
| PAIR_COVERAGE.md | 单因素pair、系列/分子、水平/差值分布、重复行与权限门 |
| TRAIN_FIT_RESPONSE.md | 训练内volume/PE响应、逐模型与等权汇总、重复行差异 |
| VALIDATION_RESPONSE.md | 两端validation泛化、混合对单列、原Row指标重现 |
| MATCHED_AL_COMPARISON.md | L525/L1005/L1485与Fully同pair、曝光分层、分子差异 |
| MODEL_RESPONSE_EXAMPLES.md | 15个metadata固定曲线pool、案例规则、插值/外推范围 |
| FINAL_DECISION.md | 五问数值结论、局限与面向Row效率的下一步 |
| RESULTS_QA.md | 独立公式、raw预测、角色与曲线支持范围复算 |

正式model seed525与AL预算L525无关。三个正式checkpoint均本地存在、原生V2，无R2转换；seed42 SHA精确匹配指定fce9…544b0。正式模型分别使用自己的3330训练scaler，AL模型使用各自L333冻结scaler，比较响应统一为mL。完整域4163行从未被误称为3330训练行，AL旧metadata少1个非共同ID也没有补造。

## 复现

仓库根目录，解释器`/Users/fish/miniforge3/envs/chromatography/bin/python`，使用`-B`和`PYTHONDONTWRITEBYTECODE=1`，OMP/BLAS/MKL/VECLIB线程环境均1；无训练入口：

```sh
python -B studies/predictor/full_training_volume_response_audit_20261117/data_boundary.py
python -B studies/predictor/full_training_volume_response_audit_20261117/checkpoint_audit.py
python -B studies/predictor/full_training_volume_response_audit_20261117/al_checkpoint_audit.py
python -B studies/predictor/full_training_volume_response_audit_20261117/predict_formal.py
python -B studies/predictor/full_training_volume_response_audit_20261117/analyze_formal.py
python -B studies/predictor/full_training_volume_response_audit_20261117/matched_and_curves.py
python -B studies/predictor/full_training_volume_response_audit_20261117/matched_summary.py
python -B studies/predictor/full_training_volume_response_audit_20261117/make_figures.py
python -B studies/predictor/full_training_volume_response_audit_20261117/qa_results.py
python -B studies/predictor/full_training_volume_response_audit_20261117/delivery.py --check
```

此处`python`须指上述科学环境。正式与AL checkpoint、图cache及历史原日志为现有本地依赖；缺失时停止，不自动训练替代。全测试命令、实际结果和SHA在`TEST_RESULT.json`，输入/输出SHA在`VERIFICATION.json`。重复生成脚本会重写本目录派生表与时间戳，核验旧封存时只用`--check`，不要重生成后覆盖旧身份。

主要数据：`formal_response_per_model.csv`、`formal_response_per_series.csv`、`formal_response_per_molecule.csv`、`formal_row_metrics.csv`、`matched_seed_metrics.csv`、`matched_paired_improvements.csv`、`controlled_volume_curves.csv`。未请求source-test凑pair，未打开历史含test truth的predictions.csv.gz用于分析。Row NRMSE初期派生公式从算术均值改为官方RMS，完整21指标重现核验与R1修复审计保留；局部差分未受该修正影响。原始split与checkpoint未变。
