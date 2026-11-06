# 冻结实验表示与 GP 开发基准

本研究仅重用已有 LCMD32 状态，训练 GP 与小型 Projection，不训练 QGeoGNN、不获取候选标签、不启动主动学习。协议、配置与代码哈希见 [PROTOCOL.md](PROTOCOL.md)、[config.json](config.json)、[protocol_freeze.json](protocol_freeze.json)；工程边界修复保留在 [engineering_amendments.json](engineering_amendments.json)。

报告入口：[审计](AUDIT.md)、[数学定义](METHOD.md)、[表示报告](REPRESENTATION_REPORT.md)、[概率报告](GP_CALIBRATION_REPORT.md)、[对照报告](COMPARISON_REPORT.md)、[AL 可行性](AL_FEASIBILITY.md)、[下一阶段裁决](NEXT_STAGE_DECISION.md)。报告只根据已完成的真实结果生成。

## 运行

在仓库根目录使用已有、包含 torch / torch-geometric / RDKit / scipy / transformers 的 Python 3.11 或更新环境。模型仅从本地缓存加载，依赖和权重均不自动安装或下载。下列 `python` 指向该环境；数据使用相对路径，另可通过 `--project-root` 配置根目录。

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Repository-compatible workaround when its installed OpenMP runtimes conflict.
export KMP_DUPLICATE_LIB_OK=TRUE
python -B -m pytest -q studies/active_learning/representation_gp_benchmark_20261104
python -B studies/active_learning/representation_gp_benchmark_20261104/run_benchmark.py --phase audit
python -B studies/active_learning/representation_gp_benchmark_20261104/run_benchmark.py --phase preflight
python -B studies/active_learning/representation_gp_benchmark_20261104/run_benchmark.py --phase full
python -B studies/active_learning/representation_gp_benchmark_20261104/summarize_results.py
python -B studies/active_learning/representation_gp_benchmark_20261104/residual_diagnostics.py
python -B studies/active_learning/representation_gp_benchmark_20261104/write_reports.py
python -B studies/active_learning/representation_gp_benchmark_20261104/write_decision.py
python -B studies/active_learning/representation_gp_benchmark_20261104/finalize_artifacts.py
python -B studies/active_learning/representation_gp_benchmark_20261104/verify_study.py
python -B studies/active_learning/representation_gp_benchmark_20261104/audit_text_cache.py
python -B studies/active_learning/representation_gp_benchmark_20261104/reproduce_gp.py
python -B studies/active_learning/representation_gp_benchmark_20261104/seal_artifacts.py
python -B studies/active_learning/representation_gp_benchmark_20261104/seal_artifacts.py --check
```

已有状态缓存仅在 SHA256、checkpoint、L/U、索引顺序和源数据身份一致时复用；完整结果有 `complete.json`，重复运行不会重新拟合这些状态。保留预检状态后，full 只补充其余状态与缺失语言向量，同状态只汇总一次。运行顺序固定为 seed 157/887/2357/6101/12203，预算 333/525/813/1005；每状态固定表示、linear 控制、三个 Projection 初始化依次执行。任何有限但未收敛的拟合仍保留，不因 validation 成绩增加起点或优化步数。

## 数据和结果

- `state_audit.json` / `representation_source_audit.json`：20 个历史状态的来源与哈希。
- `cache/*.json`：实际读取权限、特征、模型与预处理回执；`cache/*.npz` 为可重算本地缓存，不纳入 Git。
- `results/seed_*_L*/`：每个模型的真实预测、方差、MLL、参数、条件对及无标签采集诊断。
- 汇总 CSV/JSON 和图由 `summarize_results.py` 从完整状态生成；缺失状态不冒充完成。
- `summary/predictions.csv.gz`：全部实际开发评价预测；`summary/costs.json`：实测编码、特征和GP时间。
- `verification.json`：代码/缓存/预算/候选/标签边界与矩阵完整性核验。

当前本地 LLM 是明确标记的 Qwen1.5-0.5B-Chat 冻结 causal hidden-state 对照，不等于 T5 或完整 GOLLuM。每 seed 的 validation 曾用于选择 QGeoGNN checkpoint，因此全部结果是开发证据。四预算及五 seed 相关；加载溶剂体积轴只有 3 个验证对，不能作稳定结论。任何模型内信息增益都不代表已提高 AULC。
