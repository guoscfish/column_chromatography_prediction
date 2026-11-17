# EPIG v2：代码审计、实际诊断与全长效能运行

研究性质：**post-stage1/stage2 exploratory development study**。原研究及Stage1/2资格未修改。新分支为`codex/epig-prediction-oriented-v2-20261117`；协议提交`fd5d676ab`，执行freeze及启动回执提交`11980b406`。新主研究封存于`freeze.json`，旧研究2323文件的逐项不变校验见`evidence/original_preservation_check.json`。

已经实际完成：143项组合预检、另7项supervisor/代理数值测试；独立dense Gaussian135项比较；70个全量/共同抽样参考状态及60个原采集状态完全复现；60次honest OOF辅助fit和5次明确标为encoder条件诊断的BLR fit。详见下列审计文件及各自JSON/CSV回执。

完整五seed两臂L333→L1005实验已经启动，目标210次成功QGeoGNN后续fit，最多231次尝试，单key最多3次；训练72h、辅助12h累计活动预算（不把停机算训练）。保持原458952参数、scratch、1000epoch/patience100、固定L333尺度与validation checkpoint。当前真实完成数以自动刷新报告为准，**启动不等于完成，未完成不作EPIG有效/无效判断**。历史单臂105fits耗时15.66h，两臂外推约31.32h；该估计不保证完成时间。

| 文件 | 内容 |
|---|---|
| [DEEP_CODE_AUDIT.md](DEEP_CODE_AUDIT.md) | 源码位置、设置、风险、证据等级与修复 |
| [METHOD_CORRECTNESS.md](METHOD_CORRECTNESS.md) | EPIG/IVR/BALD、独立Schur oracle、multi-output与batch |
| [REFERENCE_DISTRIBUTION_AUDIT.md](REFERENCE_DISTRIBUTION_AUDIT.md) | 全量P1/P2、共同随机数、M收敛及实际RAM/时间 |
| [FEATURE_AND_SURROGATE_AUDIT.md](FEATURE_AND_SURROGATE_AUDIT.md) | 特征、优化、均值/噪声、有限OOF与encoder可行性 |
| [PREDICTIVE_DEPENDENCE_VALIDATION.md](PREDICTIVE_DEPENDENCE_VALIDATION.md) | full/block/diagonal NLPD、条件logscore、残差与条件校准 |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | 独立探索性协议与有限资源/访问预算 |
| [RESULTS.md](RESULTS.md) | 自动更新的已完成真实Row拟合；未完成明确留空 |
| [MECHANISM_ANALYSIS.md](MECHANISM_ANALYSIS.md) | 信息价值与真实学习收益之间的证据边界 |
| [LIMITATIONS.md](LIMITATIONS.md) | 未测模型因素、有限样本、复用验证与外推限制 |
| [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md) | 随实际完成状态更新的下一阶段判断 |

主要确定发现：原Gaussian噪声/条件化语义正确；独立参考抽样是实质比较混杂；M1024/2048不足以稳定B32；全量积分后EPIG/IVR差异仍在。新特征改变Top32，但有限模型族未普遍提高概率质量。以上均不是Row AULC结论。完整学习主代理预先固定为原F0 Matérn规则，不根据新OOF或学习结果挑选模型。

## 运行与恢复

本地常驻supervisor和worker状态在`runtime/supervisor_status.json`、`exploratory_status.json`；操作系统flock防止重复worker，2s活动账目心跳支持异常恢复。worker日志在`runtime/worker_*.log`。supervisor运行时每约45秒刷新已完成fit报告、五seed CSV和原始非单调曲线；训练期间不据早期得失停跑。

```bash
# 只读：检查旧/新封存依赖
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/run_exploratory.py verify
# 只读科学证据复核
/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/surrogate_verify.py
# 刷新报告：不训练，不揭示标签
/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/report_study.py
# 仅在当前supervisor停止后恢复；持锁旧worker存在时会监控它
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/supervisor.py
```

不要覆盖checkpoint或改动已封存决策文件。原初次代理汇总入口有已披露PCA缓存复用问题，完成后使用`surrogate_verify.py`，从零重跑应在隔离空结果副本中进行。主AL恢复路径有独立hash/state/partial-fit测试。

小型不可变选点/标签访问/模型审计/拟合回执自动导出到`receipts/`，完整计算账目快照为`compute_ledger_snapshot.json`，模型checkpoint保持在本地runtime并逐次哈希验证。`VERIFICATION.json`是审计交付时的静态快照；运行完成状态以自动刷新`reporting_manifest.json`为准。

导出：`validation_curves.csv`、`per_seed/seed_*_curves.csv`、`seed_metrics.csv`、`paired_differences.csv`、`compute_cost.json`及`figures/validation_learning_curves.png`。只有全部22预算点齐全时才计算完整AULC；不插值、不作best-so-far，不把fold/轮次当独立重复。未调用付费LLM API；未评估test标签；未来U标签只在不可变合法选点之后揭示。
