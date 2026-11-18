# Actual Row-level learning results

**已按用户要求暂停。完成178/210次成功拟合；当前未完成训练及部分checkpoint已保留，自动续跑进程已停止。只有收到用户明确恢复指令后才继续。**

This is a post-stage1/stage2 exploratory development study. The original five-seed probability gate remains unsatisfied. All numbers below come from completed fit receipts, with checkpoint hashes verified. No test targets are evaluated.

Completed downstream QGeoGNN fits: **178/210**. Complete L333–L1005 seed pairs: **4/5**.

**Primary five-seed efficacy is not yet evaluated.** Missing budgets are not interpolated and partial curves are not substitutes for the required learning window. No EPIG failure or success verdict follows from this snapshot.

| Seed | Arm | Completed fits | Last budget | Full AULC | L1005 NRMSE |
|---|---|---:|---:|---:|---:|
| 157 | EPIG | 21/21 | 1005 | 0.6689232 | 0.5618654 |
| 157 | IVR | 21/21 | 1005 | 0.7013625 | 0.6625346 |
| 887 | EPIG | 21/21 | 1005 | 0.4375225 | 0.3998096 |
| 887 | IVR | 21/21 | 1005 | 0.4486492 | 0.3952451 |
| 2357 | EPIG | 21/21 | 1005 | 0.4954271 | 0.4380041 |
| 2357 | IVR | 21/21 | 1005 | 0.5066270 | 0.4269038 |
| 6101 | EPIG | 21/21 | 1005 | 0.7739560 | 0.6513628 |
| 6101 | IVR | 21/21 | 1005 | 0.7800512 | 0.6283102 |
| 12203 | EPIG | 5/21 | 493 | pending | pending |
| 12203 | IVR | 5/21 | 493 | pending | pending |

Files: `validation_curves.csv` (including physical V1/V2 MAE/RMSE in ml), `seed_metrics.csv` (early/middle/late and milestone metrics), `paired_differences.csv`, `compute_cost.json`, `reporting_manifest.json`. Curves are original nonmonotone checkpoint metrics. Full AULC is only defined for all22 expected budgets. Initial checkpoints are reused, never counted as new fits.

The two arms use identical predefined GP modeling rules. Once selections diverge their training sets and posteriors differ. Offline common-state comparisons isolate the objective; trajectory comparisons measure the entire acquisition-and-retraining procedure. Historical methods remain in the original baseline inventory and are not silently pooled as protocol-identical controls.

Resume/run: `/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/run_exploratory.py run`. Refresh this report: same Python with `report_study.py`. Do not change sealed decision files or delete partial attempts.
