# Actual Row-level learning results

This is a post-stage1/stage2 exploratory development study. The original five-seed probability gate remains unsatisfied. All numbers below come from completed fit receipts, with checkpoint hashes verified. No test targets are evaluated.

Completed downstream QGeoGNN fits: **5/210**. Complete L333–L1005 seed pairs: **0/5**.

**Primary five-seed efficacy is not yet evaluated.** Missing budgets are not interpolated and partial curves are not substitutes for the required learning window. No EPIG failure or success verdict follows from this snapshot.

| Seed | Arm | Completed fits | Last budget | Full AULC | L1005 NRMSE |
|---|---|---:|---:|---:|---:|
| 157 | EPIG | 3/21 | 429 | pending | pending |
| 157 | IVR | 2/21 | 397 | pending | pending |
| 887 | EPIG | 0/21 | pending | pending | pending |
| 887 | IVR | 0/21 | pending | pending | pending |
| 2357 | EPIG | 0/21 | pending | pending | pending |
| 2357 | IVR | 0/21 | pending | pending | pending |
| 6101 | EPIG | 0/21 | pending | pending | pending |
| 6101 | IVR | 0/21 | pending | pending | pending |
| 12203 | EPIG | 0/21 | pending | pending | pending |
| 12203 | IVR | 0/21 | pending | pending | pending |

Files: `validation_curves.csv` (including physical V1/V2 MAE/RMSE in ml), `seed_metrics.csv` (early/middle/late and milestone metrics), `paired_differences.csv`, `compute_cost.json`, `reporting_manifest.json`. Curves are original nonmonotone checkpoint metrics. Full AULC is only defined for all22 expected budgets. Initial checkpoints are reused, never counted as new fits.

The two arms use identical predefined GP modeling rules. Once selections diverge their training sets and posteriors differ. Offline common-state comparisons isolate the objective; trajectory comparisons measure the entire acquisition-and-retraining procedure. Historical methods remain in the original baseline inventory and are not silently pooled as protocol-identical controls.

Resume/run: `/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/run_exploratory.py run`. Refresh this report: same Python with `report_study.py`. Do not change sealed decision files or delete partial attempts.
