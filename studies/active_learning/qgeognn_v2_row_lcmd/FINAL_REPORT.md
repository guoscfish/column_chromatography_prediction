# Final report — QGeoGNN-V2 4g row Gradient-LCMD-TP

Decision: **`STRONG_POSITIVE`**. The preregistered one-step experiment is complete and stops here.

## Primary result

Across five independent outer row seeds, LCMD beat the within-seed Random median in **5/5** seeds. Mean LCMD after-batch combined normalized RMSE was **0.440213**, versus **0.644425** across the 25 Random controls, an improvement of **31.69%**. Mean `LCMD gain − Random mean gain` was **0.204212** and the cross-seed median `LCMD gain − Random median gain` was **0.198739**.

Mean endpoint RMSE changes for LCMD relative to Random were **-30.86%** for V1 and **-32.39%** for V2 (negative is better).

| Outer seed | L0 baseline NRMSE | LCMD after NRMSE | Random mean after | Random median after | LCMD beat count |
|---:|---:|---:|---:|---:|---:|
| 73 | 0.906007 | 0.493123 | 0.792571 | 0.799820 | 5/5 |
| 311 | 0.837026 | 0.485974 | 0.737122 | 0.738546 | 5/5 |
| 1297 | 0.708525 | 0.446939 | 0.602884 | 0.602910 | 5/5 |
| 4093 | 0.653849 | 0.374220 | 0.519991 | 0.521007 | 5/5 |
| 8191 | 0.774395 | 0.400809 | 0.569557 | 0.599547 | 5/5 |

## Baseline and error shape

At 10% of outer-training labels (333 active labels, plus 416 shared validation labels), mean test combined normalized RMSE was **0.775960**. Mean V1/V2 RMSE was **6.104/11.184 mL**. Mean baseline test RMSE/MAE ratios were **2.026/2.010**, compared with training ratios **1.698/1.812**. These unusually high ratios support a heavy-tail/outlier-sensitive error diagnosis, but do not by themselves prove a particular error distribution. This diagnostic was not used to tune acquisition.

## What LCMD selected

Relative to the mean Random batch, LCMD's selected rows had mean gradient norm **206.514** versus **119.990**, mean nearest-L0 gradient distance **125.929** versus **60.480**, and covered **137.4** versus **176.8** initial gradient clusters. It selected **160.8** unique compounds per batch versus **164.4** for Random, with repeated-row fractions **0.817** versus **0.796**. Mean nearest-L0 standardized condition distance was **0.230** versus **0.199**; mean unique eluent-ratio counts were **8.40** versus **8.76**, and both covered all **3** loading solvents. Thus LCMD emphasized gradient-space distance and norm rather than maximizing raw compound or categorical-condition counts. These are label-free mechanism descriptions, not post-hoc selection changes.

## Scientific interpretation

The overall decision follows the frozen gate exactly. Both endpoints improved substantially: mean V1 RMSE fell from **5.165** to **3.571 mL**, and mean V2 RMSE fell from **9.018** to **6.097 mL**. The relative reduction is slightly larger for V2, but the result is not driven by only one endpoint. Five outer seeds are the independent replication units; the five Random controls within each seed estimate the conditional Random distribution and are not 25 independent datasets.

This current-V2 result must not be numerically pooled with Wu et al.'s full 4g result or with legacy E2/A1a experiments. It answers whether labels treated as initially unknown are chosen more effectively for this qualified canonical V2 row protocol. It measures label efficiency, data efficiency, and experimental-data acquisition efficiency—not wall-clock training acceleration.

The validation labels are a fixed shared auxiliary cost and are excluded from the active label count. Test truth was first accessed only after every acquisition, checkpoint, and prediction was frozen and hashed. No test result changed seeds, sketch size, gradient scope, batch size, distance, model, or training settings.

## Next stage

Full row learning curve recommended: **true**. Regardless of this result, no BAIT, Condition-LCMD, UCB/EI/TS, Hybrid continuation, or automatic tuning is launched.

## Method references

- Holzmüller et al., [A Framework and Benchmark for Deep Batch Active Learning for Regression](https://www.jmlr.org/papers/v24/22-0937.html), JMLR 24(164), 2023.
- Author implementation: [`dholzmueller/bmdal_reg`](https://github.com/dholzmueller/bmdal_reg). This study uses the corrected TP initialization semantics and an explicitly documented two-output QGeoGNN-V2 adaptation.
