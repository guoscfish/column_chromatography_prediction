# S1 — Source-to-Target Structural Shift Audit

Status: `COMPLETED_STOPPED`. Exploratory/hypothesis-generating, not confirmatory.

## Contract

The truth-blind compound split assigned 62 compounds/402 rows to analysis and 26 compounds/172 rows to the S1-unconsumed reserved set. No reserved labels were parsed. Frozen 4g checkpoints 42/525/1101 matched the protected policy and historical source audit. Conditions use the frozen 9D representation standardized only by 4g source statistics. Simple corrections use nested GroupKFold by compound; outer validation compounds never enter training or Ridge-alpha selection.

## Domain overlap and conditions

8g contains 88 unique compounds; 87 occur exactly in 4g and one is target-only. Across compounds, the median number of matching 4g rows is 20 (IQR 12–21; range 0–42). Of 574 target rows, 567 have a same-compound source reference.

Condition means are close after source standardization (largest absolute mean shift 0.119 SD, loading-solvent code). Eluent-feature spread is about 1.012× source; target loading-amount and loading-solvent-volume spreads are 0.403× and 0.683× source. Global nearest-condition distance has median 0 and mean 0.0143; same-compound distance has median 0 and mean 0.1328. These are descriptive geometry, not causal physical variables.

## Frozen-source residual structure

| Target | mean | median | SD | IQR | P10 | P90 | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 10.158 | 5.748 | 12.603 | 6.328 | 3.224 | 22.314 | 10.357 | 16.187 |
| V2 | 15.519 | 10.159 | 17.823 | 10.202 | 5.178 | 30.733 | 15.759 | 23.633 |

Median target/source ratios are 2.043 (V1) and 1.790 (V2), using the preregistered 0.5 mL denominator floor. Mean source-ensemble row SD is only 0.774/1.311 mL, much smaller than residual RMSE. Residual V1/V2 correlation is Pearson 0.923 and Spearman 0.758.

Source prediction is strongly associated with residual (Pearson 0.801/0.766 for V1/V2). Same-compound distance has Spearman 0.116/0.115 and global distance 0.111/0.072: weak descriptive association. PE/EA log-ratio is associated with residual, but no causal interpretation is allowed. Flow is constant, so its correlation is undefined.

## Compound versus condition variation

Between-compound variance is 40.60 for V1 and 86.07 for V2; within-compound variance is 118.25 and 231.59. Between/total ratios are 0.256 and 0.271. Most residual variance occurs within compound across rows/conditions, while a smaller compound-level component remains. This measures residual clustering, not training reliability.

## Simple correction GroupKFold

| Model | V1 RMSE | V2 RMSE | combined NRMSE | SD | improvement vs zero-shot |
|---|---:|---:|---:|---:|---:|
| S0 zero-shot | 15.609 | 22.566 | 0.8027 | 0.2660 | 0.0000 |
| S1 global offset | 12.136 | 16.547 | 0.6076 | 0.2485 | 0.1950 |
| S2 affine | 7.634 | 11.450 | **0.3993** | 0.1299 | **0.4033** |
| S3 condition Ridge | 7.976 | 12.843 | 0.4317 | 0.1753 | 0.3709 |

Affine won three of five outer folds; condition Ridge won two. S1 resembles Case A with substantial remaining variation: simple affine correction captures a large part of shift but does not completely explain it. Condition-aware Ridge improves zero-shot yet supplies no stable incremental gain over affine.

## Manual implication and limits

T1 should include affine calibration, condition-aware residual, target readout, and current last2-head. S1 does not show that deep transfer is unnecessary, identify a physical mechanism, select a final model, consume reserved truth, alter historical E4/A2a null evidence, or authorize T1/active transfer.
