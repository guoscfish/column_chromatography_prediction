# QGeoGNN point-prediction regression report

Status: `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC_COMPLETE / CLEAN_ARCHITECTURE_REGRESSION_CONFIRMED / POINT_PERFORMANCE_QUALIFICATION_REOPENED`.

## Executive result

R0 reproduced historical E0 almost exactly. R1 showed that the current Clean training protocol does not cause the regression. R2 showed that completing the condition path does not cause the regression. R3 reproduced the low Clean performance on the identical E0 split. The first regression is therefore R2→R3.

## Frozen control

All variants used the historical `experiments/e0_4g_baseline/split_seed_42.csv` without regeneration: 3,330 train, 416 validation, 417 test, total 4,163. Source rows obeyed V1≤60 and V2≤120. Scaling was train-only, no 8g data or downstream module was read, validation alone selected checkpoints, and test was evaluated after reload. Scientific role is `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC`, not confirmatory comparison.

## Complete point metrics

| Variant | Split | V1 R² | V1 RMSE | V1 MAE | V2 R² | V2 RMSE | V2 MAE | Combined norm. RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R0 Legacy exact | Train | 0.901 | 2.446 | 1.335 | 0.944 | 3.788 | 2.160 | 0.279 |
| R0 Legacy exact | Validation | 0.847 | 2.903 | 1.458 | 0.871 | 5.502 | 2.717 | 0.359 |
| R0 Legacy exact | Test | 0.867 | 2.971 | 1.604 | 0.903 | 4.969 | 2.868 | 0.348 |
| R1 Legacy + Clean protocol | Train | 0.928 | 2.089 | 1.122 | 0.966 | 2.959 | 1.689 | 0.231 |
| R1 Legacy + Clean protocol | Validation | 0.851 | 2.864 | 1.393 | 0.881 | 5.267 | 2.349 | 0.349 |
| R1 Legacy + Clean protocol | Test | 0.866 | 2.980 | 1.502 | 0.930 | 4.208 | 2.341 | 0.329 |
| R2 Condition Completion V2 | Train | 0.935 | 1.977 | 1.042 | 0.968 | 2.867 | 1.616 | 0.220 |
| R2 Condition Completion V2 | Validation | 0.858 | 2.793 | 1.300 | 0.883 | 5.230 | 2.365 | 0.344 |
| R2 Condition Completion V2 | Test | 0.889 | 2.711 | 1.374 | 0.942 | 3.854 | 2.136 | 0.300 |
| R3 Clean current | Train | 0.553 | 5.201 | 3.865 | 0.583 | 10.318 | 7.221 | 0.657 |
| R3 Clean current | Validation | 0.412 | 5.685 | 4.399 | 0.451 | 11.324 | 7.953 | 0.720 |
| R3 Clean current | Test | 0.525 | 5.614 | 4.044 | 0.607 | 9.997 | 7.160 | 0.676 |

## R0 sanity

Historical versus reproduced test R²:

| Target | Historical | Reproduced | Absolute difference | Gate |
|---|---:|---:|---:|---|
| V1 | 0.866943895 | 0.866943896 | 1.08×10⁻⁹ | PASS |
| V2 | 0.902874617 | 0.902874589 | 2.81×10⁻⁸ | PASS |

Best epoch was 91 and the run stopped at epoch 191, exactly matching the historical record. The historical train-only scaler also matched exactly. `R0_SANITY_CHECK_FAILED` does not apply.

## Adjacent ladder changes

Test R² changes were:

- R0→R1: V1 −0.0008, V2 +0.0275. `TRAINING_PROTOCOL_REGRESSION` is not supported.
- R1→R2: V1 +0.0230, V2 +0.0112. `CONDITION_COMPLETION_IMPLEMENTATION_REGRESSION` is not supported.
- R2→R3: V1 −0.3642, V2 −0.3347. `CLEAN_ARCHITECTURE_REGRESSION_CONFIRMED` applies to the R3 architecture/forward-path package.

R3 train R² is already only 0.553/0.583 versus R2's 0.935/0.968. The dominant symptom is training-set underfit/representation-optimization limitation, not a high-train/low-test overfit or merely stricter generalization split.

## R3 output and representation diagnostics

| Split | Pooled molecule 128D L2 | Projected 64D pre-LN L2 | Molecular post-LN L2 | Condition post-LN L2 |
|---|---:|---:|---:|---:|
| Train | 73.734 | 72.081 | 10.956 | 11.894 |
| Validation | 70.822 | 69.045 | 10.830 | 11.864 |
| Test | 75.363 | 73.028 | 11.045 | 11.897 |

On test, truth/predicted q50 standard deviations were 8.146/7.403 for V1 and 15.944/14.268 for V2. The prediction-to-truth ratios, 0.909 and 0.895, do not support a global low-variance collapse. Predicted q50 means were lower than truth (6.905 versus 8.839; 16.161 versus 18.607), maxima were compressed (29.548 versus 53.417; 58.426 versus 101.442), and q50≤1e−6 occurred in 8.87%/7.91% of rows. q50 clamp-to-zero was zero; all-quantile clamp rate was 7.59%.

Condition perturbations still strongly degrade R3 test performance: permutation changes V1/V2 RMSE by +4.595/+7.867 and disabling changes them by +3.354/+8.165. This confirms learned condition use, but does not establish that the Clean interaction architecture is adequate.

## Mechanism assessment

- `CONFIRMED`: the regression first appears in the R2→R3 architecture/forward-path package; R3 substantially underfits even its training rows.
- `SUPPORTED`: removal of Legacy's early molecule–condition message-passing interaction and replacement by a pre-softplus additive late-fusion head is the leading mechanism.
- `PLAUSIBLE`: the unvalidated 128→64 molecular bottleneck, LayerNorm after sum pooling, and softplus head interaction with Clean latents may contribute. They were not individually isolated.
- `NOT_SUPPORTED`: a different split, the Clean training protocol, V2 condition completion, global q50 variance collapse, or q50 clamp-to-zero as the primary cause.

## Baseline decision

`NO — point-performance qualification reopened.`

The prior six-run Clean study remains valid evidence for engineering integrity, condition reachability/use, numerical stability, and its measured results. Its designation `POINT_PREDICTOR_BASELINE_READY` is withdrawn because controlled same-split performance is far below reproducible Legacy/V2 behavior.

UQ qualification, 4g→8g transfer, active learning, and active transfer are paused.

## Subsequent decision

The former Clean MLP/FiLM/message-passing repair proposals are superseded. [R2-pruned requalification](../r2_pruned_requalification/R2_PRUNED_COMPARISON.md) removed only audited dead modules and exactly reproduced R2 checkpoint outputs, all point metrics and the complete retraining history. The current mainline is Legacy historical → Condition Completion V2 → R2-pruned candidate baseline.

Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`: engineering reachability PASS, performance FAIL. Its historical measurements above remain intact. The next separate step is `R2_PRUNED_QUANTILE_HEAD_QUALIFICATION`; no head comparison or Clean repair was executed in this round.
