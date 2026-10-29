# 研究记录：performance_regression_audit

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# QGeoGNN point-prediction performance regression audit

Status: `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC_COMPLETE / PROTOCOL_FROZEN / ARTIFACT_AUDIT_PASS`.

This study locates the first controlled step at which the historical E0 Legacy point-prediction performance regresses. It reuses `experiments/e0_4g_baseline/split_seed_42.csv` byte-for-byte and does not generate a split. The ladder is R0 Legacy exact, R1 Legacy with the Clean training protocol, R2 Condition Completion V2 with that protocol, and R3 current Clean-QGeoGNN with that protocol.

R0 must reproduce both historical test R² values within absolute tolerance 0.03 before R1–R3 may run. All test outcomes are developmental diagnostics and are evaluated only after validation checkpoint selection. No UQ calibration, 8g access, transfer, active learning, hyperparameter sweep, architecture search, or model repair is authorized.

Runtime checkpoints and resume state are gitignored under `runtime/`. Compact formal diagnostic artifacts belong under `results/`.

The completed ladder found the first regression at `R2_CONDITION_COMPLETE_V2 -> R3_CLEAN_CURRENT`: R0/R1/R2 test R2 values remained approximately 0.866-0.942, while R3 reached 0.525 (V1) and 0.607 (V2). The leading supported mechanism is loss of Legacy early molecule-condition interaction together with additive late fusion; the 128D -> 64D projection, LayerNorm, and softplus head remain plausible but unisolated contributors. Clean point-performance qualification is reopened.

Subsequent decision: [R2-pruned](../r2_pruned_requalification/README.md) passed function-preserving pruning and exactly reproduced R2 retraining. Mainline: Legacy historical → Condition Completion V2 → R2-pruned candidate baseline. Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE` (engineering reachability PASS, performance FAIL). The earlier Clean repair proposal is superseded by a separate next quantile-head controlled test; it has not been executed. Historical ladder results remain unchanged.

<a id="记录-2"></a>

## 记录 2：`ARCHITECTURE_DIFF_AUDIT.md`

# Legacy/V2/Clean architecture-difference audit

Status: `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC`. This is a code and mathematics audit of the frozen variants; no architecture was changed.

## Shared controlled boundary

R0–R3 used the same 4,163-row historical E0 dataset, byte-identical frozen split (`9a758e…b198`), graph/conformer cache, train-only molecular/eluent scaling, seed 42, batch size 2,048, learning rate 0.001, maximum 1,000 epochs, patience 100, and post-selection test evaluation. R1–R3 used identical loss weights, deterministic epoch shuffling, zero weight decay, and validation combined normalized RMSE selection. R1 and R2 had function-identical initialization: R2 wrapped the same-seed random R1 Legacy backbone with a zero-output condition residual.

## Architecture summary

| Property | Legacy R0/R1 | Condition Completion R2 | Clean R3 |
|---|---|---|---|
| Nominal parameters | 775,476 | 777,808 | 413,732 |
| Graph representation | 128D sum-pooled | Same 128D Legacy path | 128D sum-pooled, then 64D projection |
| Conditions entering message passing | Eluent ExactMolWt, TPSA, nRotB, HBD | Same four Legacy inputs | None |
| Other conditions | Forward-unreachable in Legacy | Typed residual after pooling | Typed separate condition encoder |
| Fusion | Conditions affect edge/message representations from layer 1 | Early Legacy interaction plus additive post-pool residual | Concatenate 64D molecule and 64D condition |
| Prediction head | Linear(128,6) + ReLU | Same Legacy head | Linear(128,6), then componentwise softplus monotonic parameterization |

## A. Condition–molecule interaction

The Legacy data builder places bond length, six eluent descriptors, loading-solvent code, loading mass, and loading volume after the three categorical bond fields. `application/QGeoGNN.py` passes the continuous tail to `BondFloatRBF`, whose five configured inputs consume bond length followed by the first four eluent dimensions: ExactMolWt, TPSA, nRotB, and HBD. These edge representations enter every Legacy node-message layer. Thus the executed Legacy representation can learn molecule–condition interactions during graph message passing.

R2 retains this path and adds a zero-output-initialized typed residual after sum pooling for HBA, LogP, loading solvent, loading mass, and loading volume. Its high performance shows that completing the missing conditions is not itself the regression.

Clean removes all experimental conditions from molecular message passing. It computes `z_mol` and `z_cond` independently, concatenates them, then applies one linear head. For each raw head logit the operation is exactly:

`raw_j = W_m,j z_mol + W_c,j z_cond + b_j`.

Softplus subsequently transforms individual logits, but there are no learned bilinear, product, hidden-MLP, modulation, or message-passing cross terms before the output parameterization. `ADDITIVE_LATE_FUSION_WITHOUT_EXPLICIT_INTERACTION` is therefore confirmed as a structural fact. Because R1 and R2 retain early condition interaction and remain strong while R3 removes it and collapses, loss of early interaction is a `SUPPORTED` regression mechanism, but it is not individually isolated from the other simultaneous Clean changes.

## B. 128D → 64D molecular projection

Legacy sends the 128D sum-pooled representation directly to its prediction head. Clean inserts `Linear(128,64)` (8,256 parameters) and `LayerNorm(64)` (128 parameters), then uses only the 64D result. No performance experiment supported choosing 64 dimensions before this audit. It is an `UNVALIDATED_DESIGN_CHOICE` and a plausible information bottleneck.

The R3 test-set mean L2 norms were 75.363 before projection and 73.028 after the linear projection but before LayerNorm. Thus the linear map does not simply collapse vector magnitude. This audit does not measure retained information rank or sufficiency, so the bottleneck remains `PLAUSIBLE`, not confirmed.

## C. LayerNorm after sum pooling

R3 test mean L2 changes from 73.028 before LayerNorm to 11.045 after LayerNorm. Train/validation show the same pattern (72.081→10.956 and 69.045→10.830). LayerNorm intentionally removes per-sample location/scale information and gives the latent a constrained magnitude, so it can plausibly discard information encoded in sum-pooling magnitude. The observed norm change proves normalization is active, not that it causes the R² loss. Classification: `PLAUSIBLE_REGRESSION_MECHANISM_REQUIRING_CONTROLLED_TEST`.

## D. Loss

- R0: `V1_loss + 0.5 × V2_loss`.
- R1/R2/R3: `V1_loss + V2_loss`.

R1 and R2 remain at or above historical performance under the current loss. Therefore the loss-weight change is `NOT_SUPPORTED` as the primary regression mechanism. It may still alter target trade-offs inside another architecture.

## E. Shuffle

- R0: `shuffle=False`.
- R1/R2/R3: deterministic reshuffle each epoch.

R1 retains V1 performance and improves V2 under reshuffling. Shuffle is `NOT_SUPPORTED` as the regression source.

## F. Weight decay

- R0: `1e-5`.
- R1/R2/R3: `0`.

R1/R2 remain strong with zero weight decay. This change is `NOT_SUPPORTED` as the regression source.

## G. Checkpoint selection

- R0: minimum validation `V1_RMSE² + 0.5 × V2_RMSE²`.
- R1/R2/R3: minimum validation combined normalized RMSE using train-only target scales.

R1/R2 remain strong with the normalized selection metric. Checkpoint selection is `NOT_SUPPORTED` as the regression source.

## H. Monotonic softplus head and output compression

On R3 test, prediction/truth q50 standard-deviation ratios were 0.909 for V1 and 0.895 for V2; neither met the prespecified diagnostic flag of below 0.1. q50 clamp-to-zero was exactly zero for both targets. Therefore global low-variance compression and q50 clamping are `NOT_SUPPORTED` as primary explanations.

There is nevertheless a local tail symptom: 8.87% of V1 and 7.91% of V2 test q50 predictions were at or below `1e-6`, produced by strongly negative raw median logits before softplus. Across all six quantiles, 7.59% were clamped to zero, principally lower quantiles. Predicted maxima were also lower than truth maxima. The head–latent interaction remains `PLAUSIBLE`, but the present ladder does not isolate it.

## Evidence classification

- `CONFIRMED`: R2→R3 Clean architecture/forward-path package regression; R3 training-set underfit relative to R2; additive pre-softplus late-fusion form.
- `SUPPORTED`: removal of Legacy early molecule–condition interaction plus replacement by additive late fusion is the leading mechanism.
- `PLAUSIBLE`: unvalidated 128→64 bottleneck; LayerNorm removal of pooled magnitude; monotonic softplus head interaction with Clean latents.
- `NOT_SUPPORTED`: split difference, current training protocol, V2 condition completion, global q50 variance collapse, or q50 evaluation clamp as the primary cause.

These levels deliberately do not claim that any single Clean subcomponent has been causally isolated.

<a id="记录-3"></a>

## 记录 3：`REGRESSION_REPORT.md`

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

The former Clean MLP/FiLM/message-passing repair proposals are superseded. [R2-pruned requalification](../r2_pruned_requalification/README.md#记录-2) removed only audited dead modules and exactly reproduced R2 checkpoint outputs, all point metrics and the complete retraining history. The current mainline is Legacy historical → Condition Completion V2 → R2-pruned candidate baseline.

Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`: engineering reachability PASS, performance FAIL. Its historical measurements above remain intact. The next separate step is `R2_PRUNED_QUANTILE_HEAD_QUALIFICATION`; no head comparison or Clean repair was executed in this round.
