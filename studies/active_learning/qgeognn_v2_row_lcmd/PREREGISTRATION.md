# Preregistration — QGeoGNN-V2 4g row Gradient-LCMD-TP

Frozen before formal test evaluation on 2026-09-14. This file must not be changed in response to the formal results.

## Scientific question

On the current qualified standalone QGeoGNN-V2 and canonical 4g row-interpolation domain, does Gradient-LCMD-TP choose one additional 10%-of-outer-train label batch that reduces prediction error more than equal-budget Random sampling?

This is a one-step mechanism test, not a full learning curve. BAIT, Condition-LCMD, UCB, EI, Thompson sampling, the historical Hybrid strategy, uncertainty shortlists, and test-dependent selection are out of scope.

## Frozen design

- Predictor: `src/qgeognn_al/models/qgeognn_v2.py`, `MODEL_VARIANT=qgeognn_v2`; no architecture, head, condition-branch, output, quantile, loss, or hyperparameter change.
- Canonical dataset: repository-qualified 4g data, validated from the current file and its hash rather than a hard-coded paper row count.
- Outer seeds: `[73, 311, 1297, 4093, 8191]`; these may not be replaced.
- Row split integer rule for 4,163 rows: 3,330 outer training, 416 fixed validation, 417 fixed test. A seeded PCG64 permutation uses canonical row order; training is first, validation next, test last.
- L0: 333 rows selected from outer train with PCG64 seed `outer_seed + 104729`. U0: the remaining 2,997 outer-train rows.
- Query batch B: 333 rows from the full U0.
- Arms per outer seed: one Gradient-LCMD-TP batch and five independent equal-size Random batches. Random control `j` uses PCG64 seed `outer_seed * 1000003 + 50000 + j`.
- Fixed validation labels are shared auxiliary cost and are not included in the active training-label budget. Active labels are 333 at L0 and 666 after the batch.
- Every baseline and after-batch arm within an outer seed uses identical covariate preprocessing, L0-derived endpoint scales, direct initialization seed, training shuffle seed, validation IDs, and training protocol. Only selected training IDs differ.
- Seed derivation is frozen: initialization `outer_seed + 2000003`, epoch-shuffle/training `outer_seed + 3000017`, and CountSketch `outer_seed + 4000037`.
- Training: Adam, learning rate 0.001, weight decay 0, batch size 2048, maximum 1000 epochs, patience 100, deterministic epoch shuffle, validation combined normalized RMSE checkpoint selection, and no test during training.
- Every after-batch arm retrains from the same initialization. No arm warm-starts from L0.

## Frozen acquisition representation

For every L0 and U0 row, at the frozen L0 checkpoint:

`phi(x) = CountSketch_512(concat(grad_theta[V1_q50(x)/s_V1], grad_theta[V2_q50(x)/s_V2]))`

`s_V1` and `s_V2` are population-standard-deviation scales computed only from L0 labels and frozen across arms. `theta` is every `requires_grad` parameter in the standalone QGeoGNN-V2. The two-output concatenated gradient is sketched with a deterministic seed derived only from the outer seed. At most one row's full gradient is materialized; the forbidden `N_pool × 458,952` Jacobian is never formed.

This is an explicit multi-output adaptation for QGeoGNN-V2, not a row-for-row reproduction of the paper's scalar-regression implementation. LCMD uses squared Euclidean distances in the sketched gradient space and corrected TP semantics: all L0 rows are installed as selected centers before the first pool point is chosen.

The sketch dimension, parameter scope, output columns, batch size, seeds, distance, LCMD formula, training protocol, and decision gate are frozen and cannot be changed after formal test evaluation.

## Leakage boundary

Pool covariates and graph structures are observable. U0 truth cannot enter preprocessing, target scaling, fitting, acquisition, or checkpoint selection. Test truth is unavailable until all acquisition IDs, checkpoints, and predictions for all five seeds are frozen and hashed. Only then is test truth read once for final scoring. Validation truth is permitted only for the shared early-stopping rule.

## Outcomes and gate

Primary outcome: test `combined_normalized_RMSE`, using the fixed L0 endpoint scales for all comparisons in an outer seed. Secondary outcomes: V1/V2 RMSE, MAE, and R2.

For each arm, `gain = baseline_L0_NRMSE - after_batch_NRMSE`; larger is better.

`STRONG_POSITIVE` requires all of:

1. LCMD gain exceeds the Random median in at least 4/5 outer seeds.
2. Mean across seeds of `LCMD gain - Random mean gain` is positive.
3. Median across seeds of `LCMD gain - Random median gain` is positive.
4. Mean after-batch combined NRMSE improves by at least 5% relative to the Random-control mean.
5. Neither endpoint's mean LCMD RMSE is more than 2% worse than its Random-control mean.

`PROMISING_BUT_NOT_CONFIRMED` applies when LCMD has at least 4/5 paired directional wins or at least 3% mean NRMSE improvement, but misses any strong-positive condition. All other completed outcomes are `NO_STABLE_BENEFIT`.

Whatever the outcome, this experiment stops. A full 10%→20%→30%→40%→50% row learning curve is recommended only after `STRONG_POSITIVE`; no failed result automatically launches another method.
