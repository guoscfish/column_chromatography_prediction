# 4g predictor benchmark protocol draft

Status: `PROTOCOL_DRAFTED / FORMAL_RUN_NOT_AUTHORIZED`.

This document defines the next scientific gate. It does not authorize training, architecture selection, or test inspection.

## Models

The future benchmark must compare:

1. frozen Legacy QGeoGNN under its audited contract;
2. `qgeognn_condition_complete_v2` as the controlled completion ablation;
3. `qgeognn_clean_fusion_v1`;
4. a strong ANN/MLP baseline using an explicit, comparable molecular/condition contract.

Additional simple baselines require preregistration. The benchmark must not silently tune Clean-QGeoGNN against test outcomes.

## Estimands

**Row interpolation:** the same compound may have other conditions in training. This estimates prediction under different experimental conditions for seen or similar molecules.

**Compound generalization:** split or cross-validate by `canonical_smiles` using Group split/GroupKFold. This estimates prediction for compounds absent from training.

Row-split performance must never be described as generalization to new compounds. Scaffold OOD may be added by a later preregistration but is not part of this draft.

## Data-use boundary

The historical 4g test has been repeatedly viewed. It may support legacy comparability but is not a pristine untouched confirmatory test. A formal benchmark should use a frozen resampling protocol, GroupKFold, nested validation, or a new preregistered holdout policy. The final choice must be frozen before execution.

Historical 4g applies `V1 <= 60` and `V2 <= 120`, whereas authoritative 8g is no-threshold. Before the benchmark, a source-only audit must determine whether the thresholds express physical/instrument censoring, a legitimate domain restriction, outlier cleanup, or only a legacy implementation choice. This draft neither removes nor endorses them as scientific truth.

## Frozen comparisons and diagnostics

Within-target monotonic quantiles remain required. No hard `V1 <= V2` constraint is allowed. Configured V1/V2 loss weights remain 1:1; per-target gradient contribution must be audited because raw-mL scales may make effective contributions unequal. Alternative weights require a separate preregistered ablation.

Report modality latent norms, projection/encoder gradient norms, V1/V2 gradient contributions, condition permutation, molecule-only, condition-only/disabled, and cross-target-order diagnostics. These analyses describe behavior; they do not create a new model-selection loop on held-out test data.

## Gate

Formal execution requires a separate preregistration that freezes splits, preprocessing, training budget, seeds, checkpoint selection, metrics, uncertainty summaries, and decision rules. Until then, only unit, fixture, synthetic, and tiny no-performance engineering smoke tests are permitted.
