# Metric Comparability Audit

## Problem

The formal P0-P3 score used an outer-split, gradient-train-only standard deviation for each target. Historical ROW reference scores used the fixed qualified 4g source-train scales. Test IDs were matched, but the normalization denominator was not.

The discrepancy is visible in the 40g example: paper-style has lower raw V1/V2 RMSE than P1, while its historical source-scale combined NRMSE is much larger. This is a denominator difference, not evidence that P1 dominates paper-style.

## Contract

For every column and outer seed, this audit computes `s_V1` and `s_V2` from that context's `gradient_train` labels with population standard deviation (`ddof=0`). All seven methods use the same scales and the same frozen outer test IDs. No test label is used to define scales or select a method.

Source-scale values retained for diagnosing the old reports: `V1=7.8796590346`, `V2=16.0765095536`.

## Findings

- Raw RMSE, MAE, and R2 were already directly comparable on matched test IDs.
- Historical combined NRMSE values in `reference_comparison.csv` are not directly comparable to formal P0-P3 combined NRMSE values and must not be used for cross-study ranking.
- The unified values are in `summary.csv`; per-seed scales and metrics are in `train_only_scales.csv` and `per_seed_metrics.csv`.
- This is a re-score of frozen prediction artifacts, not a new fit and not a new test-driven method selection.

## Scope limitation

Only methods with local frozen ROW prediction artifacts were included: P0-P3, paper-style, HIER_CW_SHARED_LAMBDA_CORRECTED, and BALANCED_JOINT_FULL128. Other historical methods without matched prediction artifacts are marked unavailable rather than reconstructed.

