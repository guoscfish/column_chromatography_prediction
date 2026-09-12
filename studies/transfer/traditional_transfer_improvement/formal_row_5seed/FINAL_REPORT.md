# Formal traditional parameter-transfer: 5-seed ROW report

## Integrity

All 40 Phase-1 fits completed before the global prediction freeze. The freeze manifest contains 10 contexts and 40 method-context fits. Test truth was then read once for the unified score table; all 40 metric rows are finite and each column/method has five frozen seeds.

## Test results (mean ± SD)

| column | method | V1_r2_mean | V1_r2_std | V1_rmse_mean | V1_rmse_std | V1_mae_mean | V1_mae_std | V2_r2_mean | V2_r2_std | V2_rmse_mean | V2_rmse_std | V2_mae_mean | V2_mae_std | combined_normalized_rmse_mean | combined_normalized_rmse_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P0 | 0.377 | 0.333 | 7.278 | 1.290 | 4.397 | 0.345 | 0.546 | 0.168 | 10.895 | 2.167 | 6.891 | 0.493 | 0.666 | 0.118 |
| 25g | P1 | 0.358 | 0.354 | 7.359 | 1.426 | 4.466 | 0.367 | 0.560 | 0.174 | 10.708 | 2.223 | 6.916 | 0.589 | 0.666 | 0.128 |
| 25g | P2 | 0.418 | 0.251 | 7.137 | 1.039 | 4.457 | 0.459 | 0.593 | 0.118 | 10.355 | 1.642 | 6.918 | 0.534 | 0.646 | 0.092 |
| 25g | P3 | 0.380 | 0.293 | 7.330 | 1.120 | 4.607 | 0.567 | 0.516 | 0.230 | 11.158 | 2.820 | 7.124 | 0.652 | 0.677 | 0.125 |
| 40g | P0 | 0.613 | 0.081 | 15.533 | 1.801 | 10.832 | 0.605 | 0.730 | 0.039 | 18.356 | 2.353 | 13.254 | 1.113 | 0.537 | 0.070 |
| 40g | P1 | 0.686 | 0.090 | 13.944 | 2.213 | 8.948 | 0.501 | 0.779 | 0.042 | 16.606 | 2.559 | 11.326 | 1.307 | 0.484 | 0.080 |
| 40g | P2 | 0.610 | 0.061 | 15.617 | 1.558 | 10.159 | 0.750 | 0.720 | 0.029 | 18.713 | 2.247 | 13.305 | 1.571 | 0.544 | 0.065 |
| 40g | P3 | 0.676 | 0.110 | 14.064 | 2.158 | 8.878 | 1.047 | 0.764 | 0.057 | 16.940 | 1.088 | 11.448 | 0.928 | 0.490 | 0.053 |

## Paired combined-NRMSE comparisons (negative delta favors candidate)

| column | candidate | reference | mean_paired_delta | median_paired_delta | std | seed_wins | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P1 | P0 | -0.001 | -0.004 | 0.012 | 3 | 5 |
| 25g | P2 | P1 | -0.020 | -0.003 | 0.051 | 3 | 5 |
| 25g | P2 | P0 | -0.021 | -0.018 | 0.042 | 3 | 5 |
| 25g | P3 | P2 | 0.032 | 0.014 | 0.044 | 1 | 5 |
| 25g | P3 | P0 | 0.011 | 0.015 | 0.012 | 1 | 5 |
| 40g | P1 | P0 | -0.053 | -0.056 | 0.019 | 5 | 5 |
| 40g | P2 | P1 | 0.060 | 0.067 | 0.033 | 0 | 5 |
| 40g | P2 | P0 | 0.006 | 0.013 | 0.020 | 2 | 5 |
| 40g | P3 | P2 | -0.053 | -0.050 | 0.050 | 4 | 5 |
| 40g | P3 | P0 | -0.047 | -0.025 | 0.046 | 5 | 5 |

## Convergence

`TRAINING_BUDGET_POSSIBLY_INSUFFICIENT`: 41/50 Adam stages selected their best checkpoint at epoch 150. No budget was changed after test evaluation.

| column | method | stage | fits | hit_max_epoch | early_stopped | best_epoch_mean | epochs_run_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P0 | B | 5 | 3 | 1 | 135.800 | 147.200 |
| 25g | P1 | A | 5 | 5 | 0 | 150.000 | 150.000 |
| 25g | P1 | B | 5 | 1 | 3 | 91.000 | 118.200 |
| 25g | P2 | A | 5 | 5 | 0 | 150.000 | 150.000 |
| 25g | P2 | B | 5 | 3 | 1 | 136.000 | 145.200 |
| 40g | P0 | B | 5 | 5 | 0 | 150.000 | 150.000 |
| 40g | P1 | A | 5 | 5 | 0 | 150.000 | 150.000 |
| 40g | P1 | B | 5 | 5 | 0 | 150.000 | 150.000 |
| 40g | P2 | A | 5 | 5 | 0 | 150.000 | 150.000 |
| 40g | P2 | B | 5 | 4 | 1 | 139.800 | 147.800 |

## Staged adaptation validation behavior

| column | method | contexts | stage_b_improves | mean_stage_b_minus_a |
| --- | --- | --- | --- | --- |
| 25g | P1 | 5 | 5 | -1.071 |
| 25g | P2 | 5 | 5 | -1.035 |
| 25g | P3 | 5 | 4 | -0.087 |
| 40g | P1 | 5 | 5 | -1.633 |
| 40g | P2 | 5 | 5 | -1.580 |
| 40g | P3 | 5 | 4 | -0.050 |

## Reference comparability

The reference split manifests have exactly matching ROW test sample-ID sets for all 10 contexts; the following frozen methods are directly comparable on the same test populations.

| column | method | V1_rmse_mean | V1_mae_mean | V1_r2_mean | V2_rmse_mean | V2_mae_mean | V2_r2_mean | combined_normalized_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | BALANCED_JOINT_FULL128 | 7.694 | 4.536 | 0.284 | 10.997 | 7.045 | 0.534 | 0.843 |
| 25g | HIER_CW_SHARED_LAMBDA_CORRECTED | 8.009 | 4.521 | 0.191 | 11.031 | 7.113 | 0.530 | 0.868 |
| 25g | P0 | 7.278 | 4.397 | 0.377 | 10.895 | 6.891 | 0.546 | 0.666 |
| 25g | P1 | 7.359 | 4.466 | 0.358 | 10.708 | 6.916 | 0.560 | 0.666 |
| 25g | P2 | 7.137 | 4.457 | 0.418 | 10.355 | 6.918 | 0.593 | 0.646 |
| 25g | P3 | 7.330 | 4.607 | 0.380 | 11.158 | 7.124 | 0.516 | 0.677 |
| 25g | paper_style_current_v2 | 7.409 | 4.462 | 0.349 | 11.111 | 6.967 | 0.525 | 0.826 |
| 40g | BALANCED_JOINT_FULL128 | 13.425 | 8.165 | 0.703 | 16.559 | 11.444 | 0.780 | 1.410 |
| 40g | HIER_CW_SHARED_LAMBDA_CORRECTED | 14.348 | 8.938 | 0.664 | 17.435 | 12.556 | 0.757 | 1.500 |
| 40g | P0 | 15.533 | 10.832 | 0.613 | 18.356 | 13.254 | 0.730 | 0.537 |
| 40g | P1 | 13.944 | 8.948 | 0.686 | 16.606 | 11.326 | 0.779 | 0.484 |
| 40g | P2 | 15.617 | 10.159 | 0.610 | 18.713 | 13.305 | 0.720 | 0.544 |
| 40g | P3 | 14.064 | 8.878 | 0.676 | 16.940 | 11.448 | 0.764 | 0.490 |
| 40g | paper_style_current_v2 | 13.109 | 7.987 | 0.721 | 15.897 | 10.643 | 0.797 | 1.370 |

P3 is a complete L-BFGS recipe (including its frozen learning-rate/weight-decay recipe), not a standalone causal optimizer comparison. Quantile crossings are diagnostics, not calibrated uncertainty claims.
