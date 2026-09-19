# Phase 1: matched-total-budget batch/adaptivity results

The preregistered endpoint conditions support a feedback benefit in this cohort.
The combined endpoint-plus-curve conditions do not pass; inspect each component below.

These are iterative development results on five previously evaluated row splits, not independent confirmation. All methods use 333 initial + 320 acquired = 653 active labels and the same 416 validation labels. No weighted overall score is used.

## Prediction quality

| method | combined_normalized_RMSE | V1_RMSE | V2_RMSE | V1_R2 | V2_R2 |
|---|---|---|---|---|---|
| adaptive_lcmd | 0.436791 | 3.400738 | 5.426925 | 0.776048 | 0.857337 |
| random | 0.629362 | 4.740613 | 8.367543 | 0.567952 | 0.664049 |
| static_lcmd | 0.451013 | 3.512849 | 5.687079 | 0.763893 | 0.844417 |
| oneshot_b320 (same final model) | 0.451013 | 3.512849 | 5.687079 | 0.763893 | 0.844417 |

OneShot-B320 and Static-B320-prefix share exactly the same 653-label model. Static intermediate fits are scratch fits and never influence acquisition. Their equality is by construction, not a second independent experimental replicate.

### Per-seed endpoint NRMSE

| outer_seed | adaptive_lcmd | random | static_lcmd | adaptive_minus_static |
|---|---|---|---|---|
| 157 | 0.559336 | 0.741877 | 0.481144 | 0.078193 |
| 887 | 0.326176 | 0.487975 | 0.333903 | -0.007727 |
| 2357 | 0.348123 | 0.527542 | 0.440712 | -0.092589 |
| 6101 | 0.566694 | 0.793659 | 0.588748 | -0.022054 |
| 12203 | 0.383627 | 0.595755 | 0.410558 | -0.026930 |

### Paired Adaptive minus Static

For endpoint and AULC, negative differences favor Adaptive for error metrics and positive differences favor it for R2. For gap_closed, positive differences favor Adaptive for every metric. Intervals are descriptive 95% paired bootstrap intervals over five overlapping splits.

| metric | statistic | directional_wins | n_pairs | mean_difference | median_difference | sample_std_difference | bootstrap_low | bootstrap_high |
|---|---|---|---|---|---|---|---|---|
| V1_R2 | aulc | 2 | 5 | -0.006368 | -0.027155 | 0.050601 | -0.040557 | 0.033771 |
| V1_R2 | endpoint | 4 | 5 | 0.012156 | 0.034276 | 0.059132 | -0.037998 | 0.049636 |
| V1_R2 | gap_closed | 4 | 5 | 0.016727 | 0.080776 | 0.190584 | -0.153363 | 0.133615 |
| V1_RMSE | aulc | 2 | 5 | 0.026150 | 0.162061 | 0.347073 | -0.256902 | 0.263465 |
| V1_RMSE | endpoint | 4 | 5 | -0.112111 | -0.241284 | 0.441489 | -0.407972 | 0.268633 |
| V1_RMSE | gap_closed | 4 | 5 | 0.026033 | 0.094419 | 0.215876 | -0.167511 | 0.161599 |
| V2_R2 | aulc | 2 | 5 | -0.005778 | -0.010186 | 0.028036 | -0.027241 | 0.015463 |
| V2_R2 | endpoint | 3 | 5 | 0.012920 | 0.009497 | 0.048870 | -0.022291 | 0.054943 |
| V2_R2 | gap_closed | 3 | 5 | 0.033244 | 0.027840 | 0.147050 | -0.076434 | 0.160136 |
| V2_RMSE | aulc | 2 | 5 | 0.066639 | 0.125525 | 0.479151 | -0.296229 | 0.436861 |
| V2_RMSE | endpoint | 3 | 5 | -0.260153 | -0.161260 | 0.975262 | -1.109421 | 0.446315 |
| V2_RMSE | gap_closed | 3 | 5 | 0.041913 | 0.035415 | 0.181944 | -0.094364 | 0.198734 |
| combined_normalized_RMSE | aulc | 2 | 5 | 0.004854 | 0.005567 | 0.039248 | -0.028326 | 0.035460 |
| combined_normalized_RMSE | endpoint | 4 | 5 | -0.014222 | -0.022054 | 0.061129 | -0.061510 | 0.038094 |
| combined_normalized_RMSE | gap_closed | 4 | 5 | 0.035284 | 0.048976 | 0.185663 | -0.128504 | 0.177461 |

## Label efficiency

AULC is the trapezoidal integral divided by 653-333. This interval differs from Phase 0's 333..1005 interval, so those AULCs must not be ranked directly.

| method | combined_normalized_RMSE | V1_RMSE | V2_RMSE | V1_R2 | V2_R2 |
|---|---|---|---|---|---|
| adaptive_lcmd | 0.544359 | 4.120808 | 6.966689 | 0.668088 | 0.757842 |
| random | 0.669095 | 4.967291 | 9.073244 | 0.522937 | 0.602343 |
| static_lcmd | 0.539505 | 4.094658 | 6.900050 | 0.674455 | 0.763619 |

### Targets

The following targets are computed on cohort-mean curves. Per-seed targets remain in the CSV. First crossing is not persistence. Sustained means every remaining observed point meets the threshold; a final-point crossing alone has no future validation. Censored values are not imputed or extrapolated.

| method | metric | target | interpolated_labels | first_observed_labels | sustained_labels | status |
|---|---|---|---|---|---|---|
| adaptive_lcmd | combined_normalized_RMSE | Random@653 | 405.850698 | 429.000000 | 429.000000 | REACHED |
| adaptive_lcmd | combined_normalized_RMSE | N80 | 651.188602 | 653.000000 | 653.000000 | REACHED |
| adaptive_lcmd | combined_normalized_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | combined_normalized_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | combined_normalized_RMSE | Random@653 | 618.299040 | 621.000000 | 621.000000 | REACHED |
| random | combined_normalized_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | combined_normalized_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | combined_normalized_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | combined_normalized_RMSE | Random@653 | 400.119471 | 429.000000 | 429.000000 | REACHED |
| static_lcmd | combined_normalized_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | combined_normalized_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | combined_normalized_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V1_RMSE | Random@653 | 402.909149 | 429.000000 | 429.000000 | REACHED |
| adaptive_lcmd | V1_RMSE | N80 | 639.277113 | 653.000000 | 653.000000 | REACHED |
| adaptive_lcmd | V1_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V1_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V1_RMSE | Random@653 | 653.000000 | 653.000000 | 653.000000 | REACHED |
| random | V1_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V1_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V1_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V1_RMSE | Random@653 | 400.298402 | 429.000000 | 429.000000 | REACHED |
| static_lcmd | V1_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V1_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V1_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V2_RMSE | Random@653 | 397.649720 | 429.000000 | 429.000000 | REACHED |
| adaptive_lcmd | V2_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V2_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V2_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V2_RMSE | Random@653 | 617.232307 | 621.000000 | 621.000000 | REACHED |
| random | V2_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V2_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V2_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V2_RMSE | Random@653 | 394.103850 | 397.000000 | 397.000000 | REACHED |
| static_lcmd | V2_RMSE | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V2_RMSE | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V2_RMSE | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V1_R2 | Random@653 | 402.069146 | 429.000000 | 429.000000 | REACHED |
| adaptive_lcmd | V1_R2 | N80 | 588.589748 | 589.000000 | 589.000000 | REACHED |
| adaptive_lcmd | V1_R2 | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V1_R2 | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V1_R2 | Random@653 | 653.000000 | 653.000000 | 653.000000 | REACHED |
| random | V1_R2 | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V1_R2 | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V1_R2 | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V1_R2 | Random@653 | 399.621524 | 429.000000 | 429.000000 | REACHED |
| static_lcmd | V1_R2 | N80 | 591.833708 | 621.000000 | 621.000000 | REACHED |
| static_lcmd | V1_R2 | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V1_R2 | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V2_R2 | Random@653 | 398.514075 | 429.000000 | 429.000000 | REACHED |
| adaptive_lcmd | V2_R2 | N80 | 580.848476 | 589.000000 | 589.000000 | REACHED |
| adaptive_lcmd | V2_R2 | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| adaptive_lcmd | V2_R2 | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V2_R2 | Random@653 | 617.935652 | 621.000000 | 621.000000 | REACHED |
| random | V2_R2 | N80 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V2_R2 | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| random | V2_R2 | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V2_R2 | Random@653 | 395.615617 | 397.000000 | 397.000000 | REACHED |
| static_lcmd | V2_R2 | N80 | 588.022196 | 589.000000 | 653.000000 | REACHED |
| static_lcmd | V2_R2 | N90 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |
| static_lcmd | V2_R2 | N95 | censored / unavailable | censored / unavailable | censored / unavailable | CENSORED |

### Saving at Random@653

| method | metric | random_labels | method_labels | saved_labels | incremental_saving | status |
|---|---|---|---|---|---|---|
| adaptive_lcmd | V1_R2 | 653.000000 | 429.000000 | 224.000000 | 0.700000 | BOTH_REACHED |
| random | V1_R2 | 653.000000 | 653.000000 | 0.000000 | 0.000000 | BOTH_REACHED |
| static_lcmd | V1_R2 | 653.000000 | 429.000000 | 224.000000 | 0.700000 | BOTH_REACHED |
| adaptive_lcmd | V1_RMSE | 653.000000 | 429.000000 | 224.000000 | 0.700000 | BOTH_REACHED |
| random | V1_RMSE | 653.000000 | 653.000000 | 0.000000 | 0.000000 | BOTH_REACHED |
| static_lcmd | V1_RMSE | 653.000000 | 429.000000 | 224.000000 | 0.700000 | BOTH_REACHED |
| adaptive_lcmd | V2_R2 | 621.000000 | 429.000000 | 192.000000 | 0.666667 | BOTH_REACHED |
| random | V2_R2 | 621.000000 | 621.000000 | 0.000000 | 0.000000 | BOTH_REACHED |
| static_lcmd | V2_R2 | 621.000000 | 397.000000 | 224.000000 | 0.777778 | BOTH_REACHED |
| adaptive_lcmd | V2_RMSE | 621.000000 | 429.000000 | 192.000000 | 0.666667 | BOTH_REACHED |
| random | V2_RMSE | 621.000000 | 621.000000 | 0.000000 | 0.000000 | BOTH_REACHED |
| static_lcmd | V2_RMSE | 621.000000 | 397.000000 | 224.000000 | 0.777778 | BOTH_REACHED |
| adaptive_lcmd | combined_normalized_RMSE | 621.000000 | 429.000000 | 192.000000 | 0.666667 | BOTH_REACHED |
| random | combined_normalized_RMSE | 621.000000 | 621.000000 | 0.000000 | 0.000000 | BOTH_REACHED |
| static_lcmd | combined_normalized_RMSE | 621.000000 | 429.000000 | 192.000000 | 0.666667 | BOTH_REACHED |

## Robustness and cost

| method | fit_count | epochs | training_seconds | gradient_seconds | new_fit_count | new_training_seconds |
|---|---|---|---|---|---|---|
| adaptive_lcmd | 55 | 12578 | 22376.723224 | 5896.184522 | 0 | 0.000000 |
| oneshot_b320 | 10 | 2021 | 2733.332416 | 616.670690 | 0 | 0.000000 |
| random | 55 | 10818 | 25558.666566 | 0.000000 | 0 | 0.000000 |
| static_lcmd | 55 | 11387 | 13884.153842 | 616.670690 | 45 | 11125.803793 |

Training seconds include validation and final prediction. These are summed task times, not calendar time. Logical OneShot effort includes L0 plus the final model; its final model is already charged to Static in newly incurred work, so logical arm costs must not be added together. Historical machine loads differ. Full-data references are shared evaluation overhead, excluded from method operating costs. Interrupted-attempt compute was not fully measured; see EXECUTION_NOTES.md. The successful invocation's elapsed time is in execution_time.json, not a full wall-clock total.

### Frozen gate components

```json
{
  "evidence": "iterative_development",
  "endpoint_conditions": {
    "mean_NRMSE_improves": true,
    "paired_NRMSE_wins_at_least_4": true,
    "V1_RMSE_guard": true,
    "V2_RMSE_guard": true,
    "V1_R2_nonlower": true,
    "V2_R2_nonlower": true
  },
  "curve_conditions": {
    "NRMSE_AULC_improves": false,
    "V1_R2_AULC_nonlower": false,
    "V2_R2_AULC_nonlower": false
  },
  "feedback_signal": true,
  "curve_efficiency_signal": false,
  "paired_NRMSE_wins": 4,
  "oneshot_static_endpoint_identical_by_construction": true,
  "new_fits": 45
}
```

## Selection mechanism

| outer_seed | active_label_count | selected_intersection | selected_overlap_fraction |
|---|---|---|---|
| 157 | 653 | 133 | 0.415625 |
| 887 | 653 | 147 | 0.459375 |
| 2357 | 653 | 125 | 0.390625 |
| 6101 | 653 | 121 | 0.378125 |
| 12203 | 653 | 114 | 0.356250 |

The first 32 selected IDs match exactly for all five seeds. Later final membership differs. Because every evaluated model starts from the same initialization and uses the same fitting rule, the contrast measures the effect of the resulting acquisition trajectory and ordered training set. It is not an effect of warm-start or additional optimization applied to the final model. Training order is part of the fixed protocol; this experiment does not separately randomize set membership and order.

## Interpretation and next action

This control isolates fixed initial LCMD ranking versus B32 feedback at a matched final budget. It does not establish a monotone relationship across batch sizes; B64x5 and B160x2 were not run. It does not distinguish performance on new compounds or independent future experiments.

The next acquisition implementation remains the separately specified matched B32 Gradient-MaxDet screen. Its advancement must depend on actual paired prediction and endpoint results. Do not automatically promote noise weighting, IVR variants, manual strategy switching or dynamic batches. See the Phase 2 implementation plan for prerequisites and stopping gates.

Artifacts: `results/complete_results_with_oneshot.csv`, all per-seed tables, `figures/learning_curves.png`, and `figures/paired_endpoints.png`.
