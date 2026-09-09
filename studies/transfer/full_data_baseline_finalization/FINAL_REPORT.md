# Final filtered FULL-data transfer baseline

## Decision

**CENTER_WIDTH_PROMOTED_AS_WORKING_FULL_DATA_BASELINE**. Selected working baseline: **M3_CENTER_WIDTH_FULL**. No Active Learning or learning curve was started.

FULL means every eligible operationally filtered row outside the inherited frozen validation/test identities is in gradient-train. It never means training on test. Compound is primary; row is secondary.

## Compound-primary results

| column | protocol | method | V1_rmse_mean | V1_mae_mean | V1_r2_mean | V2_rmse_mean | V2_mae_mean | V2_r2_mean | combined_normalized_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | ALL_CONTEXT_SHARED_CENTER_WIDTH | 6.615 | 4.427 | 0.569 | 9.310 | 6.719 | 0.745 | 0.722 |
| 25g | compound | M3_CENTER_WIDTH_FULL | 6.624 | 4.516 | 0.568 | 9.199 | 6.893 | 0.750 | 0.719 |
| 25g | compound | MASS_FLOW_SHARED | 7.123 | 4.690 | 0.503 | 10.966 | 7.602 | 0.648 | 0.801 |
| 25g | compound | conditional_EA | 7.139 | 4.903 | 0.501 | 9.584 | 7.167 | 0.720 | 0.768 |
| 25g | compound | paper_style_current_v2 | 8.041 | 5.836 | 0.366 | 10.959 | 8.338 | 0.646 | 0.868 |
| 25g | compound | scale_only | 7.711 | 5.351 | 0.417 | 9.039 | 6.946 | 0.757 | 0.798 |
| 40g | compound | ALL_CONTEXT_SHARED_CENTER_WIDTH | 18.263 | 11.415 | 0.560 | 22.929 | 14.775 | 0.623 | 1.925 |
| 40g | compound | M3_CENTER_WIDTH_FULL | 15.994 | 10.400 | 0.662 | 21.411 | 14.749 | 0.671 | 1.718 |
| 40g | compound | MASS_FLOW_SHARED | 18.133 | 11.369 | 0.566 | 22.679 | 14.587 | 0.631 | 1.909 |
| 40g | compound | conditional_EA | 16.372 | 10.476 | 0.646 | 22.261 | 14.918 | 0.645 | 1.767 |
| 40g | compound | paper_style_current_v2 | 16.240 | 11.001 | 0.650 | 22.083 | 15.389 | 0.647 | 1.753 |
| 40g | compound | scale_only | 18.537 | 12.848 | 0.547 | 21.144 | 13.843 | 0.679 | 1.907 |


## Row-secondary results

| column | protocol | method | V1_rmse_mean | V1_mae_mean | V1_r2_mean | V2_rmse_mean | V2_mae_mean | V2_r2_mean | combined_normalized_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | row | ALL_CONTEXT_SHARED_CENTER_WIDTH | 8.005 | 4.534 | 0.192 | 10.930 | 7.063 | 0.540 | 0.865 |
| 25g | row | M3_CENTER_WIDTH_FULL | 8.081 | 4.621 | 0.175 | 11.646 | 7.571 | 0.476 | 0.889 |
| 25g | row | MASS_FLOW_SHARED | 7.957 | 4.537 | 0.208 | 10.855 | 6.985 | 0.549 | 0.860 |
| 25g | row | conditional_EA | 8.461 | 4.853 | 0.056 | 12.096 | 7.876 | 0.431 | 0.929 |
| 25g | row | paper_style_current_v2 | 7.409 | 4.462 | 0.349 | 11.111 | 6.967 | 0.525 | 0.826 |
| 25g | row | scale_only | 8.754 | 5.250 | 0.052 | 11.056 | 7.306 | 0.533 | 0.924 |
| 40g | row | ALL_CONTEXT_SHARED_CENTER_WIDTH | 14.813 | 9.897 | 0.647 | 17.538 | 12.784 | 0.754 | 1.538 |
| 40g | row | M3_CENTER_WIDTH_FULL | 14.296 | 8.890 | 0.666 | 17.406 | 12.527 | 0.757 | 1.496 |
| 40g | row | MASS_FLOW_SHARED | 14.821 | 9.918 | 0.647 | 17.554 | 12.855 | 0.753 | 1.539 |
| 40g | row | conditional_EA | 15.338 | 9.671 | 0.613 | 18.517 | 13.195 | 0.727 | 1.606 |
| 40g | row | paper_style_current_v2 | 13.109 | 7.987 | 0.721 | 15.897 | 10.643 | 0.797 | 1.370 |
| 40g | row | scale_only | 17.461 | 12.332 | 0.511 | 18.159 | 13.021 | 0.735 | 1.760 |


## Ranking

| method | 25g_compound_combined_nrmse | 40g_compound_combined_nrmse | mean_25g_40g_compound_nrmse | V1_mean_rmse | V2_mean_rmse | mean_mae | seed_stability | wins_vs_conditional_ea | primary_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3_CENTER_WIDTH_FULL | 0.719 | 1.718 | 1.218 | 11.309 | 15.305 | 9.140 | 9/10 combined-NRMSE wins vs conditional_EA | 9 | 1 |
| conditional_EA | 0.768 | 1.767 | 1.267 | 11.755 | 15.923 | 9.366 | reference | 0 | 2 |
| paper_style_current_v2 | 0.868 | 1.753 | 1.310 | 12.140 | 16.521 | 10.141 | 3/10 combined-NRMSE wins vs conditional_EA | 3 | 3 |
| ALL_CONTEXT_SHARED_CENTER_WIDTH | 0.722 | 1.925 | 1.323 | 12.439 | 16.119 | 9.334 | 5/10 combined-NRMSE wins vs conditional_EA | 5 | 4 |
| scale_only | 0.798 | 1.907 | 1.352 | 13.124 | 15.092 | 9.747 | 1/10 combined-NRMSE wins vs conditional_EA | 1 | 5 |
| MASS_FLOW_SHARED | 0.801 | 1.909 | 1.355 | 12.628 | 16.822 | 9.562 | 1/10 combined-NRMSE wins vs conditional_EA | 1 | 6 |


## Corrected M3 versus Conditional EA

On the compound-primary endpoint, M3 improves combined normalized RMSE by 6.32% on 25g (5/5 wins) and 2.76% on 40g (4/5 wins). Across both columns the mean gain is 3.84% with 9/10 paired wins. All four mean endpoint RMSEs improve; aggregate MAE improves 2.42%. R2 moves in the same favorable direction for all four endpoint/column means.

The row-secondary comparison is directionally consistent versus Conditional EA: combined normalized RMSE improves 4.36% on 25g and 6.87% on 40g. Paper-style remains the best row-only method, but row interpolation does not override the compound-primary selection.

## Shared-context increment

ALL_CONTEXT and MASS_FLOW_SHARED use the same 8g/25g/40g gradient-train supervision and the same shared model. Their only difference is adding `legacy_column_dia`, `legacy_column_len`, and `legacy_column_den`. Negative deltas favor ALL_CONTEXT.

| column | protocol | candidate | reference | n_pairs | V1_rmse_delta_mean | V1_rmse_relative_gain | V1_rmse_wins | V1_mae_delta_mean | V1_mae_relative_gain | V1_mae_wins | V2_rmse_delta_mean | V2_rmse_relative_gain | V2_rmse_wins | V2_mae_delta_mean | V2_mae_relative_gain | V2_mae_wins | combined_normalized_rmse_delta_mean | combined_normalized_rmse_relative_gain | combined_normalized_rmse_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | ALL_CONTEXT_SHARED_CENTER_WIDTH | MASS_FLOW_SHARED | 5 | -0.507 | 0.071 | 5 | -0.263 | 0.056 | 5 | -1.656 | 0.151 | 5 | -0.882 | 0.116 | 5 | -0.080 | 0.100 | 5 |
| 40g | compound | ALL_CONTEXT_SHARED_CENTER_WIDTH | MASS_FLOW_SHARED | 5 | 0.130 | -0.007 | 0 | 0.046 | -0.004 | 0 | 0.250 | -0.011 | 0 | 0.187 | -0.013 | 0 | 0.016 | -0.008 | 0 |


Against Conditional EA, ALL_CONTEXT improves 25g compound combined normalized RMSE by 6.00% but worsens 40g by 8.95%. Relative to MASS_FLOW_SHARED, adding the three legacy descriptors improves 25g by 9.96% (5/5) but worsens 40g by 0.82% (0/5). They therefore do not provide stable cross-column incremental predictive value.

## Interpretation

M3 is the exact corrected six-coefficient Center/Width formulation. Its actual-data equivalence to the historical corrected implementation is below `1e-10`. The shared model is `Ct=[a0+aEA*EA+aZ^T Z]Cs+bC` and `Wt=[d0+dEA*EA+dZ^T Z]Ws+bW`, with Z standardized on shared gradient-train only. MASS_FLOW uses two Z variables; ALL_CONTEXT uses exactly five.

Shared candidates receive more historical target supervision than single-column Conditional EA because they pool authorized gradient-train rows across 8g/25g/40g. No donor validation or test labels enter fitting. Removing 8g would still leave a mathematically definable 25g+40g fit, but only two column contexts and an even weaker basis for extrapolating column effects; this was not run as an ablation.

RMSE and combined normalized RMSE determine the ranking, MAE checks whether typical absolute error reverses direction, and R2 is secondary. A higher R2 with worse RMSE is not promotion evidence: R2 depends on the held-out target variance and can preserve ranking trends while absolute calibration remains worse.

The legacy descriptor coefficients are not independently identifiable from packing mass, flow, or column identity. Any improvement is predictive association only; no flow, diameter, length, or density causal effect is claimed.

All candidate predictions were globally frozen and hash-verified before test truth evaluation. This remains developmental evidence because these outer tests have historical repository exposure. The selected working baseline may now be frozen for subsequent AL baseline construction, but this study itself does not start AL.
