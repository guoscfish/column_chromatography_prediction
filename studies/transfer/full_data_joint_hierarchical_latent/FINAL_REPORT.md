# Final report: full-data joint hierarchical latent transfer

## Decision

**HIER_CW_25G_40G_RETAINED**. Selected working baseline: **HIER_CW_25G_40G**. Controlled shallow adaptation was skipped because the preregistered train-only joint gate did not pass.

## COMPOUND 25g

| method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIRECT_LATENT_PLS | 7.871 | 5.681 | 0.351 | 9.309 | 7.277 | 0.738 | 0.817 |
| DIRECT_LATENT_RIDGE | 7.647 | 5.453 | 0.382 | 9.125 | 7.032 | 0.747 | 0.796 |
| HIER_CW_25G_40G | 6.556 | 4.463 | 0.574 | 8.770 | 6.667 | 0.771 | 0.704 |
| HIER_CW_V2 | 6.550 | 4.493 | 0.574 | 9.294 | 6.728 | 0.746 | 0.717 |
| JOINT_HIER_LATENT_FULL128 | 6.647 | 4.653 | 0.556 | 9.194 | 6.760 | 0.752 | 0.722 |
| JOINT_HIER_LATENT_PCA16 | 6.545 | 4.482 | 0.573 | 9.248 | 6.644 | 0.749 | 0.715 |
| M3_CENTER_WIDTH_FULL | 6.624 | 4.516 | 0.568 | 9.199 | 6.893 | 0.750 | 0.719 |
| paper_style_current_v2 | 8.041 | 5.836 | 0.366 | 10.959 | 8.338 | 0.646 | 0.868 |

## COMPOUND 40g

| method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIRECT_LATENT_PLS | 16.686 | 10.241 | 0.632 | 19.904 | 13.068 | 0.716 | 1.735 |
| DIRECT_LATENT_RIDGE | 16.810 | 10.286 | 0.627 | 19.474 | 12.696 | 0.728 | 1.735 |
| HIER_CW_25G_40G | 15.426 | 9.944 | 0.685 | 20.113 | 13.351 | 0.710 | 1.644 |
| HIER_CW_V2 | 15.576 | 9.986 | 0.679 | 20.041 | 13.153 | 0.712 | 1.654 |
| JOINT_HIER_LATENT_FULL128 | 15.257 | 9.718 | 0.692 | 19.595 | 12.770 | 0.725 | 1.619 |
| JOINT_HIER_LATENT_PCA16 | 15.493 | 9.908 | 0.683 | 19.923 | 13.086 | 0.715 | 1.645 |
| M3_CENTER_WIDTH_FULL | 15.994 | 10.400 | 0.662 | 21.411 | 14.749 | 0.671 | 1.718 |
| paper_style_current_v2 | 16.240 | 11.001 | 0.650 | 22.083 | 15.389 | 0.647 | 1.753 |

## ROW 25g

| method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIRECT_LATENT_PLS | 9.109 | 5.658 | -0.053 | 11.926 | 7.732 | 0.444 | 0.972 |
| DIRECT_LATENT_RIDGE | 8.815 | 5.366 | 0.007 | 11.525 | 7.398 | 0.480 | 0.940 |
| HIER_CW_25G_40G | 8.132 | 4.518 | 0.151 | 11.231 | 7.213 | 0.510 | 0.882 |
| HIER_CW_V2 | 8.001 | 4.538 | 0.194 | 11.076 | 7.125 | 0.533 | 0.869 |
| JOINT_HIER_LATENT_FULL128 | 8.126 | 4.766 | 0.165 | 11.221 | 7.277 | 0.520 | 0.882 |
| JOINT_HIER_LATENT_PCA16 | 8.004 | 4.543 | 0.192 | 11.064 | 7.065 | 0.534 | 0.869 |
| M3_CENTER_WIDTH_FULL | 8.081 | 4.621 | 0.175 | 11.646 | 7.571 | 0.476 | 0.889 |
| paper_style_current_v2 | 7.409 | 4.462 | 0.349 | 11.111 | 6.967 | 0.525 | 0.826 |

## ROW 40g

| method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIRECT_LATENT_PLS | 14.175 | 8.349 | 0.674 | 16.472 | 11.167 | 0.782 | 1.465 |
| DIRECT_LATENT_RIDGE | 13.653 | 7.962 | 0.698 | 15.988 | 10.757 | 0.795 | 1.414 |
| HIER_CW_25G_40G | 14.270 | 8.895 | 0.668 | 17.441 | 12.568 | 0.756 | 1.494 |
| HIER_CW_V2 | 14.275 | 8.992 | 0.670 | 17.300 | 12.350 | 0.761 | 1.491 |
| JOINT_HIER_LATENT_FULL128 | 13.803 | 8.462 | 0.689 | 16.819 | 11.662 | 0.773 | 1.444 |
| JOINT_HIER_LATENT_PCA16 | 14.158 | 8.801 | 0.674 | 17.156 | 12.166 | 0.765 | 1.479 |
| M3_CENTER_WIDTH_FULL | 14.296 | 8.890 | 0.666 | 17.406 | 12.527 | 0.757 | 1.496 |
| paper_style_current_v2 | 13.109 | 7.987 | 0.721 | 15.897 | 10.643 | 0.797 | 1.370 |

## Ranking

| method | compound_25g_nrmse | compound_40g_nrmse | compound_mean_nrmse | row_25g_nrmse | row_40g_nrmse | row_mean_nrmse | compound_wins | row_wins | aggregate_mae | complexity | promotion_gate_passed | selected_working_baseline | final_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIER_CW_25G_40G | 0.704 | 1.644 | 1.174 | 0.882 | 1.494 | 1.188 | 0 | 3 | 8.453 | 18 coefficients | False | True | 1 |
| JOINT_HIER_LATENT_FULL128 | 0.722 | 1.619 | 1.170 | 0.882 | 1.444 | 1.163 | 6 | 3 | 8.259 | 274 coefficients | False | False | 2 |
| JOINT_HIER_LATENT_PCA16 | 0.715 | 1.645 | 1.180 | 0.869 | 1.479 | 1.174 | 5 | 2 | 8.337 | 50 coefficients | False | False | 3 |
| HIER_CW_V2 | 0.717 | 1.654 | 1.185 | 0.869 | 1.491 | 1.180 | 3 | 2 | 8.421 | 18 coefficients | False | False | 4 |
| M3_CENTER_WIDTH_FULL | 0.719 | 1.718 | 1.218 | 0.889 | 1.496 | 1.192 | 2 | 2 | 8.771 | 6 coefficients | False | False | 5 |
| DIRECT_LATENT_RIDGE | 0.796 | 1.735 | 1.265 | 0.940 | 1.414 | 1.177 | 1 | 2 | 8.369 | 262 coefficients | False | False | 6 |
| DIRECT_LATENT_PLS | 0.817 | 1.735 | 1.276 | 0.972 | 1.465 | 1.218 | 1 | 1 | 8.647 | selected PLS components | False | False | 7 |
| paper_style_current_v2 | 0.868 | 1.753 | 1.310 | 0.826 | 1.370 | 1.098 | 1 | 0 | 8.828 | neural partial fine-tune | False | False | 8 |

## Promotion gate

| method | compound_gain | compound_wins | row_change_vs_paper | max_compound_endpoint_deterioration | compound_mae_change | row_mae_change_vs_paper | promotion_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIER_CW_V2 | -0.010 | 3 | 0.075 | 0.060 | -0.002 | 0.098 | False |
| DIRECT_LATENT_RIDGE | -0.078 | 1 | 0.072 | 0.166 | 0.030 | 0.047 | False |
| DIRECT_LATENT_PLS | -0.087 | 1 | 0.110 | 0.201 | 0.053 | 0.095 | False |
| JOINT_HIER_LATENT_FULL128 | 0.003 | 6 | 0.059 | 0.048 | -0.015 | 0.070 | False |
| JOINT_HIER_LATENT_PCA16 | -0.005 | 5 | 0.069 | 0.054 | -0.009 | 0.084 | False |

## Interpretation

HIER_CW_V2 uses independent Center/Width deviation penalties but a single endpoint-space weighted objective. Direct latent models use only `[C_source, W_source, EA, h]`. Joint models fit HIER and latent blocks simultaneously; they are not post-hoc residual fits. FULL128 and PCA16 were the only latent variants. All preprocessing, PCA/PLS fitting, and selection were train-only.

Center/Width error variance and covariance are reported in `summary.csv`; train-quantile low/mid/high diagnostics are in `retention_regime_metrics.csv`. The decision requires both compound and row stability, endpoint and MAE guards, and therefore cannot be driven by one endpoint's covariance cancellation.

## Answers to the preregistered questions

1. HIER_CW_V2 does not beat HIER_CW_25G_40G: compound mean NRMSE worsens 0.96%. Separate shrinkage therefore has no promotion value in this benchmark; compound folds selected 0.1/0.1 throughout.

2. Direct latent Ridge and PLS worsen compound mean NRMSE by 7.80% and 8.71%. Ridge is better than PLS overall, but neither learns a competitive anchor-free mapping.

3. Joint FULL128 changes compound mean NRMSE by +0.30% (positive is improvement) but worsens row mean NRMSE by 5.94% versus matched paper-style. PCA16 is slightly weaker than FULL128. The joint model therefore exceeds neither HIER across both protocols nor paper-style on row.

4. The latent increment is column-dependent: for 40g compound, FULL128 improves Center/Width RMSE by 1.76%/5.48%; for 25g both worsen. This is not a stable cross-column latent gain.

5. No candidate simultaneously improves row and compound, no candidate passes the RMSE/MAE/endpoint gate, and the conditional shallow-adaptation gate is below 3% train-only gain. HIER_CW_25G_40G is retained.

6. The direction is not explained by covariance cancellation: the promoted state is unchanged, and the only useful joint slice (40g compound) reduces Center RMSE, Width RMSE, and error covariance together; its opposite 25g direction blocks promotion.

7. Matched paper-style remains strongest on row (25g R2 0.349/0.525; 40g 0.721/0.797). None approaches both paper-reported pairs 0.747/0.840 and 0.826/0.824 under these frozen FULL-data row splits. Paper reported no compound benchmark, so no compound-to-paper claim is made.

Historical test exposure makes this a developmental matched benchmark, not external validation. No column descriptors, feature sweep, ensemble, arbitrary neural network, or Active Learning run was added.
