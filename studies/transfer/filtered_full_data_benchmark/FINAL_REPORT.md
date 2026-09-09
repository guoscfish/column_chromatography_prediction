# Filtered full-data transfer pilot: final report

## A. Audited facts

The fixed 4g source checkpoint is present and hash-locked to `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. Its qualified source has 4,163 effective rows under V1 ≤ 60 mL and V2 ≤ 120 mL. The original QGeoGNN constructors and the data manifest agree with 25g=60/120 and 40g=150/200; these are recorded here as legacy operational-domain thresholds, not physical laws.

## B. Filtered population

| column | reader_compatible_rows | filtered_rows | removed_rows | removed_percent | unique_compounds_before | unique_compounds_after | compounds_completely_removed | compounds_with_some_rows_removed | removed_V1_median_ml | removed_V2_median_ml |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | 490 | 408 | 82 | 16.735 | 78 | 78 | 0 | 49 | 85.219 | 137.894 |
| 40g | 529 | 456 | 73 | 13.800 | 80 | 80 | 0 | 47 | 202.875 | 304.300 |

Every full-data context retains the frozen B=100 outer validation/test identities after threshold intersection. Historical `pool` plus `gradient_train` rows become the full gradient-train set. This preserves test identity where rows remain and avoids re-drawing favorable seeds. The prespecified minimums were train ≥300, validation ≥5, test ≥60; all contexts pass. Compound validation has only one compound per context, so neural checkpoint selection is necessarily noisy.

## C. Primary compound-split results (mean across five seeds; SD/median/min/max are in summary.csv)

| column | method | n_train_mean | n_valid_mean | n_test_mean | V1_rmse_mean | V1_mae_mean | V1_r2_mean | V2_rmse_mean | V2_mae_mean | V2_r2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | conditional_EA | 315.800 | 6.200 | 86.000 | 7.139 | 4.903 | 0.501 | 9.584 | 7.167 | 0.720 |
| 25g | paper_style_current_v2 | 315.800 | 6.200 | 86.000 | 8.041 | 5.836 | 0.366 | 10.959 | 8.338 | 0.646 |
| 25g | scale_only | 315.800 | 6.200 | 86.000 | 7.711 | 5.351 | 0.417 | 9.039 | 6.946 | 0.757 |
| 25g | target_only_full | 315.800 | 6.200 | 86.000 | 8.820 | 6.168 | 0.209 | 15.013 | 11.078 | 0.332 |
| 25g | zero_shot | 315.800 | 6.200 | 86.000 | 22.645 | 20.647 | -4.030 | 34.898 | 31.660 | -2.598 |
| 40g | conditional_EA | 360.800 | 7.000 | 88.200 | 16.372 | 10.476 | 0.646 | 22.261 | 14.918 | 0.645 |
| 40g | paper_style_current_v2 | 360.800 | 7.000 | 88.200 | 16.240 | 11.001 | 0.650 | 22.083 | 15.389 | 0.647 |
| 40g | scale_only | 360.800 | 7.000 | 88.200 | 18.537 | 12.848 | 0.547 | 21.144 | 13.843 | 0.679 |
| 40g | target_only_full | 360.800 | 7.000 | 88.200 | 20.423 | 13.799 | 0.445 | 29.488 | 20.109 | 0.374 |
| 40g | zero_shot | 360.800 | 7.000 | 88.200 | 53.106 | 46.266 | -2.734 | 72.180 | 63.958 | -2.740 |

## D. Secondary row-split results

| column | method | n_train_mean | n_valid_mean | n_test_mean | V1_rmse_mean | V1_mae_mean | V1_r2_mean | V2_rmse_mean | V2_mae_mean | V2_r2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | conditional_EA | 321.600 | 6.400 | 80.000 | 8.461 | 4.853 | 0.056 | 12.096 | 7.876 | 0.431 |
| 25g | paper_style_current_v2 | 321.600 | 6.400 | 80.000 | 7.409 | 4.462 | 0.349 | 11.111 | 6.967 | 0.525 |
| 25g | scale_only | 321.600 | 6.400 | 80.000 | 8.754 | 5.250 | 0.052 | 11.056 | 7.306 | 0.533 |
| 25g | target_only_full | 321.600 | 6.400 | 80.000 | 7.629 | 4.588 | 0.353 | 11.670 | 7.485 | 0.488 |
| 25g | zero_shot | 321.600 | 6.400 | 80.000 | 22.169 | 20.377 | -4.349 | 33.663 | 30.934 | -3.251 |
| 40g | conditional_EA | 357.200 | 6.800 | 92.000 | 15.338 | 9.671 | 0.613 | 18.517 | 13.195 | 0.727 |
| 40g | paper_style_current_v2 | 357.200 | 6.800 | 92.000 | 13.109 | 7.987 | 0.721 | 15.897 | 10.643 | 0.797 |
| 40g | scale_only | 357.200 | 6.800 | 92.000 | 17.461 | 12.332 | 0.511 | 18.159 | 13.021 | 0.735 |
| 40g | target_only_full | 357.200 | 6.800 | 92.000 | 15.574 | 9.405 | 0.614 | 20.197 | 12.711 | 0.677 |
| 40g | zero_shot | 357.200 | 6.800 | 92.000 | 49.939 | 43.896 | -2.990 | 69.595 | 62.164 | -2.919 |

## E. Post-hoc no-threshold prediction sensitivity

The following recomputes frozen historical B=100 predictions on the threshold-intersection test rows only. Predictions were unchanged; it isolates evaluation-population tail exposure rather than filtered retraining.

| column | protocol | method | seed | n_no_threshold_test | n_threshold_intersection_test | V1_rmse_no_threshold | V1_rmse_threshold_intersection | V1_rmse_delta_filtered_minus_full | V1_mae_no_threshold | V1_mae_threshold_intersection | V1_mae_delta_filtered_minus_full | V1_r2_no_threshold | V1_r2_threshold_intersection | V1_r2_delta_filtered_minus_full | V2_rmse_no_threshold | V2_rmse_threshold_intersection | V2_rmse_delta_filtered_minus_full | V2_mae_no_threshold | V2_mae_threshold_intersection | V2_mae_delta_filtered_minus_full | V2_r2_no_threshold | V2_r2_threshold_intersection | V2_r2_delta_filtered_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | conditional_EA | 1372220817.200 | 102.400 | 86.000 | 13.007 | 7.659 | -5.348 | 8.172 | 5.561 | -2.611 | 0.864 | 0.405 | -0.458 | 19.759 | 11.979 | -7.780 | 12.821 | 9.337 | -3.483 | 0.878 | 0.562 | -0.316 |
| 25g | compound | paper_style_current_v2 | 1372220817.200 | 102.400 | 86.000 | 19.558 | 9.506 | -10.051 | 11.883 | 7.208 | -4.675 | 0.690 | 0.088 | -0.602 | 26.270 | 12.935 | -13.335 | 15.931 | 9.932 | -5.999 | 0.782 | 0.494 | -0.289 |
| 25g | compound | scale_only | 1372220817.200 | 102.400 | 86.000 | 14.759 | 9.774 | -4.985 | 10.331 | 7.879 | -2.452 | 0.823 | 0.036 | -0.786 | 21.341 | 13.832 | -7.508 | 15.008 | 11.586 | -3.422 | 0.856 | 0.405 | -0.451 |
| 25g | compound | zero_shot | 1372220817.200 | 102.400 | 86.000 | 44.228 | 22.645 | -21.583 | 31.855 | 20.647 | -11.208 | -0.605 | -4.030 | -3.425 | 66.971 | 34.898 | -32.073 | 48.412 | 31.660 | -16.752 | -0.423 | -2.598 | -2.174 |
| 25g | row | conditional_EA | 1372220817.200 | 98.000 | 80.000 | 17.782 | 9.753 | -8.029 | 9.132 | 5.507 | -3.625 | 0.768 | -0.323 | -1.091 | 25.298 | 15.955 | -9.343 | 15.037 | 10.820 | -4.217 | 0.815 | -0.003 | -0.818 |
| 25g | row | paper_style_current_v2 | 1372220817.200 | 98.000 | 80.000 | 17.372 | 11.016 | -6.356 | 10.147 | 6.859 | -3.288 | 0.775 | -0.695 | -1.470 | 25.652 | 15.796 | -9.856 | 14.885 | 9.903 | -4.982 | 0.809 | -0.017 | -0.826 |
| 25g | row | scale_only | 1372220817.200 | 98.000 | 80.000 | 18.882 | 10.637 | -8.244 | 10.687 | 7.192 | -3.495 | 0.738 | -0.443 | -1.181 | 26.822 | 15.175 | -11.647 | 15.815 | 11.127 | -4.689 | 0.792 | 0.101 | -0.690 |
| 25g | row | zero_shot | 1372220817.200 | 98.000 | 80.000 | 46.443 | 22.169 | -24.274 | 33.089 | 20.377 | -12.712 | -0.591 | -4.349 | -3.758 | 70.424 | 33.663 | -36.761 | 50.007 | 30.934 | -19.073 | -0.427 | -3.251 | -2.824 |
| 40g | compound | conditional_EA | 1372220817.200 | 101.800 | 88.200 | 31.845 | 18.208 | -13.637 | 18.640 | 13.082 | -5.558 | 0.760 | 0.561 | -0.198 | 43.455 | 32.311 | -11.144 | 28.433 | 23.982 | -4.451 | 0.768 | 0.249 | -0.519 |
| 40g | compound | paper_style_current_v2 | 1372220817.200 | 101.800 | 88.200 | 37.404 | 25.339 | -12.065 | 21.909 | 16.267 | -5.642 | 0.674 | 0.140 | -0.534 | 48.167 | 33.443 | -14.724 | 28.407 | 21.134 | -7.273 | 0.721 | 0.192 | -0.528 |
| 40g | compound | scale_only | 1372220817.200 | 101.800 | 88.200 | 34.739 | 23.676 | -11.063 | 24.334 | 19.670 | -4.664 | 0.715 | 0.259 | -0.456 | 42.709 | 29.652 | -13.057 | 28.515 | 23.624 | -4.890 | 0.775 | 0.370 | -0.405 |
| 40g | compound | zero_shot | 1372220817.200 | 101.800 | 88.200 | 91.155 | 53.106 | -38.049 | 66.894 | 46.266 | -20.628 | -0.927 | -2.734 | -1.807 | 123.844 | 72.180 | -51.664 | 91.749 | 63.958 | -27.791 | -0.832 | -2.740 | -1.907 |
| 40g | row | conditional_EA | 1372220817.200 | 106.000 | 92.000 | 29.986 | 17.949 | -12.037 | 17.311 | 12.173 | -5.138 | 0.817 | 0.466 | -0.351 | 36.191 | 23.408 | -12.782 | 20.712 | 16.080 | -4.631 | 0.857 | 0.556 | -0.302 |
| 40g | row | paper_style_current_v2 | 1372220817.200 | 106.000 | 92.000 | 29.787 | 19.633 | -10.155 | 17.579 | 12.607 | -4.972 | 0.820 | 0.357 | -0.463 | 36.884 | 24.847 | -12.036 | 22.177 | 16.217 | -5.960 | 0.855 | 0.475 | -0.380 |
| 40g | row | scale_only | 1372220817.200 | 106.000 | 92.000 | 35.428 | 23.133 | -12.295 | 24.639 | 19.181 | -5.459 | 0.744 | 0.125 | -0.619 | 42.661 | 28.177 | -14.483 | 28.528 | 22.712 | -5.816 | 0.803 | 0.333 | -0.470 |
| 40g | row | zero_shot | 1372220817.200 | 106.000 | 92.000 | 93.698 | 49.939 | -43.760 | 65.865 | 43.896 | -21.969 | -0.765 | -2.990 | -2.225 | 126.971 | 69.595 | -57.376 | 91.315 | 62.164 | -29.151 | -0.702 | -2.919 | -2.216 |

## F. Historical B=100 context (not a threshold-only comparison)

This comparison changes both operational domain and revealed training-label count, so it must not be attributed wholly to threshold filtering.

| column | protocol | method | n_revealed | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | conditional_EA | 99.000 | 13.007 | 8.172 | 0.864 | 19.759 | 12.821 | 0.878 |
| 25g | compound | paper_style_current_v2 | 99.000 | 19.558 | 11.883 | 0.690 | 26.270 | 15.931 | 0.782 |
| 25g | compound | scale_only | 99.000 | 14.759 | 10.331 | 0.823 | 21.341 | 15.008 | 0.856 |
| 25g | compound | zero_shot | 99.000 | 44.228 | 31.855 | -0.605 | 66.971 | 48.412 | -0.423 |
| 25g | row | conditional_EA | 100.000 | 17.782 | 9.132 | 0.768 | 25.298 | 15.037 | 0.815 |
| 25g | row | paper_style_current_v2 | 100.000 | 17.372 | 10.147 | 0.775 | 25.652 | 14.885 | 0.809 |
| 25g | row | scale_only | 100.000 | 18.882 | 10.687 | 0.738 | 26.822 | 15.815 | 0.792 |
| 25g | row | zero_shot | 100.000 | 46.443 | 33.089 | -0.591 | 70.424 | 50.007 | -0.427 |
| 40g | compound | conditional_EA | 98.200 | 31.845 | 18.640 | 0.760 | 43.455 | 28.433 | 0.768 |
| 40g | compound | paper_style_current_v2 | 98.200 | 37.404 | 21.909 | 0.674 | 48.167 | 28.407 | 0.721 |
| 40g | compound | scale_only | 98.200 | 34.739 | 24.334 | 0.715 | 42.709 | 28.515 | 0.775 |
| 40g | compound | zero_shot | 98.200 | 91.155 | 66.894 | -0.927 | 123.844 | 91.749 | -0.832 |
| 40g | row | conditional_EA | 100.000 | 29.986 | 17.311 | 0.817 | 36.191 | 20.712 | 0.857 |
| 40g | row | paper_style_current_v2 | 100.000 | 29.787 | 17.579 | 0.820 | 36.884 | 22.177 | 0.855 |
| 40g | row | scale_only | 100.000 | 35.428 | 24.639 | 0.744 | 42.661 | 28.528 | 0.803 |
| 40g | row | zero_shot | 100.000 | 93.698 | 65.865 | -0.765 | 126.971 | 91.315 | -0.702 |

## G. Decision

Primary compound-split endpoint winners span: conditional_EA, paper_style_current_v2, scale_only. No single universal winner is supported across both columns, outputs and MAE/RMSE.

`target_only_full` is a genuine target-data ceiling-like reference: it loads neither source weights nor source predictions/replay and fits all scalers on target gradient-train only. `paper_style_current_v2` keeps the historical matched shallow scope and uses the qualified source preprocessing; its scope, optimizer and selection rule were fixed before test evaluation. Simple calibration uses fixed source q50 predictions; conditional-EA selects only its predeclared regularization grid on validation.

The full-data result can indicate whether label scarcity remains plausible, but it is not itself evidence that active learning will help. We therefore do not enter active learning automatically. The next high-value experiment is a preregistered learning curve on this same filtered compound split (nested target-label budgets, target-only and the selected transfer baseline), followed by an acquisition-vs-random test only if the curve shows a material label-scarcity regime.

25g ≈5.91/8.21 mL and 40g ≈12.85/14.81 mL V1/V2 RMSE in the legacy paper reconstruction; it has an old source, distinct preprocessing/split and about 80% target rows, so it is NOT A HEAD-TO-HEAD COMPARISON.

## H. Limits

- Threshold filtering can alter compound-row and condition composition; consult FILTER_AUDIT.json rather than calling removed rows generic outliers.
- Target-compound holdout does not imply source-unseen chemistry.
- Small frozen validation sets, especially one validation compound, make neural checkpoint comparisons less precise.
- Existing reconstructed paper-transfer results remain descriptive context only.
