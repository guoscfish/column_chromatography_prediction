# Matched cross-column absolute-error benchmark

Decision is based on RMSE/MAE in mL; R2 is secondary.  Every method uses the same qualified current-V2 source checkpoint, canonical no-threshold targets, frozen split schedule, target-label budget and test population.

## Protocol

Source: `studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt` (SHA-256 `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`). Columns: 8g, 25g, 40g. Protocols: row and target-compound holdout. Outer seeds: `769539383, 1425370602, 536279090, 2767143051, 1362771960`. Budgets: `30, 50, 70, 100`. Target threshold: `NONE`.

Simple calibration fits use gradient-train labels only. Neural checkpoint selection uses frozen validation labels only. The paper-style runner builds zero-labelled graphs, reveals only train/validation labels, freezes and hashes all 120 prediction files, and only then reads test truth. The schedule is inherited byte-for-byte from `cross_column`; no split was rerandomized.

## B=100 primary table

The CSV contains mean, sample SD, median, minimum and maximum across the five seeds. The table below shows mean values and keeps row/compound protocols explicit.

| column | protocol | method | B | V1 RMSE | V2 RMSE | V1 MAE | V2 MAE | V1 R2 | V2 R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25g | compound | affine | 100 | 14.14 | 20.13 | 8.84 | 12.61 | 0.837 | 0.873 |
| 25g | compound | conditional_EA | 100 | 13.01 | 19.76 | 8.17 | 12.82 | 0.864 | 0.878 |
| 25g | compound | local_identity_shrinkage | 100 | 13.95 | 19.76 | 8.91 | 12.49 | 0.842 | 0.878 |
| 25g | compound | paper_style_current_v2 | 100 | 19.56 | 26.27 | 11.88 | 15.93 | 0.690 | 0.782 |
| 25g | compound | scale_only | 100 | 14.76 | 21.34 | 10.33 | 15.01 | 0.823 | 0.856 |
| 25g | compound | standard_shallow_finetune | 100 | 19.55 | 26.28 | 11.94 | 16.03 | 0.690 | 0.782 |
| 25g | compound | target_head_only | 100 | 25.35 | 46.04 | 14.50 | 28.92 | 0.484 | 0.336 |
| 25g | compound | zero_shot | 100 | 44.23 | 66.97 | 31.86 | 48.41 | -0.605 | -0.423 |
| 25g | row | affine | 100 | 18.61 | 25.22 | 9.71 | 14.36 | 0.747 | 0.816 |
| 25g | row | conditional_EA | 100 | 17.78 | 25.30 | 9.13 | 15.04 | 0.768 | 0.815 |
| 25g | row | local_identity_shrinkage | 100 | 18.61 | 25.53 | 9.92 | 15.15 | 0.746 | 0.812 |
| 25g | row | paper_style_current_v2 | 100 | 17.37 | 25.65 | 10.15 | 14.89 | 0.775 | 0.809 |
| 25g | row | scale_only | 100 | 18.88 | 26.82 | 10.69 | 15.82 | 0.738 | 0.792 |
| 25g | row | standard_shallow_finetune | 100 | 17.38 | 25.66 | 10.16 | 14.90 | 0.775 | 0.809 |
| 25g | row | target_head_only | 100 | 28.28 | 50.36 | 16.04 | 30.68 | 0.411 | 0.270 |
| 25g | row | zero_shot | 100 | 46.44 | 70.42 | 33.09 | 50.01 | -0.591 | -0.427 |
| 40g | compound | affine | 100 | 33.57 | 40.97 | 20.08 | 23.08 | 0.732 | 0.792 |
| 40g | compound | conditional_EA | 100 | 31.84 | 43.46 | 18.64 | 28.43 | 0.760 | 0.768 |
| 40g | compound | local_identity_shrinkage | 100 | 33.57 | 40.97 | 20.08 | 23.08 | 0.732 | 0.792 |
| 40g | compound | paper_style_current_v2 | 100 | 37.40 | 48.17 | 21.91 | 28.41 | 0.674 | 0.721 |
| 40g | compound | scale_only | 100 | 34.74 | 42.71 | 24.33 | 28.51 | 0.715 | 0.775 |
| 40g | compound | standard_shallow_finetune | 100 | 37.46 | 48.27 | 21.99 | 28.53 | 0.673 | 0.720 |
| 40g | compound | target_head_only | 100 | 70.99 | 102.75 | 47.33 | 71.60 | -0.169 | -0.261 |
| 40g | compound | zero_shot | 100 | 91.15 | 123.84 | 66.89 | 91.75 | -0.927 | -0.832 |
| 40g | row | affine | 100 | 32.01 | 37.02 | 18.86 | 20.68 | 0.791 | 0.851 |
| 40g | row | conditional_EA | 100 | 29.99 | 36.19 | 17.31 | 20.71 | 0.817 | 0.857 |
| 40g | row | local_identity_shrinkage | 100 | 32.91 | 37.06 | 20.00 | 20.69 | 0.779 | 0.850 |
| 40g | row | paper_style_current_v2 | 100 | 29.79 | 36.88 | 17.58 | 22.18 | 0.820 | 0.855 |
| 40g | row | scale_only | 100 | 35.43 | 42.66 | 24.64 | 28.53 | 0.744 | 0.803 |
| 40g | row | standard_shallow_finetune | 100 | 29.84 | 36.95 | 17.67 | 22.28 | 0.820 | 0.855 |
| 40g | row | target_head_only | 100 | 73.48 | 105.72 | 46.41 | 71.16 | -0.083 | -0.179 |
| 40g | row | zero_shot | 100 | 93.70 | 126.97 | 65.86 | 91.32 | -0.765 | -0.702 |
| 8g | compound | affine | 100 | 7.39 | 9.18 | 3.88 | 4.73 | 0.804 | 0.899 |
| 8g | compound | conditional_EA | 100 | 6.10 | 8.64 | 3.17 | 4.75 | 0.872 | 0.909 |
| 8g | compound | local_identity_shrinkage | 100 | 5.99 | 8.59 | 3.26 | 4.75 | 0.877 | 0.909 |
| 8g | compound | paper_style_current_v2 | 100 | 6.85 | 10.27 | 3.52 | 5.14 | 0.834 | 0.873 |
| 8g | compound | scale_only | 100 | 6.53 | 8.72 | 3.78 | 4.56 | 0.853 | 0.909 |
| 8g | compound | standard_shallow_finetune | 100 | 6.86 | 10.27 | 3.52 | 5.15 | 0.834 | 0.873 |
| 8g | compound | target_head_only | 100 | 5.85 | 9.88 | 3.04 | 4.70 | 0.882 | 0.887 |
| 8g | compound | zero_shot | 100 | 13.97 | 19.94 | 9.16 | 13.24 | 0.333 | 0.535 |
| 8g | row | affine | 100 | 6.61 | 9.49 | 3.56 | 5.65 | 0.842 | 0.867 |
| 8g | row | conditional_EA | 100 | 6.51 | 9.42 | 3.47 | 5.55 | 0.849 | 0.869 |
| 8g | row | local_identity_shrinkage | 100 | 6.47 | 9.46 | 3.55 | 5.64 | 0.851 | 0.868 |
| 8g | row | paper_style_current_v2 | 100 | 6.22 | 9.71 | 3.56 | 5.75 | 0.860 | 0.868 |
| 8g | row | scale_only | 100 | 6.20 | 9.24 | 3.82 | 5.52 | 0.853 | 0.875 |
| 8g | row | standard_shallow_finetune | 100 | 6.21 | 9.70 | 3.56 | 5.74 | 0.860 | 0.868 |
| 8g | row | target_head_only | 100 | 6.07 | 9.12 | 3.34 | 5.30 | 0.863 | 0.879 |
| 8g | row | zero_shot | 100 | 14.32 | 19.45 | 9.50 | 13.62 | 0.304 | 0.496 |

## Label efficiency

AULC is the trapezoidal area of the project-compatible arithmetic mean source-normalized RMSE over planned budgets 30/50/70/100; lower is better.

| column | protocol | method | AULC mean | sample SD | median | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 25g | compound | conditional_EA | 1.479 | 0.333 | 1.576 | 0.953 | 1.833 |
| 25g | compound | local_identity_shrinkage | 1.524 | 0.351 | 1.631 | 0.957 | 1.905 |
| 25g | compound | scale_only | 1.607 | 0.317 | 1.653 | 1.076 | 1.883 |
| 25g | compound | affine | 1.675 | 0.594 | 1.631 | 0.969 | 2.611 |
| 25g | compound | paper_style_current_v2 | 2.064 | 0.341 | 2.102 | 1.523 | 2.432 |
| 25g | compound | standard_shallow_finetune | 2.067 | 0.341 | 2.101 | 1.531 | 2.436 |
| 25g | compound | target_head_only | 3.050 | 0.605 | 3.224 | 2.032 | 3.503 |
| 25g | compound | zero_shot | 4.889 | 0.748 | 5.061 | 3.663 | 5.550 |
| 25g | row | conditional_EA | 1.966 | 0.412 | 1.857 | 1.425 | 2.525 |
| 25g | row | local_identity_shrinkage | 2.021 | 0.399 | 1.962 | 1.502 | 2.562 |
| 25g | row | scale_only | 2.060 | 0.299 | 1.971 | 1.674 | 2.419 |
| 25g | row | affine | 2.088 | 0.513 | 1.879 | 1.555 | 2.902 |
| 25g | row | standard_shallow_finetune | 2.117 | 0.461 | 1.876 | 1.646 | 2.724 |
| 25g | row | paper_style_current_v2 | 2.117 | 0.461 | 1.874 | 1.648 | 2.720 |
| 25g | row | target_head_only | 3.376 | 0.423 | 3.309 | 2.913 | 3.908 |
| 25g | row | zero_shot | 5.137 | 0.623 | 5.034 | 4.512 | 5.995 |
| 40g | compound | conditional_EA | 3.417 | 0.409 | 3.561 | 2.952 | 3.843 |
| 40g | compound | scale_only | 3.601 | 0.441 | 3.856 | 3.043 | 3.952 |
| 40g | compound | local_identity_shrinkage | 3.704 | 0.620 | 3.782 | 3.053 | 4.619 |
| 40g | compound | affine | 3.916 | 0.980 | 3.865 | 3.053 | 5.526 |
| 40g | compound | paper_style_current_v2 | 3.934 | 0.406 | 3.994 | 3.306 | 4.321 |
| 40g | compound | standard_shallow_finetune | 3.959 | 0.384 | 4.022 | 3.373 | 4.327 |
| 40g | compound | target_head_only | 7.719 | 0.389 | 7.986 | 7.149 | 7.998 |
| 40g | compound | zero_shot | 9.636 | 0.463 | 9.886 | 9.021 | 10.093 |
| 40g | row | conditional_EA | 3.089 | 0.735 | 2.632 | 2.503 | 3.918 |
| 40g | row | local_identity_shrinkage | 3.286 | 0.765 | 2.799 | 2.669 | 4.253 |
| 40g | row | affine | 3.320 | 0.869 | 2.733 | 2.669 | 4.517 |
| 40g | row | paper_style_current_v2 | 3.475 | 0.535 | 3.344 | 3.111 | 4.402 |
| 40g | row | standard_shallow_finetune | 3.480 | 0.530 | 3.347 | 3.119 | 4.397 |
| 40g | row | scale_only | 3.635 | 0.666 | 3.206 | 3.118 | 4.588 |
| 40g | row | target_head_only | 7.964 | 1.122 | 8.217 | 6.775 | 9.463 |
| 40g | row | zero_shot | 9.895 | 1.286 | 10.127 | 8.480 | 11.520 |
| 8g | compound | target_head_only | 0.725 | 0.084 | 0.771 | 0.579 | 0.774 |
| 8g | compound | local_identity_shrinkage | 0.728 | 0.184 | 0.657 | 0.543 | 1.022 |
| 8g | compound | scale_only | 0.735 | 0.211 | 0.681 | 0.528 | 1.065 |
| 8g | compound | conditional_EA | 0.752 | 0.214 | 0.673 | 0.529 | 1.096 |
| 8g | compound | standard_shallow_finetune | 0.823 | 0.109 | 0.815 | 0.678 | 0.969 |
| 8g | compound | paper_style_current_v2 | 0.824 | 0.108 | 0.815 | 0.677 | 0.968 |
| 8g | compound | affine | 0.850 | 0.305 | 0.812 | 0.518 | 1.340 |
| 8g | compound | zero_shot | 1.507 | 0.105 | 1.505 | 1.387 | 1.647 |
| 8g | row | target_head_only | 0.679 | 0.113 | 0.653 | 0.538 | 0.825 |
| 8g | row | standard_shallow_finetune | 0.725 | 0.103 | 0.719 | 0.629 | 0.881 |
| 8g | row | paper_style_current_v2 | 0.725 | 0.103 | 0.719 | 0.629 | 0.882 |
| 8g | row | scale_only | 0.730 | 0.136 | 0.718 | 0.617 | 0.955 |
| 8g | row | local_identity_shrinkage | 0.754 | 0.092 | 0.746 | 0.635 | 0.891 |
| 8g | row | conditional_EA | 0.755 | 0.102 | 0.767 | 0.604 | 0.890 |
| 8g | row | affine | 0.789 | 0.104 | 0.774 | 0.666 | 0.953 |
| 8g | row | zero_shot | 1.513 | 0.203 | 1.506 | 1.263 | 1.760 |

## Tail diagnosis

For each seed and budget, q50 and q80 are computed from that context's gradient-train labels only. Test rows are then classified as low, mid or high-tail using true test volume. `squared_error_fraction` is that stratum's contribution to total SSE. `tail_error_summary.csv` retains mean, sample SD, median, min and max for all low/mid/high strata.

| column | protocol | method | target | tail RMSE | tail MAE | tail SSE fraction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 25g | compound | conditional_EA | V1 | 24.99 | 19.45 | 73.4% |
| 25g | compound | scale_only | V1 | 26.14 | 19.81 | 62.8% |
| 25g | compound | local_identity_shrinkage | V1 | 26.31 | 20.67 | 70.7% |
| 25g | compound | affine | V1 | 27.00 | 21.10 | 72.0% |
| 25g | compound | standard_shallow_finetune | V1 | 40.23 | 32.79 | 82.9% |
| 25g | compound | paper_style_current_v2 | V1 | 40.33 | 32.87 | 83.3% |
| 25g | compound | target_head_only | V1 | 54.08 | 46.59 | 90.2% |
| 25g | compound | zero_shot | V1 | 89.62 | 81.06 | 80.7% |
| 25g | compound | scale_only | V2 | 35.31 | 26.25 | 60.4% |
| 25g | compound | conditional_EA | V2 | 35.39 | 26.36 | 67.6% |
| 25g | compound | local_identity_shrinkage | V2 | 35.87 | 26.68 | 69.9% |
| 25g | compound | affine | V2 | 36.69 | 27.15 | 70.3% |
| 25g | compound | standard_shallow_finetune | V2 | 51.42 | 39.32 | 78.7% |
| 25g | compound | paper_style_current_v2 | V2 | 51.59 | 39.49 | 79.2% |
| 25g | compound | target_head_only | V2 | 94.81 | 83.00 | 88.0% |
| 25g | compound | zero_shot | V2 | 133.07 | 120.14 | 81.4% |
| 25g | row | paper_style_current_v2 | V1 | 29.41 | 21.65 | 71.8% |
| 25g | row | standard_shallow_finetune | V1 | 29.42 | 21.68 | 71.8% |
| 25g | row | conditional_EA | V1 | 31.81 | 21.06 | 78.2% |
| 25g | row | local_identity_shrinkage | V1 | 33.24 | 21.70 | 77.4% |
| 25g | row | affine | V1 | 33.43 | 21.97 | 78.5% |
| 25g | row | scale_only | V1 | 33.58 | 21.60 | 76.1% |
| 25g | row | target_head_only | V1 | 55.87 | 46.15 | 92.8% |
| 25g | row | zero_shot | V1 | 87.94 | 78.11 | 85.2% |
| 25g | row | conditional_EA | V2 | 48.50 | 33.95 | 69.7% |
| 25g | row | paper_style_current_v2 | V2 | 49.21 | 36.82 | 71.2% |
| 25g | row | standard_shallow_finetune | V2 | 49.21 | 36.84 | 71.1% |
| 25g | row | local_identity_shrinkage | V2 | 49.32 | 35.08 | 71.5% |
| 25g | row | affine | V2 | 50.11 | 34.62 | 75.3% |
| 25g | row | scale_only | V2 | 53.24 | 35.37 | 75.1% |
| 25g | row | target_head_only | V2 | 108.18 | 92.38 | 89.0% |
| 25g | row | zero_shot | V2 | 144.66 | 128.98 | 81.8% |
| 40g | compound | scale_only | V1 | 61.55 | 43.55 | 67.5% |
| 40g | compound | conditional_EA | V1 | 61.98 | 43.95 | 81.9% |
| 40g | compound | local_identity_shrinkage | V1 | 64.76 | 47.15 | 80.3% |
| 40g | compound | affine | V1 | 64.76 | 47.15 | 80.3% |
| 40g | compound | paper_style_current_v2 | V1 | 68.53 | 50.64 | 72.6% |
| 40g | compound | standard_shallow_finetune | V1 | 68.61 | 50.74 | 72.6% |
| 40g | compound | target_head_only | V1 | 142.53 | 130.00 | 87.6% |
| 40g | compound | zero_shot | V1 | 177.43 | 164.54 | 82.3% |
| 40g | compound | conditional_EA | V2 | 76.42 | 52.26 | 60.4% |
| 40g | compound | scale_only | V2 | 77.35 | 50.58 | 63.7% |
| 40g | compound | local_identity_shrinkage | V2 | 79.92 | 51.10 | 73.6% |
| 40g | compound | affine | V2 | 79.92 | 51.10 | 73.6% |
| 40g | compound | paper_style_current_v2 | V2 | 88.58 | 65.20 | 65.6% |
| 40g | compound | standard_shallow_finetune | V2 | 88.72 | 65.34 | 65.5% |
| 40g | compound | target_head_only | V2 | 213.35 | 197.23 | 83.9% |
| 40g | compound | zero_shot | V2 | 250.90 | 234.23 | 79.9% |
| 40g | row | paper_style_current_v2 | V1 | 54.24 | 39.83 | 63.1% |
| 40g | row | standard_shallow_finetune | V1 | 54.32 | 39.90 | 63.0% |
| 40g | row | conditional_EA | V1 | 58.68 | 42.12 | 70.6% |
| 40g | row | affine | V1 | 62.72 | 44.35 | 70.6% |
| 40g | row | local_identity_shrinkage | V1 | 64.17 | 46.15 | 69.9% |
| 40g | row | scale_only | V1 | 64.98 | 47.07 | 62.4% |
| 40g | row | target_head_only | V1 | 161.17 | 144.08 | 87.7% |
| 40g | row | zero_shot | V1 | 199.05 | 180.97 | 82.3% |
| 40g | row | conditional_EA | V2 | 63.79 | 39.02 | 61.8% |
| 40g | row | paper_style_current_v2 | V2 | 64.84 | 48.59 | 64.7% |
| 40g | row | standard_shallow_finetune | V2 | 64.91 | 48.65 | 64.6% |
| 40g | row | local_identity_shrinkage | V2 | 65.94 | 40.29 | 63.1% |
| 40g | row | affine | V2 | 66.47 | 40.51 | 64.3% |
| 40g | row | scale_only | V2 | 72.76 | 47.86 | 58.9% |
| 40g | row | target_head_only | V2 | 220.59 | 196.52 | 86.0% |
| 40g | row | zero_shot | V2 | 258.81 | 233.62 | 82.0% |
| 8g | compound | target_head_only | V1 | 11.35 | 7.28 | 83.5% |
| 8g | compound | local_identity_shrinkage | V1 | 11.62 | 8.03 | 83.1% |
| 8g | compound | conditional_EA | V1 | 12.08 | 8.34 | 86.7% |
| 8g | compound | scale_only | V1 | 12.47 | 8.83 | 80.8% |
| 8g | compound | paper_style_current_v2 | V1 | 13.62 | 9.07 | 87.4% |
| 8g | compound | standard_shallow_finetune | V1 | 13.63 | 9.08 | 87.3% |
| 8g | compound | affine | V1 | 14.68 | 10.43 | 85.6% |
| 8g | compound | zero_shot | V1 | 27.61 | 22.76 | 88.0% |
| 8g | compound | local_identity_shrinkage | V2 | 16.94 | 11.28 | 78.9% |
| 8g | compound | conditional_EA | V2 | 17.06 | 11.26 | 79.1% |
| 8g | compound | scale_only | V2 | 17.54 | 11.04 | 82.2% |
| 8g | compound | affine | V2 | 18.65 | 11.72 | 83.3% |
| 8g | compound | target_head_only | V2 | 20.44 | 12.39 | 86.8% |
| 8g | compound | paper_style_current_v2 | V2 | 20.95 | 12.58 | 85.0% |
| 8g | compound | standard_shallow_finetune | V2 | 20.95 | 12.59 | 85.0% |
| 8g | compound | zero_shot | V2 | 40.10 | 32.33 | 83.4% |
| 8g | row | scale_only | V1 | 9.81 | 7.07 | 65.4% |
| 8g | row | target_head_only | V1 | 10.12 | 6.67 | 69.1% |
| 8g | row | standard_shallow_finetune | V1 | 10.55 | 7.54 | 73.0% |
| 8g | row | paper_style_current_v2 | V1 | 10.61 | 7.59 | 73.4% |
| 8g | row | local_identity_shrinkage | V1 | 10.69 | 7.66 | 71.9% |
| 8g | row | conditional_EA | V1 | 10.98 | 7.84 | 74.7% |
| 8g | row | affine | V1 | 11.02 | 7.86 | 73.1% |
| 8g | row | zero_shot | V1 | 26.74 | 21.96 | 87.1% |
| 8g | row | scale_only | V2 | 14.59 | 10.24 | 56.1% |
| 8g | row | conditional_EA | V2 | 14.94 | 10.60 | 56.5% |
| 8g | row | local_identity_shrinkage | V2 | 15.10 | 10.68 | 57.0% |
| 8g | row | affine | V2 | 15.21 | 10.69 | 57.2% |
| 8g | row | standard_shallow_finetune | V2 | 15.81 | 11.18 | 59.8% |
| 8g | row | paper_style_current_v2 | V2 | 15.87 | 11.25 | 60.0% |
| 8g | row | target_head_only | V2 | 16.04 | 11.26 | 66.7% |
| 8g | row | zero_shot | V2 | 37.14 | 30.98 | 78.9% |

## Reuse and protocol boundaries

Existing baseline and current-V2 source-anchored predictions are included only as `REUSED_MATCHED_ARTIFACT` after source/canonical/schedule/role/hash checks. The new paper arm is `RERUN_MATCHED_ARTIFACT`. Reused source-anchored results retain their `DEVELOPMENTAL` caveat because their historical test populations had already been examined. The legacy paper reproduction remains `NOT_PROTOCOL_MATCHED` because it uses threshold filtering, a larger label fraction, an old E0 source checkpoint and different test population; it is not ranked here.

Paper-style current-V2 starts from the qualified source and keeps the six-output current-V2 contract. It trains the final message layer, condition branch, transferred target head and a zero-initialized three-dimensional context adapter. It uses only nominal mass and column identity from the audited canonical targets; it does not use the historical hardcoded `(2.15, 15.6, 0.5248)` tuple and makes no verified physical-geometry claim. Since context is constant within each independently fit target column, its learned context shift is not separately identifiable from a target-specific latent shift; this is a paper-inspired strategy control, not proof of a physical column mechanism.

## Direct answers to the scientific questions

### 1–3. B=100 absolute-error floor

The following lists the lowest matched mean RMSE for each column/protocol/target; these are comparative observations, not operational acceptance thresholds.

- `8g/row` V1: `target_head_only` has the lowest B=100 mean RMSE (6.07 mL; MAE 3.34 mL; R2 0.863).
- `8g/row` V2: `target_head_only` has the lowest B=100 mean RMSE (9.12 mL; MAE 5.30 mL; R2 0.879).
- `8g/compound` V1: `target_head_only` has the lowest B=100 mean RMSE (5.85 mL; MAE 3.04 mL; R2 0.882).
- `8g/compound` V2: `local_identity_shrinkage` has the lowest B=100 mean RMSE (8.59 mL; MAE 4.75 mL; R2 0.909).
- `25g/row` V1: `paper_style_current_v2` has the lowest B=100 mean RMSE (17.37 mL; MAE 10.15 mL; R2 0.775).
- `25g/row` V2: `affine` has the lowest B=100 mean RMSE (25.22 mL; MAE 14.36 mL; R2 0.816).
- `25g/compound` V1: `conditional_EA` has the lowest B=100 mean RMSE (13.01 mL; MAE 8.17 mL; R2 0.864).
- `25g/compound` V2: `local_identity_shrinkage` has the lowest B=100 mean RMSE (19.76 mL; MAE 12.49 mL; R2 0.878).
- `40g/row` V1: `paper_style_current_v2` has the lowest B=100 mean RMSE (29.79 mL; MAE 17.58 mL; R2 0.820).
- `40g/row` V2: `conditional_EA` has the lowest B=100 mean RMSE (36.19 mL; MAE 20.71 mL; R2 0.857).
- `40g/compound` V1: `conditional_EA` has the lowest B=100 mean RMSE (31.84 mL; MAE 18.64 mL; R2 0.760).
- `40g/compound` V2: `local_identity_shrinkage` has the lowest B=100 mean RMSE (40.97 mL; MAE 23.08 mL; R2 0.792).

### 4. Equal-budget paper-style strategy

The fixed material-gain rule is `NOT PASSED`. It requires >=5% B=100 combined-normalized-RMSE and AULC gain with >=4/5 paired wins versus scale-only, affine and local shrinkage in both 25g and 40g. See `paired_comparisons.csv`; a one-seed win is not treated as evidence.

### 5. Why the historical paper reproduction can look much lower

The paper-aligned reproduction is not an estimand in this table: it confounds legacy V1/V2 filtering, roughly 80% target training labels, an old E0 source, different random splits/test population and a monotonic-head architecture. Its apparent low RMSE cannot be causally apportioned among those differences from the existing artifacts, so it cannot establish a matched strategy advantage.

### 6. Is high-volume tail the primary driver?

The predeclared B=100 tail SSE ranges are recorded in `SCIENTIFIC_DECISION.json` and the full tail table above. A tail is called dominant only at >=50% SSE. Results below that threshold show a material tail contribution but do not support attributing the entire large-column RMSE to a small tail alone.

### 7. Why high R2 can coexist with large RMSE

R2 is normalized by the variance of the evaluated target population. The B=100 test-distribution summary below shows the target-scale context; a larger test standard deviation permits a larger absolute RMSE at the same R2. It is therefore secondary to mL RMSE/MAE for operational interpretation.

| column | protocol | target | mean test sample SD (mL) | mean test variance (mL²) |
| --- | --- | --- | ---: | ---: |
| 25g | compound | V1 | 35.32 | 1281.41 |
| 25g | compound | V2 | 56.56 | 3273.28 |
| 25g | row | V1 | 37.11 | 1400.44 |
| 25g | row | V2 | 59.37 | 3573.62 |
| 40g | compound | V1 | 66.03 | 4369.79 |
| 40g | compound | V2 | 92.14 | 8528.98 |
| 40g | row | V1 | 70.85 | 5089.07 |
| 40g | row | V2 | 97.79 | 9684.49 |
| 8g | compound | V1 | 17.22 | 297.91 |
| 8g | compound | V2 | 29.51 | 884.24 |
| 8g | row | V1 | 17.39 | 313.03 |
| 8g | row | V2 | 27.66 | 775.50 |

### 8–9. Bottleneck and next-model decision

`SIMPLE_CALIBRATION_REMAINS_COMPETITIVE`. This benchmark can distinguish matched predictive error patterns and paired reproducibility, but it cannot causally isolate calibration bias, tail extrapolation, representation capacity, low-label estimation, or measurement noise. Unless the preregistered paper-style material-gain rule passes, the evidence does not justify appending a more complex model solely to chase this test result; independent batches, replicated measurements and tail coverage remain the cleaner next evidence.

## Scientific decision

Primary outcome: `SIMPLE_CALIBRATION_REMAINS_COMPETITIVE`. Tags: `SIMPLE_CALIBRATION_REMAINS_COMPETITIVE, TAIL_ERROR_DOMINATES_LARGE_COLUMN_RMSE`. RMSE/MAE, tail fractions and paired five-seed comparisons should be read together. High R2 alone is not evidence of operationally small absolute error.
