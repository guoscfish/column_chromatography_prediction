# Stage 1 Handoff: Formal ROW Transfer

## Integrity

- HEAD: `90879eab13f3e1ece9d44c48d54e4395ea923680`
- Experiment complete: 40/40 method-context fits.
- Prediction freeze: successful, 10 contexts and 40 method-context predictions.
- One-shot test scoring: successful; 40 finite metric rows after the freeze boundary.
- No test truth was used during fit or freeze. No seed, split, threshold, checkpoint, or optimizer recipe was changed.

## Fixed methods

- P0: direct historical-shallow adaptation, current BN, Adam.
- P1: head-only Stage A -> historical-shallow Stage B, current BN, Adam.
- P2: head-only Stage A -> historical-shallow Stage B, source BN running statistics, Adam.
- P3: head-only Stage A -> historical-shallow Stage B, source BN running statistics, L-BFGS.

Historical trainable scope is `backbone.convs.4 + condition_branch + head`, 36,387 parameters. The frozen source checkpoint is seed 42.

## Test summary

Values below are mean combined normalized RMSE; lower is numerically better.

| column | P0 | P1 | P2 | P3 |
| --- | ---: | ---: | ---: | ---: |
| 25g | 0.666 | 0.666 | 0.646 | 0.677 |
| 40g | 0.537 | 0.484 | 0.544 | 0.490 |

Mean endpoint metrics (V1 RMSE / MAE / R2; V2 RMSE / MAE / R2):

| column | method | V1 RMSE | V1 MAE | V1 R2 | V2 RMSE | V2 MAE | V2 R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25g | P0 | 7.278 | 4.397 | 0.377 | 10.895 | 6.891 | 0.546 |
| 25g | P1 | 7.359 | 4.466 | 0.358 | 10.708 | 6.916 | 0.560 |
| 25g | P2 | 7.137 | 4.457 | 0.418 | 10.355 | 6.918 | 0.593 |
| 25g | P3 | 7.330 | 4.607 | 0.380 | 11.158 | 7.124 | 0.516 |
| 40g | P0 | 15.533 | 10.832 | 0.613 | 18.356 | 13.254 | 0.730 |
| 40g | P1 | 13.944 | 8.948 | 0.686 | 16.606 | 11.326 | 0.779 |
| 40g | P2 | 15.617 | 10.159 | 0.610 | 18.713 | 13.305 | 0.720 |
| 40g | P3 | 14.064 | 8.878 | 0.676 | 16.940 | 11.448 | 0.764 |

## Per-seed combined NRMSE

| column | seed | P0 | P1 | P2 | P3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25g | 769539383 | 0.625 | 0.620 | 0.656 | 0.636 |
| 25g | 1425370602 | 0.754 | 0.771 | 0.693 | 0.768 |
| 25g | 536279090 | 0.819 | 0.823 | 0.754 | 0.834 |
| 25g | 2767143051 | 0.609 | 0.605 | 0.618 | 0.632 |
| 25g | 1362771960 | 0.525 | 0.510 | 0.507 | 0.516 |
| 40g | 769539383 | 0.543 | 0.505 | 0.519 | 0.537 |
| 40g | 1425370602 | 0.608 | 0.578 | 0.621 | 0.549 |
| 40g | 536279090 | 0.448 | 0.392 | 0.467 | 0.424 |
| 40g | 2767143051 | 0.486 | 0.411 | 0.511 | 0.460 |
| 40g | 1362771960 | 0.602 | 0.534 | 0.601 | 0.481 |

## Paired comparisons

Combined-NRMSE mean paired delta (candidate minus reference), with seed wins out of 5:

| column | comparison | delta | wins/5 |
| --- | --- | ---: | ---: |
| 25g | P1 vs P0 | -0.001 | 3 |
| 25g | P2 vs P1 | -0.020 | 3 |
| 25g | P2 vs P0 | -0.021 | 3 |
| 25g | P3 vs P2 | 0.032 | 1 |
| 25g | P3 vs P0 | 0.011 | 1 |
| 40g | P1 vs P0 | -0.053 | 5 |
| 40g | P2 vs P1 | 0.060 | 0 |
| 40g | P2 vs P0 | 0.006 | 2 |
| 40g | P3 vs P2 | -0.053 | 4 |
| 40g | P3 vs P0 | -0.047 | 5 |

## Diagnostics

- BN running-stat drift was 0 for all P2/P3 contexts. P0/P1 current-BN drift is nonzero by design of the frozen recipe.
- Quantile crossings are present in some frozen predictions; counts and rates are recorded in `summary.csv`, `test_metrics.csv`, and `bn_analysis.csv`.
- 41/50 Adam stages selected their best checkpoint at epoch 150, so the fixed budget is flagged as potentially insufficient. No budget was changed.
- No runtime, numerical, or data-integrity anomaly was found in the completed artifacts. All predictions are finite, all contexts have complete hash-verified artifacts, and graph coverage is complete.

The statements above are data facts only. Scientific interpretation and Stage 2 experiment decisions are deferred to Stage 2.
