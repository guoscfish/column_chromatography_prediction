# Paper-transfer reproduction report

## Protocol

This is a paper-aligned reconstructed reproduction, not an exact direct execution of a complete 25g/40g author training path. It loads the fixed 4g source, appends repository-original column values, function-preservingly initializes new adapters, uses the G0-4 monotonic head and last2-plus-head transfer scope, and selects the checkpoint using validation only. Direct training is random initialization with all parameters trainable. Both use identical random row splits, source preprocessing, loss, and held-out test rows.

## Main results

| 柱规格 | 方法 | V1 R2 (mean +/- SD) | V1 RMSE mL (mean +/- SD) | V1 MAE mL (mean +/- SD) | V2 R2 (mean +/- SD) | V2 RMSE mL (mean +/- SD) | V2 MAE mL (mean +/- SD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25g | direct | -0.076 +/- 0.731 | 9.11 +/- 3.55 | 6.65 +/- 2.98 | 0.169 +/- 0.684 | 13.79 +/- 6.15 | 9.73 +/- 3.95 |
| 25g | paper_transfer | 0.597 +/- 0.087 | 5.91 +/- 1.03 | 4.09 +/- 0.60 | 0.717 +/- 0.067 | 8.21 +/- 0.95 | 5.79 +/- 0.78 |
| 40g | direct | 0.547 +/- 0.232 | 15.93 +/- 3.52 | 10.68 +/- 2.40 | 0.650 +/- 0.122 | 19.36 +/- 2.83 | 13.24 +/- 1.72 |
| 40g | paper_transfer | 0.697 +/- 0.187 | 12.85 +/- 3.81 | 7.98 +/- 2.16 | 0.789 +/- 0.099 | 14.81 +/- 2.31 | 9.56 +/- 2.23 |

The full seed-level values, medians, minima, and maxima are in `all_metrics.csv` and `PAPER_TRANSFER_RMSE_SUMMARY.csv`. Per-run selection metadata and scalar validation/test metrics are retained in `run_summary.csv`; checkpoints, histories and per-sample predictions are reproducible runtime only.

## Figure 4 R2 sanity check

| column | paper R2 V1 | reproduction R2 V1 | paper R2 V2 | reproduction R2 V2 |
| --- | ---: | ---: | ---: | ---: |
| 8g | 0.759 | 0.761 | 0.752 | 0.806 |
| 25g | 0.747 | 0.597 | 0.840 | 0.717 |
| 40g | 0.826 | 0.697 | 0.824 | 0.789 |

## Interpretation boundary

R2 measures explained variation within a target test distribution, whereas RMSE is an absolute deviation in mL. Larger-column data can have larger output variance, so an apparently high R2 can coexist with an operationally large RMSE. The 25g and 40g legacy input tuples are identical repository constants, not verified physical metadata; see `COLUMN_SPEC_PROVENANCE_AUDIT.md`.

## Reproduction gap

Differences from Figure 4 are descriptive only. Plausible non-identifiable sources include dataset version, legacy filtering, random split, unavailable exact freeze map, source checkpoint, preprocessing, output-head implementation, epoch/patience, and unverified column geometry. No item in this list was changed after observing a test metric.

Shared column-spec marker: `25G_AND_40G_SHARE_LEGACY_COLUMN_SPEC_VALUES`.
