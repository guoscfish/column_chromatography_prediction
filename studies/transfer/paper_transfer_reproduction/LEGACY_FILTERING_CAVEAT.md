# Legacy filtering caveat

The legacy constructors filter target labels after the release-code reader has
discarded records with invalid `t1` values. Volumes are computed as
`V_ml = t_raw * Flow_mL_min / 1200`.

| Column | Raw rows | Legacy thresholds | Reader-compatible rows | Rows after legacy filter |
| --- | ---: | --- | ---: | ---: |
| 8g | 574 | `V1 <= 60`, `V2 <= 120` | 574 | 552 |
| 25g | 569 | `V1 <= 60`, `V2 <= 120` | 490 | 408 |
| 40g | 531 | `V1 <= 150`, `V2 <= 200` | 529 | 456 |

These counts come from the committed `experiments/data_audit/data_manifest.csv`
and are checked again by the reproduction runner before training. The different 40g
threshold is preserved; thresholds are not harmonized.

Consequences:

- Primary `legacy_filtered` RMSE is an estimate for the post-filter target
  population, not for all valid raw measurements.
- Filtering removes high-volume tails disproportionately. Because squared error is
  tail-sensitive, a no-threshold RMSE can be materially larger even when R2 remains
  visually favorable.
- Results cannot be treated as a strictly fair comparison with the current Scale
  budget-100 studies, which use no target threshold and only 100 revealed target
  labels rather than about 80% target rows.

The runner exposes `--protocols no_threshold` for a separately labelled sensitivity
run. That option cannot alter the frozen primary model, trainable scope, split logic,
or reported primary conclusion.
