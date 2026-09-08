# Reproduction gap


R2 measures explained variation within a target test distribution, whereas RMSE is an absolute deviation in mL. Larger-column data can have larger output variance, so an apparently high R2 can coexist with an operationally large RMSE. The 25g and 40g legacy input tuples are identical repository constants, not verified physical metadata; see `COLUMN_SPEC_PROVENANCE_AUDIT.md`.

## Reproduction gap

Differences from Figure 4 are descriptive only. Plausible non-identifiable sources include dataset version, legacy filtering, random split, unavailable exact freeze map, source checkpoint, preprocessing, output-head implementation, epoch/patience, and unverified column geometry. No item in this list was changed after observing a test metric.

Shared column-spec marker: `25G_AND_40G_SHARE_LEGACY_COLUMN_SPEC_VALUES`.
