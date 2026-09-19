# 4 g Error Landscape

Status: `DIAGNOSTIC_COMPLETE / POST_HOC_DESCRIPTIVE`.

This artifact is built from the six frozen QGeoGNN-V2 qualification prediction
files, the canonical 4 g metadata table, and the frozen split manifests. The
existing E1 uncertainty table is checked for identity compatibility but is not
joined because its source manifest has zero sample-id overlap with these
qualification predictions. It does not train a model, reveal a new label, change
a split, or select an acquisition parameter.

Run from the repository root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish \
  python scripts/analyze_4g_error_landscape.py
```

Key outputs are `row_level_metrics.csv.gz`, `bin_summary.csv`,
`interaction_summary.csv`, `calibration_summary.csv`,
`correlation_summary.csv`, the local sensitivity tables, and
`repeated_condition_summary.csv`. The row-level table contains 24,978 records:
4,163 rows for each of three row and three compound qualification runs.

All test-derived values are descriptive. They are not used to choose a new model,
acquisition strategy, or hyperparameter.
