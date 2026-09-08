# Matched cross-column absolute-error benchmark

`MATCHED_CROSS_COLUMN_ABSOLUTE_ERROR_BENCHMARK` is the authoritative current
comparison of cross-column transfer strategies.  It inherits the frozen
no-threshold 8g/25g/40g schedule from `../cross_column/` and treats RMSE and
MAE in mL as primary outcomes; R2 is a secondary trend measure.

The fixed source is
`studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt`
(SHA-256 `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`).
It evaluates row and target-compound splits over five frozen outer seeds and
budgets 30/50/70/100.  Calibration only sees gradient-train labels; neural
selection only sees frozen validation labels.  No target threshold is applied.

Primary methods are `zero_shot`, `scale_only`, `affine`,
`local_identity_shrinkage`, `conditional_EA`, `target_head_only`,
`standard_shallow_finetune`, and `paper_style_current_v2`.  Existing exact
matches are audit-reused; only the last arm is newly fit.  The paper-style arm
keeps the current six-output contract, starts from current-V2, and trains its
final shallow layer/condition/head plus a zero-initialized nominal-mass and
column-identity adapter.  It does **not** use the legacy 25g/40g tuple.
Because each target column is independently adapted, that constant context is
a paper-inspired strategy control rather than identified physical geometry.

Run from the repository root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish \
  python scripts/studies/run_matched_rmse_benchmark.py --prepare
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish \
  python scripts/studies/run_matched_rmse_benchmark.py --execute
```

The runner first freezes and hashes all 120 paper-style prediction files under
ignored `runtime/`, then evaluates test truth once.  Compact outputs at this
study root include `all_metrics.csv`, `budget100_summary.csv`,
`aulc_summary.csv`, `paired_comparisons.csv`, `tail_error_metrics.csv`,
`artifact_audit.json`, and `MATCHED_RMSE_REPORT.md`.

`../paper_transfer_reproduction/` remains a
`PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION`: its filtered data, old source,
larger label fraction and different test population exclude it from this
ranking.  Reused source-anchored artifacts retain their historical
developmental/test-exposure caveat; this benchmark is not an independent new
test set for those arms.
