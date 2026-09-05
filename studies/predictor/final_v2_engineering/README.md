# Standalone QGeoGNN-V2 engineering

PASS: full frozen E0 (4,163 rows) six-output maximum absolute difference is 0, including point metrics. Nominal, requires-grad and gradient-bearing counts are all 458,952; unreachable count is 0. See `equivalence_audit.json` and `reachability_audit.json`.

Reproduce from repository root in conda fish:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/studies/run_final_v2_engineering.py
```

The model directly builds five effective node layers and four geometry updates, retains early eluent interaction, typed condition completion, sum pooling, 128D representation and the original six-output head. Conversion lives in the historical namespace. Active checkpoint loading strictly validates schema, input contract and normalization; its state round trip is audited. Runtime checkpoints are ignored binary artifacts, while hashes and measured evidence are tracked. See `results/test_report.json` and `results/pytest_output.txt` for final verification.
