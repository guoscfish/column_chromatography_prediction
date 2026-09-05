# Final standalone 4g qualification

All six frozen runs completed. [Formal report](FINAL_4G_QUALIFICATION_REPORT.md) includes every Train/Validation/Test metric, mean/sample-standard-deviation/min/max and generalization gaps. [Quantile audit](QUANTILE_AUDIT.md) is descriptive and nonblocking for point transfer.

Reproduction in conda fish from repository root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/studies/run_final_4g_qualification.py --execute --workers 2
KMP_DUPLICATE_LIB_OK=TRUE python scripts/studies/summarize_final_4g_qualification.py
```

Read `--help` and the frozen protocol before execution. Completed matching runs are reused. The interrupted compound-seed-1101 run was restarted from its original fixed seed and protocol because no resumable optimizer checkpoint existed; all other complete runs were preserved. Its first 70 epochs matched the interrupted run. No seed was retried for poor performance.

The fixed transfer source is `runtime/row/seed_42/best.pt`, SHA256 `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. It is available locally as an ignored runtime artifact; a fresh clone must reproduce or separately restore that exact checkpoint before the hash-locked transfer study can run. Binary runtime is not published by the Git push.
