# QGeoGNN-V2 row Hybrid extension

This is a **POST-PRIMARY MATCHED EXTENSION**, not a retrospective modification of `qgeognn_v2_row_small_batch_benchmark`.

The frozen primary B=32 study and its decision, `BEST_CURRENT_ROW_ACQUISITION`, remain unchanged.  This extension reuses its matched five-seed Random and Gradient-LCMD results only after a strict hash audit, then adds exactly one new arm: historical-principle Hybrid implemented entirely with current QGeoGNN-V2 uncertainty and `extract_representation` latent space.

Run the strict audit first, then the formal extension:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_hybrid_extension.py --audit
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_hybrid_extension.py --run
```

`reuse_audit.json` is the authority for whether old Random and LCMD results may be reused.  A failed audit stops execution rather than retraining or reinterpreting the primary study.
