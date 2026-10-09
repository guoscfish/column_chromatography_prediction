# CW factorial continuation to1005

User-authorized extension of all eight arms × two seeds through1005. Three new arms continue their own frozen653 histories with11 further batches each (66 fits); five controls reuse110 historical budget points after protocol/hash/lineage audit. Including the preceding24-fit653 continuation,90 new fits extend the original525 study. Zero paid LLM calls, zero new test labels.

Status: queued after the active653 run. Frozen protocol: protocol.json; reuse manifest: REUSE_MANIFEST.json. Pipeline status: pipeline_state.json; per-seed progress: progress_seed_*.json; fit reservations/checkpoints remain under ignored runtime.

Primary descriptive extension metric: validation AULC525–1005, fixed before1005 extension outcomes. Secondary AULC653–1005, AULC333–1005, NRMSE1005 and V1/V2 RMSE/MAE/R². Original333–525 gate and intermediate525–653 result remain unchanged. Post-development exploration is not independent confirmation. Hard stop1005 regardless of effect.

Two independent fits may run concurrently after653, two CPU threads per fit. Per-trajectory quota11; interrupted attempts retain reservations. Training otherwise follows the original common frozen protocol.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/continue_cw_factorial_to1005.py
# Final verification only:
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_to1005.py
```

The supervisor refuses to silently restart an interrupted653 fit. Inspect its log and ledger before recovery; never erase a reservation to bypass the fit limit. Completed fits are reused without retraining.
