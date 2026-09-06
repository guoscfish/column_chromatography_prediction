# Current research decision

The latest [two controlled residual diagnostics](../studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md), based on cross-column commit `61f20c9`, conclude **`NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA`**. Monotone nonlinear calibration has no stable material benefit. Shared calibration improves compound portfolio AULC by 9.04% versus affine, but only 1.42% versus scale-only and 1.64% versus independent shrinkage. Neither family passes the preregistered cross-seed/cross-column gate. The remaining error cannot yet be attributed uniquely to readout or data limitations. No extra model is appended to obtain a positive result.

This stage does not start Active Learning, ordinary full fine-tuning, an adapter width sweep, Clean, or adaptive readout. The older cross-column report's `ACTIVE_CALIBRATION` proposal is historical, not the current execution instruction. Further mechanism identification would benefit from independent batches/compounds, replication, crossed mass/flow settings and tail coverage. Target-compound holdout is not source-unseen OOD.

The final standalone QGeoGNN-V2 passed exact R2-pruned equivalence and six-run 4g qualification: `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`. Predictor architecture is no longer the default research object.

The current quantile head is `CURRENT_HEAD_RETAINED_FOR_POINT_TRANSFER`. Compound-seed-525 V1 crossing is 15.22%, and nominal 80% intervals under-cover, especially on unseen compounds. Width remains positively associated with absolute error. `MONOTONIC_HEAD_CONTROL_REQUIRED_BEFORE_ACTIVE_TRANSFER` therefore applies to future UQ/active-transfer qualification; it does not block ordinary point transfer. No replacement head was trained.

The [final-source baseline report](../studies/transfer/4g_to_8g/TRANSFER_BASELINE_REPORT.md) and [machine-readable decision](../studies/transfer/4g_to_8g/decision.json) own the measured transfer ranking. The comparison uses one preregistered standalone source, five existing frozen target row partitions, four nested random-label budgets and five fixed adaptation families. This is developmental evidence: independent target/compound/column validation remains necessary.

Ordinary transfer and UQ are parallel workstreams. Independent transfer validation plus UQ qualification precede active transfer. No active acquisition, adapter sweep, Clean repair or predictor benchmark was executed. Historical T1/T1b/G0/S1 are `HISTORICAL_LEGACY_PREDICTOR_EVIDENCE`; their rankings are not imported into the new decision.

Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / HISTORICAL_NEGATIVE_RESULT`. Historical decision JSONs retain their original next-stage proposals to preserve provenance; this document supersedes those proposals, including the former requirement to finish a quantile-head experiment before ordinary transfer.
