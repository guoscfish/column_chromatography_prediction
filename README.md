# Column chromatography prediction

Retention-volume prediction with molecular geometry and experimental conditions, followed by low-label 4g→8g transfer.

Historical Legacy → condition-complete correction → function-preserving pruning → standalone QGeoGNN-V2 → current predictor.

The standalone QGeoGNN-V2 is `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`. Predictor architecture is no longer the default research target. Ordinary transfer proceeds independently of UQ qualification; active transfer requires independent transfer validation and an adequate uncertainty contract.

The final model has 458,952 parameters, all gradient-bearing. Six-output equivalence to R2-pruned on all 4,163 frozen E0 rows is exact (maximum absolute difference 0). Final 4g qualification completed all six frozen row/compound runs without failures. The existing quantile head is retained for point transfer; its audit motivates a head/UQ control before active transfer.

## Current evidence

- [Scaling failure audit](studies/transfer/scaling_failure_audit/SCALING_FAILURE_AUDIT.md) and [model decision](studies/transfer/scaling_failure_audit/NEXT_MODEL_DECISION.md): reproducible EA/V1 structure, one controlled conditional-scaling experiment, `STRUCTURED_FAILURE_BUT_NO_MATERIAL_MODEL_GAIN`.
- [Research status and backlog](docs/research/CROSS_COLUMN_TRANSFER_STATUS.md): precise scope of earlier negative results and future hypotheses.
- [Historical transfer residual diagnostics](studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md): two controlled experiments; the historical `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA` applies only to tested calibration extensions. Active learning remains deferred.
- [Cross-column validation](studies/transfer/cross_column/CROSS_COLUMN_TRANSFER_REPORT.md): matched 8g/25g/40g row and target-compound splits.
- [Standalone engineering](studies/predictor/final_v2_engineering/README.md): equivalence, reachability and checkpoint contract.
- [Final 4g qualification](studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md): all Train/Validation/Test metrics and seed aggregates.
- [Quantile audit](studies/predictor/final_4g_qualification/QUANTILE_AUDIT.md): descriptive uncertainty assessment, without head retraining.
- [Final-source 4g→8g baseline](studies/transfer/4g_to_8g/TRANSFER_BASELINE_REPORT.md): five adaptation families, five frozen target partitions, budgets 30/50/70/100.
- [Next decision](docs/NEXT_STAGE_DECISION.md), [documentation](docs/README.md), [study index](studies/README.md).

## Active code

`src/qgeognn_al/models/qgeognn_v2.py` owns `build_predictor`, `load_predictor_checkpoint`, `forward` and `extract_representation`. Shared data, training, transfer, evaluation and uncertainty code lives in the corresponding `src/qgeognn_al/` packages. New studies use this single predictor API.

Run verification in the validated conda environment:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
```

[Historical evidence](studies/predictor/historical/README.md) remains at its original paths for provenance. Historical T1/T1b/G0/S1 conclusions do not establish rankings for the corrected predictor.
