# Column chromatography prediction

Retention-volume prediction with molecular geometry and experimental conditions, followed by matched low-label cross-column transfer evaluation.

Historical Legacy → condition-complete correction → function-preserving pruning → standalone QGeoGNN-V2 → current predictor.

The standalone QGeoGNN-V2 is `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`. Predictor architecture is no longer the default research target. Ordinary transfer proceeds independently of UQ qualification; active transfer requires independent transfer validation and an adequate uncertainty contract.

The final model has 458,952 parameters, all gradient-bearing. Six-output equivalence to R2-pruned on all 4,163 frozen E0 rows is exact (maximum absolute difference 0). Final 4g qualification completed all six frozen row/compound runs without failures. The existing quantile head is retained for point transfer; its audit motivates a head/UQ control before active transfer.

## Current evidence

The authoritative transfer result is the [matched absolute-error benchmark](studies/transfer/matched_rmse_benchmark/MATCHED_RMSE_REPORT.md). It uses the qualified final 4g source checkpoint (`fce9…544b`), identical no-threshold 8g/25g/40g row and target-compound splits, five fixed seeds, and 30/50/70/100 revealed-label budgets. RMSE/MAE in mL are primary; R² is secondary.

`SIMPLE_CALIBRATION_REMAINS_COMPETITIVE`: at equal budget, paper-style current-V2 adaptation does not meet the preregistered paired B=100-plus-AULC gain rule against scale, affine, and shrinkage controls on both 25g and 40g. Large-column error is tail-dominated, so high R² does not establish operationally small retention-volume error. The current simple baseline family is conditional EA / local identity shrinkage; conclusions remain developmental because reused source-anchored evidence had historical test exposure.

The [paper-transfer reconstruction](studies/transfer/paper_transfer_reproduction/REPRODUCTION_REPORT.md) is a `PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION`, not a ranked matched comparator: it uses legacy filtering, larger label fractions, an old source checkpoint, and different splits. Historical studies remain available through the [study index](studies/README.md); the current decision and backlog are in [docs/NEXT_STAGE_DECISION.md](docs/NEXT_STAGE_DECISION.md).

## Active code

`src/qgeognn_al/models/qgeognn_v2.py` owns `build_predictor`, `load_predictor_checkpoint`, `forward` and `extract_representation`. Shared data, training, transfer, evaluation and uncertainty code lives in the corresponding `src/qgeognn_al/` packages. New studies use this single predictor API.

Run verification in the validated conda environment:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
```

[Historical evidence](studies/predictor/historical/README.md) remains at its original paths for provenance. Historical T1/T1b/G0/S1 conclusions do not establish rankings for the corrected predictor.
