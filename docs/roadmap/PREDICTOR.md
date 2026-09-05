# Predictor roadmap

Historical Legacy → condition-complete correction → function-preserving pruning → standalone QGeoGNN-V2 → current predictor.

The standalone QGeoGNN-V2 is `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`. Predictor architecture is no longer the default research target. Ordinary transfer proceeds independently of UQ qualification; active transfer requires independent transfer validation and an adequate uncertainty contract.

[Engineering gate](../../studies/predictor/final_v2_engineering/README.md): zero full-E0 six-output difference; 458,952 nominal/trainable/gradient-bearing parameters; zero unreachable. The final model directly constructs its effective network and preserves the R2 output head and preprocessing semantics.

[Final 4g evidence](../../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md): 4,163 rows / 217 compounds, V1≤60 and V2≤120; three frozen seeds per row and compound estimand. Row generalization is stronger than novel-compound generalization, without numerical failures or training collapse.

[Quantile audit](../../studies/predictor/final_4g_qualification/QUANTILE_AUDIT.md): `CURRENT_HEAD_RETAINED_FOR_POINT_TRANSFER`; `MONOTONIC_HEAD_CONTROL_REQUIRED_BEFORE_ACTIVE_TRANSFER`. These are separate point and uncertainty decisions. No new head was trained.

Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / HISTORICAL_NEGATIVE_RESULT`. Its reproduction implementation is under `src/qgeognn_al/historical/`; old diagnostic V2/pruned files retain their original paths and hashes. No backbone, fusion, adapter or hyperparameter sweep is scheduled.
