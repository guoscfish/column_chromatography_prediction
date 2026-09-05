# Column chromatography prediction

This repository studies retention-volume prediction for column chromatography, with emphasis on molecular geometry, experimental conditions, data-efficient source-to-target adaptation, and active learning.

## Current scientific state

| Direction | Status | Boundary |
|---|---|---|
| Predictor | `POINT_PREDICTOR_CANDIDATE_BASELINE` | R2-pruned removes 318,856 prediction-unreachable parameters and retains 458,952 gradient-bearing parameters. Function gates and frozen E0 retraining exactly reproduce R2 (test R² 0.889218 / 0.941563). Clean is a failed architecture experiment, retained for provenance. |
| 4g in-domain active learning | `PAUSED / historical evidence retained` | E2 and A1a results remain measured historical evidence. No new acquisition run is authorized. |
| 4g -> 8g transfer | `PAUSED_PENDING_HEAD_AND_UQ_QUALIFICATION` | No Clean transfer run has been executed. Transfer and UQ work remain paused pending separate quantile-head and uncertainty-contract qualification. |
| Active transfer | `DEFERRED` | It remains downstream of predictor and transfer qualification. |

The predictor is being requalified because the I0 semantic audit found that the legacy graph builder constructs ten continuous edge features while the legacy encoder consumes only five. The controlled regression audit then confirmed that the first large performance loss appears between Condition Completion V2 and the current Clean architecture on the identical historical E0 split. These facts narrow the interpretation of historical evidence; they do not erase the measured results.

## Next step

The predictor mainline is **Legacy historical → Condition Completion V2 → R2-pruned candidate baseline**. The next separate controlled test should compare the current head with a monotonic quantile head, changing only the output head. This round did not execute that test, UQ, 8g transfer, or active learning. See the [R2-pruned comparison](studies/predictor/r2_pruned_requalification/R2_PRUNED_COMPARISON.md) and [predictor roadmap](docs/roadmap/PREDICTOR.md).

## Start reading

- [`docs/README.md`](docs/README.md): documentation map and current contracts.
- [`studies/README.md`](studies/README.md): semantic navigation for current and historical studies.
- [`docs/model/LEGACY_QGEOGNN_AUDIT.md`](docs/model/LEGACY_QGEOGNN_AUDIT.md): audited legacy implementation boundary.
- [`docs/model/INPUT_SCHEMA.md`](docs/model/INPUT_SCHEMA.md): explicit feature semantics and units.
- [`docs/protocols/4G_PREDICTOR_BENCHMARK.md`](docs/protocols/4G_PREDICTOR_BENCHMARK.md): draft benchmark protocol.
- [`studies/predictor/clean_qgeognn/preflight/`](studies/predictor/clean_qgeognn/preflight/README.md): Clean-QGeoGNN engineering preflight.
- [`studies/predictor/clean_4g_baseline_qualification/`](studies/predictor/clean_4g_baseline_qualification/QUALIFICATION_REPORT.md): completed formal 4g qualification and next-gate decision.
- [`studies/predictor/performance_regression_audit/`](studies/predictor/performance_regression_audit/REGRESSION_REPORT.md): controlled Legacy/V2/Clean regression ladder and diagnosis.
- [`experiments/INDEX.md`](experiments/INDEX.md): frozen historical experiment index.

## Code and validation

Reusable scientific code lives in `src/qgeognn_al/`; historical runners and compatibility shims remain under `scripts/`. The validated environment is conda `fish`:

```bash
conda run --no-capture-output -n fish pytest -q
```
