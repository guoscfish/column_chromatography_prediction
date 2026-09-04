# Column chromatography prediction

This repository studies retention-volume prediction for column chromatography, with emphasis on molecular geometry, experimental conditions, data-efficient source-to-target adaptation, and active learning.

## Current scientific state

| Direction | Status | Boundary |
|---|---|---|
| Predictor | `CLEAN_4G_BASELINE_QUALIFICATION_COMPLETE` | Clean-QGeoGNN preflight revision 2 and six-run 4g baseline qualification are complete. It is point-baseline ready, demonstrates learned condition reliance, and still requires UQ qualification. Legacy is frozen historical evidence; V2 is an archived engineering-qualified ablation. |
| 4g in-domain active learning | `PAUSED / historical evidence retained` | E2 and A1a results remain measured historical evidence. No new acquisition run is authorized. |
| 4g -> 8g transfer | `PAUSED_PENDING_UQ_REVIEW` | T1/T1b-1 remain valid under the audited legacy predictor contract. No Clean transfer run has been executed; UQ qualification/calibration is the next gate. |
| Active transfer | `DEFERRED` | It remains downstream of predictor and transfer qualification. |

The predictor is being requalified because the I0 semantic audit found that the legacy graph builder constructs ten continuous edge features while the legacy encoder consumes only five. It also found a substantial gap between nominal and gradient-bearing parameter counts. These facts narrow the interpretation of historical evidence; they do not erase the measured results.

## Next step

The sole recommended next gate is Clean-QGeoGNN UQ qualification/calibration before transfer or active-learning work. A multi-model predictor benchmark is not required for the current mainline; Legacy/V2/MLP/Paper-ANN comparisons remain optional future architecture-ablation work. No 8g labels or active-learning runs were used in the completed 4g qualification.

## Start reading

- [`docs/README.md`](docs/README.md): documentation map and current contracts.
- [`studies/README.md`](studies/README.md): semantic navigation for current and historical studies.
- [`docs/model/LEGACY_QGEOGNN_AUDIT.md`](docs/model/LEGACY_QGEOGNN_AUDIT.md): audited legacy implementation boundary.
- [`docs/model/INPUT_SCHEMA.md`](docs/model/INPUT_SCHEMA.md): explicit feature semantics and units.
- [`docs/protocols/4G_PREDICTOR_BENCHMARK.md`](docs/protocols/4G_PREDICTOR_BENCHMARK.md): draft benchmark protocol.
- [`studies/predictor/clean_qgeognn/preflight/`](studies/predictor/clean_qgeognn/preflight/README.md): Clean-QGeoGNN engineering preflight.
- [`studies/predictor/clean_4g_baseline_qualification/`](studies/predictor/clean_4g_baseline_qualification/QUALIFICATION_REPORT.md): completed formal 4g qualification and next-gate decision.
- [`experiments/INDEX.md`](experiments/INDEX.md): frozen historical experiment index.

## Code and validation

Reusable scientific code lives in `src/qgeognn_al/`; historical runners and compatibility shims remain under `scripts/`. The validated environment is conda `fish`:

```bash
conda run --no-capture-output -n fish pytest -q
```
