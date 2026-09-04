# Column chromatography prediction

This repository studies retention-volume prediction for column chromatography, with emphasis on molecular geometry, experimental conditions, data-efficient source-to-target adaptation, and active learning.

## Current scientific state

| Direction | Status | Boundary |
|---|---|---|
| Predictor | `UNDER_REDESIGN_AND_REQUALIFICATION` | Legacy is frozen historical evidence; Predictor V2 is an engineering-qualified ablation; Clean-QGeoGNN v1 has completed implementation preflight but remains performance-unqualified. |
| 4g in-domain active learning | `PAUSED / historical evidence retained` | E2 and A1a results remain measured historical evidence. No new acquisition run is authorized. |
| 4g -> 8g transfer | `PAUSED_PENDING_PREDICTOR_QUALIFICATION` | T1/T1b-1 remain valid under the audited legacy predictor contract; a new predictor must first qualify on 4g. |
| Active transfer | `DEFERRED` | It remains downstream of predictor and transfer qualification. |

The predictor is being requalified because the I0 semantic audit found that the legacy graph builder constructs ten continuous edge features while the legacy encoder consumes only five. It also found a substantial gap between nominal and gradient-bearing parameter counts. These facts narrow the interpretation of historical evidence; they do not erase the measured results.

## Next step

The next scientific action is to preregister and execute a formal 4g predictor benchmark comparing the frozen Legacy QGeoGNN, the condition-completion ablation, Clean-QGeoGNN, and a strong ANN/MLP baseline. The benchmark has not been authorized or run.

## Start reading

- [`docs/README.md`](docs/README.md): documentation map and current contracts.
- [`studies/README.md`](studies/README.md): semantic navigation for current and historical studies.
- [`docs/model/LEGACY_QGEOGNN_AUDIT.md`](docs/model/LEGACY_QGEOGNN_AUDIT.md): audited legacy implementation boundary.
- [`docs/model/INPUT_SCHEMA.md`](docs/model/INPUT_SCHEMA.md): explicit feature semantics and units.
- [`docs/protocols/4G_PREDICTOR_BENCHMARK.md`](docs/protocols/4G_PREDICTOR_BENCHMARK.md): draft benchmark protocol.
- [`studies/predictor/clean_qgeognn/preflight/`](studies/predictor/clean_qgeognn/preflight/README.md): Clean-QGeoGNN engineering preflight.
- [`experiments/INDEX.md`](experiments/INDEX.md): frozen historical experiment index.

## Code and validation

Reusable scientific code lives in `src/qgeognn_al/`; historical runners and compatibility shims remain under `scripts/`. The validated environment is conda `fish`:

```bash
conda run --no-capture-output -n fish pytest -q
```
