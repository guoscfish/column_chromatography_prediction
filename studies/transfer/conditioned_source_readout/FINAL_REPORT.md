# Final report — conditioned source readout

## Decision

`NO_MATERIAL_ARCHITECTURE_GAIN`

The preregistered ROW inner-CV gate found no eligible R1/R2 readout arm. Consequently this terminal report was created without reading outer validation/test truth, without making new outer predictions, and without running COMPOUND confirmation.

## ROW inner-screen evidence

| arm | column | mean relative improvement (%) | fold wins | seed wins | gate |
| --- | --- | ---: | ---: | ---: | --- |
| R1 | 25g | -0.019 | 9/25 | 1/5 | fail |
| R1 | 40g | 0.695 | 22/25 | 5/5 | fail |
| R2 | 25g | 0.048 | 11/25 | 3/5 | fail |
| R2 | 40g | 0.717 | 23/25 | 5/5 | fail |

## Outer-score artifact status

`ROW_RESULTS.csv`, `COMPOUND_RESULTS.csv`, `PAIRED_COMPARISON.csv`, and `BOOTSTRAP_CI.csv` are intentionally header-only. They record no synthetic or partially observed score; the machine-readable stopping decision and `prediction_hashes.json` attest that the outer-test score boundary was not crossed.

## Required questions

1. **Is fixed sum pooling a transfer bottleneck?** No qualifying evidence under the preregistered ROW inner screen.
2. **Does adaptive readout (R1) improve stably?** No under the fixed gate.
3. **Does condition-query readout (R2) improve over R1/R0?** No under the fixed gate.
4. **Does source prediction (R3) add benefit?** Not run: R2 did not authorize source-prediction augmentation.
5. **Does source embedding (R4) add incremental benefit?** Not run: R3 was not authorized.
6. **Is improvement present in both 25g and 40g?** No arm met the two-column inner-CV rule.
7. **Is improvement present in both ROW and COMPOUND?** Not assessed; COMPOUND is correctly blocked by the ROW stopping rule.
8. **Does the new model exceed paper-style/P0/structured baselines?** Not assessed: no outer-test score was read.
9. **Is any improvement driven by one endpoint?** Not assessed on outer truth; no endpoint-level outer score exists.
10. **Does the study meet the promotion gate?** No — `NO_MATERIAL_ARCHITECTURE_GAIN`.

## Reproducibility

Stopping-decision SHA256: `9c3ca3605999d7e1c2869a73994de888363c394f7b25f97423ae9d41666ef64d`. The inner screen remains in `INNER_SCREEN_RESULTS.csv`; protocol and frozen split provenance remain in `protocol.json` and `run_manifest.csv`. This report action performs no endpoint-data read.
