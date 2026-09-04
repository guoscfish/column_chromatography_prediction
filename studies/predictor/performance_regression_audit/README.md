# QGeoGNN point-prediction performance regression audit

Status: `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC_COMPLETE / PROTOCOL_FROZEN / ARTIFACT_AUDIT_PASS`.

This study locates the first controlled step at which the historical E0 Legacy point-prediction performance regresses. It reuses `experiments/e0_4g_baseline/split_seed_42.csv` byte-for-byte and does not generate a split. The ladder is R0 Legacy exact, R1 Legacy with the Clean training protocol, R2 Condition Completion V2 with that protocol, and R3 current Clean-QGeoGNN with that protocol.

R0 must reproduce both historical test R² values within absolute tolerance 0.03 before R1–R3 may run. All test outcomes are developmental diagnostics and are evaluated only after validation checkpoint selection. No UQ calibration, 8g access, transfer, active learning, hyperparameter sweep, architecture search, or model repair is authorized.

Runtime checkpoints and resume state are gitignored under `runtime/`. Compact formal diagnostic artifacts belong under `results/`.

The completed ladder found the first regression at `R2_CONDITION_COMPLETE_V2 -> R3_CLEAN_CURRENT`: R0/R1/R2 test R2 values remained approximately 0.866-0.942, while R3 reached 0.525 (V1) and 0.607 (V2). The leading supported mechanism is loss of Legacy early molecule-condition interaction together with additive late fusion; the 128D -> 64D projection, LayerNorm, and softplus head remain plausible but unisolated contributors. Clean point-performance qualification is reopened.
