# QGeoGNN point-prediction performance regression audit

Status: `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC / PROTOCOL_FROZEN / RUN_PENDING`.

This study locates the first controlled step at which the historical E0 Legacy point-prediction performance regresses. It reuses `experiments/e0_4g_baseline/split_seed_42.csv` byte-for-byte and does not generate a split. The ladder is R0 Legacy exact, R1 Legacy with the Clean training protocol, R2 Condition Completion V2 with that protocol, and R3 current Clean-QGeoGNN with that protocol.

R0 must reproduce both historical test R² values within absolute tolerance 0.03 before R1–R3 may run. All test outcomes are developmental diagnostics and are evaluated only after validation checkpoint selection. No UQ calibration, 8g access, transfer, active learning, hyperparameter sweep, architecture search, or model repair is authorized.

Runtime checkpoints and resume state are gitignored under `runtime/`. Compact formal diagnostic artifacts belong under `results/`.
