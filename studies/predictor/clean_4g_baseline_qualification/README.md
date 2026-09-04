# Clean-QGeoGNN 4g baseline qualification

Status: `4G_ENGINEERING_QUALIFICATION_COMPLETE / POINT_PERFORMANCE_QUALIFICATION_REOPENED / UQ_PAUSED`.

This study established numerically valid, interpretable 4g engineering and input-contract behavior for `qgeognn_clean_fusion_v1` preflight revision 2. A later controlled E0 regression audit found that its point performance is far below the reproducible Legacy/V2 controls, so this study no longer qualifies Clean as a point-prediction baseline. It does not test statistical superiority over Legacy and trains no other model.

The primary data contract is `LEGACY_THRESHOLD_DOMAIN`: `V1_ml <= 60`, `V2_ml <= 120`, exactly 4,163 rows. This is a `PROJECT_CONTINUITY_DECISION`, not a scientific or instrument-limit claim. Row interpolation and compound generalization each use fixed 80/10/10 manifests at seeds 42, 525, and 1101.

Formal checkpoints and transient state belong under gitignored `runtime/`. Only compact configs, histories, predictions, metrics, diagnostics, checkpoint metadata, and artifact manifests belong under `results/`.

All six formal runs completed and passed the numerical artifact audit. Row-interpolation combined normalized RMSE was 0.736 mean (0.066 sample SD); compound-generalization was 0.839 (0.142). Condition permutation and disabling each worsened both-target RMSE in 6/6 runs, supporting learned condition reliance. These results remain engineering/input-contract evidence, not current point-baseline qualification. The regression audit is the active next-stage gate; UQ, transfer, and active-learning work are paused. No 8g or active-learning work was executed.

See [`QUALIFICATION_REPORT.md`](QUALIFICATION_REPORT.md), [`PREREGISTRATION.md`](PREREGISTRATION.md), `protocol.json`, `data_usage.json`, `decision.json`, and the compact machine-readable artifacts under `results/`.
