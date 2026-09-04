# Clean-QGeoGNN 4g baseline qualification

Status: `PROTOCOL_FROZEN / SPLITS_PENDING / FORMAL_RUN_PENDING`.

This study asks whether `qgeognn_clean_fusion_v1` preflight revision 2 has stable, numerically valid, interpretable 4g behavior suitable for downstream UQ, transfer, and active-learning work. It does not test statistical superiority over Legacy and trains no other model.

The primary data contract is `LEGACY_THRESHOLD_DOMAIN`: `V1_ml <= 60`, `V2_ml <= 120`, exactly 4,163 rows. This is a `PROJECT_CONTINUITY_DECISION`, not a scientific or instrument-limit claim. Row interpolation and compound generalization each use fixed 80/10/10 manifests at seeds 42, 525, and 1101.

Formal checkpoints and transient state belong under gitignored `runtime/`. Only compact configs, histories, predictions, metrics, diagnostics, checkpoint metadata, and artifact manifests belong under `results/`.

See [`PREREGISTRATION.md`](PREREGISTRATION.md), `protocol.json`, `data_usage.json`, and `decision.json`.
