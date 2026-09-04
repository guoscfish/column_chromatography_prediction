# Clean-QGeoGNN engineering preflight

Model variant: `qgeognn_clean_fusion_v1`.

Status: `IMPLEMENTATION_PREFLIGHT_COMPLETE / FORMAL_PERFORMANCE_UNQUALIFIED`.

## Scope

This preflight verifies typed inputs, source-train-only normalization, internal feature reachability, gradient reachability, shapes, finite outputs, monotonic quantiles, deterministic fixture inference, loading-condition collision separation, diagnostic utilities, and checkpoint metadata. It performs no formal training, architecture selection, 4g test comparison, 8g transfer, or active learning.

The molecular branch consumes atom/bond topology, bond length, bond-angle geometry, and molecular descriptors. It explicitly ignores the historical condition columns still carried by compatibility graph fixtures. The condition branch receives `ea_fraction`, categorical loading solvent, `loading_mass_mg`, and loading-solvent volume once per graph.

## Result

- nominal parameters: 413,732
- requires-grad parameters: 413,732
- gradient-bearing parameters in the multi-fixture backward: 413,732
- forward-unreachable trainable parameters: 0
- normalization rows: 3,330 4g source-train; 0 validation; 0 test; 0 8g
- output: finite `[batch, 6]`, with `q10 <= q50 <= q90` for each target
- repeated eval inference on identical fixtures: exact tensor equality
- same-molecule/same-eluent loading-condition collision: separated in condition latent space

Machine-readable records are `config.json`, `decision.json`, `normalization_audit.json`, `parameter_reachability_audit.json`, `feature_reachability_audit.csv`, `checkpoint_contract.json`, and `determinism_and_shape_audit.json`.

## Scientific boundary

Passing means the candidate is implementable under its declared contract. It does not show lower RMSE, better calibration, better generalization, better transfer, or better active-learning behavior than Legacy QGeoGNN or Predictor V2.

The only recommended next action is to preregister and execute the formal 4g predictor benchmark.
