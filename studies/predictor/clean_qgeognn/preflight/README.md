# Clean-QGeoGNN engineering preflight

Model variant: `qgeognn_clean_fusion_v1`.

Status: `IMPLEMENTATION_PREFLIGHT_REVISION_2_COMPLETE / FORMAL_PERFORMANCE_UNQUALIFIED`.

Revision 2 supersedes revision 1 without deleting its Git history. Reason: `remove_unnecessary_molecular_activation_confound_and_align_output_contract_before_formal_benchmark`.

## Scope

This preflight verifies typed inputs, source-train-only normalization, internal feature reachability, gradient reachability, shapes, finite outputs, monotonic quantiles, deterministic fixture inference, loading-condition collision separation, diagnostic utilities, and checkpoint metadata. It performs no formal training, architecture selection, 4g test comparison, 8g transfer, or active learning.

The molecular branch consumes atom/bond topology, bond length, bond-angle geometry, and `official_code_molecular_descriptor_16`. It uses ReLU between message-passing layers, matching the effective official path and removing an unnecessary activation confound. It explicitly ignores the historical condition columns still carried by compatibility graph fixtures. The GELU condition branch receives `ea_fraction`, categorical loading solvent, `loading_mass_mg`, and loading-solvent volume once per graph.

The head uses a softplus median and softplus lower/upper offsets. During training it preserves that parameterization (ordered q10/q50/q90, while q10 may be negative); during evaluation it clamps every output to `[0, 1e8]`, matching the current cleaned Legacy inference policy.

## Result

- nominal parameters: 413,732
- requires-grad parameters: 413,732
- gradient-bearing parameters in the multi-fixture backward: 413,732
- forward-unreachable trainable parameters: 0
- revision-1 parameter count: 413,732; revision-2 parameter count: 413,732 (unchanged)
- normalization rows: 3,330 4g source-train; 0 validation; 0 test; 0 8g
- output: finite `[batch, 6]`, with `q10 <= q50 <= q90` for each target and non-negative evaluation outputs
- repeated eval inference on identical fixtures: exact tensor equality
- same-molecule/same-eluent loading-condition collision: separated in condition latent space

Machine-readable records are `config.json`, `decision.json`, `normalization_audit.json`, `parameter_reachability_audit.json`, `gradient_reachability_audit.json`, `feature_reachability_audit.csv`, `checkpoint_contract.json`, `determinism_and_shape_audit.json`, `environment.json`, and `test_report.json`.

## Scientific boundary

Passing means the candidate is implementable under its declared contract. It does not show lower RMSE, better calibration, better generalization, better transfer, or better active-learning behavior than Legacy QGeoGNN or Predictor V2.

The candidate contracts are `official_code_first_embedded_for_controlled_comparison`, `official_code_molecular_descriptor_16`, and `clean_typed_sample_level_v1`; `paper_method_equivalent` is false. Lowest-energy geometry and paper-text TPSA remain separate future sensitivity factors.

The formal benchmark is preregistered but remains unauthorized. A separate authorization is required before any performance execution.
