# Predictor V2 implementation preflight

Status: `IMPLEMENTATION_PREFLIGHT_COMPLETE`.

This is an engineering result, not a predictor-performance experiment. No formal 4g training, validation selection, 8g outcome access, transfer run, or active transfer was performed. The implementation is an engineering candidate and is not the selected final predictor.

## Implemented candidate

`qgeognn_condition_complete_v2` wraps a normally loaded legacy QGeoGNN. The legacy GNN and fixed sum pooling produce the 128D `h_G`. A separate typed branch reads only the five conditions that I0 found unreachable: eluent HBA, eluent LogP, loading solvent, loading amount (`Density * V`), and loading-solvent volume.

Loading solvent uses a 3-class (`PE`, `EA`, `DCM`) 4D embedding. HBA and LogP reuse dimensions 4 and 5 of the frozen source-train eluent min-max scaler. Loading amount and volume use a min-max scaler fit only on the 3,330 rows assigned `train` in the frozen 4g seed-42 source split. The resulting four continuous values and 4D solvent embedding enter a `Linear(8,16) -> ReLU -> Linear(16,128)` branch. The output layer is zero-initialized and the residual is added after sum pooling and before the unchanged prediction head.

The branch adds 2,332 parameters, giving 777,808 nominal V2 parameters. It does not feed loading values to the legacy RBF encoders and adds no column-specification input.

## Gate result

- All source members 42/525/1101 loaded. Across ten distinct 4g molecules with multiple eluent ratios and all three loading solvents, initialized V2 predictions were bit-identical to legacy predictions (`max_abs_difference=0.0`, tolerance `1e-7`).
- All nine intended conditions have a forward path. Each missing condition changes the branch representation; after fixed nonzero diagnostic activation, each can change prediction.
- All branch parameters had nonzero gradients by the second synthetic optimizer step.
- A real same-molecule/same-eluent loading collision had identical legacy embedding/prediction but different V2 condition representation.
- The schema and branch hashes are deterministic. The V2 loader requires the complete checkpoint metadata contract, including `input_schema_hash`; historical checkpoints remain loadable without V2 metadata.
- Multi-fixture legacy classification remained 456,620 gradient-bearing, 318,856 structurally unreachable, and 0 fixture-dependent inactive parameters. Diagnostic V2 activation added all 2,332 branch parameters to the gradient-bearing category.

Detailed evidence is stored in the JSON/CSV files in this directory. `decision.json` keeps formal source qualification unauthorized. Full repository tests are a separate completion condition and are not encoded as a scientific metric here.
