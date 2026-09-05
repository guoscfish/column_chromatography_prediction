# R2-pruned controlled requalification

Status: `FUNCTION_PRESERVING_PARAMETER_CLEANUP_SUCCESS`.

Mainline: Legacy historical → Condition Completion V2 → R2-pruned candidate baseline.

## Reachability

| Model | Nominal | Requires grad | Gradient-bearing | Unreachable |
|---|---:|---:|---:|---:|
| R2 | 777808 | 777808 | 458952 | 318856 |
| R2-pruned | 458952 | 458952 | 458952 | 0 |

Removed registrations (parameter counts exclude buffers):

| Module | Removed parameters |
|---|---:|
| `legacy_model.NN_descriptor` | 250496 |
| `legacy_model.gnn_node.bond_angle_encoder` | 6401 |
| `legacy_model.gnn_node.batch_norms` | 1280 |
| `legacy_model.gnn_node.batch_norms_ba` | 1280 |
| `legacy_model.gnn_node.convs_bond_angle.4` | 33281 |
| `legacy_model.gnn_node.convs_bond_embeding.4` | 6272 |
| `legacy_model.gnn_node.convs_bond_float.4` | 13445 |
| `legacy_model.gnn_node.convs_angle_float.4` | 6401 |

The terminal edge update executes in R2, but its result cannot reach a later node update or the prediction. Other removed modules are never called in the geometry-enhanced prediction path. Static trace, forward hooks, and three real multi-molecule/multi-condition backward batches agree. Gradient-bearing means `grad is not None`, including mathematically reachable parameters with zero-valued gradients. Counts describe registered parameters, not unregistered Legacy RBF tensor attributes.

Five node layers, four effective geometry updates, bond length, descriptor geometry path, early eluent interaction, sum pooling, 128D representation, typed completion branch and Linear(128,6)+ReLU head are retained.

## Function and initialization gates

P0 and P1 cover all 4,163 real rows, 217 molecules, PE/EA/DCM loading solvents, varying loading masses/volumes and eluent compositions. Each split's six-output maximum absolute difference is 0, and all point metric differences are 0. The trained source checkpoint SHA-256 is `2c8bbb738b7e163b53bc80786747edf661df08e150103b0bc7611d9240456072`.

Mapped initial-state hash: `bc967d4a50f53c88a8c14e0c38ce9a1fe4ba8f0f09fa3a42899fd9d257b73b37`. All retained initial values and parameter traversal order are exact; construction preserves the canonical post-construction RNG state. Two real Adam steps have zero retained-state differences.

Legacy RBF centers are non-leaf unregistered tensor views, so generic deepcopy fails. The constructor rebuilds an independent source inside an isolated RNG context, loads the full canonical state, and transfers only retained modules. The resulting model registers no deleted parameters; original R2 remains intact.

## Controlled retraining

Best epoch: **162**; total epochs: **262**. Frozen split SHA-256: `9a758e115c63cc9de2491d483b224d2e4c4b88fd6aadabaf7a45d4e73263b198`. Train/validation/test: 3330/416/417; thresholds V1 ≤ 60, V2 ≤ 120. Adam, lr 0.001, weight decay 0, batch 2048, seed 42, maximum 1000 epochs, patience 100, epoch-deterministic shuffle, equal target loss weights and train-only normalization match R2. Only validation combined normalized RMSE selects the checkpoint; test is evaluated after selection.

| Split | V1 R² | V1 RMSE | V1 MAE | V2 R² | V2 RMSE | V2 MAE | Combined normalized RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.935359299 | 1.976880789 | 1.041515946 | 0.967799962 | 2.867027283 | 1.615695000 | 0.220046318 |
| validation | 0.858151913 | 2.793196440 | 1.300172329 | 0.883010030 | 5.229596615 | 2.365057468 | 0.343643388 |
| test | 0.889217973 | 2.711170435 | 1.374214053 | 0.941562533 | 3.854372978 | 2.135871887 | 0.299813310 |

All metric deltas versus the full-precision historical R2 artifacts: `{'train': {'V1_rmse': 0.0, 'V1_mae': 0.0, 'V1_r2': 0.0, 'V2_rmse': 0.0, 'V2_mae': 0.0, 'V2_r2': 0.0, 'combined_normalized_rmse': 0.0}, 'validation': {'V1_rmse': 0.0, 'V1_mae': 0.0, 'V1_r2': 0.0, 'V2_rmse': 0.0, 'V2_mae': 0.0, 'V2_r2': 0.0, 'combined_normalized_rmse': 0.0}, 'test': {'V1_rmse': 0.0, 'V1_mae': 0.0, 'V1_r2': 0.0, 'V2_rmse': 0.0, 'V2_mae': 0.0, 'V2_r2': 0.0, 'combined_normalized_rmse': 0.0}}`.

Entire training history exactly equal: `True`. Retrained retained-checkpoint maximum difference: `0.0`. Retrained six-output prediction maximum difference: `0.0`.

## Decision

`POINT_PREDICTOR_CANDIDATE_BASELINE`; this is a candidate, not a formally qualified UQ/baseline contract. Dead trainable registrations can be removed while reproducing the effective R2 model; they contributed no predictive ability. The next separate controlled study should change only the quantile head. No head alternative, UQ, 8g, transfer, AL, sweep, or Clean repair was executed.

`qgeognn_clean_fusion_v1`: `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`. Engineering reachability PASS; performance FAIL. Preserve it and all historical R0/R1/R2/R3 results for regression provenance. No further MLP/FiLM/LayerNorm/bottleneck repair is part of this mainline.

Validation details and the complete pytest output are stored in `test_report.json` and `results/pytest_output.txt`.
