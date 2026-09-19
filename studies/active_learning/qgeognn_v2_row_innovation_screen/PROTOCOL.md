# Protocol

## Existing implementation summary

- Authoritative predictor: standalone `QGeoGNNV2`, 458,952 parameters, five
  node layers/four edge updates, typed condition residual, 128D
  `global_add_pool(backbone) + condition_branch` representation, and a six
  quantile-output head.
- Raw Gradient-LCMD feature: the two full-network gradients of
  `V1_q50/s_V1` and `V2_q50/s_V2`, mapped through the two fixed halves of a
  seeded 512D CountSketch and summed. Endpoint scales are L0 population
  standard deviations. One row gradient is materialized at a time.
- Split: canonical row order, 333 L0 rows and 2,997 U0 rows inside the 3,330-row
  outer train. L0 precedes U0 in every feature matrix. Validation/test rows are
  excluded from acquisition.
- LCMD-TP: install all L0 points as centers; assign U0 to nearest centers;
  repeatedly choose the cluster with largest sum of squared nearest-center
  distances and add its farthest available point. `argmax` gives canonical-order
  stable ties.
- CoreSet: existing `coreset_tp_select`, greedy farthest-first Euclidean
  k-center with all L0 rows installed before selection and each selected U0 row
  immediately added as a center.
- Safe reuse: development-only split CSVs, label-scrubbed graph tensors, L0
  checkpoints, raw gradient caches/receipts, label-free ensemble pool
  predictions, and prior B32 selected-ID tables. Frozen historical files remain
  read-only.

## Fixed methods

Batch size is 32. Every method uses the same seed, split, L0, U0, checkpoint,
CountSketch seed where applicable, Euclidean distance and canonical tie order.

1. `latent_farthest_first`: existing current-V2 CoreSet control.
2. `latent_lcmd`: LCMD-TP on the unnormalized authoritative 128D representation.
3. `gradient_farthest_first`: farthest-first on the raw 512D gradient sketch.
4. `raw_gradient_lcmd`: unchanged historical raw Gradient-LCMD control.
5. `gradient_norm_topB`: stable descending raw sketch norm.
6. `direction_lcmd`: LCMD-TP on `g/||g||`; exact zero rows remain zero and are audited.
7. `tempered_lcmd_alpha_0p5`: LCMD-TP on `g/sqrt(||g||)`.
8. `output_whitened_lcmd`: define `z=[V1/s_V1,V2/s_V2]`, compute L0 population
   covariance, floor eigenvalues at `max(1e-8, 1e-6*mean(eigenvalues))`, and use
   `Sigma_z^(-1/2) diag(1/s_V1,1/s_V2)` as the output-gradient transform.
9. `center_width_lcmd`: use output coordinates `(V1+V2)/(2s_C)` and
   `(V2-V1)/s_W`, where both population scales use L0 labels only.

No alpha sweep, threshold tuning, output-loss change, predictor change, or
post-selection training is permitted.

## Label and artifact firewall

Only seeds 73, 311, 1297, 4093 and 8191 are accepted. Confirmation seeds 157,
887, 2357, 6101 and 12203 raise before artifact or label loading. Each seed's
partition is checked against canonical identities. `RestrictedLabelStore`
receives one request containing L0 IDs only. U0 inputs, model predictions,
gradients and representations are allowed; U0/test labels and all confirmation
metrics are forbidden.

The external artifact repository is explicit and read-only. Every source cache
receipt, content hash, checkpoint binding, row identity and sketch seed is
verified. Raw B32 selected IDs must exactly match the frozen development table.
New transformed caches live only under this study's ignored `runtime/`.

## Outputs and stop rule

Committed outputs are `selected_batches.csv`, `selection_overlap.csv`,
`method_profiles.csv`, `transform_audit.json`, label-access/raw-compatibility
audits, three selection-mechanism figures, `mechanism_summary.md`, `RESULTS.md`,
environment metadata and an artifact manifest.

The terminal status is **selection-only innovation screening complete**. The
runner cannot reveal selected labels, retrain, evaluate test truth, launch
sequential AL, access confirmation results, or execute Target-IVR/MaxDet/BAIT.
