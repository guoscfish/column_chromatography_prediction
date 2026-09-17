# Results - QGeoGNN-V2 row AL innovation screen

Status: **selection-only innovation screening complete**.

The screen used only seeds [73, 311, 1297, 4093, 8191], B=32, L0 labels,
frozen L0 checkpoints, and label-free U0 inputs/model outputs. No selected label,
test truth, confirmation result, retraining, or test metric was accessed.

## Mathematical definitions

- `raw_gradient_lcmd`: LCMD-TP on the existing 512D CountSketch of
  `grad(V1_q50/s_V1)` and `grad(V2_q50/s_V2)`.
- `latent_lcmd`: the same LCMD-TP selector on the authoritative 128D pre-head
  QGeoGNN-V2 representation.
- `gradient_farthest_first`: TP greedy k-center on the raw gradient sketch.
- `gradient_norm_topB`: stable descending `||g||_2` ranking.
- `direction_lcmd`: LCMD-TP on `g/||g||` with audited zero vectors retained.
- `tempered_lcmd_alpha_0p5`: LCMD-TP on `g/sqrt(||g||)`.
- `output_whitened_lcmd`: LCMD-TP on gradients after the fixed L0 population
  covariance transform `Sigma_z^(-1/2) diag(1/s_V1,1/s_V2)`.
- `center_width_lcmd`: LCMD-TP on gradients of `(V1+V2)/(2s_C)` and
  `(V2-V1)/s_W`, with L0 population scales.
- `latent_farthest_first`: existing current-V2 CoreSet control.

## Raw compatibility

All five ordered raw Gradient-LCMD B=32 selections exactly match the frozen
historical development selection artifacts. The legacy raw extractor was not
rewritten.

## Selection interpretation

- Latent -> gradient at fixed farthest-first retained 0.194 of each batch on average.
- Farthest-first -> LCMD at fixed gradient retained 0.450.
- Farthest-first -> LCMD at fixed latent retained 0.406.
- Gradient-norm Top-B retained 0.181 of raw Gradient-LCMD.

These contrasts show how strongly selections change under one controlled
representation or selector substitution. They do not identify a causal
performance mechanism.

Full seed-by-method overlaps are in `results/selection_overlap.csv`; profiles
and transform numerics are in `results/method_profiles.csv` and
`results/transform_audit.json`.

## Next-stage candidates

For a separately authorized one-step training comparison, the mechanism-only
shortlist is `direction_lcmd`, `latent_lcmd`, `center_width_lcmd` because these
prespecified methods produced the lowest mean membership overlap with the raw
control. Hidden-label performance played no role in this ranking.

## Infrastructure boundary

The current code now supports arbitrary fixed 2x2 linear output-gradient
coordinates and reusable LCMD/farthest-first controls. Target-IVR still needs a
predictive covariance/influence operator and a candidate-to-target utility
contract. MaxDet still needs a numerically stable incremental log-determinant
selector (with explicit regularization and tie rules). Neither method is
implemented or executed here.
