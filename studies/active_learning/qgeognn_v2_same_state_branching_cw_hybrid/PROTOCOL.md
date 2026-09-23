# Frozen protocol: CW/Hybrid same-state branching

## Evidence class

Development / mechanism pilot, not final confirmation. No controller, bandit, RL, threshold search, or adaptive model is trained.

## Exact design

Eight anchors (`2 seeds × 2 sources × 2 budgets`) branch into three strategies. Each branch acquires one batch of 32, reveals only the selected batch, scratch-fits the fixed QGeoGNN-V2 protocol, recomputes label-free features, and repeats once. Endpoints are 461/493 or 685/717. Planned new branch fits: 48.

Primary metric is short-AULC over anchor→+64. Secondary metrics are NRMSE at +32/+64, winner counts, paired short-AULC deltas, and local-oracle gain over always-CW.

## Firewall

Historical source checkpoints and states are read-only. Branch stores are isolated. Selection receives no labels. Every acquisition, checkpoint, prediction, and audit file is recursively hashed before `global_pre_test_freeze.json` authorizes the single unified test reveal. `test_label_access_audit.csv` is mandatory.

## Compatibility note

The historical Hybrid source trajectory uses K=3 uncertainty shortlist plus latent k-center. Its artifacts are reused only for the source state. For new branch selection, the frozen compatibility implementation uses a current gradient-norm top-25% shortlist plus current-L farthest-first. This avoids extra branch fitting while preserving the mechanism contrast; it is explicitly treated as development evidence.

## Diagnostics

At each anchor and first branch state record gradient norm mean/std/p90/p95, participation-ratio effective rank, nearest-current-L coverage distances, recent validation NRMSE and slopes, IVR predicted reducible variance, and strategy preference correlations. Correlations are descriptive only (`n=8`).
