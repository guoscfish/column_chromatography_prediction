# Protocol — QGeoGNN-V2 Gradient-LCMD-TP

The preregistration is the frozen scientific contract. The executable protocol is isolated under `src/qgeognn_al/active_learning_v2/` and `scripts/studies/run_qgeognn_v2_4g_row_lcmd_pilot.py`; legacy `acquisition.py` is not modified.

The feature adapter evaluates V1 q50 and V2 q50 gradients at the L0 checkpoint, divides each endpoint by its L0-only label scale, concatenates both full-network gradients conceptually, and applies one fixed CountSketch into 512 dimensions. It stores only an `N × 512` matrix and one row-gradient workspace. The LCMD implementation first assigns every U0 point to its nearest L0 center, repeatedly chooses the cluster with the largest sum of squared distances, and chooses that cluster's farthest point.

Covariate min/max preprocessing uses only outer-training graph and condition inputs, which are label-free and identical for every arm. Endpoint scales use L0 truth only. Training and validation graphs receive only their authorized labels; all other graph labels are zero sentinels. Test predictions are written and hashed before the test-label gate can open.

The performance preflight must pass finite/noncollapsed gradients, repeatability, exact batch size, uniqueness, role isolation, and a complete seed-73 U0+L0 extraction benchmark. If full-network extraction is infeasible, the study terminates with `BLOCKER_REPORT.md`; a head-only diagnostic cannot answer the primary question.
