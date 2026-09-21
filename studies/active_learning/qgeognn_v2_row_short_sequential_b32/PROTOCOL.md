# Protocol

- Split: ROW.
- Seeds: 157 and 6101.
- New methods: Center/Width-LCMD and Direction-LCMD.
- Budgets: 333, 365, 397, and 429 active labels.
- Batch size: 32; exactly three adaptive acquisitions.
- Hard stop: 429. No continuation to 461, 525, 653, or 1005.
- Model and training: historical matched QGeoGNN-V2 scratch-retraining protocol.
- Center/Width: gradients of `(V1+V2)/(2*s_C)` and `(V2-V1)/s_W`, with fixed L0 population scales.
- Direction: existing raw scaled q50 CountSketch followed by `g/||g||`, retaining zero rows.
- Selector: existing LCMD-TP, with every current labeled row installed as a center.
- Primary metric: normalized trapezoidal combined-NRMSE AULC over 333--429.
- Test firewall: all 16 method/seed/budget predictions freeze before test truth access.
- Evidence status: truncated developmental screen, not independent confirmation.
