# Frozen protocol: pure CW-LCMD L525 to L653

- Evidence class: developmental continuation.
- Seeds: 157 and 6101 only.
- Source: exact frozen Center/Width-LCMD L525 state from `qgeognn_v2_row_cw_lcmd_to_525`.
- Budgets: 525, 557, 589, 621, 653; batch size 32; hard stop at 653.
- Work: four current-checkpoint acquisitions and four scratch QGeoGNN-V2 fits per seed.
- Method: frozen full-network Center/Width gradients, CountSketch-512, existing sketch seed, LCMD-TP, all current L_t centers, current U_t candidates.
- Center/Width scales: original L0 population scales; no whitening, fusion, uncertainty, MaxDet, or tuning.
- Comparators: historical read-only reuse; no comparator training.
- Test firewall: all eight new checkpoint/prediction pairs must be globally frozen before one final test evaluation.
- Primary metric: normalized trapezoidal combined-NRMSE AULC 525-653.
- Secondary metrics: AULC 429-653 and complete-grid AULC 333-653.
- Forbidden: continuation to 685/1005 and MaxDet-to-CW switching.

## Frozen decision implementation

- `STRONG_SUSTAINED_CW`: CW AULC 525-653 is lower for both seeds than each of Gradient-MaxDet, Hybrid, and Kernel-IVR, and for neither seed is CW NRMSE@653 more than `+0.03` above the best of those three endpoints.
- `SUSTAINED_AGAINST_SOME`: the strong rule is not met, but at least one of those three comparators has negative CW-minus-comparator AULC 525-653 deltas in both seeds.
- `STOP_CW`: for every strong comparator both seed AULC deltas exceed `+0.03`, and CW is not the best strong-comparator endpoint in either seed.
- `MIXED_LATE_SIGNAL`: neither prior rule applies and at least one strong comparator has opposite seed directions with both absolute AULC deltas greater than `0.01`.
- `MIDSTAGE_ONLY`: none of the above applies; no strong comparator shows a two-seed late CW advantage.

These thresholds and precedence are frozen before any new L557-L653 test metric is revealed.
