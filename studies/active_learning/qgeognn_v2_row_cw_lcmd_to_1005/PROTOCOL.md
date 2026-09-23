# Frozen protocol: pure CW-LCMD L653 to L1005

- Developmental continuation; seeds 157 and 6101 only.
- Exact source: frozen pure-CW L653 state from `qgeognn_v2_row_cw_lcmd_to_653`.
- Budgets: 653, 685, 717, 749, 781, 813, 845, 877, 909, 941, 973, 1005.
- Eleven B=32 acquisitions and eleven scratch QGeoGNN-V2 fits per seed; no acquisition at L1005.
- Frozen Center/Width full-network gradients, original L0 population scales, CountSketch-512, existing sketch seed, LCMD-TP, all current L_t centers, current U_t candidates.
- No warm start, tuning, uncertainty, whitening, fusion, MaxDet, comparator training, or method search.
- Test firewall: all 22 new checkpoint/prediction pairs must be globally frozen before one final test evaluation.
- Primary metric: normalized trapezoidal combined-NRMSE AULC 653-1005.
- Secondary: AULC 525-1005, 429-1005, and complete 22-point AULC 333-1005.
- Hard stop: L1005.

## Frozen decision rule

- `SUSTAINED_TO_1005`: both seeds beat MaxDet, Hybrid, and IVR on AULC 653-1005.
- `SUSTAINED_AGAINST_SOME`: both seeds beat at least one of those three.
- `STOP_CW`: both seeds lose to all three by more than `0.03` and have no endpoint advantage.
- `MIXED_LONG_HORIZON`: neither prior rule applies and a strong comparator has opposite seed directions with both absolute deltas above `0.01`.
- `NO_CONSISTENT_ADVANTAGE`: none of the preceding rules applies.

The rule and thresholds are frozen before any new L685-L1005 test metric is revealed.
