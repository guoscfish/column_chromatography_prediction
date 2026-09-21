# Short sequential B=32 developmental screen

This screen ran only Center/Width-LCMD and Direction-LCMD for seeds 157 and 6101, with adaptive scratch retraining at 333 -> 365 -> 397 -> 429. Ensemble Uncertainty was explicitly excluded. This is developmental evidence because these historical row-test cohorts have already been inspected.

## Primary results

| Seed | Method | NRMSE@365 | NRMSE@429 | AULC 333-429 |
|---:|---|---:|---:|---:|
| 157 | Center/Width-LCMD | 0.695087 | 0.640645 | 0.696086 |
| 157 | Direction-LCMD | 0.718815 | 0.711268 | 0.735171 |
| 6101 | Center/Width-LCMD | 0.980531 | 0.812892 | 0.913828 |
| 6101 | Direction-LCMD | 0.922994 | 0.908861 | 0.923925 |

Among the two new methods, round 1 is led by **Direction-LCMD**, round 3 by **Center/Width-LCMD**, and AULC by **Center/Width-LCMD**. The round-1 and AULC rankings do not agree, so a crossing/ranking reversal occurred.

Across both new methods and all five matched historical comparators, the two-seed mean leader at 365 is **Gradient-MaxDet**, at 429 is **Hybrid**, and over AULC is **Gradient-MaxDet**. These are descriptive two-seed rankings, not significance tests.

The earlier development one-step Center/Width negative signal (about 0.728810 versus raw Gradient-LCMD 0.708760) does not hold uniformly here: CW beats Gradient-LCMD on both seeds' 333-429 AULC and @429, but trails the stronger Gradient-MaxDet on both seeds for both measures. In seed 6101, CW starts behind Direction at 365 (0.980531 versus 0.922994) and overtakes it at 429 (0.812892 versus 0.908861); seed 157 already favors CW at 365. Thus a one-step comparison does not reliably capture their short sequential ranking.

## Paired differences

Positive values mean the new method is worse.

| Seed | Method | Comparator | AULC delta | NRMSE@429 delta |
|---:|---|---|---:|---:|
| 157 | Center/Width-LCMD | Gradient-LCMD | -0.028820 | -0.044598 |
| 157 | Center/Width-LCMD | Gradient-MaxDet | +0.009547 | +0.027630 |
| 157 | Direction-LCMD | Gradient-LCMD | +0.010264 | +0.026025 |
| 157 | Direction-LCMD | Gradient-MaxDet | +0.048632 | +0.098253 |
| 6101 | Center/Width-LCMD | Gradient-LCMD | -0.010841 | -0.056031 |
| 6101 | Center/Width-LCMD | Gradient-MaxDet | +0.069938 | +0.018931 |
| 6101 | Direction-LCMD | Gradient-LCMD | -0.000744 | +0.039938 |
| 6101 | Direction-LCMD | Gradient-MaxDet | +0.080035 | +0.114900 |

Two-seed descriptive mean deltas:

- Center/Width-LCMD minus Gradient-MaxDet: AULC +0.039743; NRMSE@429 +0.023281.
- Center/Width-LCMD minus Gradient-LCMD: AULC -0.019831; NRMSE@429 -0.050315.
- Direction-LCMD minus Gradient-MaxDet: AULC +0.064333; NRMSE@429 +0.106577.
- Direction-LCMD minus Gradient-LCMD: AULC +0.004760; NRMSE@429 +0.032981.

Direction does not translate its different first-batch geometry into a stable predictive gain: its AULC is worse than Gradient-MaxDet on both seeds and worse than Gradient-LCMD in seed 157 (only marginally better in seed 6101), while its @429 NRMSE is worse than both baselines on both seeds.

## Selection overlap

Mean intersection per batch over six seed-round pairs (per-pair detail is in `results/selection_overlap.csv`):

- Center/Width-LCMD vs Direction-LCMD: 0.83/32 (0.026).
- Center/Width-LCMD vs Gradient-MaxDet: 5.33/32 (0.167).
- Center/Width-LCMD vs Gradient-LCMD: 6.33/32 (0.198).
- Direction-LCMD vs Gradient-MaxDet: 1.17/32 (0.036).
- Direction-LCMD vs Gradient-LCMD: 0.83/32 (0.026).

CW and Direction choose notably different batches. Neither low nor high overlap establishes superiority without the paired predictive results above.

## Continue/stop assessment

- Center/Width-LCMD: **STOP** versus Gradient-MaxDet (AULC deltas {'157': 0.009547438371950379, '6101': 0.06993782475778942}; endpoint deltas {'157': 0.027630130218013638, '6101': 0.018931414070145536}).
- Direction-LCMD: **STOP** versus Gradient-MaxDet (AULC deltas {'157': 0.04863190622472957, '6101': 0.08003502437637877}; endpoint deltas {'157': 0.09825274050799448, '6101': 0.11490038034405581}).

No method is automatically continued to 525.

The explicit STOP rule takes precedence when both seeds have worse AULC and @429 than the strongest baseline, even if a method improves over the weaker Gradient-LCMD. Ensemble Uncertainty was not run, so its sequential conclusion and cost cannot be evaluated by this study.

## Compute cost

- Center/Width-LCMD: 1387.6s new training, 349.7s gradient extraction, 0.575s selection.
- Direction-LCMD: 1526.2s new training, 326.4s gradient extraction, 0.533s selection.

These costs describe computation, not label efficiency. Both methods use one reused round-0 fit and three new evaluation fits per seed; neither runs an ensemble.

## Integrity and interpretation

Every acquisition used the current trajectory-specific checkpoint, recomputed its representation, installed all current L_t rows as LCMD centers, selected exactly 32 unique current-U_t IDs, and revealed their labels only after selection was frozen. All 16 prediction points were frozen before test truth was read. Comparator tables and batch IDs were read only from exact seed/budget historical trajectories. Low selection overlap establishes a different strategy, not predictive superiority.
