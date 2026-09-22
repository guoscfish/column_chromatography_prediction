# Pure Center/Width-LCMD continuation: L525 to L653

This developmental continuation starts from each seed's exact frozen pure-CW L525 state. It performs four current-checkpoint CW acquisitions and four scratch QGeoGNN-V2 fits per seed, then hard-stops at L653. Historical comparators are read-only; no other acquisition method was trained. Test truth remained inaccessible until all eight new checkpoint/prediction pairs were globally frozen.

## Primary AULC 525-653

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.506915 | 0.514650 | 0.584167 | 0.524261 | 0.590839 |
| 6101 | 0.660010 | 0.732008 | 0.639155 | 0.722848 | 0.634775 |
| Mean | 0.583463 | 0.623329 | 0.611661 | 0.623554 | 0.612807 |

## AULC 429-653

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.529641 | 0.553571 | 0.595492 | 0.549356 | 0.617654 |
| 6101 | 0.694351 | 0.747294 | 0.688691 | 0.734648 | 0.707751 |
| Mean | 0.611996 | 0.650433 | 0.642092 | 0.642002 | 0.662703 |

## AULC 333-653

Every method/seed integral contains exactly the 11 preregistered points from 333 through 653.

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.579574 | 0.593462 | 0.626214 | 0.604759 | 0.649830 |
| 6101 | 0.760194 | 0.776273 | 0.746821 | 0.762930 | 0.772826 |
| Mean | 0.669884 | 0.684867 | 0.686518 | 0.683845 | 0.711328 |

## Fixed-budget endpoint

| Seed | Method | NRMSE | V1 RMSE | V1 MAE | V1 R2 | V2 RMSE | V2 MAE | V2 R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 157 | Center/Width-LCMD | 0.494809 | 3.660338 | 1.700103 | 0.757962 | 5.470631 | 2.815812 | 0.868962 |
| 157 | Gradient-MaxDet | 0.484709 | 3.669656 | 1.876473 | 0.756728 | 5.120744 | 2.882280 | 0.885187 |
| 157 | Hybrid | 0.576106 | 4.478396 | 2.044550 | 0.637685 | 5.728562 | 2.790783 | 0.856314 |
| 157 | Kernel-IVR | 0.520384 | 4.058873 | 1.963725 | 0.702386 | 5.130579 | 2.804649 | 0.884746 |
| 157 | Gradient-LCMD | 0.559336 | 4.223722 | 1.984876 | 0.677721 | 5.941090 | 2.820596 | 0.845455 |
| 6101 | Center/Width-LCMD | 0.607372 | 3.855565 | 1.807198 | 0.754699 | 6.003932 | 2.852489 | 0.830472 |
| 6101 | Gradient-MaxDet | 0.718619 | 4.359861 | 1.956872 | 0.686333 | 7.716615 | 3.479819 | 0.719958 |
| 6101 | Hybrid | 0.536177 | 3.217983 | 1.603822 | 0.829120 | 5.855889 | 2.840252 | 0.838729 |
| 6101 | Kernel-IVR | 0.722873 | 4.330557 | 1.963664 | 0.690535 | 7.916810 | 3.566666 | 0.705239 |
| 6101 | Gradient-LCMD | 0.566694 | 3.299186 | 1.668175 | 0.820387 | 6.461545 | 3.110792 | 0.803645 |

## Rankings

- AULC 525-653: Center/Width-LCMD 0.583463, Hybrid 0.611661, Gradient-LCMD 0.612807, Gradient-MaxDet 0.623329, Kernel-IVR 0.623554
- AULC 429-653: Center/Width-LCMD 0.611996, Kernel-IVR 0.642002, Hybrid 0.642092, Gradient-MaxDet 0.650433, Gradient-LCMD 0.662703
- AULC 333-653: Center/Width-LCMD 0.669884, Kernel-IVR 0.683845, Gradient-MaxDet 0.684867, Hybrid 0.686518, Gradient-LCMD 0.711328
- NRMSE @653: Center/Width-LCMD 0.551091, Hybrid 0.556141, Gradient-LCMD 0.563015, Gradient-MaxDet 0.601664, Kernel-IVR 0.621628

## Decision and interpretation

Decision: **SUSTAINED_AGAINST_SOME**. The frozen two-seed AULC advantage flags versus MaxDet, Hybrid, and IVR are `{'gradient_maxdet': True, 'hybrid': False, 'kernel_ivr': True}`. Paired deltas for every comparator and metric are in `results/paired_comparison.csv`; NRMSE/AULC use negative-is-CW-better, while R2 uses positive-is-CW-better.

CW late marginal deltas are: seed 157: 525->557 -0.009623, 557->589 -0.004319, 589->621 +0.002051, 621->653 -0.012074 / seed 6101: 525->557 -0.055179, 557->589 +0.016758, 589->621 -0.015094, 621->653 -0.048675. The 493->525 seed-6101 rebound is therefore evaluated against the actual 525->653 sequence rather than a single endpoint.

Component changes from L525 to L653:
- Seed 157: V1 RMSE 3.816886->3.660338 (-0.156548); V2 RMSE 5.791986->5.470631 (-0.321355).
- Seed 6101: V1 RMSE 4.332549->3.855565 (-0.476984); V2 RMSE 7.539949->6.003932 (-1.536017).

Selection overlap is diagnostic only. Against Gradient-MaxDet, mean/max overlap fractions are 0.0195/0.0312; against Gradient-LCMD they are 0.0352/0.0625.

Compute cost was exactly 8 new evaluation fits and 0 ensemble fits. Totals: training 4498.0s, gradient extraction 931.3s, selection 1.7s, recorded combined elapsed 5431.0s. Per-seed and per-round values, epochs, and checkpoint reuse flags are in `results/compute_cost.csv`.

No MaxDet-to-CW switch was implemented: its L_t state differs from pure CW. The stage pattern may motivate a separately preregistered matched-state switch study, but this continuation cannot establish switch efficacy. No automatic continuation to 685 or 1005 is authorized; any further pure-CW experiment requires a new protocol and cost/benefit decision.
