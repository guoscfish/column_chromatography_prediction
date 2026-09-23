# Pure Center/Width-LCMD continuation: L653 to L1005

This developmental continuation starts from each seed's exact frozen pure-CW L653 state, performs eleven current-checkpoint CW acquisitions and eleven scratch QGeoGNN-V2 fits per seed, and hard-stops at L1005 without an L1005 acquisition. Historical comparators are read-only. Test truth remained inaccessible until all 22 new checkpoint/prediction pairs were globally frozen.

## Primary AULC 653-1005

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.476857 | 0.489395 | 0.545033 | 0.497618 | 0.522292 |
| 6101 | 0.554151 | 0.635710 | 0.518974 | 0.605140 | 0.548031 |
| Mean | 0.515504 | 0.562552 | 0.532004 | 0.551379 | 0.535162 |

Paired primary deltas (negative means CW is better):
- CW minus Gradient-MaxDet: seed 157 -0.012538; seed 6101 -0.081559.
- CW minus Hybrid: seed 157 -0.068176; seed 6101 +0.035176.
- CW minus Kernel-IVR: seed 157 -0.020761; seed 6101 -0.050989.
- CW minus Gradient-LCMD: seed 157 -0.045435; seed 6101 +0.006119.

## Longer-horizon AULCs

- AULC 525-1005: Center/Width-LCMD 0.533626, Hybrid 0.553246, Gradient-LCMD 0.555867, Kernel-IVR 0.570626, Gradient-MaxDet 0.578759
- AULC 429-1005: Center/Width-LCMD 0.553028, Hybrid 0.574816, Gradient-LCMD 0.584761, Kernel-IVR 0.586621, Gradient-MaxDet 0.596728
- AULC 333-1005: Center/Width-LCMD 0.589018, Hybrid 0.605582, Kernel-IVR 0.614458, Gradient-LCMD 0.619050, Gradient-MaxDet 0.620797

Every full AULC uses exactly all 22 budgets from 333 through 1005 for every seed and method.

## Endpoint at L1005

| Seed | Method | NRMSE | V1 RMSE | V1 MAE | V1 R2 | V2 RMSE | V2 MAE | V2 R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 157 | Center/Width-LCMD | 0.469339 | 3.537422 | 1.584773 | 0.773944 | 5.004628 | 2.385629 | 0.890335 |
| 157 | Gradient-MaxDet | 0.484456 | 3.801134 | 1.834227 | 0.738983 | 4.702728 | 2.521176 | 0.903167 |
| 157 | Hybrid | 0.545465 | 4.331021 | 1.861319 | 0.661138 | 5.121646 | 2.360630 | 0.885147 |
| 157 | Kernel-IVR | 0.479036 | 3.687103 | 1.786899 | 0.754409 | 4.878869 | 2.557628 | 0.895777 |
| 157 | Gradient-LCMD | 0.509225 | 3.865644 | 1.711852 | 0.730049 | 5.349156 | 2.532810 | 0.874716 |
| 6101 | Center/Width-LCMD | 0.514631 | 3.172856 | 1.586975 | 0.833879 | 5.378877 | 2.416805 | 0.863933 |
| 6101 | Gradient-MaxDet | 0.534559 | 3.332455 | 1.661326 | 0.816746 | 5.476139 | 2.750980 | 0.858968 |
| 6101 | Hybrid | 0.508853 | 3.065289 | 1.522327 | 0.844952 | 5.526015 | 2.520493 | 0.856387 |
| 6101 | Kernel-IVR | 0.544745 | 3.393455 | 1.655328 | 0.809976 | 5.588146 | 2.696279 | 0.853139 |
| 6101 | Gradient-LCMD | 0.528277 | 3.208923 | 1.508070 | 0.830081 | 5.661672 | 2.464328 | 0.849249 |

Two-seed mean NRMSE@1005 ranking: Center/Width-LCMD 0.491985, Gradient-MaxDet 0.509508, Kernel-IVR 0.511890, Gradient-LCMD 0.518751, Hybrid 0.527159.

## Decision and trajectory analysis

Decision: **SUSTAINED_AGAINST_SOME**. Two-seed AULC 653-1005 advantage flags versus MaxDet, Hybrid, and IVR are `{'gradient_maxdet': True, 'hybrid': False, 'kernel_ivr': True}`. Endpoint rankings are reported separately and do not override curve efficiency.

CW marginal NRMSE changes after L653:
- Seed 157: 653->685 +0.004891, 685->717 -0.013485, 717->749 +0.000675, 749->781 -0.012297, 781->813 +0.005694, 813->845 -0.009010, 845->877 -0.002643, 877->909 +0.002519, 909->941 -0.004637, 941->973 -0.008434, 973->1005 +0.011256
- Seed 6101: 653->685 +0.019590, 685->717 -0.079881, 717->749 +0.025907, 749->781 -0.038456, 781->813 -0.003987, 813->845 +0.006620, 845->877 +0.007327, 877->909 -0.005385, 909->941 +0.015072, 941->973 -0.006586, 973->1005 -0.032963

CW component changes from L653 to L1005:
- Seed 157: V1 RMSE 3.660338->3.537422 (-0.122915); V2 RMSE 5.470631->5.004628 (-0.466002).
- Seed 6101: V1 RMSE 3.855565->3.172856 (-0.682709); V2 RMSE 6.003932->5.378877 (-0.625055).

Selection overlap remained diagnostic only. Versus Gradient-MaxDet, mean/max fractions are 0.0057/0.0312; versus Gradient-LCMD they are 0.0284/0.0938.

Compute cost was exactly 22 new evaluation fits and 0 ensemble fits: training 11714.9s, gradient extraction 1842.2s, selection 3.0s, recorded combined elapsed 13560.0s. Per-round epochs and timing are in `results/compute_cost.csv`.

This study ends at the established L1005 endpoint. No additional method search, switch experiment, or later acquisition was run.
