# Final 4g qualification

Decision: `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`.

All six frozen runs completed with finite metrics and validation-only checkpoint selection. The decision qualifies the point predictor for transfer studies; it does not qualify a final UQ model.

## Row complete metrics

| mode | seed | split | V1_r2 | V1_rmse | V1_mae | V2_r2 | V2_rmse | V2_mae | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| row | 42 | train | 0.927988291 | 2.11450624 | 1.17624998 | 0.954072297 | 3.4453156 | 1.94963753 | 0.24283676 |
| row | 42 | validation | 0.865635037 | 2.4895494 | 1.38111711 | 0.935464621 | 3.80542397 | 2.19203711 | 0.279152592 |
| row | 42 | test | 0.857115626 | 2.97305226 | 1.36419368 | 0.898454785 | 4.91012239 | 2.33261681 | 0.343251686 |
| row | 525 | train | 0.945157409 | 1.821908 | 0.88969028 | 0.972073972 | 2.63973093 | 1.30884171 | 0.203431278 |
| row | 525 | validation | 0.903955758 | 2.50962043 | 1.27647924 | 0.957216978 | 3.53030133 | 1.76491308 | 0.277494803 |
| row | 525 | test | 0.841235161 | 2.96536279 | 1.34991443 | 0.870739222 | 5.60666227 | 2.22746468 | 0.368282475 |
| row | 1101 | train | 0.904455662 | 2.43329954 | 1.36713922 | 0.944786727 | 3.80533838 | 2.22652602 | 0.274551976 |
| row | 1101 | validation | 0.735375524 | 3.46039701 | 1.63605058 | 0.8156389 | 5.31244802 | 2.41377234 | 0.387836568 |
| row | 1101 | test | 0.875490785 | 2.81969213 | 1.53848231 | 0.867610514 | 6.05836153 | 2.96882224 | 0.366227618 |

### train aggregate (sample std, ddof=1)

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| V1_r2 | 0.92586712 | 0.0204336138 | 0.904455662 | 0.945157409 |
| V1_rmse | 2.12323793 | 0.305789285 | 1.821908 | 2.43329954 |
| V1_mae | 1.14435983 | 0.240316685 | 0.88969028 | 1.36713922 |
| V2_r2 | 0.956977665 | 0.0138736912 | 0.944786727 | 0.972073972 |
| V2_rmse | 3.29679497 | 0.596828255 | 2.63973093 | 3.80533838 |
| V2_mae | 1.82833509 | 0.470714179 | 1.30884171 | 2.22652602 |
| combined_normalized_rmse | 0.240273338 | 0.035629577 | 0.203431278 | 0.274551976 |

### validation aggregate (sample std, ddof=1)

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| V1_r2 | 0.834988773 | 0.0883697854 | 0.735375524 | 0.903955758 |
| V1_rmse | 2.81985561 | 0.554815889 | 2.4895494 | 3.46039701 |
| V1_mae | 1.43121564 | 0.184946707 | 1.27647924 | 1.63605058 |
| V2_r2 | 0.902773499 | 0.0762405411 | 0.8156389 | 0.957216978 |
| V2_rmse | 4.21605778 | 0.959414821 | 3.53030133 | 5.31244802 |
| V2_mae | 2.12357418 | 0.329802918 | 1.76491308 | 2.41377234 |
| combined_normalized_rmse | 0.314827987 | 0.0632327187 | 0.277494803 | 0.387836568 |

### test aggregate (sample std, ddof=1)

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| V1_r2 | 0.857947191 | 0.0171429451 | 0.841235161 | 0.875490785 |
| V1_rmse | 2.91936906 | 0.0864083291 | 2.81969213 | 2.97305226 |
| V1_mae | 1.41753014 | 0.104990688 | 1.34991443 | 1.53848231 |
| V2_r2 | 0.87893484 | 0.016976996 | 0.867610514 | 0.898454785 |
| V2_rmse | 5.52504873 | 0.578453849 | 4.91012239 | 6.05836153 |
| V2_mae | 2.50963457 | 0.401128692 | 2.22746468 | 2.96882224 |
| combined_normalized_rmse | 0.359253926 | 0.0138963799 | 0.343251686 | 0.368282475 |

## Compound complete metrics

| mode | seed | split | V1_r2 | V1_rmse | V1_mae | V2_r2 | V2_rmse | V2_mae | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compound | 42 | train | 0.904654145 | 2.39813089 | 1.30488575 | 0.950005412 | 3.59517837 | 1.97964978 | 0.269574167 |
| compound | 42 | validation | 0.679651618 | 3.81809568 | 2.18547058 | 0.70053035 | 7.10385895 | 4.0285182 | 0.467376179 |
| compound | 42 | test | 0.406218171 | 6.74492598 | 3.35824037 | 0.448918104 | 12.6347103 | 6.21231318 | 0.828162787 |
| compound | 525 | train | 0.687240601 | 4.36903667 | 2.25571179 | 0.798392475 | 7.13990021 | 3.76926899 | 0.507132582 |
| compound | 525 | validation | 0.328696787 | 6.61075354 | 3.35360098 | 0.411162019 | 13.5244923 | 7.45428467 | 0.84835797 |
| compound | 525 | test | 0.529119492 | 4.9133606 | 2.4625895 | 0.489437044 | 9.86576557 | 4.86694813 | 0.624690864 |
| compound | 1101 | train | 0.901754022 | 2.48125243 | 1.35926151 | 0.93583703 | 4.15082502 | 2.34399915 | 0.284963957 |
| compound | 1101 | validation | 0.52797544 | 4.90389967 | 2.3508966 | 0.642237544 | 7.52441216 | 4.13148642 | 0.545252418 |
| compound | 1101 | test | 0.50170517 | 5.16679811 | 2.78428626 | 0.52307272 | 10.4385834 | 5.80524921 | 0.644900582 |

### train aggregate (sample std, ddof=1)

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| V1_r2 | 0.831216256 | 0.124695007 | 0.687240601 | 0.904654145 |
| V1_rmse | 3.08280667 | 1.11468292 | 2.39813089 | 4.36903667 |
| V1_mae | 1.63995302 | 0.533955367 | 1.30488575 | 2.25571179 |
| V2_r2 | 0.894744972 | 0.0837438871 | 0.798392475 | 0.950005412 |
| V2_rmse | 4.96196787 | 1.9064962 | 3.59517837 | 7.13990021 |
| V2_mae | 2.69763931 | 0.945769659 | 1.97964978 | 3.76926899 |
| combined_normalized_rmse | 0.353890236 | 0.13293466 | 0.269574167 | 0.507132582 |

### validation aggregate (sample std, ddof=1)

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| V1_r2 | 0.512107948 | 0.176014648 | 0.328696787 | 0.679651618 |
| V1_rmse | 5.1109163 | 1.40779131 | 3.81809568 | 6.61075354 |
| V1_mae | 2.62998939 | 0.632101056 | 2.18547058 | 3.35360098 |
| V2_r2 | 0.584643304 | 0.153040289 | 0.411162019 | 0.70053035 |
| V2_rmse | 9.38425446 | 3.59171172 | 7.10385895 | 13.5244923 |
| V2_mae | 5.20476309 | 1.948823 | 4.0285182 | 7.45428467 |
| combined_normalized_rmse | 0.620328856 | 0.201281233 | 0.467376179 | 0.84835797 |

### test aggregate (sample std, ddof=1)

| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| V1_r2 | 0.479014277 | 0.0645162079 | 0.406218171 | 0.529119492 |
| V1_rmse | 5.60836156 | 0.992417072 | 4.9133606 | 6.74492598 |
| V1_mae | 2.86837204 | 0.453707429 | 2.4625895 | 3.35824037 |
| V2_r2 | 0.487142622 | 0.0371305134 | 0.448918104 | 0.52307272 |
| V2_rmse | 10.9796864 | 1.46162859 | 9.86576557 | 12.6347103 |
| V2_mae | 5.62817017 | 0.689941677 | 4.86694813 | 6.21231318 |
| combined_normalized_rmse | 0.699251411 | 0.112096901 | 0.624690864 | 0.828162787 |

## Generalization and decision boundary

| mode | seed | best_epoch | epochs | V1_train_minus_validation_R2 | V1_train_minus_test_R2 | V2_train_minus_validation_R2 | V2_train_minus_test_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| row | 42 | 95 | 195 | 0.0623532534 | 0.0708726645 | 0.018607676 | 0.0556175113 |
| row | 525 | 318 | 418 | 0.0412016511 | 0.103922248 | 0.0148569942 | 0.101334751 |
| row | 1101 | 67 | 167 | 0.169080138 | 0.0289648771 | 0.129147828 | 0.0771762133 |
| compound | 42 | 70 | 170 | 0.225002527 | 0.498435974 | 0.249475062 | 0.501087308 |
| compound | 525 | 69 | 169 | 0.358543813 | 0.158121109 | 0.387230456 | 0.308955431 |
| compound | 1101 | 65 | 165 | 0.373778582 | 0.400048852 | 0.293599486 | 0.412764311 |

Row interpolation has predictive signal for both targets. Compound splits hold out entire molecules and expose a larger generalization gap; this limits claims about unseen molecules. The gap is reported rather than tuned away. All runs have finite outputs and nonconstant predictions; the engineering equivalence gate confirms the corrected R2 semantics. This supports ordinary transfer studies, not universal extrapolation or qualified uncertainty. No seed was rerun because of poor performance.

One interrupted compound-1101 attempt was rerun from the same seed/protocol because no optimizer-resume state existed. The five completed runs were reused with checkpoint hash checks. Frozen manifests and numerical training protocol remain unchanged.

Source for transfer is preregistered row seed 42: `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. The E0 engineering fixture differs from the already-frozen six-run qualification splits, and standalone initialization directly samples effective modules; these runs are not an R2 architecture benchmark.
