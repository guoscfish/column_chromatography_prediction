# Balanced shared-context CW + IVR portfolio

Decision: **NO_CLEAR_PORTFOLIO_GAIN**. Development / exposed cohort; not independent confirmation.
No p-values or generalization/significance claims are made from two exposed seeds.

## Definition and provenance

At every current checkpoint, one full-network q50 derivative pass generates both
historical CountSketch-512 geometries. CW uses the frozen L333 Center/Width scales;
IVR uses endpoint scales then all-reference RMS, unit prior and unit noise.
Exactly 32 alternating steps begin with CW. Every pick becomes a CW center and
an IVR observation location through a rank-one covariance update. No labels enter
selection. The whole batch is frozen before reveal; the next predictor is trained
from the identical historical scratch initialization, never warm-started.
L333 checkpoints and test-X predictions are reused for both seeds. L333 gradients
are re-extracted once in shared multi-transform form, because endpoint-only
compressed sketches cannot reconstruct CW. `engineering_smoke.json` records
numerical agreement and unchanged historical pure-selector L333 batch IDs.

New fits: 20; new L333 fits: 0; reused anchor checkpoints: 2.
Execution wall time (summed sequential seed windows): 6712.4s;
training 5532.4s; all gradient extraction including smoke
1416.0s; selectors/diagnostics 27.6s.
Smoke precedes the execution wall window, so these timing quantities differ.
Both complete trajectories and 22 checkpoint/test-X prediction points were
recursively frozen before 2 test accesses
(one per seed); pre-global-freeze access count was zero. Source hashes revalidated.

## Per-seed metrics

| outer_seed | method | AULC_333_653 | AULC_333_525 | AULC_429_653 | AULC_525_653 | AULC_333_429 | AULC_429_525 | combined_normalized_RMSE_653 | V1_RMSE_653 | V2_RMSE_653 | V1_R2_653 | V2_R2_653 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | cw_ivr_portfolio | 0.579977 | 0.608486 | 0.547146 | 0.537213 | 0.656583 | 0.560389 | 0.522079 | 4.017922 | 5.318726 | 0.708362 | 0.876138 |
| 157 | center_width_lcmd | 0.579574 | 0.628014 | 0.529641 | 0.506915 | 0.696086 | 0.559942 | 0.494809 | 3.660338 | 5.470631 | 0.757962 | 0.868962 |
| 157 | kernel_ivr | 0.604759 | 0.658425 | 0.549356 | 0.524261 | 0.734032 | 0.582817 | 0.520384 | 4.058873 | 5.130579 | 0.702386 | 0.884746 |
| 157 | lcmd | 0.649830 | 0.689157 | 0.617654 | 0.590839 | 0.724906 | 0.653408 | 0.559336 | 4.223722 | 5.941090 | 0.677721 | 0.845455 |
| 157 | hybrid | 0.626214 | 0.654246 | 0.595492 | 0.584167 | 0.697898 | 0.610593 | 0.576106 | 4.478396 | 5.728562 | 0.637685 | 0.856314 |
| 157 | gradient_maxdet | 0.593462 | 0.646002 | 0.553571 | 0.514650 | 0.686539 | 0.605466 | 0.484709 | 3.669656 | 5.120744 | 0.756728 | 0.885187 |
| 6101 | cw_ivr_portfolio | 0.729867 | 0.794014 | 0.664253 | 0.633647 | 0.882968 | 0.705060 | 0.614226 | 3.594777 | 6.954624 | 0.786760 | 0.772534 |
| 6101 | center_width_lcmd | 0.760194 | 0.826984 | 0.694351 | 0.660010 | 0.913828 | 0.740139 | 0.607372 | 3.855565 | 6.003932 | 0.754699 | 0.830472 |
| 6101 | kernel_ivr | 0.762930 | 0.789651 | 0.734648 | 0.722848 | 0.828922 | 0.750381 | 0.722873 | 4.330557 | 7.916810 | 0.690535 | 0.705239 |
| 6101 | lcmd | 0.772826 | 0.864861 | 0.707751 | 0.634775 | 0.924669 | 0.805052 | 0.566694 | 3.299186 | 6.461545 | 0.820387 | 0.803645 |
| 6101 | hybrid | 0.746821 | 0.818599 | 0.688691 | 0.639155 | 0.882458 | 0.754740 | 0.536177 | 3.217983 | 5.855889 | 0.829120 | 0.838729 |
| 6101 | gradient_maxdet | 0.776273 | 0.805782 | 0.747294 | 0.732008 | 0.843890 | 0.767675 | 0.718619 | 4.359861 | 7.716615 | 0.686333 | 0.719958 |

## Two-seed development mean

| method | AULC_333_653 | AULC_333_525 | AULC_429_653 | AULC_525_653 | AULC_333_429 | AULC_429_525 | combined_normalized_RMSE_653 | V1_RMSE_653 | V2_RMSE_653 | V1_R2_653 | V2_R2_653 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| center_width_lcmd | 0.669884 | 0.727499 | 0.611996 | 0.583463 | 0.804957 | 0.650040 | 0.551091 | 3.757951 | 5.737281 | 0.756330 | 0.849717 |
| cw_ivr_portfolio | 0.654922 | 0.701250 | 0.605699 | 0.585430 | 0.769775 | 0.632725 | 0.568153 | 3.806350 | 6.136675 | 0.747561 | 0.824336 |
| gradient_maxdet | 0.684867 | 0.725892 | 0.650433 | 0.623329 | 0.765214 | 0.686570 | 0.601664 | 4.014759 | 6.418679 | 0.721530 | 0.802573 |
| hybrid | 0.686518 | 0.736422 | 0.642092 | 0.611661 | 0.790178 | 0.682667 | 0.556141 | 3.848190 | 5.792225 | 0.733402 | 0.847522 |
| kernel_ivr | 0.683845 | 0.724038 | 0.642002 | 0.623554 | 0.781477 | 0.666599 | 0.621628 | 4.194715 | 6.523694 | 0.696461 | 0.794992 |
| lcmd | 0.711328 | 0.777009 | 0.662703 | 0.612807 | 0.824788 | 0.729230 | 0.563015 | 3.761454 | 6.201317 | 0.749054 | 0.824550 |

## Component oracle regret

Regret is AULC(method) minus min(AULC(CW),AULC(IVR)) for the same seed.
Negative values are permitted; no clipping. Historical CW wins both seeds on
333-653, so its oracle regret is zero. This is a strict hedging benchmark,
not evidence that pure CW wins all future states.

| method | mean_regret | max_seed_regret |
| --- | --- | --- |
| center_width_lcmd | 0.000000 | 0.000000 |
| cw_ivr_portfolio | -0.014962 | 0.000402 |
| kernel_ivr | 0.013960 | 0.025185 |

## Answers to the scientific questions

1. Mean AULC 333-653 is 0.654922; delta to best component mean
(center_width_lcmd) is -0.014962. The frozen closeness
tolerance is 0.01, approximately 1.5% of the historical scale.
2. Stability within the frozen per-seed AULC and endpoint oracle tolerance:
False. Exact per-seed deltas:
[{'outer_seed': 157, 'AULC_delta_to_oracle': 0.00040249526379987355, 'AULC_delta_to_worse': -0.02478223232395793, 'endpoint_delta_to_oracle': 0.027269685260613796}, {'outer_seed': 6101, 'AULC_delta_to_oracle': -0.030326744659670557, 'AULC_delta_to_worse': -0.033062708338116864, 'endpoint_delta_to_oracle': 0.006854174967801163}]. No seed was dropped based on performance.
3. Worst-regret reductions versus fixed components (positive is improvement):
{'center_width_lcmd': -0.00040249526379987355, 'kernel_ivr': 0.02478223232395793}. The decision uses both comparisons.
4. AULC 333-525=0.701250, 429-653=0.605699,
525-653=0.585430. The tables include every seed and comparator.
5. Disjoint early/middle/late-middle AULC deltas to the best full-trajectory
component are {'333-429': -0.035181592628377056, '429-525': -0.01731592993835185, '525-653': 0.0019678301802084075}. Negative identifies the stage with predictive gain;
overlapping requested windows must not be summed as independent contributions.
6. Shared conditioning prevents duplicates by construction and updates both
objectives after every pick. Independent CW16/IVR16 proposals overlap by
2.10 rows on average. Before/after geometry and union diagnostics appear
below. Duplicate prevention does not by itself establish optimal redundancy
reduction: independent unions have fewer locations when overlap is nonzero,
and no equal-budget no-conditioning ablation was trained.
7. Expert-specific nearest-center distances, marginal variance contributions and
same-state proposal agreement quantify differing selection profiles below.
Both experts supply 16 rows per batch, but this alone does not establish two
disjoint chemical regions; geometric complementarity is only descriptive.
8. Predictive gain must be assessed by the frozen metrics/regrets above, not
selection diversity. Acquisition differences do not automatically imply gains.

## Acquisition mechanism diagnostics, all 20 rounds

| outer_seed | round | active_labels_before | gradient_norm_mean | gradient_norm_std | gradient_norm_p50 | gradient_norm_p90 | gradient_norm_p95 | gradient_norm_max | gradient_effective_rank | coverage_nearest_distance_mean | coverage_nearest_distance_p90 | coverage_after_mean | coverage_after_p90 | ivr_relative_reducible_variance | ivr_integrated_variance_before | ivr_integrated_variance_after | independent_proposal_overlap_count | independent_union_size | independent_union_coverage_mean | independent_union_integrated_variance | shared_proposal_agreement_steps | cw_turns | ivr_turns | batch_size | unique_ids | gradient_seconds | selector_seconds | cw_marginal_variance_reduction | ivr_marginal_variance_reduction | cw_selected_nearest_distance_mean | ivr_selected_nearest_distance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | 1 | 333 | 122.588822 | 90.940771 | 89.162876 | 237.358141 | 323.339140 | 643.552635 | 8.501932 | 65.275556 | 134.175767 | 57.334781 | 118.858201 | 0.228634 | 0.199183 | 0.153643 | 5 | 27 | 58.939839 | 0.158674 | 4 | 16 | 16 | 32 | 32 | 139.629078 | 2.242947 | 0.018731 | 0.026809 | 234.783659 | 216.230078 |
| 157 | 2 | 365 | 142.537144 | 109.584154 | 104.015744 | 284.003688 | 394.248666 | 849.921792 | 8.177802 | 72.124242 | 145.224686 | 65.114172 | 129.000002 | 0.167085 | 0.173085 | 0.144165 | 5 | 27 | 66.380261 | 0.147494 | 0 | 16 | 16 | 32 | 32 | 82.134025 | 1.559988 | 0.012265 | 0.016655 | 272.954272 | 250.257412 |
| 157 | 3 | 397 | 153.806574 | 130.389367 | 108.649816 | 302.052006 | 435.374579 | 1111.051250 | 6.716822 | 71.870932 | 152.113866 | 65.540923 | 135.016277 | 0.120687 | 0.139781 | 0.122912 | 5 | 27 | 66.468909 | 0.125092 | 0 | 16 | 16 | 32 | 32 | 87.176622 | 1.690056 | 0.006484 | 0.010386 | 278.099502 | 218.622556 |
| 157 | 4 | 429 | 153.772943 | 133.434642 | 107.749596 | 318.065506 | 434.421843 | 1091.059493 | 8.079243 | 75.970928 | 152.986047 | 70.776602 | 145.842951 | 0.087904 | 0.137865 | 0.125746 | 3 | 29 | 71.468105 | 0.126453 | 0 | 16 | 16 | 32 | 32 | 97.175206 | 1.853466 | 0.004280 | 0.007839 | 215.675839 | 243.402653 |
| 157 | 5 | 461 | 166.210964 | 149.227407 | 111.325238 | 355.789971 | 490.097151 | 1231.758378 | 7.788636 | 74.536181 | 155.968117 | 68.120017 | 141.178416 | 0.080822 | 0.126658 | 0.116421 | 1 | 31 | 68.775808 | 0.116610 | 0 | 16 | 16 | 32 | 32 | 100.241352 | 1.861237 | 0.003504 | 0.006733 | 246.361749 | 265.828600 |
| 157 | 6 | 493 | 170.378266 | 140.006123 | 120.872270 | 341.623670 | 470.685257 | 1246.122647 | 8.453275 | 81.642098 | 165.778737 | 75.058091 | 156.430211 | 0.066483 | 0.138614 | 0.129398 | 0 | 32 | 75.058096 | 0.129398 | 0 | 16 | 16 | 32 | 32 | 93.231975 | 1.438372 | 0.002544 | 0.006672 | 187.981938 | 287.376809 |
| 157 | 7 | 525 | 166.545747 | 148.793368 | 113.167042 | 336.716816 | 487.805662 | 1315.534556 | 9.208883 | 67.045615 | 136.511256 | 62.317277 | 129.009238 | 0.052325 | 0.120913 | 0.114586 | 1 | 31 | 62.569302 | 0.114714 | 0 | 16 | 16 | 32 | 32 | 96.981911 | 2.231682 | 0.001639 | 0.004687 | 163.396083 | 199.644826 |
| 157 | 8 | 557 | 163.396961 | 166.216078 | 101.035348 | 366.445207 | 505.843488 | 1356.141741 | 6.405256 | 60.931390 | 125.050261 | 56.542549 | 116.433935 | 0.051949 | 0.104714 | 0.099274 | 3 | 29 | 57.161158 | 0.099704 | 1 | 16 | 16 | 32 | 32 | 66.619025 | 1.770650 | 0.001839 | 0.003601 | 175.535296 | 141.555196 |
| 157 | 9 | 589 | 182.136325 | 157.955662 | 124.965674 | 383.997837 | 518.139108 | 1220.319622 | 7.839057 | 72.256019 | 149.548518 | 68.203550 | 142.351235 | 0.042998 | 0.116690 | 0.111672 | 1 | 31 | 68.632017 | 0.111837 | 0 | 16 | 16 | 32 | 32 | 57.852531 | 1.232016 | 0.001676 | 0.003342 | 180.672998 | 130.588657 |
| 157 | 10 | 621 | 185.453352 | 147.557698 | 133.965241 | 371.245910 | 493.705770 | 1146.379996 | 9.145203 | 74.246926 | 154.136294 | 70.082235 | 147.711974 | 0.036943 | 0.124735 | 0.120127 | 0 | 32 | 69.939888 | 0.120133 | 0 | 16 | 16 | 32 | 32 | 69.002281 | 1.647304 | 0.001123 | 0.003485 | 144.323306 | 151.912075 |
| 6101 | 1 | 333 | 147.644942 | 103.376046 | 108.199690 | 291.816437 | 382.810184 | 775.308308 | 6.416337 | 70.990725 | 156.883753 | 64.162979 | 136.043681 | 0.191938 | 0.215980 | 0.174525 | 5 | 27 | 65.437153 | 0.179016 | 0 | 16 | 16 | 32 | 32 | 124.345958 | 2.701040 | 0.014123 | 0.027332 | 201.314363 | 228.522216 |
| 6101 | 2 | 365 | 157.464165 | 110.962980 | 118.162508 | 315.846149 | 386.896760 | 769.203663 | 8.542251 | 78.325055 | 158.427797 | 70.808087 | 143.246883 | 0.170774 | 0.211866 | 0.175684 | 4 | 28 | 71.652874 | 0.177997 | 0 | 16 | 16 | 32 | 32 | 45.333550 | 0.695834 | 0.012968 | 0.023213 | 224.348688 | 268.393814 |
| 6101 | 3 | 397 | 163.870841 | 126.833050 | 121.443074 | 324.248838 | 440.406386 | 1239.561744 | 9.123045 | 69.475455 | 145.917366 | 62.678989 | 126.651731 | 0.119566 | 0.162502 | 0.143072 | 1 | 31 | 63.647131 | 0.144159 | 0 | 16 | 16 | 32 | 32 | 45.300830 | 0.780143 | 0.007812 | 0.011618 | 208.210985 | 229.179657 |
| 6101 | 4 | 429 | 165.805953 | 128.323462 | 124.208441 | 312.483240 | 408.197701 | 1263.380003 | 10.710345 | 70.808812 | 144.723274 | 64.953115 | 130.773313 | 0.090179 | 0.161914 | 0.147313 | 1 | 31 | 65.441753 | 0.147892 | 1 | 16 | 16 | 32 | 32 | 44.864741 | 0.718074 | 0.005774 | 0.008827 | 218.912469 | 218.949909 |
| 6101 | 5 | 461 | 176.685161 | 161.668480 | 113.334834 | 396.634732 | 538.396475 | 1365.604372 | 8.272813 | 69.353890 | 150.226747 | 63.112383 | 138.873348 | 0.085872 | 0.134943 | 0.123355 | 3 | 29 | 63.728530 | 0.124368 | 0 | 16 | 16 | 32 | 32 | 44.802197 | 0.696836 | 0.004506 | 0.007082 | 222.504645 | 233.016172 |
| 6101 | 6 | 493 | 208.737471 | 172.328862 | 148.341474 | 410.248187 | 580.922464 | 1483.647590 | 9.414574 | 81.623967 | 171.373444 | 74.944636 | 158.149543 | 0.064302 | 0.143185 | 0.133978 | 1 | 31 | 75.070896 | 0.134341 | 0 | 16 | 16 | 32 | 32 | 41.802608 | 0.895497 | 0.003780 | 0.005427 | 247.681074 | 249.131464 |
| 6101 | 7 | 525 | 193.250733 | 179.109474 | 131.597739 | 390.310412 | 594.416952 | 1837.960588 | 7.639206 | 65.231130 | 137.070049 | 60.497173 | 126.709289 | 0.052921 | 0.117166 | 0.110965 | 0 | 32 | 60.653069 | 0.110793 | 0 | 16 | 16 | 32 | 32 | 44.684325 | 0.893279 | 0.001858 | 0.004343 | 159.681238 | 155.949288 |
| 6101 | 8 | 557 | 203.898051 | 181.630117 | 145.678620 | 405.051702 | 598.042344 | 1537.886021 | 8.367932 | 64.459182 | 132.608507 | 60.093303 | 127.033812 | 0.053778 | 0.115245 | 0.109048 | 0 | 32 | 60.135163 | 0.109277 | 0 | 16 | 16 | 32 | 32 | 44.945490 | 0.915714 | 0.002505 | 0.003692 | 220.429418 | 89.085327 |
| 6101 | 9 | 589 | 199.986713 | 170.205827 | 140.845741 | 395.918809 | 556.270783 | 1441.543328 | 9.425938 | 60.380880 | 127.305277 | 56.338463 | 117.066538 | 0.048123 | 0.115251 | 0.109705 | 2 | 30 | 56.673001 | 0.110139 | 0 | 16 | 16 | 32 | 32 | 44.946946 | 0.943999 | 0.002036 | 0.003511 | 165.284760 | 76.652123 |
| 6101 | 10 | 621 | 180.041667 | 186.790829 | 110.366350 | 391.324934 | 586.321415 | 1855.759847 | 6.674871 | 49.380284 | 106.714513 | 45.820324 | 100.609004 | 0.048980 | 0.094279 | 0.089661 | 1 | 31 | 46.099410 | 0.089737 | 1 | 16 | 16 | 32 | 32 | 44.970310 | 0.808339 | 0.001377 | 0.003240 | 164.085150 | 59.229849 |

Full 640-step expert trace, IDs, canonical indices, checkpoint/bank hashes,
cluster scores and variance updates are in `results/selection_trace.csv`.
Per-round label-free banks and immutable contracts are under `runtime/`.

## Interpretation and next step

The frozen decision does not justify proceeding directly to adaptive quota.
(1) Independent overlap and contribution profiles test geometric complementarity,
but cannot establish universal predictive complementarity. (2) Regression and
explicit covariance checks preserve both mathematical objectives; conditioning
changes their greedy path intentionally, not their definition. (3) A 16/16
bottleneck cannot be identified from a single fixed quota; no quota sweep is
warranted by these data alone. (4) CW path dependence remains plausible because
the portfolio changes the trajectory from the first batch. (5) Matched scratch
initialization controls one source of variation but does not estimate optimization
variance; repeated predictor fits would be needed. (6) Use the paired metrics to
distinguish observed selection diversity from actual predictive improvement.
There is no controlled evidence here that allocation is the principal bottleneck.
A future experiment, if pursued, should first isolate shared-conditioning and
path effects with an equal-budget preregistered comparison; it is not run here.

## Artifact index

- `PROTOCOL.md`, `protocol.json`, `seal.json`, `reuse_audit.json`: frozen design and provenance.
- `preflight_tests.xml`, `engineering_smoke.json`: tests and selection-only regression.
- `global_pre_test_freeze.json`: both complete trajectories bound by content hashes.
- `results/learning_curve_metrics.csv`: 11-point learning curves for all six methods.
- `results/summary_per_seed.csv`, `results/summary_mean.csv`: all AULC windows and endpoint metrics.
- `results/component_regret.csv`, `results/regret_summary.csv`: per-seed/mean/worst regret.
- `results/acquisition_diagnostics.csv`, `results/selection_trace.csv`: round and step audits.
- `results/fit_audit.csv`, `results/test_label_access_audit.csv`: cost and firewall audits.
- `figures/learning_curves.png`, `figures/component_regret.png`: scientific figures.
- `runtime/seed_{157,6101}/cw_ivr_portfolio/round_{00..10}/`: states, selections,
  gradient banks, model checkpoints, predictions and nested freezes (local, git-ignored).
- `decision.json`, `artifact_manifest.json`: decision and committed artifact hashes.
