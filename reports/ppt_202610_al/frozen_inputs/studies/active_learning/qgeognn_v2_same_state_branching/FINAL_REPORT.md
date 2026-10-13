# Same-state short-rollout branching：最终开发研究报告

## 结论摘要

本研究在 8 个严格复用的 anchor state 上完成了 24 个分支、48 次新的 scratch fit。所有 acquisition、fit 与 test-X prediction 均先冻结，随后才进行一次统一 test truth reveal。证据等级是 **development / mechanism**，不是独立确认。

核心结论是：从完全相同的 `(L_t, U_t, checkpoint)` 出发，三种 acquisition mechanism 的确会产生不同的后续收益，但 winner 不是 label budget 的稳定函数。8 个 anchor 内 short-AULC 的策略跨度为 0.008944–0.047463（均值 0.023867）；第一批到第二批的 winner 在 5/8 个 anchor 中改变，说明 one-step 比较会误判相当一部分短 rollout。

- short-AULC winner：Kernel-IVR 4/8，Gradient-LCMD 4/8，Gradient-MaxDet 0/8。
- budget 从 429 变到 653 时，winner 仅在 2/4 个 seed-source 配对中改变，且方向不一致；不支持一个稳健的固定 stage switch。
- 同一 budget 改变 source trajectory 时，winner 在 2/4 个 seed-budget 配对中改变。两个变化都发生在 budget 429：Gradient-LCMD 来源 state 偏向 IVR，而 Kernel-IVR 来源 state 偏向 LCMD，呈现 trajectory/state crossover。
- 改变 seed 时，winner 在 2/4 个 source-budget 配对中改变；两个变化都发生在 budget 653。两个开发 seed 不足以学习 controller。
- validation +64 与 test +64 的完整三策略排序只在 1/8 个 anchor 完全一致。该 validation 已参与 checkpoint selection，只能作为开发期 state observable，不能当成独立泛化证据。

因此，更值得保留的工作假设是 `a_t = pi(s_t)`，而不是 `a_t = pi(n_t)`；但当前数据只支持继续验证这个假设，不支持训练或部署 adaptive selector。

## 执行与泄漏控制

| item | value |
| --- | --- |
| global freeze status | ALL_24_BRANCHES_48_FITS_FROZEN_BEFORE_TEST_TRUTH |
| anchor states | 8 |
| branches | 24 |
| actual new fits | 48 |
| summed branch wall time | 13060.947 s (3.628 h) |
| summed model-training time | 11718.018 s |
| summed epochs run | 13790 |
| test accesses before global freeze | 0 |
| unified post-freeze test accesses | 2 (one per seed) |
| controller trained | False |

`test_label_access_audit.csv` 的两次访问均显示 `acquisitions_frozen=True` 且 `predictions_frozen=True`。历史 source artifacts 保持只读；封印验证负责检查其哈希及所有递归 branch prediction 哈希。

## Anchor lineage audit

429 与 653 分别从历史 source trajectory 的 round 3 与 round 10 复用，没有重训 anchor。下表哈希显示前 12 位；完整值见 `anchor_lineage_audit.csv`。

| seed | source_trajectory | anchor_budget | source_round | source_study | checkpoint_sha256 | ordered_L_hash | ordered_U_hash | gradient_bank_hash | test_truth_access_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | 3 | qgeognn_v2_row_sequential_b32 | 46a0ec59e006 | 693d0436496d | de5ab8e77831 | 3449ab6321d0 | 0 |
| 157 | gradient_lcmd | 653 | 10 | qgeognn_v2_row_sequential_b32 | dde10fa5a1ef | 90ddbbdc1bb3 | 81f8657f69b8 | 76e6520c10e0 | 0 |
| 157 | kernel_ivr | 429 | 3 | qgeognn_v2_row_kernel_ivr_b32 | a71877576d40 | 1e5d5d15db14 | de08af0ffe8d | 5cf88f240679 | 0 |
| 157 | kernel_ivr | 653 | 10 | qgeognn_v2_row_kernel_ivr_b32 | 2944251764f7 | 46652d1d2d6f | 51f088c00d74 | 52ba04962309 | 0 |
| 6101 | gradient_lcmd | 429 | 3 | qgeognn_v2_row_sequential_b32 | f3ffd93fdbf6 | c853e21b5b84 | 553a1a66a781 | 620ca917eb3e | 0 |
| 6101 | gradient_lcmd | 653 | 10 | qgeognn_v2_row_sequential_b32 | e858389dfe01 | 3552c013a65d | 1775f50dd93b | 66b29f204969 | 0 |
| 6101 | kernel_ivr | 429 | 3 | qgeognn_v2_row_kernel_ivr_b32 | 760e891999a6 | d6d70de419e2 | b69250e0b1ea | a3ed602ccd36 | 0 |
| 6101 | kernel_ivr | 653 | 10 | qgeognn_v2_row_kernel_ivr_b32 | 31f5deda6869 | 32252dafe947 | c0a1dcbc1c36 | 9bfb0f7c8d5b | 0 |

## State diagnostics（test-blind）

### 历史与 validation observable

| seed | source_trajectory | anchor_budget | active_label_count | candidate_pool_size | recent_validation_nrmse | recent_validation_last_step_delta | recent_validation_2step_linear_slope_per_label | recent_validation_3step_linear_slope_per_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | 429 | 2901 | 0.651545 | -0.089352 | -0.002057 | -0.002029 |
| 157 | gradient_lcmd | 653 | 653 | 2677 | 0.403946 | -0.035751 | -0.000722 | -0.000775 |
| 157 | kernel_ivr | 429 | 429 | 2901 | 0.630171 | -0.116665 | -0.002009 | -0.002135 |
| 157 | kernel_ivr | 653 | 653 | 2677 | 0.450012 | -0.028662 | -0.001039 | -0.000857 |
| 6101 | gradient_lcmd | 429 | 429 | 2901 | 0.800604 | -0.109678 | -0.001672 | -0.002057 |
| 6101 | gradient_lcmd | 653 | 653 | 2677 | 0.462123 | -0.018016 | -0.001735 | -0.002088 |
| 6101 | kernel_ivr | 429 | 429 | 2901 | 0.832530 | -0.062256 | -0.001311 | -0.001833 |
| 6101 | kernel_ivr | 653 | 653 | 2677 | 0.559875 | -0.005913 | -0.000226 | -0.000851 |

### Gradient geometry 与 coverage

Effective rank 使用 participation ratio `(sum lambda)^2 / sum(lambda^2)`。

| seed | source_trajectory | anchor_budget | gradient_norm_mean | gradient_norm_std | gradient_norm_p50 | gradient_norm_p90 | gradient_norm_p95 | gradient_norm_max | gradient_effective_rank_participation_ratio | coverage_nearest_distance_mean | coverage_nearest_distance_median | coverage_nearest_distance_p90 | coverage_nearest_distance_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | 146.464598 | 142.338032 | 97.430254 | 310.248724 | 440.764662 | 1345.93 | 5.918337 | 60.659816 | 47.179501 | 132.867129 | 539.054377 |
| 157 | gradient_lcmd | 653 | 183.726253 | 149.795748 | 124.750306 | 376.540775 | 505.738885 | 1274.48 | 8.712205 | 54.248558 | 58.705121 | 113.012712 | 369.058461 |
| 157 | kernel_ivr | 429 | 151.205048 | 146.423715 | 100.411502 | 324.405576 | 474.324254 | 1449.58 | 5.993277 | 62.182607 | 51.541227 | 129.206395 | 426.084863 |
| 157 | kernel_ivr | 653 | 174.841402 | 154.049335 | 117.465984 | 370.231946 | 515.846650 | 1238.89 | 7.368645 | 59.154019 | 58.969958 | 121.198816 | 284.218017 |
| 6101 | gradient_lcmd | 429 | 184.873403 | 175.870123 | 115.333129 | 409.798145 | 568.172140 | 1483.15 | 7.024454 | 79.576164 | 62.369179 | 184.233171 | 531.791848 |
| 6101 | gradient_lcmd | 653 | 213.408388 | 208.745898 | 133.921713 | 496.859383 | 690.252013 | 1553.08 | 8.659718 | 53.192856 | 53.283054 | 118.184222 | 333.704099 |
| 6101 | kernel_ivr | 429 | 164.791245 | 169.218615 | 101.186251 | 358.461866 | 510.140251 | 1715.17 | 5.036329 | 65.128060 | 53.899807 | 143.749896 | 484.917445 |
| 6101 | kernel_ivr | 653 | 169.670440 | 182.366298 | 99.767344 | 383.618180 | 578.545197 | 1697.46 | 6.420957 | 51.220590 | 47.675052 | 108.733459 | 370.349377 |

### 三种 acquisition objective

| seed | source_trajectory | anchor_budget | ivr_integrated_variance_before | ivr_predicted_total_reduction_B32 | ivr_predicted_relative_reduction_B32 | maxdet_total_marginal_logdet_gain_B32 | lcmd_largest_cluster_score_first |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | 0.130615 | 0.017964 | 0.137532 | 14.068118 | 640383.16 |
| 157 | gradient_lcmd | 653 | 0.121953 | 0.007384 | 0.060549 | 10.539751 | 240270.01 |
| 157 | kernel_ivr | 429 | 0.119342 | 0.012410 | 0.103986 | 11.620350 | 501278.06 |
| 157 | kernel_ivr | 653 | 0.100281 | 0.003977 | 0.039660 | 6.456757 | 377184.80 |
| 6101 | gradient_lcmd | 429 | 0.140939 | 0.015417 | 0.109391 | 13.555454 | 1336790.59 |
| 6101 | gradient_lcmd | 653 | 0.106568 | 0.007545 | 0.070799 | 11.347170 | 253591.84 |
| 6101 | kernel_ivr | 429 | 0.115079 | 0.010787 | 0.093738 | 11.332837 | 697898.05 |
| 6101 | kernel_ivr | 653 | 0.080639 | 0.003674 | 0.045563 | 5.888446 | 354880.33 |

三种 selector 使用同一当前 full-network q50、512D CountSketch gradient bank。历史方法定义中的归一化差异被保留：LCMD 使用 raw squared-Euclidean geometry；IVR 使用 all-R RMS normalization 和既有 prior/noise；MaxDet 使用 current-L RMS normalization 与 D-opt/logdet objective。它们是机制定义的一部分，而不是未受控的 representation 差异。

## Pairwise branch results 与 short AULC

`delta_NRMSE` 为 anchor NRMSE 减去未来 NRMSE，越大越好；NRMSE 与 AULC 越小越好。

| seed | source_trajectory | anchor_budget | strategy | NRMSE_anchor | NRMSE_plus32 | NRMSE_plus64 | short_AULC | delta_NRMSE_32 | delta_NRMSE_64 | V1_RMSE_plus64 | V1_R2_plus64 | V2_RMSE_plus64 | V2_R2_plus64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | gradient_lcmd | 0.685243 | 0.700982 | 0.617361 | 0.676142 | -0.015739 | 0.067883 | 4.612470 | 0.615666 | 6.698955 | 0.803511 |
| 157 | gradient_lcmd | 429 | kernel_ivr | 0.685243 | 0.632866 | 0.563741 | 0.628679 | 0.052378 | 0.121502 | 4.171283 | 0.685673 | 6.229949 | 0.830061 |
| 157 | gradient_lcmd | 429 | gradient_maxdet | 0.685243 | 0.664846 | 0.664698 | 0.669908 | 0.020397 | 0.020546 | 4.844110 | 0.576094 | 7.544750 | 0.750762 |
| 157 | gradient_lcmd | 653 | gradient_lcmd | 0.559336 | 0.545190 | 0.521745 | 0.542865 | 0.014146 | 0.037591 | 4.053125 | 0.703229 | 5.196622 | 0.881760 |
| 157 | gradient_lcmd | 653 | kernel_ivr | 0.559336 | 0.505838 | 0.522220 | 0.523308 | 0.053499 | 0.037117 | 4.088750 | 0.697989 | 5.097922 | 0.886208 |
| 157 | gradient_lcmd | 653 | gradient_maxdet | 0.559336 | 0.512451 | 0.533568 | 0.529452 | 0.046885 | 0.025769 | 4.159855 | 0.687393 | 5.266533 | 0.878557 |
| 157 | kernel_ivr | 429 | gradient_lcmd | 0.645848 | 0.585780 | 0.558644 | 0.594013 | 0.060068 | 0.087204 | 4.224971 | 0.677530 | 5.914835 | 0.846818 |
| 157 | kernel_ivr | 429 | kernel_ivr | 0.645848 | 0.589921 | 0.583595 | 0.602321 | 0.055926 | 0.062253 | 4.530262 | 0.629244 | 5.823299 | 0.851522 |
| 157 | kernel_ivr | 429 | gradient_maxdet | 0.645848 | 0.594665 | 0.576650 | 0.602957 | 0.051183 | 0.069198 | 4.313296 | 0.663906 | 6.243123 | 0.829342 |
| 157 | kernel_ivr | 653 | gradient_lcmd | 0.520384 | 0.512316 | 0.536534 | 0.520388 | 0.008068 | -0.016150 | 4.147410 | 0.689261 | 5.409165 | 0.871890 |
| 157 | kernel_ivr | 653 | kernel_ivr | 0.520384 | 0.514622 | 0.509838 | 0.514866 | 0.005762 | 0.010546 | 3.918558 | 0.722608 | 5.209938 | 0.881153 |
| 157 | kernel_ivr | 653 | gradient_maxdet | 0.520384 | 0.525258 | 0.531425 | 0.525582 | -0.004874 | -0.011041 | 4.129364 | 0.691959 | 5.289709 | 0.877486 |
| 6101 | gradient_lcmd | 429 | gradient_lcmd | 0.868923 | 0.806955 | 0.776637 | 0.814868 | 0.061968 | 0.092287 | 4.769861 | 0.624564 | 8.171607 | 0.685960 |
| 6101 | gradient_lcmd | 429 | kernel_ivr | 0.868923 | 0.756497 | 0.779397 | 0.790329 | 0.112426 | 0.089527 | 4.542400 | 0.659518 | 8.874182 | 0.629638 |
| 6101 | gradient_lcmd | 429 | gradient_maxdet | 0.868923 | 0.814495 | 0.792661 | 0.822643 | 0.054429 | 0.076263 | 4.652110 | 0.642872 | 8.940853 | 0.624052 |
| 6101 | gradient_lcmd | 653 | gradient_lcmd | 0.566694 | 0.559976 | 0.550537 | 0.559296 | 0.006718 | 0.016158 | 3.332011 | 0.816795 | 5.934716 | 0.834358 |
| 6101 | gradient_lcmd | 653 | kernel_ivr | 0.566694 | 0.604618 | 0.566675 | 0.585651 | -0.037924 | 0.000020 | 3.475452 | 0.800682 | 5.976789 | 0.832001 |
| 6101 | gradient_lcmd | 653 | gradient_maxdet | 0.566694 | 0.578956 | 0.563166 | 0.571943 | -0.012261 | 0.003529 | 3.361807 | 0.813504 | 6.200636 | 0.819182 |
| 6101 | kernel_ivr | 429 | gradient_lcmd | 0.786210 | 0.739752 | 0.735206 | 0.750230 | 0.046457 | 0.051004 | 4.421812 | 0.677355 | 8.003721 | 0.698732 |
| 6101 | kernel_ivr | 429 | kernel_ivr | 0.786210 | 0.792505 | 0.723492 | 0.773678 | -0.006296 | 0.062717 | 4.259984 | 0.700539 | 8.124250 | 0.689590 |
| 6101 | kernel_ivr | 429 | gradient_maxdet | 0.786210 | 0.784827 | 0.784537 | 0.785100 | 0.001382 | 0.001672 | 4.724966 | 0.631599 | 8.522740 | 0.658392 |
| 6101 | kernel_ivr | 653 | gradient_lcmd | 0.722873 | 0.669968 | 0.684374 | 0.686796 | 0.052905 | 0.038499 | 4.145018 | 0.716485 | 7.369013 | 0.744619 |
| 6101 | kernel_ivr | 653 | kernel_ivr | 0.722873 | 0.690194 | 0.686780 | 0.697510 | 0.032679 | 0.036092 | 4.198206 | 0.709162 | 7.284034 | 0.750475 |
| 6101 | kernel_ivr | 653 | gradient_maxdet | 0.722873 | 0.684825 | 0.681460 | 0.693495 | 0.038048 | 0.041413 | 4.127397 | 0.718890 | 7.337554 | 0.746795 |

### 两 seed 平均

| source_trajectory | anchor_budget | strategy | NRMSE_anchor | NRMSE_plus32 | NRMSE_plus64 | short_AULC | delta_NRMSE_32 | delta_NRMSE_64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_lcmd | 429 | gradient_lcmd | 0.777083 | 0.753969 | 0.696999 | 0.745505 | 0.023115 | 0.080085 |
| gradient_lcmd | 429 | gradient_maxdet | 0.777083 | 0.739670 | 0.728679 | 0.746276 | 0.037413 | 0.048404 |
| gradient_lcmd | 429 | kernel_ivr | 0.777083 | 0.694681 | 0.671569 | 0.709504 | 0.082402 | 0.105514 |
| gradient_lcmd | 653 | gradient_lcmd | 0.563015 | 0.552583 | 0.536141 | 0.551081 | 0.010432 | 0.026874 |
| gradient_lcmd | 653 | gradient_maxdet | 0.563015 | 0.545703 | 0.548367 | 0.550697 | 0.017312 | 0.014649 |
| gradient_lcmd | 653 | kernel_ivr | 0.563015 | 0.555228 | 0.544447 | 0.554480 | 0.007787 | 0.018568 |
| kernel_ivr | 429 | gradient_lcmd | 0.716029 | 0.662766 | 0.646925 | 0.672121 | 0.053263 | 0.069104 |
| kernel_ivr | 429 | gradient_maxdet | 0.716029 | 0.689746 | 0.680594 | 0.694029 | 0.026282 | 0.035435 |
| kernel_ivr | 429 | kernel_ivr | 0.716029 | 0.691213 | 0.653544 | 0.688000 | 0.024815 | 0.062485 |
| kernel_ivr | 653 | gradient_lcmd | 0.621628 | 0.591142 | 0.610454 | 0.603592 | 0.030486 | 0.011174 |
| kernel_ivr | 653 | gradient_maxdet | 0.621628 | 0.605042 | 0.606442 | 0.609538 | 0.016587 | 0.015186 |
| kernel_ivr | 653 | kernel_ivr | 0.621628 | 0.602408 | 0.598309 | 0.606188 | 0.019221 | 0.023319 |

### 成对差值

差值定义为 left 减 right；负值表示 left 更好。

| seed | source_trajectory | anchor_budget | left_strategy | right_strategy | plus32_left_minus_right | plus64_left_minus_right | short_AULC_left_minus_right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | gradient_lcmd | kernel_ivr | 0.068117 | 0.053619 | 0.047463 |
| 157 | gradient_lcmd | 429 | gradient_lcmd | gradient_maxdet | 0.036136 | -0.047337 | 0.006234 |
| 157 | gradient_lcmd | 429 | kernel_ivr | gradient_maxdet | -0.031981 | -0.100956 | -0.041229 |
| 157 | gradient_lcmd | 653 | gradient_lcmd | kernel_ivr | 0.039353 | -0.000474 | 0.019558 |
| 157 | gradient_lcmd | 653 | gradient_lcmd | gradient_maxdet | 0.032739 | -0.011823 | 0.013414 |
| 157 | gradient_lcmd | 653 | kernel_ivr | gradient_maxdet | -0.006614 | -0.011348 | -0.006144 |
| 157 | kernel_ivr | 429 | gradient_lcmd | kernel_ivr | -0.004142 | -0.024951 | -0.008309 |
| 157 | kernel_ivr | 429 | gradient_lcmd | gradient_maxdet | -0.008886 | -0.018006 | -0.008944 |
| 157 | kernel_ivr | 429 | kernel_ivr | gradient_maxdet | -0.004744 | 0.006945 | -0.000636 |
| 157 | kernel_ivr | 653 | gradient_lcmd | kernel_ivr | -0.002306 | 0.026697 | 0.005521 |
| 157 | kernel_ivr | 653 | gradient_lcmd | gradient_maxdet | -0.012942 | 0.005109 | -0.005194 |
| 157 | kernel_ivr | 653 | kernel_ivr | gradient_maxdet | -0.010637 | -0.021587 | -0.010715 |
| 6101 | gradient_lcmd | 429 | gradient_lcmd | kernel_ivr | 0.050458 | -0.002760 | 0.024539 |
| 6101 | gradient_lcmd | 429 | gradient_lcmd | gradient_maxdet | -0.007540 | -0.016024 | -0.007776 |
| 6101 | gradient_lcmd | 429 | kernel_ivr | gradient_maxdet | -0.057997 | -0.013264 | -0.032315 |
| 6101 | gradient_lcmd | 653 | gradient_lcmd | kernel_ivr | -0.044643 | -0.016138 | -0.026356 |
| 6101 | gradient_lcmd | 653 | gradient_lcmd | gradient_maxdet | -0.018980 | -0.012629 | -0.012647 |
| 6101 | gradient_lcmd | 653 | kernel_ivr | gradient_maxdet | 0.025663 | 0.003509 | 0.013709 |
| 6101 | kernel_ivr | 429 | gradient_lcmd | kernel_ivr | -0.052753 | 0.011713 | -0.023448 |
| 6101 | kernel_ivr | 429 | gradient_lcmd | gradient_maxdet | -0.045075 | -0.049331 | -0.034870 |
| 6101 | kernel_ivr | 429 | kernel_ivr | gradient_maxdet | 0.007678 | -0.061045 | -0.011422 |
| 6101 | kernel_ivr | 653 | gradient_lcmd | kernel_ivr | -0.020226 | -0.002406 | -0.010714 |
| 6101 | kernel_ivr | 653 | gradient_lcmd | gradient_maxdet | -0.014857 | 0.002914 | -0.006700 |
| 6101 | kernel_ivr | 653 | kernel_ivr | gradient_maxdet | 0.005369 | 0.005320 | 0.004015 |

## Strategy winner matrix

`validation_test_ranking_exact_match` 比较 validation +64 与 test +64 的完整排序；`test_AULC_ranking` 单独给出 short-AULC 排序。

| seed | source_trajectory | anchor_budget | winner_plus32 | winner_plus64 | winner_short_AULC | test_AULC_ranking | validation_plus64_ranking | validation_test_ranking_exact_match | within_anchor_AULC_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | kernel_ivr | kernel_ivr | kernel_ivr | kernel_ivr > gradient_maxdet > gradient_lcmd | kernel_ivr > gradient_maxdet > gradient_lcmd | no | 0.047463 |
| 157 | gradient_lcmd | 653 | kernel_ivr | gradient_lcmd | kernel_ivr | kernel_ivr > gradient_maxdet > gradient_lcmd | kernel_ivr > gradient_lcmd > gradient_maxdet | no | 0.019558 |
| 157 | kernel_ivr | 429 | gradient_lcmd | gradient_lcmd | gradient_lcmd | gradient_lcmd > kernel_ivr > gradient_maxdet | kernel_ivr > gradient_lcmd > gradient_maxdet | no | 0.008944 |
| 157 | kernel_ivr | 653 | gradient_lcmd | kernel_ivr | kernel_ivr | kernel_ivr > gradient_lcmd > gradient_maxdet | gradient_lcmd > kernel_ivr > gradient_maxdet | no | 0.010715 |
| 6101 | gradient_lcmd | 429 | kernel_ivr | gradient_lcmd | kernel_ivr | kernel_ivr > gradient_lcmd > gradient_maxdet | gradient_maxdet > kernel_ivr > gradient_lcmd | no | 0.032315 |
| 6101 | gradient_lcmd | 653 | gradient_lcmd | gradient_lcmd | gradient_lcmd | gradient_lcmd > gradient_maxdet > kernel_ivr | gradient_lcmd > kernel_ivr > gradient_maxdet | no | 0.026356 |
| 6101 | kernel_ivr | 429 | gradient_lcmd | kernel_ivr | gradient_lcmd | gradient_lcmd > kernel_ivr > gradient_maxdet | kernel_ivr > gradient_maxdet > gradient_lcmd | no | 0.034870 |
| 6101 | kernel_ivr | 653 | gradient_lcmd | gradient_maxdet | gradient_lcmd | gradient_lcmd > gradient_maxdet > kernel_ivr | gradient_maxdet > gradient_lcmd > kernel_ivr | yes | 0.010714 |

### Winner counts

| endpoint | strategy | count |
| --- | --- | --- |
| plus32 | gradient_lcmd | 5 |
| plus32 | kernel_ivr | 3 |
| plus64 | gradient_lcmd | 4 |
| plus64 | kernel_ivr | 3 |
| plus64 | gradient_maxdet | 1 |
| short_AULC | kernel_ivr | 4 |
| short_AULC | gradient_lcmd | 4 |

## Ranking 是否随 state 改变

### Source trajectory effect

| seed | anchor_budget | gradient_lcmd_source_winner | kernel_ivr_source_winner | winner_changed |
| --- | --- | --- | --- | --- |
| 157 | 429 | kernel_ivr | gradient_lcmd | yes |
| 157 | 653 | kernel_ivr | kernel_ivr | no |
| 6101 | 429 | kernel_ivr | gradient_lcmd | yes |
| 6101 | 653 | gradient_lcmd | gradient_lcmd | no |

### Budget effect

| seed | source_trajectory | winner_at_429 | winner_at_653 | winner_changed |
| --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | kernel_ivr | kernel_ivr | no |
| 157 | kernel_ivr | gradient_lcmd | kernel_ivr | yes |
| 6101 | gradient_lcmd | kernel_ivr | gradient_lcmd | yes |
| 6101 | kernel_ivr | gradient_lcmd | gradient_lcmd | no |

### Seed effect

| source_trajectory | anchor_budget | winner_seed_157 | winner_seed_6101 | winner_changed |
| --- | --- | --- | --- | --- |
| gradient_lcmd | 429 | kernel_ivr | kernel_ivr | no |
| gradient_lcmd | 653 | kernel_ivr | gradient_lcmd | yes |
| kernel_ivr | 429 | gradient_lcmd | gradient_lcmd | no |
| kernel_ivr | 653 | kernel_ivr | gradient_lcmd | yes |

stage、trajectory 与 seed 各自都造成 2/4 个 matched comparison 的 winner 改变。关键区别是变化模式：source effect 集中在 429，而 seed effect 集中在 653；budget effect 又只出现在两个不同的 seed/source 组合且方向不统一。这不符合一个简单、可复现的 label-count switch 法则。

## 第一批 selection overlap

| seed | source_trajectory | anchor_budget | left_strategy | right_strategy | intersection_count | intersection_fraction_of_32 | jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | gradient_lcmd | 429 | gradient_lcmd | kernel_ivr | 7 | 0.218750 | 0.122807 |
| 157 | gradient_lcmd | 429 | gradient_lcmd | gradient_maxdet | 7 | 0.218750 | 0.122807 |
| 157 | gradient_lcmd | 429 | kernel_ivr | gradient_maxdet | 20 | 0.625000 | 0.454545 |
| 157 | gradient_lcmd | 653 | gradient_lcmd | kernel_ivr | 5 | 0.156250 | 0.084746 |
| 157 | gradient_lcmd | 653 | gradient_lcmd | gradient_maxdet | 4 | 0.125000 | 0.066667 |
| 157 | gradient_lcmd | 653 | kernel_ivr | gradient_maxdet | 15 | 0.468750 | 0.306122 |
| 157 | kernel_ivr | 429 | gradient_lcmd | kernel_ivr | 9 | 0.281250 | 0.163636 |
| 157 | kernel_ivr | 429 | gradient_lcmd | gradient_maxdet | 6 | 0.187500 | 0.103448 |
| 157 | kernel_ivr | 429 | kernel_ivr | gradient_maxdet | 15 | 0.468750 | 0.306122 |
| 157 | kernel_ivr | 653 | gradient_lcmd | kernel_ivr | 1 | 0.031250 | 0.015873 |
| 157 | kernel_ivr | 653 | gradient_lcmd | gradient_maxdet | 0 | 0.000000 | 0.000000 |
| 157 | kernel_ivr | 653 | kernel_ivr | gradient_maxdet | 12 | 0.375000 | 0.230769 |
| 6101 | gradient_lcmd | 429 | gradient_lcmd | kernel_ivr | 8 | 0.250000 | 0.142857 |
| 6101 | gradient_lcmd | 429 | gradient_lcmd | gradient_maxdet | 6 | 0.187500 | 0.103448 |
| 6101 | gradient_lcmd | 429 | kernel_ivr | gradient_maxdet | 12 | 0.375000 | 0.230769 |
| 6101 | gradient_lcmd | 653 | gradient_lcmd | kernel_ivr | 1 | 0.031250 | 0.015873 |
| 6101 | gradient_lcmd | 653 | gradient_lcmd | gradient_maxdet | 0 | 0.000000 | 0.000000 |
| 6101 | gradient_lcmd | 653 | kernel_ivr | gradient_maxdet | 22 | 0.687500 | 0.523810 |
| 6101 | kernel_ivr | 429 | gradient_lcmd | kernel_ivr | 6 | 0.187500 | 0.103448 |
| 6101 | kernel_ivr | 429 | gradient_lcmd | gradient_maxdet | 2 | 0.062500 | 0.032258 |
| 6101 | kernel_ivr | 429 | kernel_ivr | gradient_maxdet | 13 | 0.406250 | 0.254902 |
| 6101 | kernel_ivr | 653 | gradient_lcmd | kernel_ivr | 4 | 0.125000 | 0.066667 |
| 6101 | kernel_ivr | 653 | gradient_lcmd | gradient_maxdet | 1 | 0.031250 | 0.015873 |
| 6101 | kernel_ivr | 653 | kernel_ivr | gradient_maxdet | 14 | 0.437500 | 0.280000 |

### 汇总

| left_strategy | right_strategy | intersection_mean | intersection_min | intersection_max | jaccard_mean | jaccard_min | jaccard_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_lcmd | gradient_maxdet | 3.250000 | 0 | 7 | 0.055563 | 0.000000 | 0.122807 |
| gradient_lcmd | kernel_ivr | 5.125000 | 1 | 9 | 0.089488 | 0.015873 | 0.163636 |
| kernel_ivr | gradient_maxdet | 15.375000 | 12 | 22 | 0.323380 | 0.230769 | 0.523810 |

LCMD–IVR 的平均交集为 5.125/32，LCMD–MaxDet 为 3.250/32；IVR–MaxDet 较高，为 15.375/32，但仍远非同一批次。这既验证了 branch 确实分叉，也表明 observed performance differences 对应实际 acquisition differences。

## 哪些 test-blind feature 可能预测 preference

为避免 anchor 难度支配相关性，这里相关的是**策略间 short-AULC margin**，而不是各策略的绝对 AULC。下表每个成对 margin 仅列绝对 Spearman rho 最大的三个 feature；`left_minus_right < 0` 表示 left 更好。

| outcome | feature | spearman_rho | n_anchors |
| --- | --- | --- | --- |
| IVR_minus_MaxDet_AULC | coverage_nearest_distance_p90 | -0.761905 | 8 |
| IVR_minus_MaxDet_AULC | coverage_nearest_distance_mean | -0.714286 | 8 |
| IVR_minus_MaxDet_AULC | recent_validation_last_step_delta | 0.571429 | 8 |
| LCMD_minus_IVR_AULC | gradient_norm_std | -0.642857 | 8 |
| LCMD_minus_IVR_AULC | gradient_norm_p95 | -0.571429 | 8 |
| LCMD_minus_IVR_AULC | recent_validation_last_step_delta | -0.500000 | 8 |
| LCMD_minus_MaxDet_AULC | recent_validation_3step_linear_slope_per_label | 0.595238 | 8 |
| LCMD_minus_MaxDet_AULC | recent_validation_nrmse | -0.547619 | 8 |
| LCMD_minus_MaxDet_AULC | gradient_norm_std | -0.500000 | 8 |

最值得带入下一次预注册验证的是：

1. gradient dispersion（尤其 `gradient_norm_std` / `p95`），对 LCMD–IVR preference 的描述性关联最大；
2. coverage tail（尤其 `coverage_nearest_distance_p90`，其次 mean），对 IVR–MaxDet preference 的描述性关联最大；
3. recent validation slope 可作为候选补充，但 validation reuse 使它的解释更弱。

这些都是 `n=8`、多重探索性比较、无 p-value 的线索。不能据此设 threshold、训练 controller，或声称有统计显著性。完整结果见 `results/strategy_preference_correlations.csv`。

## 对七个科学问题的直接回答

1. **同一 state 下是否有不同未来收益？** 有。策略选择批次明显不同，short-AULC 跨策略差异在每个 anchor 都非零，最大达到 0.047463；但差异大小随 state 变化。
2. **ranking 是否随 budget 改变？** 部分改变（2/4），但不一致，不能归纳成固定 stage law。
3. **同 budget 是否随 source trajectory 改变？** 会（2/4），且 429 上两个 seed 都发生 crossover，这是 state/history effect 的直接描述性证据。
4. **seed dependence 多强？** 2/4 matched comparisons 改变 winner；对只有两个 development seeds 的研究而言，这足以阻止 controller 训练或强泛化。
5. **fixed stage switch 是否成立？** 不成立为稳健规则。数据不支持继续围绕固定 653 switch 调参。
6. **是否更应写成 `a_t = pi(s_t)`？** 是，作为下一步待确认的机制假设；不是已经学到的 policy。
7. **最可能的 state features？** gradient dispersion、coverage tail/mean，以及较弱的 recent validation slope；现阶段只应预注册后验证。

## Adaptive strategy 是否值得继续、下一步最小实验

值得继续做一次**确认性验证**，不值得现在实现 RL、contextual bandit、神经 policy 或自动阈值搜索。最小下一步是：

- 在未用于这些开发决策的新 seeds，最好是 held-out 时间/实验 cohort 上，预注册同一套 3 个冻结 selector；
- 保留两个 anchor stages 和 two-round rollout，但预先固定主终点为 short-AULC、次终点为 +64 NRMSE；
- 只预注册 `gradient_norm_std` 与 `coverage_nearest_distance_p90` 两个 test-blind diagnostics（validation slope 仅作附录）；
- 先检验 ranking/state interaction 能否复现，再决定是否值得收集足够 anchor states 来训练简单、可解释的 policy。

Gradient-MaxDet 在本研究中没有赢得任何 short-AULC anchor，只在 1/8 的 +64 endpoint 获胜；下一次仍可作为冻结机制 comparator，但没有证据把它作为 adaptive policy 的优先动作。

## Artifact index

- 原始 per-seed test metrics：`results/test_metrics.csv`
- 24 条 branch summary：`results/branch_results.csv`
- 24 条 pairwise comparison：`results/paired_comparison.csv`
- 8 条 winner matrix：`results/winner_matrix.csv`
- 32 条 state diagnostics（8 anchor + 24 branch-after-step-1）：`results/state_diagnostics.csv`
- 24 条 first-batch overlap：`results/selection_overlap.csv`
- 48 条 validation fit metrics：`results/validation_metrics.csv`
- test access audit：`results/test_label_access_audit.csv`
- postfreeze 派生表与哈希清单：`results/mean_branch_results.csv`、`results/winner_counts.csv`、`results/strategy_preference_correlations.csv`、`postfreeze_artifact_manifest.json`

报告修复只改变冻结后呈现层；详见 `REPORTING_NOTE.md`。
