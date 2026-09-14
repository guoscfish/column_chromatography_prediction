# Column-conditioned multi-task QGeoGNN: final report

Decision: `REPRESENTATION_GATE_PASSED`.

25g: +5.05% mean paired-fold NRMSE gain, 5/5 seed wins, 18/25 fold wins; 40g: +8.18% mean paired-fold NRMSE gain, 5/5 seed wins, 20/25 fold wins.

## Preregistered continuation gate

| column | mean_fold_gain_pct | seed_wins | fold_wins | worst_endpoint_change_pct | worst_tail_change_pct | passed |
| --- | --- | --- | --- | --- | --- | --- |
| 25g | 5.0541 | 5/5 | 18/25 | -3.0197 | -4.6970 | True |
| 40g | 8.1766 | 5/5 | 20/25 | -5.8409 | -10.7955 | True |

Mean gain is the arithmetic mean of 25 paired fold-relative gains. Seed wins
compare each seed's mean fold NRMSE. All five criteria are required in BOTH
columns. Positive endpoint/tail changes in this table mean deterioration.

## COMPOUND inner validation metrics

| column | arm | V1_rmse | V1_mae | V2_rmse | V2_mae | combined_normalized_rmse | V1_r2 | V2_r2 | Center_rmse | Width_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | A1 | 6.7700 | 4.4942 | 10.3928 | 7.1815 | 0.6406 | 0.4894 | 0.6287 | 7.7868 | 7.7136 |
| 25g | A2 | 6.5655 | 4.3366 | 9.5123 | 6.6764 | 0.6061 | 0.5163 | 0.6864 | 7.2560 | 7.1888 |
| 40g | A1 | 16.1390 | 10.4006 | 21.4666 | 14.4362 | 0.5954 | 0.6175 | 0.6484 | 18.0502 | 11.7355 |
| 40g | A2 | 15.1964 | 9.9930 | 18.7798 | 13.1496 | 0.5413 | 0.6586 | 0.7300 | 16.2523 | 10.5437 |

RMSE/MAE are mL. Combined NRMSE uses inner-train-only endpoint scales.
The values average 25 folds; folds across repeated outer seeds overlap and
are not 25 independent experiments. These inner-selected metrics are
developmental selection evidence, not an unbiased external generalization estimate.

## Tail and retention regimes

| column | arm | endpoint | tail_rmse | tail_count | tail_sse_fraction |
| --- | --- | --- | --- | --- | --- |
| 25g | A1 | V1 | 9.4899 | 63.8000 | 0.3803 |
| 25g | A1 | V2 | 16.7829 | 63.6000 | 0.5090 |
| 25g | A2 | V1 | 9.0442 | 63.8000 | 0.3653 |
| 25g | A2 | V2 | 14.5781 | 63.6000 | 0.4586 |
| 40g | A1 | V1 | 30.9252 | 72.4000 | 0.7189 |
| 40g | A1 | V2 | 39.8782 | 72.4000 | 0.6781 |
| 40g | A2 | V1 | 27.5866 | 72.4000 | 0.6521 |
| 40g | A2 | V2 | 31.5269 | 72.4000 | 0.5650 |

Thresholds are per-endpoint outer gradient_train 80th percentiles. Tail SSE and
count are pooled over each seed's five OOF folds, then seed RMSEs are averaged.
Displayed counts are mean per-seed counts, not unique observations across seeds.
Full thresholds/counts/SSE are in `TAIL_METRICS.csv` and `TAIL_SEED_METRICS.csv`.

| column | arm | regime | V1_rmse | V2_rmse | n |
| --- | --- | --- | --- | --- | --- |
| 25g | A1 | high | 8.9635 | 14.1219 | 21.1200 |
| 25g | A1 | low | 4.3193 | 7.1553 | 21.0400 |
| 25g | A1 | mid | 5.2669 | 7.4467 | 21.0000 |
| 25g | A2 | high | 8.8771 | 12.4154 | 21.1200 |
| 25g | A2 | low | 3.8430 | 6.7802 | 21.0400 |
| 25g | A2 | mid | 5.2269 | 7.3190 | 21.0000 |
| 40g | A1 | high | 24.9006 | 32.7165 | 24.1200 |
| 40g | A1 | low | 6.3576 | 8.8488 | 24.0400 |
| 40g | A1 | mid | 9.5101 | 14.3561 | 24.0000 |
| 40g | A2 | high | 22.9898 | 27.2176 | 24.1200 |
| 40g | A2 | low | 6.3171 | 10.5275 | 24.0400 |
| 40g | A2 | mid | 10.0733 | 13.7978 | 24.0000 |

Regimes use outer gradient_train center tertiles. They are descriptive slices,
not selectors. Center/Width error variance and covariance are retained in
`INNER_RESULTS.csv`; endpoint changes must accompany any Center/Width claim.

## Gradient compatibility

| arm | pair | observations | finite_cosine_observations | mean_cosine | median_cosine | negative_fraction | seeds_with_majority_negative_cosine | seed_count | median_magnitude_ratio | zero_norm_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 4g_25g | 946 | 946 | 0.0011 | -0.0361 | 0.5455 | 5 | 5 | 1.5169 | 0 |
| A1 | 4g_40g | 946 | 946 | -0.3002 | -0.3118 | 0.7336 | 5 | 5 | 2.1991 | 0 |
| A1 | 25g_40g | 946 | 946 | -0.0382 | -0.1257 | 0.5708 | 5 | 5 | 2.2779 | 0 |
| A2 | 4g_25g | 486 | 486 | -0.0174 | -0.0362 | 0.5720 | 3 | 5 | 1.3183 | 0 |
| A2 | 4g_40g | 486 | 486 | -0.1868 | -0.2170 | 0.7613 | 5 | 5 | 2.3826 | 0 |
| A2 | 25g_40g | 486 | 486 | 0.0587 | -0.0753 | 0.5720 | 4 | 5 | 2.0855 | 0 |

At least one task pair has negative cosine on a majority of sampled steps, consistent with persistent conflict under this recipe. No pair has a median gradient-magnitude ratio above 10. These are descriptive thresholds, not training rules or causal proof. The diagnostics use shared late-backbone parameters, excluding heads, FiLM generators, and condition-completion parameters.

Sampling cadence was fixed at epoch 1 and every 10 epochs. The raw per-task
losses and all per-task validation metrics for every epoch are retained in
`INNER_EPOCH_HISTORY.csv.gz`. Gradient diagnostics did not change training.

## Convergence

| arm | fits | mean_best_epoch | min_best_epoch | max_best_epoch | mean_epochs_run | selections_at_ceiling | runs_at_ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 25 | 304.9600 | 104 | 500 | 371.4000 | 1 | 7 |
| A2 | 25 | 109.8800 | 45 | 449 | 188.7200 | 0 | 1 |

The budget and patience were fixed before fitting. Ceiling selections are
reported as possible budget censoring; they do not authorize more epochs or
different learning rates. A1 and A2 always select one joint checkpoint per fold.

## Outer confirmation

All outer results below are DEVELOPMENTAL CONFIRMATION. Compound P0 is unavailable and was not retrained.

| protocol | column | method | V1_rmse | V1_mae | V2_rmse | V2_mae | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compound | 25g | A1 | 6.7885 | 4.0617 | 9.7734 | 6.7433 | 0.6230 |
| compound | 25g | A2 | 6.7168 | 4.3635 | 9.5381 | 6.8487 | 0.6138 |
| compound | 25g | HIER_CW_SHARED_LAMBDA_CORRECTED | 6.5948 | 4.4675 | 8.9780 | 6.7478 | 0.5920 |
| compound | 25g | paper_style_current_v2 | 8.0408 | 5.8364 | 10.9594 | 8.3380 | 0.7227 |
| compound | 40g | A1 | 16.8523 | 9.9518 | 21.7066 | 12.9035 | 0.6099 |
| compound | 40g | A2 | 16.5299 | 10.4744 | 20.9381 | 13.9721 | 0.5938 |
| compound | 40g | HIER_CW_SHARED_LAMBDA_CORRECTED | 15.4212 | 9.9483 | 20.0801 | 13.3097 | 0.5614 |
| compound | 40g | paper_style_current_v2 | 16.2397 | 11.0012 | 22.0833 | 15.3895 | 0.6034 |
| row | 25g | A1 | 7.1913 | 3.7458 | 10.1394 | 5.8350 | 0.6433 |
| row | 25g | A2 | 7.3085 | 4.2799 | 10.1739 | 6.4816 | 0.6504 |
| row | 25g | HIER_CW_SHARED_LAMBDA_CORRECTED | 8.0087 | 4.5211 | 11.0307 | 7.1133 | 0.7087 |
| row | 25g | P0 | 7.2714 | 4.3738 | 10.9866 | 6.9250 | 0.6683 |
| row | 25g | paper_style_current_v2 | 7.4088 | 4.4615 | 11.1114 | 6.9671 | 0.6788 |
| row | 40g | A1 | 11.9158 | 7.1466 | 14.7453 | 9.3414 | 0.4214 |
| row | 40g | A2 | 13.0467 | 7.9289 | 16.1155 | 10.6111 | 0.4609 |
| row | 40g | HIER_CW_SHARED_LAMBDA_CORRECTED | 14.3476 | 8.9379 | 17.4355 | 12.5556 | 0.5030 |
| row | 40g | P0 | 13.7451 | 8.7661 | 16.4034 | 11.1422 | 0.4778 |
| row | 40g | paper_style_current_v2 | 13.1091 | 7.9872 | 15.8968 | 10.6430 | 0.4591 |

| column | reference | mean_nrmse_gain | seed_wins | worst_endpoint_deterioration | worst_tail_deterioration | row_nrmse_deterioration | row_endpoint_deterioration | row_tail_deterioration | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | A1 | 0.0147 | 3 | -0.0106 | -0.0280 | 0.0110 | 0.0163 | 0.1504 | False |
| 25g | HIER_CW_SHARED_LAMBDA_CORRECTED | -0.0369 | 2 | 0.0624 | 0.0813 | -0.0823 | -0.0777 | 0.0443 | False |
| 25g | paper_style_current_v2 | 0.1506 | 4 | -0.1297 | -0.1388 | -0.0418 | -0.0135 | -0.0397 | True |
| 40g | A1 | 0.0265 | 4 | -0.0191 | -0.0728 | 0.0937 | 0.0949 | 0.0726 | False |
| 40g | HIER_CW_SHARED_LAMBDA_CORRECTED | -0.0576 | 2 | 0.0719 | 0.0853 | -0.0836 | -0.0757 | -0.0216 | False |
| 40g | paper_style_current_v2 | 0.0160 | 2 | 0.0179 | 0.1310 | 0.0040 | 0.0138 | 0.0829 | False |

The inherited COMPOUND outer partitions are per-column. Identity-only overlap:
25g: 7-11 test molecules per seed seen in the other target's gradient_train; 40g: 7-10 test molecules per seed seen in the other target's gradient_train. Exact required outer roles
were preserved; donor rows were not silently purged. Thus any outer result is
within-column compound holdout with related-task supervision, not a globally
unseen target-molecule test. This does not affect the globally grouped inner
continuation gate. Details are in `OUTER_CROSS_TASK_MOLECULE_OVERLAP.csv`.

## Required scientific questions

1. Does A1 outperform existing target-only transfer? See the matched DEVELOPMENTAL CONFIRMATION table: A1 can be compared with paper-style and corrected HIER on both protocols and P0 on ROW. No compound converged-P0 comparison is available, so a universal target-only superiority claim remains limited.

2. Does column FiLM improve over A1? 25g: +5.05% mean paired-fold NRMSE gain, 5/5 seed wins, 18/25 fold wins; 40g: +8.18% mean paired-fold NRMSE gain, 5/5 seed wins, 20/25 fold wins. The complete continuation gate passes.

3. Is gain replicated in both columns? Both columns satisfy the inner gate; outer interpretation is given above.

4. Is it replicated across seeds? 25g: +5.05% mean paired-fold NRMSE gain, 5/5 seed wins, 18/25 fold wins; 40g: +8.18% mean paired-fold NRMSE gain, 5/5 seed wins, 20/25 fold wins. Wins alone do not satisfy the magnitude and endpoint/tail guards.

5. Is it driven by only V1 or V2? Inner absolute-RMSE relative gains: 25g: V1 +3.02%, V2 +8.47%; 40g: V1 +5.84%, V2 +12.52%. Use these endpoint-specific changes rather than the combined score alone.

6. Does it reduce absolute RMSE, not only improve R2? 25g: V1 +3.02%, V2 +8.47%; 40g: V1 +5.84%, V2 +12.52%. These are direct RMSE comparisons; R2 is secondary and did not select a candidate.

7. Does it improve the high-volume tail? 25g: V1 -4.70%, V2 -13.14% RMSE change (positive worsens); 40g: V1 -10.80%, V2 -20.94% RMSE change (positive worsens). The gate requires both endpoint tail guards for both columns.

8. Does it improve both Center and Width? Inner RMSE relative gains: 25g: Center +6.82%, Width +6.80%; 40g: Center +9.96%, Width +10.16%. Center and Width are algebraic surrogates; the covariance diagnostics do not establish a physical mechanism.

9. What do gradient cosines imply? At least one task pair has negative cosine on a majority of sampled steps, consistent with persistent conflict under this recipe. No pair has a median gradient-magnitude ratio above 10. These are descriptive thresholds, not training rules or causal proof. The diagnostics use shared late-backbone parameters, excluding heads, FiLM generators, and condition-completion parameters.

10. Does A2 exceed the strongest legitimate baseline? PROJECT_TRANSFER_GAIN=False under the conservative all-compatible-reference guard. Compound P0 remains unavailable.

11. Must column identity affect the molecular representation before readout? The inner gate passed; outer REPRESENTATION_SIGNAL=False. This is predictive evidence on historically exposed identities, not physical or source-unseen OOD evidence.

12. Is complexity scientifically justified? Project-level promotion is not supported; independent confirmation is still needed.

13. Recommended next stage: a) a separately preregistered gradient-conflict handling control. The same task pair has majority-negative sampled gradients in at least 3/5 seeds for both architectures. This supports isolating gradient conflict as the next computational mechanism; it does not establish that a correction will improve prediction. Independent/crossed data collection remains the external-validation priority. Domain-specific normalization has not been isolated here. Uncontrolled neural architecture expansion should stop pending new evidence. No next-stage method was implemented.

## Provenance and restrictions

Source SHA256: `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. Split SHA256: `33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3`.
Protocol SHA256: `fb7595d661ccaf7a228453536f0f34c891900e76015c93d50a043ff5b63bf567`.
Decision SHA256: `c65d52edb669c77a33b0325c6e20355dd97c6f9214efab6221fe3433f2e47189`.

The source replay covers all 4,163 qualified rows, with maximum frozen-CSV
difference 3.8086e-6. Active-source/A1 and A1/A2 epoch-zero differences are zero.
Execution code/environment hashes, exact outer roles, inner identities, fold
hashes, histories, and completion digests are retained. Qualified 4g labels
intentionally overlap target compounds. Task embeddings represent known task
identity only. No 8g, PCGrad, GradNorm, domain-specific BN, calibration, scaling,
geometry descriptor, loss/width/embedding sweep, ensemble, or Active Learning
was added, and historical artifacts were preserved.
