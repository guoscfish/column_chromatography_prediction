# Clean-QGeoGNN 4g baseline qualification report

Status: `4G_BASELINE_QUALIFICATION_COMPLETE / POINT_PREDICTOR_BASELINE_READY / UQ_REQUIRES_FURTHER_QUALIFICATION`.

## A. Protocol

The only trained model was `qgeognn_clean_fusion_v1`, preflight revision 2. The frozen 4g continuity domain was `V1_ml <= 60` and `V2_ml <= 120`, retaining exactly 4,163 rows and 217 canonical compounds. This is a `PROJECT_CONTINUITY_DECISION`, not evidence that the thresholds are physically or scientifically optimal.

Each of the row-interpolation and compound-generalization estimands used seeds 42, 525, and 1101 with fixed approximately 80/10/10 train/validation/test manifests. Training used Adam, learning rate 0.001, batch size 2,048, maximum 1,000 epochs, patience 100, no scheduler, equal V1/V2 weights, and validation-only selection by combined normalized RMSE. Every data-dependent scale was fit on that run's training rows. Validation, test, and 8g rows used in normalization were all zero. Test was evaluated once after the validation-best checkpoint was frozen and reloaded; no test metric appears in training history.

## B. Split audit

The source SHA256 was `d485d3d46a96458d1baac11b2a21cd33374b1be42ba2303e7ae5823cc8ee553a`; the qualification dataset-manifest SHA256 was `8a291e7cbe0e89659940785a6e957f747cfe37ea2786c4280b654afcd531490b`.

| Estimand | Seed | Train/validation/test rows | Train/validation/test compounds | Compound overlaps (T–V/T–test/V–test) |
|---|---:|---:|---:|---:|
| Row interpolation | 42 | 3330 / 416 / 417 | 217 / 182 / 181 | 182 / 181 / 154 (allowed) |
| Row interpolation | 525 | 3330 / 416 / 417 | 217 / 185 / 182 | 185 / 182 / 157 (allowed) |
| Row interpolation | 1101 | 3330 / 416 / 417 | 217 / 178 / 185 | 178 / 185 / 152 (allowed) |
| Compound generalization | 42 | 3401 / 389 / 373 | 173 / 22 / 22 | 0 / 0 / 0 |
| Compound generalization | 525 | 3321 / 428 / 414 | 173 / 22 / 22 | 0 / 0 / 0 |
| Compound generalization | 1101 | 3275 / 430 / 458 | 173 / 22 / 22 | 0 / 0 / 0 |

The compound manifests prove pairwise-empty canonical-SMILES intersections. The row task intentionally permits overlap and is not an unseen-molecule estimate.

## C. 4g point prediction

### Row interpolation

| Seed/statistic | V1 R² | V1 RMSE | V1 MAE | V2 R² | V2 RMSE | V2 MAE | Normalized RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.482 | 5.660 | 4.235 | 0.501 | 10.883 | 7.720 | 0.698 |
| 525 | 0.471 | 5.412 | 4.290 | 0.495 | 11.080 | 8.289 | 0.699 |
| 1101 | 0.355 | 6.417 | 4.131 | 0.380 | 13.109 | 7.451 | 0.812 |
| Mean | 0.436 | 5.829 | 4.219 | 0.459 | 11.691 | 7.820 | 0.736 |
| Median | 0.471 | 5.660 | 4.235 | 0.495 | 11.080 | 7.720 | 0.699 |
| Sample SD | 0.070 | 0.524 | 0.081 | 0.068 | 1.232 | 0.428 | 0.066 |

### Compound generalization

| Seed/statistic | V1 R² | V1 RMSE | V1 MAE | V2 R² | V2 RMSE | V2 MAE | Normalized RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.186 | 7.897 | 5.130 | 0.203 | 15.197 | 9.201 | 0.982 |
| 525 | 0.383 | 5.626 | 3.921 | 0.396 | 10.732 | 6.364 | 0.698 |
| 1101 | 0.156 | 6.723 | 4.156 | 0.194 | 13.571 | 7.352 | 0.839 |
| Mean | 0.242 | 6.749 | 4.403 | 0.264 | 13.167 | 7.639 | 0.839 |
| Median | 0.186 | 6.723 | 4.156 | 0.203 | 13.571 | 7.352 | 0.839 |
| Sample SD | 0.123 | 1.136 | 0.641 | 0.114 | 2.260 | 1.440 | 0.142 |

## D. Quantile/UQ

Nominal q10–q90 coverage is 0.8. `Cross` is the within-target quantile crossing rate; `Median flip` is q50(V1)>q50(V2); `Interval overlap` is q90(V1)>q10(V2).

| Estimand | Seed/statistic | V1 coverage/width/pinball | V2 coverage/width/pinball | Cross | Median flip | Interval overlap |
|---|---|---|---|---:|---:|---:|
| Row | 42 | 0.813 / 7.617 / 1.173 | 0.801 / 15.717 / 2.228 | 0 | 0.192 | 0.731 |
| Row | 525 | 0.787 / 8.237 / 1.152 | 0.767 / 16.643 / 2.356 | 0 | 0.204 | 0.715 |
| Row | 1101 | 0.758 / 10.539 / 1.287 | 0.688 / 16.548 / 2.528 | 0 | 0.000 | 0.878 |
| Row | Mean | 0.786 / 8.798 / 1.204 | 0.752 / 16.302 / 2.371 | 0 | 0.132 | 0.775 |
| Compound | 42 | 0.657 / 10.191 / 1.637 | 0.611 / 17.610 / 3.161 | 0 | 0.000 | 0.960 |
| Compound | 525 | 0.790 / 9.141 / 1.156 | 0.710 / 15.973 / 2.082 | 0 | 0.000 | 0.792 |
| Compound | 1101 | 0.653 / 9.864 / 1.453 | 0.605 / 15.189 / 2.590 | 0 | 0.000 | 0.707 |
| Compound | Mean | 0.700 / 9.732 / 1.415 | 0.642 / 16.257 / 2.611 | 0 | 0.000 | 0.820 |

The monotonic head maintained within-target ordering in every prediction. Intervals are non-degenerate, but V2 is under-covered in row interpolation and both targets are materially under-covered in compound generalization. Point-baseline readiness therefore coexists with `UQ_REQUIRES_FURTHER_QUALIFICATION`.

## E. Condition usage

Each cell for permutation/disabled is the change from full prediction `(ΔRMSE, ΔMAE, ΔR²)`; positive error changes and negative R² changes mean degradation. The optional condition-only columns report RMSE and are explanatory, not a separate baseline.

| Estimand/seed | Full RMSE V1/V2 | Permuted Δ V1 | Permuted Δ V2 | Disabled Δ V1 | Disabled Δ V2 | Condition-only RMSE V1/V2 |
|---|---:|---:|---:|---:|---:|---:|
| Row 42 | 5.660 / 10.883 | +4.694, +3.070, −1.215 | +8.705, +6.206, −1.117 | +3.527, +2.428, −0.846 | +7.939, +6.235, −0.993 | 8.638 / 15.871 |
| Row 525 | 5.412 / 11.080 | +5.033, +3.361, −1.441 | +8.889, +6.566, −1.135 | +3.089, +2.362, −0.776 | +6.967, +5.513, −0.834 | 8.749 / 17.274 |
| Row 1101 | 6.417 / 13.109 | +1.894, +1.626, −0.437 | +2.676, +2.608, −0.279 | +2.811, +2.316, −0.689 | +5.447, +5.037, −0.622 | 8.029 / 17.322 |
| Compound 42 | 7.897 / 15.197 | +1.950, +1.600, −0.452 | +2.659, +2.417, −0.303 | +2.956, +2.606, −0.723 | +6.368, +6.003, −0.808 | 9.175 / 17.432 |
| Compound 525 | 5.626 / 10.732 | +2.180, +1.592, −0.571 | +2.808, +2.715, −0.358 | +2.769, +2.426, −0.757 | +5.480, +5.115, −0.774 | 7.806 / 14.240 |
| Compound 1101 | 6.723 / 13.571 | +0.877, +0.812, −0.235 | +1.092, +1.116, −0.135 | +1.447, +0.883, −0.402 | +3.186, +2.216, −0.423 | 7.822 / 17.372 |

Condition permutation increased both target RMSEs in 6/6 runs; condition disabling did the same in 6/6. Every corresponding MAE increased and R² decreased. The effects are largest in row interpolation and smaller but directionally consistent in compound generalization. Clean therefore shows learned predictive reliance on experimental conditions, not merely structural forward reachability.

## F. Representation/gradient diagnostics

Gradient norms use the validation rows at the validation-best model and are explanatory only.

| Estimand/seed | Molecule latent L2 | Condition latent L2 | V1 molecule grad | V1 condition grad | V2 molecule grad | V2 condition grad |
|---|---:|---:|---:|---:|---:|---:|
| Row 42 | 10.949 | 12.175 | 3.118 | 25.312 | 20.146 | 48.023 |
| Row 525 | 11.910 | 12.513 | 9.235 | 27.955 | 30.994 | 202.899 |
| Row 1101 | 10.155 | 10.249 | 6.230 | 12.603 | 6.170 | 22.812 |
| Compound 42 | 10.148 | 10.815 | 21.570 | 14.824 | 38.586 | 58.083 |
| Compound 525 | 10.171 | 10.570 | 16.468 | 37.134 | 44.885 | 76.239 |
| Compound 1101 | 9.809 | 9.656 | 6.933 | 9.754 | 16.894 | 26.751 |

Both latent branches are non-zero and comparably scaled. All molecular-projection and condition-encoder gradient norms are non-zero for both targets. V2 gradients are generally larger, especially the row/525 condition gradient; under the frozen loss contract this is recorded as a diagnostic and does not trigger reweighting.

## G. Stability

Gate A passed: 6/6 runs completed; all histories, predictions, and metrics were finite; all artifact hashes matched; checkpoint hashes matched runtime files; reload validation passed; prediction shapes matched test counts; histories contained no test columns; and within-target crossing was zero. Best epochs were 340, 503, and 115 for row splits and 144, 129, and 75 for compound splits.

There was no failed seed or catastrophic learning failure: all twelve target-level test R² values were positive. Row interpolation was stronger and more stable. Compound generalization showed the expected performance gap and more seed variation (normalized-RMSE SD 0.142 versus 0.066), with seed 42 the weakest, but not a collapse. This describes usability and stability; it is not a Legacy-superiority test.

## H. Scientific conclusion

### This round establishes

- The frozen Clean implementation completes a leakage-controlled six-run formal 4g qualification with numerically valid artifacts.
- It provides usable point predictions for row interpolation and positive-R², weaker but non-catastrophic predictions for unseen compounds.
- It has consistent learned reliance on experimental conditions across both estimands and all seeds.
- Its monotonic quantile parameterization preserves within-target order, while empirical interval calibration—especially compound-generalization coverage—requires more work.
- Clean-QGeoGNN may be designated `4G_BASELINE_QUALIFICATION_COMPLETE` and `POINT_PREDICTOR_BASELINE_READY`, with `UQ_REQUIRES_FURTHER_QUALIFICATION` retained.

### This round does not establish

- Clean is statistically superior to Legacy.
- A geometry-aware GNN is superior to a descriptor-only MLP.
- The paper QGeoGNN method has been reproduced exactly.
- Clean improves 4g→8g transfer.
- Clean improves active learning or active transfer.
- The historical 60/120 thresholds are scientifically optimal.

## I. Recommended next gate

`NEXT: UQ qualification/calibration before transfer-active-learning work.`

This is the sole recommended next gate because point behavior is sufficiently stable for a baseline and condition use is demonstrated, while compound-generalization q10–q90 coverage is systematically below nominal. No UQ tuning, 8g transfer, or active-learning work was executed in this study.

Machine-readable support is in `results/aggregate_summary.json`, `formal_run_audit.json`, and the four summary CSV files. Per-run metrics and predictions remain the primary evidence.
