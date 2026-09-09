# Controlled lightweight transfer audit: final report

## 1. Repository audit

The source checkpoint, five seeds, and filtered 25g/40g gradient-train identities are inherited exactly from the frozen filtered benchmark. Compound-grouped OOF is primary and row OOF is secondary. Conditional EA uses normalized source q50, the endpoint-wise standardized `u * centered_EA` interaction, `SSE/n`, and prior `(base_slope-1)^2 + intercept^2 + EA_effect^2`. The historical `fit_conditional()` implementation was not changed. No validation or outer-test truth was read.

## 2. Previous mismatch and correction

The historical generic varying-coefficient fit used unnormalized SSE, shrank the base slope toward zero, and left the intercept unpenalized. Its M2 magnitude also averaged V1 and V2 magnitude. Therefore the old M2 was not a strict nested comparison. The corrected extension retains Conditional EA's feature construction, normalization, priors, penalty grid, mass ratio, and source scales, and appends exactly one endpoint-specific effect. M3 now uses the same regularization contract in center/width coordinates and is correctly counted as 3 + 3 = 6 coefficients, not 8. Historical numerical artifacts remain untouched.

## 3. Nested-equivalence audit

```json
{
  "status": "PASS",
  "max_abs_nested_difference": 7.105427357601002e-15,
  "nested_match": true,
  "center_width_inverse_max_abs": 7.105427357601002e-15,
  "same_penalty": true,
  "same_source_scales": true,
  "same_mass_ratio": true,
  "same_EA": true
}
```

With a zero extra column, the corrected head matches historical Conditional EA to `7.105e-15`, comfortably below `1e-10`.

## 4. Primary compound GroupKFold OOF metrics

| column | method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | M1_conditional_EA | 8.0195 | 5.0208 | 0.3591 | 10.6237 | 7.0064 | 0.6268 | 0.8582 |
| 25g | M2_LOADING_SOLVENT | 7.9184 | 4.9987 | 0.3755 | 10.5129 | 7.1582 | 0.6346 | 0.8479 |
| 25g | M2_MAG | 7.3635 | 4.8355 | 0.4573 | 10.8905 | 7.4735 | 0.6071 | 0.8172 |
| 25g | M2_WIDTH | 8.0886 | 4.9331 | 0.3472 | 10.4910 | 6.9161 | 0.6362 | 0.8604 |
| 25g | M3_CENTER_WIDTH | 7.7158 | 4.5230 | 0.4068 | 10.7170 | 6.9810 | 0.6196 | 0.8377 |
| 40g | M1_conditional_EA | 15.3148 | 10.0923 | 0.6712 | 19.1812 | 13.8303 | 0.7316 | 1.6128 |
| 40g | M2_LOADING_SOLVENT | 15.2765 | 10.0639 | 0.6729 | 19.2667 | 13.8875 | 0.7292 | 1.6118 |
| 40g | M2_MAG | 15.3111 | 10.1777 | 0.6714 | 19.4891 | 14.0878 | 0.7229 | 1.6196 |
| 40g | M2_WIDTH | 15.4367 | 9.9631 | 0.6658 | 18.9025 | 13.9003 | 0.7394 | 1.6158 |
| 40g | M3_CENTER_WIDTH | 14.5559 | 9.4568 | 0.7030 | 18.7603 | 13.5758 | 0.7433 | 1.5451 |

These are means across the five fixed seeds. RMSE and MAE are in mL; R2 is secondary.

## 5. Relative gains, wins, and frozen gate

| method | aggregate_normalized_rmse | aggregate_relative_gain | endpoints_at_least_3pct | passes | 25g_V1_gain | 25g_V1_wins | 25g_V2_gain | 25g_V2_wins | 40g_V1_gain | 40g_V1_wins | 40g_V2_gain | 40g_V2_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2_MAG | 1.2184 | 0.0138 | 1 | False | 0.0818 | 4/5 | -0.0251 | 2/5 | 0.0002 | 3/5 | -0.0161 | 0/5 |
| M2_WIDTH | 1.2381 | -0.0021 | 0 | False | -0.0086 | 2/5 | 0.0125 | 4/5 | -0.0080 | 1/5 | 0.0145 | 5/5 |
| M2_LOADING_SOLVENT | 1.2298 | 0.0046 | 0 | False | 0.0126 | 4/5 | 0.0104 | 4/5 | 0.0025 | 3/5 | -0.0045 | 0/5 |
| M3_CENTER_WIDTH | 1.1914 | 0.0357 | 2 | False | 0.0379 | 5/5 | -0.0088 | 3/5 | 0.0496 | 5/5 | 0.0219 | 5/5 |

Relative gain is `1 - candidate_RMSE / M1_RMSE`; positive is better. Aggregate normalized RMSE is averaged across the two column contexts and five seeds. The M1 reference aggregate is 1.2355. The frozen gate requires at least 3% aggregate gain, at least 3 of 4 endpoints at 3% gain, and no context deterioration beyond 15%.

Decision: **NO_CONTROLLED_LIGHTWEIGHT_CALIBRATION_HEADROOM**. No candidate was promoted, so outer evaluation remained unauthorized and unread. The scalar audit followed the frozen sequence MAG -> WIDTH -> LOADING_SOLVENT; each failure authorized only the next stage. M3 was the independent parameterization check.

## 6. Center/width error decomposition

| column | protocol | center_error_variance | width_error_variance | center_width_error_covariance | center_width_error_correlation | raw_projection_frequency |
| --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | 72.4547 | 58.8778 | 27.6279 | 0.4279 | 0.0000 |
| 25g | row | 60.8575 | 59.0029 | 23.0487 | 0.3857 | 0.0000 |
| 40g | compound | 254.2135 | 111.1267 | 69.8034 | 0.4164 | 0.0000 |
| 40g | row | 263.2406 | 109.4256 | 75.5665 | 0.4462 | 0.0000 |

For M3, `eV1 = eC - 0.5 eW` and `eV2 = eC + 0.5 eW`, verified to floating-point precision in every seed. Thus `Var(eV1)=Var(eC)+0.25Var(eW)-Cov(eC,eW)` while `Var(eV2)=Var(eC)+0.25Var(eW)+Cov(eC,eW)`. The positive covariance (27.63 for 25g compound; 69.80 for 40g compound) suppresses V1 error variance but increases V2 error variance. This is the clearest diagnostic explanation for the endpoint asymmetry; it is not evidence of a causal chromatographic mechanism. Projection frequency is zero in every context, so the M3 gains do not come from post-hoc clipping.

## 7. Paired seed comparison against Conditional EA

| column | candidate | V1_rmse_delta_mean | V1_rmse_wins | V1_mae_delta_mean | V1_mae_wins | V2_rmse_delta_mean | V2_rmse_wins | V2_mae_delta_mean | V2_mae_wins | combined_normalized_rmse_delta_mean | combined_normalized_rmse_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | M2_MAG | -0.6560 | 4 | -0.1853 | 5 | 0.2668 | 2 | 0.4671 | 0 | -0.0410 | 4 |
| 40g | M2_MAG | -0.0037 | 3 | 0.0854 | 0 | 0.3079 | 0 | 0.2574 | 0 | 0.0068 | 0 |
| 25g | M2_WIDTH | 0.0691 | 2 | -0.0877 | 4 | -0.1327 | 4 | -0.0903 | 5 | 0.0022 | 2 |
| 40g | M2_WIDTH | 0.1219 | 1 | -0.1293 | 4 | -0.2788 | 5 | 0.0700 | 2 | 0.0031 | 1 |
| 25g | M2_LOADING_SOLVENT | -0.1011 | 4 | -0.0222 | 4 | -0.1108 | 4 | 0.1518 | 1 | -0.0103 | 4 |
| 40g | M2_LOADING_SOLVENT | -0.0383 | 3 | -0.0285 | 4 | 0.0855 | 0 | 0.0572 | 1 | -0.0010 | 3 |
| 25g | M3_CENTER_WIDTH | -0.3037 | 5 | -0.4978 | 5 | 0.0932 | 3 | -0.0254 | 4 | -0.0206 | 5 |
| 40g | M3_CENTER_WIDTH | -0.7589 | 5 | -0.6355 | 5 | -0.4209 | 5 | -0.2546 | 5 | -0.0676 | 5 |

Negative deltas favor the candidate. M3 is the only broad positive signal: V1 improves 5/5 seeds in both columns, and 40g V2 also improves 5/5. It still misses the endpoint-count gate because 25g V2 worsens and 40g V2 gains only 2.19%, below 3%. M2_MAG is mixed, while width and solvent effects are small and inconsistent.

## 8. Explicit scientific answers (Q1-Q11)

**Q1. How much did the old M2 result depend on mismatch?** The old aggregate gain was -3.80% (worse); the corrected endpoint-specific nested M2_MAG is +1.38%, a 5.18 percentage-point swing. Because regularization semantics and magnitude definition were corrected together, their individual contributions cannot be identified. The old negative result is therefore not a fair quantitative estimate, although the corrected candidate still fails the gate.

**Q2. Does corrected M2_MAG still fail?** Yes. Endpoint gains are +8.18%, -2.51%, +0.02%, and -1.61% for 25g V1/V2 and 40g V1/V2, with 4/5, 2/5, 3/5, and 0/5 wins. Aggregate gain is only +1.38%; 1/4 endpoints passes.

**Q3. Is endpoint-specific magnitude preferable?** It is mathematically more faithful to the stated endpoint hypothesis and is a strict nested extension. It is more favorable than the old averaged result, but not stable across endpoints or columns; no material incremental gain is established in this domain and formulation.

**Q4. Does source width add stable value?** No. Gains are -0.86%, +1.25%, -0.80%, and +1.45%, with 2/5, 4/5, 1/5, and 5/5 wins. Aggregate gain is -0.21%; 0/4 endpoints passes.

**Q5. Does loading solvent add value?** No material value. Gains are +1.26%, +1.04%, +0.25%, and -0.45%, with 4/5, 4/5, 3/5, and 0/5 wins. Aggregate gain is +0.46%; 0/4 endpoints passes. The retained indicator is categorical `I(DCM)`, not an ordinal numeric code.

**Q6. Do corrected Center/Width V1 gains remain?** Yes: +3.79% for 25g V1 and +4.96% for 40g V1.

**Q7. Are M3 gains seed-consistent?** V1 is 5/5 in both columns. V2 is 3/5 at 25g and 5/5 at 40g. Only 2/4 endpoint means exceed 3%, so consistency does not rescue promotion.

**Q8. What drives the Center/Width pattern?** The positive center-width error covariance subtracts from V1 variance and adds to V2 variance. The available decomposition supports a covariance-based explanation of the asymmetry, but cannot uniquely attribute improvement versus M1 to center, width, or covariance because it is diagnostic rather than a model-selection contrast.

**Q9. Did any candidate pass?** No. M3 reached +3.57% aggregate gain but passed only 2/4 endpoint gates; all scalar extensions were below the aggregate and endpoint requirements.

**Q10. Is calibration-variable expansion closed?** Yes, for the current filtered operational domain, source predictor, and declared low-capacity calibration family. No fourth variable or additional model is authorized.

**Q11. Is a filtered random learning curve next?** Yes, as a separately preregistered future stage at B=30/50/100/150/200/FULL. It was not started here; active learning remains unauthorized until label scarcity is demonstrated.

## 9. Interpretation limits

The results do not show that magnitude, width, solvent, or center/width coordinates lack physical relevance. They show only that these declared low-capacity additions did not deliver gate-level incremental train-only OOF accuracy for the current predictor and retained population. No causal mechanism, outer-test performance, or independent generalization claim follows.

## 10. Reproducibility

The measured runner time was 1.66 seconds. Machine-readable metrics, paired comparisons, gate decisions, decomposition, equivalence audit, and SHA-256 manifest accompany this report.
