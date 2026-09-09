# Structured center/width follow-up: final report

## Protocol and mathematical audit

This train-only study inherits the exact filtered populations, source predictions, source scales, gradient-train identities, protocols, and five seeds from the corrected audit. M3 uses independent three-coefficient Conditional-EA center and width branches (`SSE/n`, base slope prior 1, all other priors 0). Candidate A adds one center-magnitude slope coefficient; Candidate B adds one source-center coefficient only to the width slope. All scales and effect standardization are fit-subset-only.

Equal-weight endpoint SSE is not a candidate: `eV1^2+eV2^2=2eC^2+0.5eW^2`, which remains separable for independent center/width parameter blocks. A shared optimizer alone would not create joint modeling.

Formulas: `M3: Ct=[a0+aEA*EA]Cs+bC; Wt=[d0+dEA*EA]Ws+bW`. `CENTER_MAGNITUDE` adds `aMAG*log1p(max(Cs,0)/fit_center_scale)` to the center slope. `CENTER_CONDITIONED_WIDTH` adds `dC*standardized(Cs)` to the width slope. Parameter counts are 6, 7, and 7.

## Nested-equivalence audit

```json
{
  "status": "PASS",
  "tolerance": 1e-10,
  "max_abs_prediction_difference": {
    "CENTER_MAGNITUDE": 0.0,
    "CENTER_CONDITIONED_WIDTH": 0.0
  },
  "same_loss_normalization": true,
  "same_ridge_priors": true,
  "same_EA_construction": true,
  "same_projection": true
}
```

## Compound primary metrics

| column | method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | CENTER_CONDITIONED_WIDTH | 7.3447 | 4.5412 | 0.4631 | 11.0482 | 7.0726 | 0.5964 | 0.8190 |
| 25g | CENTER_MAGNITUDE | 7.6534 | 4.7506 | 0.4165 | 10.7464 | 7.1062 | 0.6179 | 0.8338 |
| 25g | M3_CENTER_WIDTH | 7.7158 | 4.5230 | 0.4068 | 10.7170 | 6.9810 | 0.6196 | 0.8377 |
| 40g | CENTER_CONDITIONED_WIDTH | 14.6645 | 9.5210 | 0.6986 | 18.8175 | 13.6234 | 0.7417 | 1.5547 |
| 40g | CENTER_MAGNITUDE | 14.7513 | 9.5800 | 0.6950 | 18.9222 | 13.6757 | 0.7389 | 1.5637 |
| 40g | M3_CENTER_WIDTH | 14.5559 | 9.4568 | 0.7030 | 18.7603 | 13.5758 | 0.7433 | 1.5451 |

## Row secondary metrics

| column | method | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | CENTER_CONDITIONED_WIDTH | 7.1744 | 4.5354 | 0.4911 | 9.9717 | 6.6703 | 0.6913 | 0.7791 |
| 25g | CENTER_MAGNITUDE | 7.2510 | 4.6172 | 0.4785 | 9.9261 | 6.8137 | 0.6944 | 0.7836 |
| 25g | M3_CENTER_WIDTH | 7.2224 | 4.4407 | 0.4834 | 9.9065 | 6.7063 | 0.6956 | 0.7810 |
| 40g | CENTER_CONDITIONED_WIDTH | 14.6830 | 9.6432 | 0.7035 | 19.1241 | 13.5735 | 0.7352 | 1.5633 |
| 40g | CENTER_MAGNITUDE | 14.7183 | 9.6764 | 0.7020 | 19.1689 | 13.6355 | 0.7339 | 1.5670 |
| 40g | M3_CENTER_WIDTH | 14.6478 | 9.6109 | 0.7048 | 19.1283 | 13.5896 | 0.7350 | 1.5607 |

## Frozen gate

| method | aggregate_normalized_rmse | aggregate_gain | positive_endpoints | endpoints_at_3pct | passes | 25g_V1_gain | 25g_V1_wins | 25g_V2_gain | 25g_V2_wins | 40g_V1_gain | 40g_V1_wins | 40g_V2_gain | 40g_V2_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CENTER_MAGNITUDE | 1.1988 | -0.0062 | 1 | 0 | False | 0.0081 | 4/5 | -0.0027 | 2/5 | -0.0134 | 0/5 | -0.0086 | 0/5 |
| CENTER_CONDITIONED_WIDTH | 1.1869 | 0.0038 | 1 | 1 | False | 0.0481 | 5/5 | -0.0309 | 1/5 | -0.0075 | 0/5 | -0.0031 | 2/5 |

Final status: **NO_STRUCTURED_CENTER_WIDTH_HEADROOM**. Outer evaluation was not executed; outer truth remained blind.

## Compound paired five-seed comparisons

| column | protocol | candidate | reference | n_pairs | V1_rmse_delta_mean | V1_rmse_delta_median | V1_rmse_wins | V1_rmse_relative_gain | V1_mae_delta_mean | V1_mae_delta_median | V1_mae_wins | V1_mae_relative_gain | V2_rmse_delta_mean | V2_rmse_delta_median | V2_rmse_wins | V2_rmse_relative_gain | V2_mae_delta_mean | V2_mae_delta_median | V2_mae_wins | V2_mae_relative_gain | combined_normalized_rmse_delta_mean | combined_normalized_rmse_delta_median | combined_normalized_rmse_wins | combined_normalized_rmse_relative_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | CENTER_MAGNITUDE | M3_CENTER_WIDTH | 5 | -0.0625 | -0.1884 | 4 | 0.0081 | 0.2276 | 0.2285 | 0 | -0.0503 | 0.0294 | 0.1526 | 2 | -0.0027 | 0.1252 | 0.2039 | 1 | -0.0179 | -0.0038 | -0.0120 | 4 | 0.0046 |
| 40g | compound | CENTER_MAGNITUDE | M3_CENTER_WIDTH | 5 | 0.1954 | 0.2251 | 0 | -0.0134 | 0.1231 | 0.1246 | 0 | -0.0130 | 0.1619 | 0.2143 | 0 | -0.0086 | 0.1000 | 0.1174 | 1 | -0.0074 | 0.0186 | 0.0197 | 0 | -0.0120 |
| 25g | compound | CENTER_CONDITIONED_WIDTH | M3_CENTER_WIDTH | 5 | -0.3711 | -0.4548 | 5 | 0.0481 | 0.0182 | 0.0286 | 2 | -0.0040 | 0.3312 | 0.3268 | 1 | -0.0309 | 0.0916 | 0.1612 | 1 | -0.0131 | -0.0186 | -0.0163 | 5 | 0.0222 |
| 40g | compound | CENTER_CONDITIONED_WIDTH | M3_CENTER_WIDTH | 5 | 0.1086 | 0.1326 | 0 | -0.0075 | 0.0641 | 0.0802 | 1 | -0.0068 | 0.0572 | 0.0562 | 2 | -0.0031 | 0.0476 | 0.0300 | 0 | -0.0035 | 0.0096 | 0.0115 | 0 | -0.0062 |

Deltas are candidate minus M3; negative favors the candidate. Relative gain is positive when the candidate is better.

## Center/width variance-covariance diagnostics

| column | protocol | method | center_error_variance | width_error_variance | center_width_error_covariance | center_width_error_correlation | V1_error_variance | V2_error_variance | projection_frequency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | CENTER_CONDITIONED_WIDTH | 71.6508 | 65.4157 | 34.0515 | 0.5017 | 53.9533 | 122.0562 | 0.0000 |
| 25g | compound | CENTER_MAGNITUDE | 71.7354 | 60.5512 | 28.3672 | 0.4347 | 58.5059 | 115.2404 | 0.0000 |
| 25g | compound | M3_CENTER_WIDTH | 72.4547 | 58.8778 | 27.6279 | 0.4279 | 59.5462 | 114.8021 | 0.0000 |
| 25g | row | CENTER_CONDITIONED_WIDTH | 61.2161 | 58.8530 | 24.1788 | 0.4015 | 51.7506 | 100.1081 | 0.0000 |
| 25g | row | CENTER_MAGNITUDE | 61.1103 | 59.2688 | 23.0308 | 0.3837 | 52.8967 | 98.9583 | 0.0000 |
| 25g | row | M3_CENTER_WIDTH | 60.8575 | 59.0029 | 23.0487 | 0.3857 | 52.5595 | 98.6570 | 0.0000 |
| 40g | compound | CENTER_CONDITIONED_WIDTH | 256.5154 | 112.4577 | 69.2640 | 0.4086 | 215.3658 | 353.8938 | 0.0000 |
| 40g | compound | CENTER_MAGNITUDE | 260.2182 | 110.3487 | 69.9657 | 0.4140 | 217.8397 | 357.7711 | 0.0000 |
| 40g | compound | M3_CENTER_WIDTH | 254.2135 | 111.1267 | 69.8034 | 0.4164 | 212.1918 | 351.7985 | 0.0000 |
| 40g | row | CENTER_CONDITIONED_WIDTH | 263.6444 | 109.5228 | 75.0291 | 0.4420 | 215.9960 | 366.0543 | 0.0000 |
| 40g | row | CENTER_MAGNITUDE | 265.0458 | 108.8619 | 75.2961 | 0.4439 | 216.9651 | 367.5574 | 0.0000 |
| 40g | row | M3_CENTER_WIDTH | 263.2406 | 109.4256 | 75.5665 | 0.4462 | 215.0304 | 366.1635 | 0.0000 |

All error and variance identities passed at floating-point precision. Diagnostics were not used for hyperparameter selection. Projection frequency remained zero unless shown otherwise.

## Extra-coefficient stability

| column | protocol | method | coefficient_mean | coefficient_std_across_seeds | positive_seeds |
| --- | --- | --- | --- | --- | --- |
| 25g | compound | CENTER_CONDITIONED_WIDTH | 0.0141 | 0.0152 | 4 |
| 25g | compound | CENTER_MAGNITUDE | -0.0861 | 0.0439 | 0 |
| 25g | row | CENTER_CONDITIONED_WIDTH | 0.0305 | 0.0113 | 5 |
| 25g | row | CENTER_MAGNITUDE | -0.0425 | 0.0304 | 0 |
| 40g | compound | CENTER_CONDITIONED_WIDTH | -0.0011 | 0.0054 | 2 |
| 40g | compound | CENTER_MAGNITUDE | -0.0039 | 0.0069 | 2 |
| 40g | row | CENTER_CONDITIONED_WIDTH | 0.0032 | 0.0089 | 3 |
| 40g | row | CENTER_MAGNITUDE | -0.0043 | 0.0147 | 2 |

Coefficients are fold-mean values summarized across five seeds. Their scale is tied to the fit-specific standardized effect and should not be interpreted as a physical constant.

## Q1-Q11

**Q1. Is center magnitude a more stable cross-column signal than endpoint magnitude?** No. Its pattern is 25g V1 +0.81% (4/5), 25g V2 -0.27% (2/5), 40g V1 -1.34% (0/5), 40g V2 -0.86% (0/5) and aggregate gain is -0.62%. Unlike the prior endpoint-magnitude model's isolated 25g V1 signal, moving magnitude to center does not produce cross-column stability.

**Q2. Does Candidate A mainly reduce center error?** Only locally. Compound `Var(eC)` changes from 72.45 to 71.74 at 25g, but from 254.21 to 260.22 at 40g. The effect is not replicated.

**Q3. Do both columns benefit?** No. At 25g only V1 improves (+0.81%); V2 worsens (-0.27%). At 40g both V1 (-1.34%) and V2 (-0.86%) worsen.

**Q4. Do V1 and V2 both benefit?** No in either column. A improves 25g combined normalized RMSE by 0.46% but worsens the 40g value by 1.20%.

**Q5. Does source center stably help width transfer?** No. Candidate B's pattern is 25g V1 +4.81% (5/5), 25g V2 -3.09% (1/5), 40g V1 -0.75% (0/5), 40g V2 -0.31% (2/5). Its compound extra coefficient is positive in 4/5 seeds at 25g but only 2/5 at 40g.

**Q6. Does Candidate B reduce width error?** No. Compound `Var(eW)` rises from 58.88 to 65.42 at 25g and from 111.13 to 112.46 at 40g.

**Q7. Does it change covariance?** At 25g, `Cov(eC,eW)` rises from 27.63 to 34.05; at 40g it changes only from 69.80 to 69.26. This is not a stable covariance reduction.

**Q8. Is V2 improvement purchased with V1 deterioration?** Candidate B does not improve V2: it worsens 25g V2 by 3.09% while improving V1 by 4.81%. The larger positive covariance helps V1 through subtraction and hurts V2 through addition. At 40g both endpoints worsen slightly.

**Q9. Which hypothesis has the clearer chromatographic interpretation?** B more directly tests the declared location-to-spreading relationship, but its predictive evidence is negative. A is also interpretable as retention-regime-conditioned center transfer, yet its 25g-only coefficient signal does not replicate at 40g. Neither interpretation is empirically supported strongly enough to promote.

**Q10. Did either candidate pass?** A pass=False; B pass=False. Final status is `NO_STRUCTURED_CENTER_WIDTH_HEADROOM`.

**Q11. What follows if both fail?** Stop center/width calibration expansion. A separately preregistered filtered random learning curve is next; it is not part of this run, and active learning remains unauthorized.

## Interpretation limits

Center and width are algebraic surrogates constructed from V1/V2, not independently measured chromatographic moments. Any predictive association is conditional on the present source model and operational domain and is not causal evidence about scale-up or band broadening. No combined A+B model, feature sweep, random learning curve, active learning, weighted objective, or outer evaluation was run.
