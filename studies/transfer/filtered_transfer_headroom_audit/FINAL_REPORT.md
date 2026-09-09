# Filtered transfer headroom audit: final report

## A. Repository audit facts

The qualified 4g source checkpoint is hash-locked to `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. The inherited filtered benchmark uses 25g thresholds 60/120 mL and 40g thresholds 150/200 mL, with 408 and 456 retained canonical rows. Its outer B=100 validation/test identities and five seeds were reused exactly.

The current Conditional EA implementation is a normalized source-q50 varying slope with `[u, 1, u*centered_EA]`; the source QGeoGNN already receives chromatography condition inputs. Flow was not added because it is effectively column-confounded in this population.

## B. Data and protocol caveats

This is a residual re-audit after operational filtering, distinct from the historical no-threshold scaling_failure_audit. Filtering changes covariate support and can remove low-EA/high-retention rows; it is not generic outlier removal. Compound GroupKFold is the primary evidence, while row contexts are secondary diagnostics.

All target truth read by this run belongs to gradient_train. Validation and outer test labels were not read. The source magnitude variable is a descriptive normalized q50 summary, so denominator coupling and source prediction error remain possible artifacts. Associations below are observational.

## C. Inner-CV results

| column | protocol | method | n_rows | V1_rmse | V1_mae | V2_rmse | V2_mae | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | M0_scale_only | 315.8000 | 8.7261 | 5.4173 | 10.7809 | 7.0570 | 0.9155 |
| 25g | compound | M1_conditional_EA | 315.8000 | 8.0195 | 5.0208 | 10.6237 | 7.0064 | 0.8582 |
| 25g | compound | M2_EA_source_magnitude | 315.8000 | 8.6663 | 5.0998 | 12.0333 | 7.3971 | 0.9412 |
| 25g | compound | M3_center_width | 315.8000 | 7.6074 | 4.5303 | 10.6667 | 7.0204 | 0.8284 |
| 25g | row | M0_scale_only | 321.6000 | 8.3410 | 5.4254 | 10.1593 | 6.9731 | 0.8718 |
| 25g | row | M1_conditional_EA | 321.6000 | 7.6093 | 4.8846 | 9.8498 | 6.6984 | 0.8088 |
| 25g | row | M2_EA_source_magnitude | 321.6000 | 8.1143 | 4.9544 | 10.4730 | 6.8220 | 0.8620 |
| 25g | row | M3_center_width | 321.6000 | 7.1889 | 4.4439 | 9.8716 | 6.6936 | 0.7777 |
| 40g | compound | M0_scale_only | 360.8000 | 17.9541 | 12.9946 | 19.7969 | 14.1701 | 1.8315 |
| 40g | compound | M1_conditional_EA | 360.8000 | 15.3148 | 10.0923 | 19.1812 | 13.8303 | 1.6128 |
| 40g | compound | M2_EA_source_magnitude | 360.8000 | 15.4755 | 10.0546 | 19.1201 | 13.6053 | 1.6236 |
| 40g | compound | M3_center_width | 360.8000 | 14.6944 | 9.5849 | 18.7988 | 13.6127 | 1.5565 |
| 40g | row | M0_scale_only | 357.2000 | 18.0199 | 12.9533 | 20.2478 | 14.1306 | 1.8462 |
| 40g | row | M1_conditional_EA | 357.2000 | 15.2599 | 10.1793 | 19.4881 | 13.8036 | 1.6157 |
| 40g | row | M2_EA_source_magnitude | 357.2000 | 15.2706 | 10.0796 | 19.2742 | 13.5287 | 1.6115 |
| 40g | row | M3_center_width | 357.2000 | 14.7014 | 9.6813 | 19.0615 | 13.5361 | 1.5632 |

## D. Promotion decision

The preregistered gate result is **NO_LIGHTWEIGHT_CALIBRATION_HEADROOM_IDENTIFIED**. M2 and M3 were evaluated only in nested train-only GroupKFold OOF. Detailed endpoint gains, wins and deterioration checks are in `promotion_decision.json`.

## E. Residual structure

| variable | target | support | std | oof_residual_rank_corr | partial_source_rank_corr | compound_controlled_rank_corr |
| --- | --- | --- | --- | --- | --- | --- |
| EA_fraction | V1 | 338.8500 | 0.3363 | 0.0454 | 0.1812 | -0.1769 |
| EA_fraction | V2 | 338.8500 | 0.3363 | 0.0578 | 0.1620 | -0.0778 |
| loading_amount | V1 | 338.8500 | 44.1763 | -0.0339 | -0.0286 | -0.0149 |
| loading_amount | V2 | 338.8500 | 44.1763 | 0.0001 | 0.0077 | 0.0403 |
| loading_solvent | V1 | 338.8500 | 0.3958 | 0.0949 | 0.1310 | -0.0124 |
| loading_solvent | V2 | 338.8500 | 0.3958 | -0.0087 | 0.0175 | -0.0306 |
| loading_solvent_volume | V1 | 338.8500 | 100.5859 | -0.0809 | -0.0196 | 0.0195 |
| loading_solvent_volume | V2 | 338.8500 | 100.5859 | -0.1747 | -0.0592 | 0.0271 |
| source_center | V1 | 338.8500 | 3.0589 | 0.1660 | 0.1501 | 0.2175 |
| source_center | V2 | 338.8500 | 3.0589 | 0.0136 | -0.1271 | 0.0571 |
| source_magnitude | V1 | 338.8500 | 0.1261 | 0.1544 |  | 0.1873 |
| source_magnitude | V2 | 338.8500 | 0.1261 | 0.0408 |  | 0.0517 |
| source_width | V1 | 338.8500 | 3.5597 | 0.2000 | 0.1382 | 0.2997 |
| source_width | V2 | 338.8500 | 3.5597 | -0.0766 | -0.2188 | 0.0556 |

The stratification file reports EA, source magnitude, source width and loading solvent bins with RMSE, MAE, signed error, row fraction and SSE fraction. No observational association is interpreted causally.

## F. Explicit answers

- **Q1:** After filtering, Conditional EA residuals show only modest, mixed source-magnitude association; the M2 test is the stricter answer because it adds one source-magnitude slope term.

- **Q2:** **No.** M2 improves 0/4 compound column x endpoint means and decreases aggregate normalized RMSE by about 3.8%; it is not promoted.

- **Q3:** M3 is better for V1 in both columns (about 5.1% and 4.1% OOF RMSE gains) but not V2; 2/4 endpoints is below the gate. Center/width therefore does not establish a general replacement.

- **Q4:** The tentative M3 gain is concentrated in V1/center-like location, not a stable width/V2 gain. This is an association, not a physical claim.

- **Q5:** The partial V1 pattern repeats in 25g and 40g, but the required all-endpoint replication does not.

- **Q6:** Neither candidate passes the pre-registered 3%/3-of-4 material gate; no seed-level outer claim is authorized.

- **Q7:** No new lightweight method is allowed to claim superiority over Scale, Conditional EA, or Paper-style: outer paired comparisons were correctly not run after the failed gate.

- **Q8:** **Yes.** Close calibration-family expansion under the declared stop rule.

- **Q9:** OOF residual stratification shows the largest remaining low-dimensional associations in source center/width and weak EA contrasts; compound-controlled effects are inconsistent. Much error remains unstructured at this resolution.

- **Q10:** Prioritize exact replicates, then TLC Rf anchors; measured V_M and crossed mass x flow follow for physical normalization and identifiability. The ranking and rationale are in `MEASUREMENT_HEADROOM_RECOMMENDATION.md`.

- **Q11:** Proceed to a filtered random learning curve at B=30/50/100/150/200/FULL. Enter active learning only if that curve demonstrates a material label-scarcity regime.


The current data cannot identify measured hold-up volume, TLC anchors, a reliable noise floor or a causal mass-flow effect. No outer truth, unmeasured V_M/Rf, or literature-derived quantitative effect was introduced.
