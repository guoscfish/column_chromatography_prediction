# Controlled shared-backbone PCGrad: final report

Terminal decision: `PCGRAD_REDUCES_CONFLICT_BUT_DOES_NOT_MATERIALLY_IMPROVE_TRANSFER`.

PCGrad materially changed the diagnosed mechanism but failed the preregistered
COMPOUND prediction gate. ROW and new outer developmental confirmation were
therefore intentionally not run, and no prediction-freeze or outer-result
artifact was created.

## Scope and evidence boundary

`A2_ADAM_FROZEN` is the completed historical A2 column-FiLM reference.
`A2_PCGRAD` is the sole new candidate and is architecture-identical. Only the
44 parameters in the completed shared late-backbone diagnostic tuple receive
projected gradients; every other trainable parameter receives the ordinary
mean-loss gradient. Twenty-five COMPOUND inner fits were completed on the exact
historical identities, scales, source, loss, batch schedule, and selector.

Independent evidence audit: `PASS`; fits verified
25, target fold observations 50,
prediction rows 13783, global target-molecule
train/validation overlap 0.

## Mechanism result

| pair | stage | observations | mean_cosine | median_cosine | negative_fraction | negative_burden |
| --- | --- | --- | --- | --- | --- | --- |
| 4g_25g | raw | 483.0000 | -0.0783 | -0.0755 | 0.6335 | 0.1436 |
| 4g_25g | projected | 483.0000 | 0.2005 | 0.1558 | 0.0704 | 0.0028 |
| 4g_40g | raw | 483.0000 | -0.2217 | -0.2428 | 0.7950 | 0.2535 |
| 4g_40g | projected | 483.0000 | 0.2854 | 0.2723 | 0.0228 | 0.0004 |
| 25g_40g | raw | 483.0000 | 0.0577 | -0.0757 | 0.5528 | 0.1359 |
| 25g_40g | projected | 483.0000 | 0.3128 | 0.2528 | 0.0311 | 0.0006 |

Across all pairs and diagnostic steps, mean cosine changed from
-0.0808 to 0.2662
(increase 0.3470). Negative-cosine burden fell
99.28%, and the
negative fraction fell from 66.05% to
4.14%. PCGrad triggered on
66.32% of directed attempts.
The projected/original mean-update norm ratio averaged
1.172; the median raw
task-gradient magnitude ratio was 2.889,
below the preregistered severe-imbalance threshold of 10. Thus PCGrad genuinely
altered the measured conflict rather than acting as an inert control.

## COMPOUND inner prediction result

| column | method | V1_rmse | V1_mae | V2_rmse | V2_mae | combined_normalized_rmse | V1_r2 | V2_r2 | Center_rmse | Width_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | A2_ADAM_FROZEN | 6.5655 | 4.3366 | 9.5123 | 6.6764 | 0.6061 | 0.5163 | 0.6864 | 7.2560 | 7.1888 |
| 25g | A2_PCGRAD | 6.5620 | 4.3036 | 9.6055 | 6.6535 | 0.6085 | 0.5177 | 0.6807 | 7.3109 | 7.1741 |
| 40g | A2_ADAM_FROZEN | 15.1964 | 9.9930 | 18.7798 | 13.1496 | 0.5413 | 0.6586 | 0.7300 | 16.2523 | 10.5437 |
| 40g | A2_PCGRAD | 15.0893 | 9.9415 | 18.7231 | 13.1087 | 0.5384 | 0.6634 | 0.7321 | 16.1677 | 10.5434 |

| column | mean_paired_fold_nrmse_gain | seed_wins | fold_wins | V1_rmse_relative_gain | V1_mae_relative_gain | V2_rmse_relative_gain | V2_mae_relative_gain | tail_V1_rmse_relative_gain | tail_V2_rmse_relative_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | -0.0037 | 3.0000 | 12.0000 | 0.0005 | 0.0076 | -0.0098 | 0.0034 | -0.0092 | -0.0299 |
| 40g | 0.0050 | 3.0000 | 14.0000 | 0.0070 | 0.0052 | 0.0030 | 0.0031 | 0.0071 | -0.0042 |

Positive gain means A2-PCGrad is better. 25g: -0.37% mean paired-fold NRMSE gain, 3/5 seed wins, 12/25 fold wins; V1/V2 RMSE gains +0.05%/-0.98%, MAE gains +0.76%/+0.34%, tail gains -0.92%/-2.99%.
40g: +0.50% mean paired-fold NRMSE gain, 3/5 seed wins, 14/25 fold wins; V1/V2 RMSE gains +0.70%/+0.30%, MAE gains +0.52%/+0.31%, tail gains +0.71%/-0.42%. The 25g V2 tail deterioration is 2.99%, exceeding
the 2% guard. Neither column reaches the required 2% mean NRMSE improvement;
25g also misses the fold-win and tail guards. The mechanism checks all pass,
but both predictive column gates fail.

## Convergence

The 25 candidate fits selected mean best epoch 107.6
(range 50-340)
and ran a mean 187.6 epochs (range
130-420).
No learning-rate, epoch-budget, loss, or other search was performed.

## Required interpretation

1. Did PCGrad materially reduce negative gradient cosine? Yes. Mean cosine rose by 0.347, negative burden fell 99.3%, and all preregistered mechanism checks passed.

2. Did raw gradient magnitude imbalance remain modest? Yes. The median raw maximum/minimum task-gradient norm ratio was 2.89, well below 10; this study provides no GradNorm rationale.

3. Did reducing conflict improve inner generalization? No materially replicated improvement occurred. 25g worsened 0.37% in mean paired-fold NRMSE and 40g improved only 0.50%, both below the required 2%.

4. Did it reduce the inner-to-outer gain collapse? Not testable: the failed COMPOUND inner gate correctly blocked new outer prediction and scoring.

5. Did it repair the prior 40g ROW deterioration? Not testable: ROW was prohibited after COMPOUND failure. The mechanism result alone cannot establish ROW repair.

6. Did RMSE improve at the expense of MAE? No consistent trade occurred. On 25g, V1 RMSE and both MAEs improved slightly while V2 RMSE worsened; on 40g all four improved by less than 1%. None produced material NRMSE gain.

7. Did the high-volume tail improve? Not consistently. 25g V1 tail worsened 0.92% and V2 worsened 2.99%; 40g V1 improved 0.71% and V2 worsened 0.42%.

8. Did PCGrad improve both columns? No. 25g mean NRMSE worsened; 40g's 0.50% gain was sub-threshold.

9. Did it improve both V1 and V2? Only 40g showed small RMSE gains for both endpoints. 25g was mixed, with V2 RMSE worse.

10. Does the candidate now exceed A1 on ROW? Unknown and deliberately unscored; ROW was not authorized.

11. Does it exceed corrected HIER on COMPOUND? Unknown and deliberately unscored; outer confirmation was not authorized.

12. What limitation remains? Conflict was real and was almost eliminated without material prediction gain, so conflict alone is insufficient. The remaining evidence is more consistent with a combination of limited/partly confounded column context, data identifiability, and high-volume-tail scarcity; this controlled study cannot separate them causally.

13. Is another model-side experiment justified? No uncontrolled neural architecture or optimizer expansion is justified under the current dataset. Independent/crossed compound and batch data with intentional tail coverage should precede another model-side mechanism study. No next method was implemented.

## Provenance and terminal rule

Qualified source SHA256: `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`. Split SHA256:
`33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3`. Protocol SHA256: `9a7b1b5901ab5e03ccb9bd336cb76b262f975996a027ffc63d05d3417509da87`.
Decision SHA256: `8349cbad979295e6535c79f00ca9883fe45ff4acffbd5e14c76171b072050703`.

The reference artifact hashes, architecture signature, trainable parameter
identities, PCGrad scope, exact inner identities, scales, task losses, raw and
projected diagnostics, and completion hashes are retained. Outer validation or
test truth was not read. No ROW fit, new outer prediction, post-result method,
GradNorm/CAGrad/NashMTL, architecture change, calibration, 8g, physical
descriptor, or Active Learning action was performed.
