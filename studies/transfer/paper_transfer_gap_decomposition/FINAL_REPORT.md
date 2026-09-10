# Final report: paper transfer gap decomposition

This is a release-code-aligned diagnostic, not an exact paper reproduction.

## Five-seed random-row results

| column | method | seeds | best_epoch_mean | V1_rmse_mean | V1_rmse_sample_sd | V1_mae_mean | V1_mae_sample_sd | V1_r2_mean | V1_r2_sample_sd | V2_rmse_mean | V2_rmse_sample_sd | V2_mae_mean | V2_mae_sample_sd | V2_r2_mean | V2_r2_sample_sd | combined_normalized_rmse_mean | combined_normalized_rmse_sample_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | RELEASE_CODE_ALIGNED_TRANSFER | 5 | 196.000 | 5.211 | 1.350 | 3.297 | 0.715 | 0.680 | 0.121 | 8.246 | 2.058 | 5.776 | 1.206 | 0.725 | 0.067 | 0.498 | 0.125 |
| 40g | RELEASE_CODE_ALIGNED_TRANSFER | 5 | 256.000 | 14.901 | 2.847 | 9.632 | 1.620 | 0.616 | 0.133 | 20.247 | 3.007 | 13.519 | 2.715 | 0.620 | 0.129 | 0.549 | 0.084 |

## Paper Figure 4 R2 comparison

| column | paper_V1_r2 | aligned_V1_r2 | V1_gap | paper_V2_r2 | aligned_V2_r2 | V2_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 25g | 0.747 | 0.680 | -0.067 | 0.840 | 0.725 | -0.115 |
| 40g | 0.826 | 0.616 | -0.210 | 0.824 | 0.620 | -0.204 |

## Conclusions

Q1/Q2. The prior reconstruction is not exact because the public release does not recover the paper's complete source checkpoint/data/split pipeline and because the reconstruction intentionally changed trainable scope, normalization, output head, V2 loss weight, scheduler semantics, and column-input path. These are confirmed implementation mismatches, not guesses.

Q3. Alignment improves 25g V1 RMSE by 11.85% and mean V1 R2 by +0.083; V2 RMSE changes by -0.43% and R2 by +0.008. Thus the 25g gain is endpoint-specific, not a general closure of the gap.

Q4. 40g does not improve: V1/V2 RMSE changes are -15.97%/-36.74% (negative means worse). Mean R2 gaps to the paper remain about 0.210/0.204, versus seed SDs 0.133/0.129; seed variance is material but not the main identified explanation.

Q5/Q6. The complete RMSE/MAE/R2 reference is the table above. Neither column is jointly close to both paper endpoints: 25g remains -0.067/-0.115 R2 below Figure 4 and 40g remains -0.210/-0.204 below. No additional epoch or hyperparameter configurations were added after seeing test results.

Remaining unidentified factors are the exact author source checkpoint and data snapshot, the paper's split/seed map, target-transfer duration, and hidden training history. Track A results were not used to design Track B.
