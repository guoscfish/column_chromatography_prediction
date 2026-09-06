# One conditional-scaling experiment, chosen from the training audit

Only `CONDITIONAL_SCALING` passed the predeclared screen: EA fraction / V1 has replicated source-adjusted ratio association and common-support contrasts in all six column/protocol contexts. B and C did not pass; neither is trained. This file is written before new-model fitting and target-test evaluation.

For each column and output, retain fixed source q50, source standard deviation S, and descriptive mass ratio m=mass/4. Define u=source_q50/S, z=target/(m*S), and e=EA_fraction-mean_train(EA_fraction). Fit three coefficients:

`z = a*u + b + gamma*standardize_train(u*e)`.

This is a varying slope `a(EA)*source_q50` with a constant intercept, not `a*x+b+Ridge(c)`. Only one evidenced condition enters. Both outputs use the same small family for a controlled two-output comparison; V1 supplies the directional rationale, and V2 regression cannot be hidden. No new molecule embedding, backbone, pooling, head, loss or other condition feature is introduced.

Capacity-matched additive control replaces only the third regressor with `standardize_train(e)`. Each arm has three coefficients per output. Fit normalized training MSE plus `lambda*((a-1)^2+b^2+gamma^2)` using lambda in {0, 0.1, 1}; select a common lambda for V1/V2 using original focal-validation RMS source NRMSE. Affine at lambda=0 and identity shrinkage at gamma=0 retain their exact meanings. No search over features, interactions or architecture is allowed. Constant third features use unit scale and carry no information. Unconstrained empirical calibration is retained without post-hoc clipping.

Report standalone `conditional_EA` and `additive_EA_control`, plus symmetric validation-selected policies. `conditional_policy` chooses among original scale, affine, local identity shrinkage and the best conditional arm; `additive_policy` substitutes the best additive arm. Ties favor the simpler original family. Existing scale/affine/head-only/local-shrinkage frozen test predictions are reused. Local shrinkage coefficients/validation selection from the preceding study are reconstructed and checked, not retuned on test.

Original 3 columns × 2 protocols × 5 seeds × 4 budgets are unchanged. Target label cost is each context's original actual_budget (train+validation); no donor target labels or observed source-pair label features enter this experiment. Knots are not used. Source and interaction normalization are frozen source-only or gradient-train-only as appropriate. All fits and test predictions freeze before target-test evaluation.

Primary reference gate for `conditional_policy`: >=5% mean relative AULC gain, negative median paired delta and >=4/5 wins against scale, affine, local identity shrinkage **and additive_policy**. Require at least two different columns' compound protocols, or both row and compound of one column; qualifying compound columns must not degrade row AULC by >5%. Head-only remains a reported neural reference, with a separate flag for whether it is beaten. This is a research mechanism/accuracy comparison, not a claim of universal best method.

High-budget accuracy is a separate gate: at least 5% budget-100 NRMSE gain versus scale, affine, local shrinkage and additive policy, >=4/5 seed wins, and neither output's mean RMSE worsens, under the same replication rule. Classify `LABEL_EFFICIENCY_GAIN`, `ACCURACY_GAIN`, both, or neither. Small validation sets, ratio division effects and historical test exposure limit confirmation. No follow-up family, MLP, penalty expansion, pair model or readout run is added after seeing the test results.
