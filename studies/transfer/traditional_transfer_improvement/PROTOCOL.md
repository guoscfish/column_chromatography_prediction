# Traditional transfer improvement: filtered FULL-data ROW

Status: preregistered implementation and audit (2026-09-11). This study is
restricted to the inherited filtered FULL-data ROW protocol, columns 25g and
40g, and the five outer seeds. Compound, linked-compound, active-learning, and
test-guided searches are out of scope.

## Question

Can parameter/representation transfer from the qualified 4g QGeoGNN improve
both target columns without relying on target-column prediction calibration?

## Fixed references

The source checkpoint, filtered row identities, thresholds, outer seeds, and
target scales are inherited from `studies/transfer/filtered_full_data_benchmark`
and are never refit from validation or test rows. Existing references are
`head_only`, `last2`, `full`, `paper_style_current_v2`, corrected HIER, and
`BALANCED_JOINT_FULL128`; their historical numbers remain frozen artifacts.

## Ordered ladder

1. T0: reproduce the existing head-only, last2, full, and paper-style scopes.
2. T1: linear probe (head-only) followed by last1/last2 fine-tuning from the
   Stage-A best checkpoint. Stage A and B checkpoints are selected by target
   validation only.
3. T2: T1 with a preregistered discriminative LR (`head > late > early`),
   retaining a same-scope uniform-LR control.
4. T3: T2 with complete endpoint loss weighting by `1 / s_e^2`, where each
   `s_e` is computed from the current target training fold only. Quantile
   outputs are not transformed and their pinball/crossing semantics remain
   unchanged.
5. T4: T3 plus L2-SP on matching source backbone/condition parameters. The
   source state is an immutable snapshot; target heads and adapters are not
   penalized. BatchNorm running buffers are not regularized.

Progression stops when a step has no validation improvement, is unstable over
the five seeds, or violates the promotion gate. No adaptive readout or PEFT
architecture is executed in this study; those require a separate feasibility
protocol after this ladder.

## Selection and leakage controls

Only train and validation labels enter fitting, normalization, early stopping,
or hyperparameter selection. The test role is absent from all adaptation APIs
and is evaluated once after the selected checkpoint is frozen. Report V1/V2
RMSE, MAE, R2, combined normalized RMSE, seed mean/std, best epoch, trainable
parameter count, validation trajectory, and parameter drift. Quantile crossing
and finite-prediction checks are mandatory.

## Decision gate

A candidate is retained only if both columns' combined NRMSE do not materially
worsen, at least one column improves by about 3% consistently, no endpoint has
more than 2% systematic RMSE/MAE worsening, and at least four of five seeds
agree. Exceeding `paper_style_current_v2` is a separate requirement, not an
implicit consequence of beating an older fine-tune.

