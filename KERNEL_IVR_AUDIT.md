# Kernel-IVR Audit

**Audit date:** 2026-09-19  
**Decision:** keep as a documented negative/exploratory baseline; do not extend the
current scalar IVR family without a new, pre-registered scientific question.

## Theory

The implemented scalar rule is a Bayesian-linear/ridge surrogate over a fixed row
feature map `phi(x)`. With standardized features `X`, unit prior precision and unit
observation noise, the labeled-set precision and covariance are

`A_L = I + X_L^T X_L`, `C_L = A_L^-1`.

The reference risk uses the uniform second moment

`M = X_R^T X_R / |R|`.

For a candidate `x`, the conditional integrated variance reduction is

`gain(x) = (x^T C_L M C_L x) / (1 + x^T C_L x)`.

The denominator is the candidate's predictive variance under the same scalar ridge
surrogate. Greedy rank-one updates are equivalent to conditioning this linear model
on selected rows. The block version applies a Woodbury rank-two update when one
experiment is treated as a multi-output block.

This is a valid experimental-design objective under the stated surrogate. It is not
the posterior of QGeoGNN-V2, whose learned function, six-output quantile loss, and
parameter uncertainty are not represented by this fixed kernel.

## Current implementation

The scalar implementation is in
[`src/qgeognn_al/active_learning_v2/ivr.py`](src/qgeognn_al/active_learning_v2/ivr.py):

- input is a finite `N x d` feature matrix and a complete `L/U` partition;
- one global RMS scale is computed over the fixed reference matrix;
- `C` is formed by Cholesky solve in float64;
- `M` is the uniform reference second moment;
- each candidate score is evaluated, the maximum is selected, and `C`/projected
  factors are updated with Sherman-Morrison;
- selected rows are removed from the available candidate set, so there is no
  duplicate selection within a batch;
- every update is checked against the realized integrated-risk decrease.

The block implementation is in
[`src/qgeognn_al/active_learning_v2/block_ivr.py`](src/qgeognn_al/active_learning_v2/block_ivr.py).
It supports scalar features as a one-output block and multi-output features as an
`N x outputs x d` array, with one shared Frobenius RMS scale and a Woodbury update.
Its optional forward/backward mode selects `2B` forward candidates and removes the
least useful half under the same surrogate.

The feature map used in the sequential scalar experiment was the current-model,
full-network endpoint-scaled q50 gradient sketch: V1 and V2 endpoint gradients are
concatenated and CountSketched to 512 dimensions. The reference set is the fixed
original outer-training universe `R = L0 union U0`; validation and test rows are
excluded. The sketch itself is not a kernel learned from labels.

## Deviations from idealized IVR claims

1. **Scalar observation:** one experiment produces V1 and V2 together, but the main
   sequential result used a scalar concatenated representation. It therefore does
   not model an exact two-output observation covariance.
2. **Fixed unit noise:** prior precision and observation noise are both one by
   convention. They are not estimated from repeated measurements or residuals.
3. **Uniform risk:** every reference row has equal weight. This is a row-risk proxy,
   not a deployment distribution or compound-balanced risk.
4. **Linearization:** the gradient map is a local parameter sensitivity of a trained
   neural predictor, while the IVR update assumes a fixed linear feature model.
   Retraining, early stopping, parameter coupling, and representation drift are
   outside the posterior update.
5. **Sketching:** CountSketch preserves inner products approximately; it changes the
   exact parameter-space kernel. The chosen dimension and seed were fixed, not tuned
   for each outcome.
6. **Batch greediness:** sequential within-batch updates reduce redundancy under the
   surrogate. They do not account for a batch-level retraining response.

These are modeling assumptions, not implementation bugs.

## Numerical and mechanism audit

The code checks finite inputs, partition completeness, positive covariance factors,
score/risk-reduction equality, and direct posterior agreement for the block variant.
The label-free mechanism audit found stable score rankings but insufficient batch
stability under its pre-registered gates:

| representation/comparison | median score Spearman | mean batch overlap | decision |
| --- | ---: | ---: | --- |
| block, cross-seed 512 | 0.9921 | 0.6438 | gate failed |
| scalar, 512 vs 2048 same seed | 0.9946 | 0.6871 | gate failed |
| scalar, cross-seed 512 | 0.9885 | 0.5767 | gate failed |

Forward/backward improved the surrogate batch objective in only 7/25 snapshots,
with median relative gain 0. The audit therefore stopped before an additional
predictive gate; it did not identify a numerical instability.

The matched sequential scalar-IVR experiment reported:

| method | normalized AULC | NRMSE at 1,005 |
| --- | ---: | ---: |
| Gradient-LCMD | 0.479539 | 0.408614 |
| Kernel-IVR | 0.500070 | 0.395913 |

IVR beat LCMD on paired AULC in 1/5 seeds, with mean AULC 4.28% worse. Its lower
final NRMSE does not offset the pre-registered AULC failure, and the extension was
exploratory on an already evaluated cohort.

## Hyperparameter sensitivity and what is not known

The current evidence does **not** justify a broad search over ridge, noise,
reference weighting, sketch dimension, or output weights. Such a search would change
the surrogate and invite post-hoc tuning on exposed outcomes. What is known:

- ranking correlations remain high when the sketch dimension or seed changes;
- selected batch membership changes materially despite those high score correlations;
- scalar and block objective values are not directly comparable because their feature
  spaces and normalizations differ;
- no empirical estimate of experimental variance is available to set the noise term.

A future IVR study would need a new target-risk question, a fixed output/block
observation model, and a prospective calibration of noise or risk weights. It should
not be presented as a repair of the existing negative result.

## Final assessment

The implementation is internally coherent for a fixed ridge-kernel surrogate. The
main limitation is scientific alignment, not a discovered coding bug: reducing
surrogate variance did not reliably reduce retrained QGeoGNN error. Kernel-IVR is
therefore retained as a negative control and mechanism reference. The next diagnosis
should establish where errors concentrate and whether an uncertainty signal predicts
learnable error before considering a different acquisition objective.
