# Phase 2 implementation plan: matched one-step B32 screens

Status: design only. No new Phase 2 selector, ensemble, noise model, or candidate
training has been executed. Phase 0 is complete; Phase 1 must finish before this
plan becomes an execution preregistration. Candidates below are hypotheses, not
presumed improvements.

## Common contract and promotion

Start with the existing confirmation cohort seeds 157/887/2357/6101/12203,
identical dataset/source hashes, row partitions, L0, preprocessing and endpoint
scales. Reuse initial member-zero models and the existing 512D raw gradient bank.
All acquired-label evaluations use the existing member-zero scratch initialization
and unchanged training/checkpoint configuration. Do not change predictor loss.
The primary Random control is the nested sequential first B32, not the historical
mean of five unrelated one-step Random batches. Reuse Gradient-LCMD first B32.
Any additional Random controls must be declared separately, never spliced into
the sequential curve.

Before any candidate selection, freeze code, parameter choices, input hashes,
dependency versions, seed list, tests, diagnostic definitions, and a global
test-reveal barrier. Freeze every candidate selection and all matched predictions
before opening new test truth. Do not use validation/test labels as accessible
pool labels. Exposed historical test outcomes make these development screens;
passing them does not constitute independent confirmation.

Prespecify the following separate conditions for promotion:

- Mean combined NRMSE below matched Random, with at least 4/5 paired wins and
  positive median paired improvement.
- Neither endpoint mean R2 lower than Random; neither endpoint mean RMSE more
  than 2% worse than Gradient-LCMD. Show all paired endpoint differences.
- Mean NRMSE no worse than Gradient-LCMD. A candidate that only clears Random
  remains useful baseline evidence but does not justify a costly new trajectory.
- The acquisition objective differs substantively from existing LCMD/IVR.
- Costs, failures, and proxy-validation gates are reported, not hidden by the
  predictive gate. No combined weighted score is formed.

One-step screens cannot measure AULC, N80->N90 improvement, or late-stage
complementarity. Those are subsequent sequential hypotheses, not screening claims.
At most the first passing candidate advances initially to 333->1005 B32;
additional promotions need an explicit cost/benefit rationale from actual results.
Use a locked new split, new compounds, or prospectively collected experiments for
eventual independent confirmation. New random seeds alone still reuse the same
chemical dataset and should be described accordingly.

## 1. Gradient-MaxDet: first implementation

Reuse `gradient_features.py` unchanged. Add a separate `maxdet.py` selector with
an API accepting X features, labeled positions, candidate positions and batch
size, never labels. Use conditional D-optimality with all L0 observations:

`A0 = I + sum_L0 psi(x) psi(x)^T`

`gain(x | B) = log1p(psi(x)^T A_B^-1 psi(x))`

where `psi = phi / sqrt(mean_L0 ||phi||^2)`. This single L0-based scale makes the
fixed ridge 1 explicit and reproducible. Freeze it without looking at test error;
do not search ridge or sketch dimension. This optimizes the incremental log
determinant conditional on L0; it is distinct from unconditional MaxDet on only B.
LCMD uses the unchanged raw geometry as the coverage baseline.

Use float64 Cholesky/triangular solves or validated rank-one updates, canonical
input-order tie breaks, and finite/positive-pivot checks. Numerical failure stops
the arm and records evidence; no silent parameter retuning. Tests compare gains
and selected order against a direct small-matrix determinant implementation,
including duplicate/collinear rows, labeled exclusion, determinism, and resume.

Diagnostics for both methods: ordered/set batch overlap, selected gradient norm,
predicted V1/V2, mean absolute off-diagonal normalized kernel correlation, and
effective rank `exp(-sum p_i log p_i)` of the batch Gram spectrum. Preserve both
raw and conditional diagnostics with explicit names. Time feature extraction and
selection separately. Expected additional training: five B32 evaluation fits;
no new gradient extraction if initial cache contracts match.

## 2. RD-EMCM: model disagreement within representative partitions

First reuse historical round-zero Hybrid K=3 independent initialization ensemble
members after exact split/L0/scale/configuration checks. These are independent
initializations, not bootstrap samples; retain that distinction. Member zero is
the matched evaluation model. If reuse is incomplete, freeze the two additional
initialization seeds before fitting. Do not silently mix different ensembles
between acquisition methods.

Standardize outputs by the L0 endpoint scales. With `z_m = f_m / s`, define
`delta_m(x) = z_m(x) - mean_m z_m(x)`. Use endpoint Jacobians in a shared parameter
coordinate system, so the model-change vector is
`g_m(x) = J_z(x)^T delta_m(x)` and `EMCM = mean_m ||g_m(x)||_2`.
This includes endpoint cross terms. It estimates change under a normalized point
output loss surrogate, not the exact six-output quantile-plus-MSE training loss.
Multiplying the norm of the concatenated raw
512D feature by a scalar disagreement is only a different approximation and must
not be mislabeled as the above EMCM. Reuse the audited block-Jacobian machinery
where possible; test shared-parameter sketch semantics against exact small models.

Representativeness: deterministic KMeans with 32 partitions in the existing
128D pre-head latent representation, standardized using accessible training X
only. Freeze seed and n_init=10. Select the maximum-EMCM eligible point in each
nonempty partition, ties by canonical order. A deterministic farthest-cluster
repair handles duplicate centers/empty partitions while preserving 32 distinct
IDs. Record cluster occupancy, density, distance to L0, and within-batch diversity.
This defines one RD-EMCM variant. No pure-EMCM production arm is added by default.

Additional costs: five evaluation fits; up to ten reusable ensemble fits if the
historical cache cannot be used; endpoint-gradient work recorded separately.

## 3. Noise-aware acquisition: audit before weighting

Treat ensemble disagreement `E` as an epistemic proxy and q90-q10 width as an
uncalibrated noise proxy. Check interval ordering and coverage, residual-width
association, disagreement-residual association, and conditional associations
after controlling for predicted V2. Audit actual repeated-input provenance:
copied records are not independent experimental repeats. Only revealed L0 labels
may supply repeat variances; do not read all U0 labels to calibrate this model.

For residual calibration, use fixed cross-fitted L0 predictions or a separately
reserved calibration set. The checkpoint-selection validation set is not an
independent calibration set. Cross-fitting costs must be budgeted before this
stage; do not pretend it is a free diagnostic. If the available labels cannot
validate a useful proxy, record that limitation and stop this candidate.

Only after the proxy audit is sealed, preregister one reliability function:
`r = E / (E + A + epsilon)`, with E and calibrated A in the same normalized
squared-output units, and epsilon fixed from L0 scale. For a calibrated Gaussian
interpretation, width-to-variance conversion needs its distributional assumptions
stated explicitly. Width alone is not A by definition.

First candidate: `r * phi` for LCMD, or reliability-weighted EMCM inside the same
RD partitions, chosen before labels are opened. Run at most one fixed sensitivity
check, declared in advance. Audit high-V2 selection rate and early error/endpoint
trade-offs. Late N90/N95 claims require a promoted sequential run.

## 4. Target-weighted multi-output V-optimal

Reuse `block_ivr.py` and the mechanism audit's endpoint-Jacobian extraction where
their contracts match; audit the current operator before writing another one.
Each experiment observes both endpoints simultaneously. With normalized
`J_x` of shape 2xd, use the rank-two posterior update

`C_new = C - C J_x^T (R + J_x C J_x^T)^-1 J_x C`.

Minimize `trace(M_w C)`, where `M_w = sum_z w_z J_z^T W_y J_z`, `W_y=I/2` after
L0 endpoint scaling. Freeze one observation-noise convention and ridge; no
prior/noise/dimension grid. Compare uniform scalar IVR (existing), uniform
multi-output IVR, and one target-weighted multi-output arm.

Accessible X defines target weights. With no deployment distribution provided,
use equal mass per canonical compound within outer training X, divided equally
among its rows, and normalize the weights to sum to one. This changes the risk
distribution toward compound balance; it is an explicitly exploratory deployment
proxy, not guaranteed to match row-test RMSE. Report the unweighted target arm
alongside it. Never construct weights from test X error, test labels, or observed
test-curve crossings. A genuine deployment X sample would require a separately
sealed target distribution.

Test rank-two gains against direct covariance recomputation, endpoint scaling,
weight normalization, labeled conditioning, duplicates, and deterministic ties.
Do not rename existing scalar IVR as a novel "True BAIT" baseline. Two new
five-seed B32 arms cost ten evaluation fits, plus block-feature extraction.

## 5. Predictive covariance geometry

Reuse the same frozen K=3 ensemble as RD-EMCM for a first low-cost pilot. Let
`h_x` concatenate centered normalized V1/V2 predictions over members, divided by
sqrt(K-1), and use `K_ij = h_i dot h_j`. This sums endpoint covariance and has
rank at most `2(K-1)`; K=3 is strongly rank limited. Report spectrum and effective
rank, and do not claim it estimates full Bayesian uncertainty.

Apply the unchanged LCMD and the validated MaxDet selectors to this kernel.
Compare Gradient-LCMD, PredictiveCov-LCMD, Gradient-MaxDet, PredictiveCov-MaxDet.
This is an empirical predictive-covariance baseline; use the B3AL name only after
matching that algorithm's actual specification. A larger K requires a separate
costed preregistration, not an unreported response to poor test performance.

Two new five-seed B32 evaluation arms cost ten fits, with shared ensemble training
and inference charged once as incurred work and separately as logical arm cost.
Measure whether extraction savings compensate for ensemble fitting. No promise
of lower total cost follows simply from avoiding full parameter gradients.

## Execution and later adaptation

Implement and screen MaxDet first. Add RD-EMCM next only after accounting for its
reusable ensemble and endpoint-gradient requirements. The noise audit precedes
any noise-weighted candidate. Do not schedule all candidates for 21 rounds.

If multiple candidates eventually pass, a validation-only common-state branch
experiment is needed for an oracle switching/headroom analysis. Different
historical policy trajectories are different labeled states: selecting the best
validation score across those trajectories each round is not a valid reachable
policy or causal feedback estimate. At each sampled common state, freeze each
candidate batch, acquire only its authorized labels, scratch fit its branch, and
compare validation improvements. Label the oracle optimistic and never report it
as a deployable test result. Test curves cannot set a switching boundary.

Only demonstrable headroom beyond the best fixed method motivates validation
gating, a contextual bandit or learned switching. Dynamic batch sizes follow
matched-budget feedback evidence and must trade label efficiency against fit
count, total epochs, gradients, ensemble inference, and calendar time. A null or
mixed Phase 1 result leaves intermediate B64/B160 and independent confirmation
as possible targeted follow-ups, not automatic expansions.
