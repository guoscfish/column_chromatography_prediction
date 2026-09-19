# Current State Audit

**Audit date:** 2026-09-19  
**Branch:** `main` at `106674f` (`docs: refresh cleanup validation result`)  
**Scope:** repository, frozen 4 g predictor qualification, active-learning studies,
transfer studies, data audits, and the current acquisition implementations.

This is a baseline audit for the uncertainty-aware active-learning work. It records
what is already established and what remains an open diagnostic question. Historical
results are not silently pooled across protocols.

## Current most-trusted 4 g predictor

The qualified point predictor is **QGeoGNN-V2**, with 458,952 parameters and the
six-output contract `V1_q10, V1_q50, V1_q90, V2_q10, V2_q50, V2_q90`.

The qualified source data are 4,163 canonical rows from 4,243 source rows, with
217 unique structures. The current reader retains repeated experiments and the
known quality warnings (one negative-label row and eight rows with `t1 > t2`).
The volume rule is `V_ml = t_raw * flow_ml_min / 1200`, with 4 g thresholds of
60 ml for V1 and 120 ml for V2.

The six-run qualification reports the following test aggregates (mean across
seeds 42, 525, and 1101):

| split | V1 RMSE (ml) | V2 RMSE (ml) | V1 R2 | V2 R2 | combined normalized RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| row | 2.919 | 5.525 | 0.858 | 0.879 | 0.359 |
| compound | 5.608 | 10.980 | 0.479 | 0.487 | 0.699 |

The row split is an interpolation benchmark. It is not leakage-resistant chemical
generalization: 210 compounds appear in more than one row split, and 24 of 52
repeated-condition groups cross subsets in the historical E0 audit. The compound
split has zero compound overlap and is the appropriate strict reference for unseen
structures.

The point predictor is qualified for ordinary point transfer studies. The quantile
head is **not** qualified as a calibrated uncertainty model. On the qualified row
tests, mean 80% interval coverage is about 0.722 for V1 and 0.691 for V2; compound
coverage is about 0.638 and 0.463. Crossing is non-zero, and the audit explicitly
requires a monotonic-head/UQ follow-up before active transfer.

Relevant records:

- [4 g qualification](studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md)
- [quantile audit](studies/predictor/final_4g_qualification/QUANTILE_AUDIT.md)
- [dataset audit](experiments/data_audit/AUDIT_REPORT.md)

## Current active-learning baseline

The strongest established 4 g row active-learning evidence is the qualified V2
Gradient-LCMD-TP trajectory with B=32, compared with the matched Random control.
The full sequential cohort uses five overlapping row splits, 333 initial active
labels, 416 shared validation rows, 417 held-out test rows, and 21 acquisitions
to reach 1,005 active labels.

| method | normalized AULC | NRMSE at 1,005 | seeds beating Random on AULC |
| --- | ---: | ---: | ---: |
| Random | 0.624260 | 0.543301 | -- |
| Hybrid | 0.479354 | 0.409709 | 5/5 |
| Gradient-LCMD | 0.479539 | 0.408614 | 5/5 |

The evidence supports label efficiency over Random, while the frozen decision
`NO_CLEAR_SEQUENTIAL_AL_GAIN` correctly avoids ranking Hybrid versus LCMD. Both
active methods improve quickly and then flatten around the 650-label region; N90
and N95 are not stably reached within 1,005 labels.

The matched 333+320 batch/adaptivity control is now available as development
evidence. Adaptive LCMD reaches endpoint combined NRMSE 0.436791 versus 0.451013
for Static LCMD and 0.629362 for Random, but its 333--653 AULC is 0.544359 versus
0.539505 for Static LCMD. The endpoint feedback signal therefore does not establish
an AULC advantage. The result is from previously evaluated splits and is not an
independent confirmation.

Current AL code uses full-network q50 gradient sketches (512D CountSketch),
current V2 latent representations, q50 ensemble disagreement, and LCMD/coreset
selectors. The acquisition APIs do not consume labels from the pool.

Relevant records:

- [sequential result](studies/active_learning/qgeognn_v2_row_sequential_b32/FINAL_REPORT.md)
- [LCMD result](studies/active_learning/qgeognn_v2_row_lcmd/FINAL_REPORT.md)
- [matched batch/adaptivity result](studies/active_learning/qgeognn_v2_batch_adaptivity/FINAL_REPORT.md)
- [active-learning index](studies/active_learning/README.md)

## Main metrics and evaluation protocol

The predictor uses V1/V2 RMSE, MAE, R2, and a source-scale combined normalized
RMSE. The latter is

`sqrt(0.5 * ((RMSE_V1 / s_V1)^2 + (RMSE_V2 / s_V2)^2))`,

where `s_V1` and `s_V2` are fit on the active training set under the frozen AL
protocol. Active-learning studies additionally report the complete learning curve,
normalized AULC, fixed-budget endpoints, labels-to-target, endpoint metrics, and
compute cost. The test set is frozen and opened only after all selections and
predictions are sealed.

The protocol distinction matters: a row split measures interpolation over a pool
where many compounds recur; a compound split measures a harder structure holdout.
Results from the two tasks must not be ranked together.

## Already tried; do not repeat by default

The repository contains controlled evidence for the following families:

- Random, uncertainty, coverage, Hybrid, Gradient-LCMD-TP, and sequential B32
  comparisons;
- batch-size pilots at B=16/B=32 and matched-budget Static versus Adaptive
  control;
- scalar Kernel-IVR, block/multi-output IVR mechanism audit, and IVR sequential
  extension;
- quantile monotonicity/calibration diagnostics and conformal post-hoc scaling;
- ensemble disagreement, quantile width, and latent-distance signal qualification;
- transfer calibration, Scale-only/Affine/Conditional/Center-Width, nonlinear
  and shared-column/readout/FiLM/PCGrad/physics-conditioned controls.

These results do not prove that every possible variant is impossible. They do mean
that a new experiment must address a specific unresolved mechanism or fix a verified
protocol defect. Re-running the same selector with a renamed objective is not a new
scientific test.

## Main bottlenecks

1. **Late-stage label efficiency:** LCMD and Hybrid have an early AULC advantage,
   but improvement becomes small and unstable after roughly 650 active labels.
2. **Uncertainty semantics:** q10--q90 width ranks hard rows better than random in
   small signal studies, but coverage is poor and the head is not a calibrated
   epistemic/aleatoric decomposition.
3. **Chemical generalization:** compound-holdout error is about twice the row error,
   so the present row AL result cannot be presented as unseen-molecule AL.
4. **Data design:** repeated measurements exist, but their variance has not been
   estimated with a provenance-aware, cross-fitted protocol. Exact repetitions are
   not automatically independent experiments.
5. **Transfer identifiability:** the current condition contract omits flow, packing
   mass, geometry, and column identity. In target data, mass and flow are nearly
   deterministic functions of column identity, so causal scale claims are not
   identified.

## Confounding and coupling that constrain interpretation

- `column size <-> flow` is coupled in the target domains (8 g, 25 g, 40 g use
  fixed flows 10, 15, and 30 ml/min respectively).
- 25 g and 40 g share the same legacy geometry tuple in the repository without
  verified physical provenance.
- Row splits leak compound identity across train/validation/test by construction;
  compound splits remove this overlap.
- Eluent ratio, molecular structure, loading conditions, and retention magnitude
  are observed in a designed but non-random pool. A correlation between a feature
  and error is not a causal effect.
- The current q50 is trained with an MSE term inside the quantile loss wrapper,
  while q10/q90 use pinball terms. It is a point predictor with quantile-shaped
  outputs, not a complete probabilistic model.

## Research risks in the current code/results

- Treating q90-q10 as experimental noise or epistemic uncertainty without a
  repeated-measurement or cross-fitted calibration estimate.
- Comparing methods at different cumulative label budgets or mixing one-shot and
  sequential curves with different endpoint definitions.
- Using post-hoc test error strata to select a new acquisition rule on the same
  exposed cohort.
- Calling the scalar IVR surrogate a Bayesian posterior over the GNN; it is a
  ridge-kernel approximation with fixed unit prior/noise assumptions.
- Interpreting flow, mass, or legacy column descriptors as independently causal
  physical features under the current observational design.
- Generalizing row-split AL gains to compound OOD or to 4 g -> target-column active
  transfer before an independently validated transfer uncertainty contract exists.

## Audit decision

The next safe research step is a low-cost, frozen **4 g error/uncertainty landscape
analysis** using existing predictions, accessible X, repeated-condition metadata,
and the fixed row/compound manifests. It should report strata, interactions,
calibration, representation coverage, and learnability proxies before any new
acquisition function is implemented. Transfer acquisition remains deferred until
this diagnostic and a stable target-column uncertainty baseline exist.
