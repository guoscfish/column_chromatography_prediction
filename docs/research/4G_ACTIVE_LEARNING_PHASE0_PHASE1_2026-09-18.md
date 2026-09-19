# 4g active learning: evaluation and matched-budget research summary

## Repository evidence reviewed

The remote was fetched before implementation. The relevant latest study base is
`exp/qgeognn-v2-4g-row-al` at `0e7b2ee`, not `main` at `f5605ba`.
Sequential B32 results are at `1e8fe69`; Kernel-IVR at `26028aa`; the mechanism
audit at `0e7b2ee`. Separate branches were reviewed without overwriting their
worktrees: innovation screen `30da9ae`, and CW performance `6ac0f9e`.
The implementation branch is `codex/4g-evaluation-adaptivity`.

Reviewed sources include the 2026-09-17 review, 2026-09-18 next-experiment memo,
sequential/hybrid protocols and reports, original gradient extraction and LCMD,
sequential fitting and acquisition, IVR study/runtime, mechanism audit,
innovation selection-only screen, and CW performance report. The innovation
screen changed selections but did not measure predictive improvement.

## Supported in the existing row benchmark

Phase 0 reused all 440 published curve points for Random, Hybrid, LCMD and IVR.
No model was retrained and no test truth was newly accessed. Both endpoints,
all seven prediction metrics, AULC, full-data gap closure, targets and savings
are available per seed and for the cohort-mean curve. Paired effects, sample
standard deviations and descriptive bootstrap intervals are reported separately.

- LCMD and Hybrid improve NRMSE AULC by about 23% relative to Random, with 5/5
  paired wins. Both V1/V2 R2 AULCs also improve in 5/5 seeds.
- The old `NO_CLEAR_SEQUENTIAL_AL_GAIN` label does not negate these gains. Its
  frozen LCMD-versus-Hybrid rules did not pick a unique winner.
- IVR has better final mean NRMSE/RMSE/R2 than LCMD, but worse NRMSE and both R2
  AULCs. Its late endpoint advantage is not whole-curve dominance.
- The empirical first-observed Random@1005 comparison gives LCMD 73.68%
  incremental label saving, whereas the old fixed-1005/interpolated definition
  gives 79.27%. These different estimands must remain explicitly labeled.
- Hybrid requires 320 fits over five trajectories, including 210 extra ensemble
  fits; the earlier member-zero runtime table was not total training effort.

These conclusions concern repeated row splits of the same filtered dataset.
They do not establish performance on independent compounds, future laboratory
experiments, other column sizes, or an arbitrary deployment distribution.

See [Phase 0 report](../../studies/active_learning/qgeognn_v2_efficiency_review/REPORT.md)
and [complete results](../../studies/active_learning/qgeognn_v2_efficiency_review/results/complete_results.csv).

## Matched-budget control

Phase 1 is complete. The endpoint gate supports feedback value at the matched
653-label endpoint: Adaptive has mean combined NRMSE 0.436791 versus Static
0.451013, wins 4/5 paired seeds, and satisfies both endpoint RMSE guards and
non-lower endpoint R2. The curve gate does not pass: Adaptive NRMSE AULC is
0.544359 versus Static 0.539505, and both endpoint R2 AULCs are lower. Therefore
feedback improves the final prediction state in this cohort, but it is not
established as a whole-curve label-efficiency improvement. This is exploratory
development evidence on the previously exposed row splits.
All five initial LCMD B320 sequences were frozen before new selected labels;
their first 32 IDs exactly match the historical Adaptive first batches.
Static and Adaptive final acquired sets share 133/147/125/121/114 of 320 IDs for
seeds 157/887/2357/6101/12203. This establishes different selection trajectories,
not a generalization benefit.

The design fixes 653 final active labels. Static's ten B32-prefix evaluations
share a single initial acquisition order; Adaptive recomputes after each batch.
OneShot and Static share the exact final fit. Random uses the same nested
permutation control. Only 45 new fits are scheduled; all other models are reused
after provenance checks. B64/B160 remain unrun optional controls, not null results.

See the [frozen protocol](../../studies/active_learning/qgeognn_v2_batch_adaptivity/PROTOCOL.md).

## Negative results and limits of mechanism evidence

- CW-LCMD worsened mean one-step NRMSE and both endpoint RMSEs, with only 2/5
  development wins. Additional coordinate variants are not a current priority.
- Scalar IVR did not improve the overall LCMD curve despite its better endpoint.
- Forward-64/backward-32 showed essentially no median surrogate gain in the
  mechanism audit. This does not support batch-greedy optimization as the main
  bottleneck.
- Representation/ranking audits do not prove numerical instability is the main
  cause. Passing a linearized surrogate objective is not evidence of improved
  scratch-retrained neural generalization.
- B16/B32/B333 changed final data quantity. Their endpoint differences cannot
  identify a batch-size effect.

## Unvalidated hypotheses and the next stage

MaxDet's information-volume selection, RD-EMCM's disagreement-conditioned model
change, calibrated noise weighting, target-weighted multi-output V-optimality,
and predictive covariance geometry remain hypotheses for this dataset.
Quantile width is not established epistemic uncertainty or calibrated noise.
Current IVR already belongs to the regression V-optimal/BAIT-like family.

The [Phase 2 implementation plan](4G_PHASE2_IMPLEMENTATION_PLAN_2026-09-18.md)
specifies a MaxDet-first B32 screen, reuse contracts, theory distinctions,
unit/engineering checks, cost accounting and explicit advancement gates.
No Phase 2 candidate was trained. No manual LCMD-to-IVR switching threshold,
adaptive batch schedule, hyperparameter grid, BO objective or active-transfer
benchmark was added to the present 4g study.

Any later strategy-switching headroom analysis must use validation-only
counterfactual branches at common labeled states. Taking the best points from
different historical trajectories is not a reachable oracle policy. Independent
confirmation should use an untouched split/compound cohort or future data and
must acknowledge the existing development history.
