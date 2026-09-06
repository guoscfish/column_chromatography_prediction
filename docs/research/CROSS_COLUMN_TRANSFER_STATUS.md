# Cross-column transfer: evidence and open questions

Updated before the scaling-failure audit, from remote `codex/study-transfer-residual-diagnostics` at `6e10498` (preceding cross-column base `61f20c9`). Historical reports and decision JSONs retain their original provenance.

## Qualified predictor and generalization

The final QGeoGNN-V2 includes missing-condition completion and function-preserving removal of dead parameters. Exact six-output equivalence and final 4g qualification are complete. The qualified backbone/readout/head are not default modification targets. Source row test R² averages approximately 0.858/0.879 for V1/V2; compound test averages approximately 0.479/0.487. These are different tasks, not interchangeable estimates.

Always distinguish row interpolation, target-compound holdout (no target training label for that compound), and source-unseen molecular OOD. Most target compounds occurred in source training; source-unseen OOD is currently not reliably estimable. See [qualification](../../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) and [target data audit](../../studies/transfer/cross_column/data_audit/DATA_AUDIT.md).

## Actual method coverage

| Tested idea | Scope and conclusion |
| --- | --- |
| zero-shot, descriptive column-mass-ratio scaling, scale-only, affine | Current V2, 8g/25g/40g, row/compound, five seeds and four budgets. Simple calibration removes much systematic shift; scale-only is a strong reference. |
| affine + condition Ridge residual | Current V2, all six column/protocol contexts. Incremental AULC gains are small; none reaches 5%. |
| target-head-only | Current V2, all six contexts. Strong on 8g; markedly worse than calibration on 25g/40g. |
| last1/last2/full fine-tune | Historical Legacy T1/G0 and current final-V2 8g cover shallow/full adaptation. **Current 25g/40g only tested head-only; the conditional last2 trigger did not fire, and full was not run there.** Do not report unperformed 25g/40g last2/full comparisons. Increased capacity has not established a stable cross-column advantage in the experiments actually run. |
| pooled representation residual adapter | Historical T1b r8/r16/r32 after sum pooling. No stable benefit; this did not test adaptive pooling. Old source/head rankings are not current V2 rankings. |
| monotone spline, nonlinear policy | Current V2, all 120 frozen contexts. Train-only two-knot monotone q50 calibration, validation choice. No stable material improvement. |
| shared-column affine, local identity shrinkage | Current V2, equal purchased three-column portfolios and donor compound purging. Shared improves compound AULC 9.04% versus affine, but 1.42% versus scale and 1.64% versus local shrinkage. |

These ideas must not be renamed and repeated. Evidence: [cross-column report](../../studies/transfer/cross_column/CROSS_COLUMN_TRANSFER_REPORT.md), [residual diagnostics](../../studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md), [pre-experiment method audit](../../NEXT_TRANSFER_MODEL_AUDIT.md).

## What the negative controls mean

`ADDITIVE_LINEAR_CONDITION_RESIDUAL_NOT_MATERIALLY_SUPPORTED` is the conclusion for `a*x+b+Ridge(c)`. It is **not** `CONDITION_EFFECT_NOT_SUPPORTED`. Conditions already influence source q50. Residual coefficients may vary multiplicatively with condition/molecule; the 9D matrix omits explicit column mass/geometry; within-column flow is constant; low-label variance may obscure effects; residual mixes source error, transfer shift and experimental variation. The negative result neither proves conditions useless nor proves a varying-coefficient formulation correct.

`LOW_CAPACITY_1D_MONOTONE_CURVATURE_NOT_SUPPORTED` applies only to the tested source-q50 → target-volume spline family and penalties. It does not exclude general nonlinear transfer or interactions.

`AFFINE_PARAMETER_PARTIAL_POOLING_HAS_NO_MATERIAL_GAIN_BEYOND_STRONG_SHRINKAGE_CONTROLS` is the shared-column result. That model shared/shrank slopes and intercepts using a quadratic penalty. It did not learn a transferable molecular representation. Regularization can reduce low-label affine instability; the evidence does not say all shared-column models are ineffective.

Historical `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA` is retained verbatim in its original decision. Its current project-level interpretation is **`NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS`**. This restricts the tested extensions and does not close the research space.

## Why scale-only might be strong

`V_target = a * V_4g_pred` acts on a learned summary of molecule and chromatography conditions. Targets overlap source molecules strongly; target conditions form structured grids; 8g has many source-condition matches. Mass, flow and specification are confounded (target flow 10/15/30 for 8g/25g/40g). A strong empirical prediction rule is not a universal physical scaling law.

The next question is whether `target/source_q50` and `target-a*source_q50` have reproducible source-range, condition, molecule or pairing structure. Ratio denominators near zero require explicit handling. Matching must be exact on declared fields and distinguish all-source identity matches from source-train label availability. Most apparent repeats are different conditions; sparse genuine repeats cannot identify an irreducible experimental noise floor.

The [scaling-failure audit](../../studies/transfer/scaling_failure_audit/SCALING_FAILURE_AUDIT.md) will use frozen predictions, identities and splits, without QGeoGNN retraining. Training-only evidence chooses at most two supported directions among conditional scaling, molecule-dependent scaling and paired/delta learning. Test summaries follow a frozen choice and never drive method iteration. AULC/label-efficiency gains and high-budget absolute accuracy gains must be reported separately.

## Future hypotheses / experiment backlog

All items below are `FUTURE_HYPOTHESES / EXPERIMENT_BACKLOG`, not demonstrated conclusions or automatic execution instructions.

| Direction | Needed evidence or control |
| --- | --- |
| Shared QGeoGNN backbone + column-specific heads | Matched total label budgets; distinguish representation sharing from coefficient regularization. |
| Explicit column context | More specifications and crossed conditions; no causal mass/flow claim from present confounding. |
| Task / column embedding | Known-column interpolation versus genuinely held-out specifications. |
| Multi-column multitask training | Global compound isolation and equal acquired-label ledger. |
| Multi-fidelity joint learning | Source-label provenance, missing-pair controls and fidelity-aware validation. |
| Adaptive readout | Controlled readout-only intervention; no simultaneous backbone/head/loss redesign. |
| Paired / delta learning | Strict matching, source-train label availability, duplicate aggregation and unmatched coverage. |
| Conditional scaling | Reproducible multiplicative or q50×condition failure structure beyond additive Ridge. |
| Experimental noise-floor estimation | Replicated measurements at identical molecule/condition/specification and independent batches. |
| Source-unseen molecular OOD | Sufficient molecules absent from all source-training labels. |
| Crossed mass × flow experimental design | Same mass at multiple flows and same flow at multiple masses, with matched conditions. |

Related literature categories to investigate later include multi-fidelity GNN/adaptive readout, chromatographic parameter vectorization, multi-dataset retention-time learning and multi-condition/multi-column retention prediction. These are research directions, not citations asserting that a specific method will work here.
