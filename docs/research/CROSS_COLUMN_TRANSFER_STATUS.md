# Cross-column transfer: evidence and open questions

Updated after the completed [column-conditioned multi-task QGeoGNN study](../../studies/transfer/column_conditioned_multitask/FINAL_REPORT.md). Historical reports and decision JSONs retain their original provenance. The byte-exact status consumed by the earlier scaling-failure protocol is preserved in [the historical snapshot](history/CROSS_COLUMN_TRANSFER_STATUS_scaling_failure_audit.md).

## Latest representation-level study

**INNER_REPRESENTATION_GATE_PASSED_BUT_NO_ROBUST_OUTER_REPRESENTATION_SIGNAL.**
The only candidates were A1 joint 4g/25g/40g training with separate copied
heads and A2, which added zero-initialized categorical column FiLM to the final
two message-passing blocks. The qualified 4g source, raw quantile loss, frozen
FULL-data identities, globally grouped target-molecule inner folds, equal-task
batching, and one joint checkpoint selector were fixed before fitting.

A2 passed every COMPOUND inner gate in both columns: 25g improved mean
paired-fold NRMSE 5.05% with 5/5 seed and 18/25 fold wins; 40g improved 8.18%
with 5/5 seed and 20/25 fold wins. Endpoint, Center/Width, and high-volume-tail
inner metrics all improved. ROW inner evidence also favored A2 (5.10% on 25g,
6.51% on 40g). Shared-layer gradient cosines were frequently negative,
especially for 4g versus 40g, while median magnitude ratios remained below 10.

After all blind predictions were frozen, developmental outer scoring was less
supportive. A2 versus A1 improved mean COMPOUND NRMSE by 1.47% on 25g and 2.65%
on 40g, but worsened ROW by 1.10% and 9.37%. The corrected hierarchical
Center/Width reference remained stronger on both COMPOUND columns. Therefore
`REPRESENTATION_SIGNAL=False` under the cross-protocol guard and
`PROJECT_TRANSFER_GAIN=False`. Learned task embeddings have no physical
interpretation, target-compound holdout is not source-unseen OOD, and the
inherited per-column COMPOUND partitions contain limited cross-target donor
overlap. No architecture expansion, 8g addition, calibration, Active Learning,
PCGrad, or domain-specific normalization was appended.

The next isolated computational hypothesis, if separately preregistered, is a
gradient-conflict handling control. Independent/crossed compound and batch data
with intentional high-volume-tail coverage remains the higher-priority external
validation need. Uncontrolled neural architecture expansion should stop.

## Latest matched strategy benchmark

**SIMPLE_CALIBRATION_REMAINS_COMPETITIVE** under the current final-source, no-threshold protocol. The benchmark inherits the frozen 8g/25g/40g row and compound schedules, five outer seeds and B=30/50/70/100 revealed-label ledger. It reuses only artifact-verified current baseline predictions, fits the new `paper_style_current_v2` shallow transfer control, freezes all 120 new prediction files before reading test truth, and reports RMSE/MAE in mL as primary outcomes.

At B=100, the best matched 25g compound mean RMSE is 13.01/19.76 mL (V1/V2, conditional EA); the best row values are 17.37/25.22 mL (paper-style V1 / affine V2). For 40g, best compound values are 31.84/40.97 mL (conditional EA / affine-shrinkage) and best row values are 29.79/36.19 mL (paper-style V1 / conditional EA V2). The paper-style method fails the preregistered all-reference, both-endpoint replication gate: it is materially worse in compound and its isolated row B=100 gains do not reproduce in AULC.

High-volume tails dominate B=100 large-column SSE (25g: 60.4%–92.8%; 40g: 58.9%–87.7%). Thus, high R² and large RMSE coexist partly because outcome variance rises with column size, but the result is also a direct tail-extrapolation limitation, not evidence of operationally accurate large-column prediction. Conditional EA / local identity shrinkage remain the point-transfer baseline family. No current result identifies a causal failure mechanism or supports a more complex model; independent compound/batch data and intentional tail coverage are required first.

The matched ranking is developmental because reused source-anchored evidence was historically test-exposed. The [paper-transfer reconstruction](../../studies/transfer/paper_transfer_reproduction/REPRODUCTION_REPORT.md) is historical `PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION`, not a comparator: it has legacy filtering, a larger label fraction, an old E0 source and another test population.

## Qualified predictor and generalization

The final QGeoGNN-V2 includes missing-condition completion and function-preserving removal of dead parameters. Exact six-output equivalence and final 4g qualification are complete. The qualified backbone/readout/head are not default modification targets. Source row test R² averages approximately 0.858/0.879 for V1/V2; compound test averages approximately 0.479/0.487. These are different tasks, not interchangeable estimates.

Always distinguish row interpolation, target-compound holdout (no target training label for that compound), and source-unseen molecular OOD. Most target compounds occurred in source training; source-unseen OOD is currently not reliably estimable. See [qualification](../../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) and [target data audit](../../studies/transfer/cross_column/data_audit/DATA_AUDIT.md).

## Actual method coverage

| Tested idea | Scope and conclusion |
| --- | --- |
| zero-shot, descriptive column-mass-ratio scaling, scale-only, affine | Current V2, 8g/25g/40g, row/compound, five seeds and four budgets. Simple calibration removes much systematic shift; scale-only is a strong reference. |
| affine + condition Ridge residual | Current V2, all six column/protocol contexts. Incremental AULC gains are small; none reaches 5%. |
| target-head-only | Current V2, all six contexts. Strong on 8g; markedly worse than calibration on 25g/40g. |
| Historical last1/last2/full fine-tune | Legacy T1/G0 and the earlier current final-V2 8g study cover shallow/full adaptation. At the cross-column/scaling-audit stage, current 25g/40g had only tested head-only: the last2 trigger did not fire. These historical facts remain unchanged. |
| Current V2 standard shallow/full and source-anchored shallow/full | Now tested in all 120 contexts. Shallow explicitly trains `backbone.convs.4`, condition completion and target head (36,387 parameters); full trains backbone, condition completion and target head (458,952). Anchored controls match capacity exactly. No replicated material advantage over strong calibration; source preservation alone is insufficient. These are not renamed Legacy last1/last2 results. |
| pooled representation residual adapter | Historical T1b r8/r16/r32 after sum pooling. No stable benefit; this did not test adaptive pooling. Old source/head rankings are not current V2 rankings. |
| monotone spline, nonlinear policy | Current V2, all 120 frozen contexts. Train-only two-knot monotone q50 calibration, validation choice. No stable material improvement. |
| shared-column affine, local identity shrinkage | Current V2, equal purchased three-column portfolios and donor compound purging. Shared improves compound AULC 9.04% versus affine, but 1.42% versus scale and 1.64% versus local shrinkage. |
| Conditional EA / validation-selected conditional policy | Completed scaling-failure study, all 120 contexts. Reproducible EA/V1 structure and local positive signals, but no replicated material gain over strong additive/shrinkage controls. Frozen references only in the representation stage. |

These ideas must not be renamed and repeated. Evidence: [cross-column report](../../studies/transfer/cross_column/CROSS_COLUMN_TRANSFER_REPORT.md), [residual diagnostics](../../studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md), [pre-experiment method audit](../../NEXT_TRANSFER_MODEL_AUDIT.md).

## What the negative controls mean

`ADDITIVE_LINEAR_CONDITION_RESIDUAL_NOT_MATERIALLY_SUPPORTED` is the conclusion for `a*x+b+Ridge(c)`. It is **not** `CONDITION_EFFECT_NOT_SUPPORTED`. Conditions already influence source q50. Residual coefficients may vary multiplicatively with condition/molecule; the 9D matrix omits explicit column mass/geometry; within-column flow is constant; low-label variance may obscure effects; residual mixes source error, transfer shift and experimental variation. The negative result neither proves conditions useless nor proves a varying-coefficient formulation correct.

`LOW_CAPACITY_1D_MONOTONE_CURVATURE_NOT_SUPPORTED` applies only to the tested source-q50 → target-volume spline family and penalties. It does not exclude general nonlinear transfer or interactions.

`AFFINE_PARAMETER_PARTIAL_POOLING_HAS_NO_MATERIAL_GAIN_BEYOND_STRONG_SHRINKAGE_CONTROLS` is the shared-column result. That model shared/shrank slopes and intercepts using a quadratic penalty. It did not learn a transferable molecular representation. Regularization can reduce low-label affine instability; the evidence does not say all shared-column models are ineffective.

Historical `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA` is retained verbatim in its original decision. Its current project-level interpretation is **`NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS`**. This restricts the tested extensions and does not close the research space.

## Why scale-only might be strong

`V_target = a * V_4g_pred` acts on a learned summary of molecule and chromatography conditions. Targets overlap source molecules strongly; target conditions form structured grids; 8g has many source-condition matches. Mass, flow and specification are confounded (target flow 10/15/30 for 8g/25g/40g). A strong empirical prediction rule is not a universal physical scaling law.

The completed scaling-failure audit found reproducible EA/V1 structure in `target/source_q50` and `target-a*source_q50`. Ratio denominators near zero require explicit handling. Matching must be exact on declared fields and distinguish all-source identity matches from source-train label availability. Most apparent repeats are different conditions; sparse genuine repeats cannot identify an irreducible experimental noise floor.

The [scaling-failure audit](../../studies/transfer/scaling_failure_audit/SCALING_FAILURE_AUDIT.md) used frozen predictions, identities and splits without QGeoGNN retraining. Training-only evidence selected conditional scaling; molecule-dependent and paired/delta directions did not pass screening. The completed representation study then tested a separate mechanism. No polynomial/MLP calibration, condition sweep or molecule-dependent scalar extension is authorized. AULC/label-efficiency gains and high-budget absolute accuracy gains remain separate endpoints.

## Future hypotheses / experiment backlog

All items below are `FUTURE_HYPOTHESES / EXPERIMENT_BACKLOG`, not demonstrated conclusions or automatic execution instructions.

| Direction | Needed evidence or control |
| --- | --- |
| Column-conditioned shared representation | Completed. A2 column FiLM passed both inner protocols but did not survive the outer cross-protocol guard; no robust representation signal or project transfer gain. Do not expand this architecture without new evidence. |
| Explicit column context | More specifications and crossed conditions; no causal mass/flow claim from present confounding. |
| Task / column embedding | Fixed 16D categorical embedding tested in A2; inner-positive but not outer-robust. Future work requires genuinely held-out specifications, not an embedding sweep. |
| Multi-column multitask training | Equal-task 4g/25g/40g training completed. Any future study needs globally isolated outer compounds and independent data. |
| Multi-fidelity joint learning | Source-label provenance, missing-pair controls and fidelity-aware validation. |
| Adaptive readout | Controlled readout-only intervention; no simultaneous backbone/head/loss redesign. |
| Paired / delta learning | Strict matching, source-train label availability, duplicate aggregation and unmatched coverage. |
| Conditional scaling extensions | Closed for this stage: frozen EA experiment completed without replicated material gain. Do not restart as V2/V3, polynomial, MLP or feature sweep. |
| Experimental noise-floor estimation | Replicated measurements at identical molecule/condition/specification and independent batches. |
| Source-unseen molecular OOD | Sufficient molecules absent from all source-training labels. |
| Crossed mass × flow experimental design | Same mass at multiple flows and same flow at multiple masses, with matched conditions. |

Related literature categories to investigate later include multi-fidelity GNN/adaptive readout, chromatographic parameter vectorization, multi-dataset retention-time learning and multi-condition/multi-column retention prediction. These are research directions, not citations asserting that a specific method will work here.
