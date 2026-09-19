# Cross-column transfer: evidence and open questions

This is the consolidated transfer synthesis. [Current project status](../NEXT_STAGE_DECISION.md) owns the next action. Study reports, protocols, splits and measured results remain at their original paths.

## Two different questions

| Setting | Population and budget | Retained interpretation |
| --- | --- | --- |
| Matched low-label transfer | Unthresholded 8g/25g/40g; row and compound; five seeds; B=30/50/70/100 | Simple calibration remains competitive; prioritize RMSE/MAE in mL, not R2 alone |
| Filtered FULL-data transfer | Filtered 25g/40g operational domain; roughly 320-360 gradient-train rows | Corrected hierarchical Center/Width is a structural reference; neural training and selection controls have not produced a universal gain |

Neither setting is pristine external confirmation. Their populations, splits, label budgets and scales differ. The [paper reconstruction](../../studies/transfer/paper_transfer_reproduction/REPRODUCTION_REPORT.md) also uses an older source and separate filtering/splits, so it is not a matched comparator.

## Completed evidence

| Tested question | Outcome | Record |
| --- | --- | --- |
| Qualified source predictor | Standalone V2 passed exact equivalence and six-run 4g qualification | [Qualification](../../studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) |
| Low-label transfer strategy | `SIMPLE_CALIBRATION_REMAINS_COMPETITIVE`; paper-style gains do not replicate across both columns/endpoints and AULC | [Matched RMSE](../../studies/transfer/matched_rmse_benchmark/MATCHED_RMSE_REPORT.md) |
| Low-capacity monotone curvature and shared affine coefficients | No stable material gain beyond strong scale/shrinkage controls | [Residual diagnostics](../../studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md) |
| Conditional scaling | EA/V1 structure exists; no replicated material improvement over the strongest controls | [Scaling decision](../../studies/transfer/scaling_failure_audit/NEXT_MODEL_DECISION.md) |
| Source-preserving shallow/full fine-tuning | No replicated material advantage over calibration in the frozen 120 contexts | [Source anchoring](../../studies/transfer/source_anchored_shared_transfer/NEXT_STAGE_DECISION.md) |
| Calibration variables and Center/Width extensions | Corrected nested tests failed their promotion gates; stop expanding this calibration family | [Controlled audit](../../studies/transfer/controlled_lightweight_transfer_audit/FINAL_REPORT.md), [structured follow-up](../../studies/transfer/structured_center_width_followup/FINAL_REPORT.md) |
| Physical metadata | Mass/flow are confounded; no newly identifiable physical context | [Metadata audit](../../studies/transfer/physical_metadata_identifiability_audit/FINAL_REPORT.md) |
| FULL-data hierarchical/latent models | Earlier HIER source-scale semantics required repair; corrected shared-lambda HIER is the stable reference | [Semantic repair](../../studies/transfer/hier_cw_semantic_repair/FINAL_REPORT.md) |
| Converged P0/P1 staged neural fitting | P0 Stage B adequately converged; P1 fails the two-column gate and is not promoted | [N1](../../studies/transfer/traditional_transfer_converged_baseline_v1/FINAL_REPORT.md) |
| Endpoint-normalized / q50 loss controls | Neither passes the two-column rule; retain L0 raw quantile loss | [Loss screen](../../studies/transfer/conditioned_source_readout/LOSS_SCREEN_REPORT.md) |
| Adaptive and condition-query readout | R1/R2 fail the ROW inner gate; R3/R4 and outer scoring were not run | [Readout](../../studies/transfer/conditioned_source_readout/FINAL_REPORT.md) |
| Shared representation and column FiLM | A2 improves inner metrics but does not survive the outer cross-protocol guard | [Multitask](../../studies/transfer/column_conditioned_multitask/FINAL_REPORT.md) |
| Gradient conflict control | PCGrad reduces conflict but does not materially improve transfer; stops before outer evaluation | [PCGrad](../../studies/transfer/column_conditioned_pcgrad/FINAL_REPORT.md) |

## Quantitative anchors

- Matched low-label calibration comparisons retain Conditional-EA / local identity shrinkage as baseline families. Large-column B=100 SSE is tail-dominated: 60.4%-92.8% for 25g and 58.9%-87.7% for 40g in the matched benchmark.
- Converged P1 versus P0 loses mean shared NRMSE by 1.10% on 25g (2/5 wins) and improves 40g by 0.57% (3/5). Its Stage A remains budget-censored; this does not authorize automatically extending it.
- Readout R1/R2 improvements stay below the material two-column threshold; no outer result is available. Header-only result tables record that stopping decision.
- FiLM A2 improves COMPOUND inner NRMSE by 5.05%/8.18% on 25g/40g. Outer COMPOUND gains shrink to 1.47%/2.65%, while ROW worsens 1.10%/9.37%.
- PCGrad raises mean shared-gradient cosine from -0.0808 to +0.2662 and reduces negative burden by 99.28%. Relative inner NRMSE improvement is -0.37% on 25g and +0.50% on 40g, below the 2% threshold.

These anchors summarize distinct controls; they are not one pooled model leaderboard.

## What is closed, and what remains open

The negative results apply to the tested designs. Additive condition Ridge failing does not mean conditions are irrelevant; affine coefficient sharing is not the same as learning a shared representation; low-capacity monotone calibration does not exclude all nonlinear transfer.

The older blanket wording `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA` is retained in frozen records. Its scoped interpretation is `NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS`. Do not rename and rerun the same candidate as a new method.

Prioritize new evidence before further architecture or optimizer expansion:

- Independently collected compound/batch validation with deliberate high-volume-tail coverage.
- Repeated measurements at identical molecule/condition/specification to estimate experimental variability.
- Crossed mass and flow settings; current 8g/25g/40g flows are fixed by column, preventing causal separation.
- Source-aware held-out molecules; target-compound holdout alone is not source-unseen OOD.
- Verified physical metadata and exact source/target pair coverage before a physical or delta-learning claim.

Loss, readout, FiLM and conflict controls have now been tested, so older plans proposing them are no longer the next action. Hard ordered quantiles, normalized anchoring, alternative scope/LR schedules and paired/delta methods remain unexecuted hypotheses, not authorized sweeps or established gains. Active transfer still requires an independently validated transfer baseline and adequate UQ.

## Historical provenance

The exact status hashed by the earlier scaling protocol remains in [the frozen snapshot](history/CROSS_COLUMN_TRANSFER_STATUS_scaling_failure_audit.md). The root [pre-experiment method audit](../../NEXT_TRANSFER_MODEL_AUDIT.md) and the [readout design](ROW_FIRST_CONDITIONED_SOURCE_TRANSFER_PLAN_2026-09-13.md) remain original protocol references.

Superseded project plans and independent one-off scripts are recoverable through [RETIREMENTS.json](../repository/RETIREMENTS.json); their detailed result tables remain in [the transfer study index](../../studies/transfer/README.md).
