# Current research decision

## Column-conditioned multi-task result (2026-09-14)

The preregistered [column-conditioned multi-task QGeoGNN study](../studies/transfer/column_conditioned_multitask/FINAL_REPORT.md)
is complete. A2 categorical column FiLM passed the COMPOUND inner continuation
gate against A1 separate heads in both target columns: mean paired-fold NRMSE
improved 5.05% on 25g and 8.18% on 40g, with 5/5 seed wins in each. The same
inner signal appeared under ROW (5.10% and 6.51%). Gradient diagnostics show
persistent conflict, especially between 4g and 40g, without a greater-than-10x
median magnitude imbalance.

The developmental outer confirmation does not support promotion. Relative to
A1, A2 improved COMPOUND mean NRMSE only 1.47% on 25g and 2.65% on 40g, then
worsened ROW by 1.10% and 9.37%, respectively. It also remained worse than the
corrected hierarchical Center/Width reference on both COMPOUND columns. The
current decision is therefore **`INNER_REPRESENTATION_GATE_PASSED_BUT_NO_ROBUST_OUTER_REPRESENTATION_SIGNAL`**
and **`PROJECT_TRANSFER_GAIN_FALSE`**. These historically exposed outer
identities are developmental confirmation, not independent validation; the
inherited COMPOUND partitions also permit limited related-target donor overlap.

Do not expand the column-conditioned architecture, add 8g, or start Active
Learning from this result. A separately preregistered gradient-conflict control
is the next isolated computational mechanism if another model study is run;
independent/crossed compound and batch collection with deliberate tail coverage
remains the external-validation priority. No next-stage method is authorized by
the completed study itself.

## Train-only neural-baseline audit record (2026-09-12)

The historical 150-epoch Adam evidence is not a stable P0/P1 ranking basis.
The regenerated audit now distinguishes `protocol_max_epoch`, `epochs_run`,
`best_epoch`, `run_reached_protocol_ceiling`, `best_at_protocol_ceiling`, and
`best_at_run_end`: the corrected historical fact is still 41/50 Adam stages
selecting the 150-epoch **protocol ceiling**, while 44/50 actually ran to it.

`traditional_transfer_converged_baseline_v1` is the only authorized neural
follow-up in this record: P0/P1, current BN, Adam, raw quantile loss,
historical-shallow Stage B, five-fold inner canonical-smiles GroupKFold, and
fixed-epoch full-gradient refit. Validation metadata now distinguishes
gradient-fit rows from validation-selection rows; zero gradient participation
does not mean validation labels were unused for checkpoint selection. The
legacy frozen artifacts are preserved; their incorrect zero-validation wording
is documented by the audit rather than overwritten. See the
[roadmap](research/TRANSFER_RESEARCH_ROADMAP_2026-09-12.md).

### Completed result: `NO_UNIVERSAL_STAGED_TRANSFER_GAIN`

The converged ROW developmental confirmation is complete. P0 is not severely
budget-censored at 500 epochs (25g/40g Stage-B ceiling selections 0/25 and
1/25); P1 Stage B is also not severe. But P1 head-only Stage A selected epoch
500 with negative late validation slope in **25/25** inner folds for each
column. It therefore remains budget-censored and P1 cannot be promoted.

P1 versus P0 shared train-NRMSE is +1.10% on 25g (2/5 wins) and -0.57% on 40g
(3/5 wins). At 25g, V1 RMSE worsens 2.20%, exceeding the 2% endpoint gate.
P0 is slightly better than paper-style on both mean 25g endpoint RMSEs
(-0.137/-0.125 mL for V1/V2), while paper-style remains better on 40g
(P0 gap +0.636/+0.507 mL). This is developmental confirmation only; it does
not authorize COMPOUND or Active Learning. The next isolated neural action is
endpoint-normalized quantile loss using the converged P0 baseline; extending
P1 Stage A requires a separate preregistered decision rather than silently
raising its budget.

## Stage 2 audit decision: STOP BEFORE COMPOUND

The latest decision is `STOP_BEFORE_COMPOUND_AND_ACTIVE_LEARNING`. The formal ROW evidence was a pre-existing local run audited and summarized in Stage 1, not 40 newly trained models. Stage 2 found that formal P0-P3 combined NRMSE used per-context gradient-train scales, while historical reference combined NRMSE used fixed qualified 4g source-train scales. The old cross-method normalized ranking is therefore withdrawn. Raw RMSE, MAE, and R2 on matched ROW test IDs remain usable; harmonized metrics are in [row_metric_harmonization](../studies/transfer/row_metric_harmonization/METRIC_COMPARABILITY_AUDIT.md).

The harmonized formal evidence does not establish a universal new transfer recipe. P2 is best among P0-P3 on 25g combined NRMSE, while paper-style is strongest on 40g across RMSE, MAE, R2 and unified NRMSE. P1 is the most credible simple candidate for train-only confirmation because it improves formal combined NRMSE over P0 in both columns, but the 150-epoch Adam budget is inadequate for stable ranking: 41/50 Adam stages hit the maximum and 43/50 retained a negative late validation slope. The independent [convergence audit](../studies/transfer/traditional_transfer_convergence_audit/CONVERGENCE_AUDIT.md) defines a longer train-only budget audit.

P3's pilot advantage is not a stable finding; source BN statistics are column-dependent; and the current evidence cannot identify an L-BFGS advantage because P3 changes both optimizer and staged recipe while Adam is budget-censored. Do not select P2 for 25g and P1 for 40g from exposed outer-test results. COMPOUND and Active Learning remain blocked. See the [Stage 2 final report](../studies/transfer/stage2_transfer_audit/FINAL_REPORT.md) and [next-stage plan](TRANSFER_NEXT_STAGE_PLAN.md).

Latest baseline decision: **`CENTER_WIDTH_PROMOTED_AS_WORKING_FULL_DATA_BASELINE`** in the [full-data baseline finalization](../studies/transfer/full_data_baseline_finalization/FINAL_REPORT.md). Under the unchanged filtered FULL-data outer identities, corrected six-coefficient M3 improves compound-primary mean combined normalized RMSE by 3.84% versus Conditional EA, with 9/10 paired seed wins; the column gains are 6.32% on 25g and 2.76% on 40g, and all four mean endpoint RMSEs improve. M3 exactly matches the historical corrected implementation (`max_abs_difference=0`). The shared mass+flow control does not improve the primary portfolio. Adding the three unitless legacy descriptors helps 25g but worsens 40g, so they have no stable cross-column incremental predictive value. M3 is frozen as the working FULL-data transfer baseline for subsequent Active Learning baseline construction; no learning curve or Active Learning was run in this study.

Latest metadata gate: **`NO_NEW_IDENTIFIABLE_PHYSICAL_CONTEXT`** in the [physical column metadata provenance and identifiability audit](../studies/transfer/physical_metadata_identifiability_audit/FINAL_REPORT.md). The previous physics study already used nominal packing mass, flow, loading mass per packing mass, and loading-solvent volume per packing mass. The remaining hard-coded `column_dia`/`column_len`/`column_den` tuples are only `SEMANTICS_SUGGESTED`: the repository does not establish their units, measurement provenance, or why 25g and 40g share `(2.15, 15.6, 0.5248)`. Across the eligible 8g/25g/40g joint context, mass+flow has intercept-inclusive rank 3 and adding all three legacy fields remains rank 3. No physics-conditioned center/width model was trained and no outer truth was read. Physics-model expansion is closed pending verified experimental metadata and crossed column/flow evidence; the next separately preregistered computational stage remains the filtered random learning curve.

Latest completed stage: **`NO_STRUCTURED_CENTER_WIDTH_HEADROOM`** in the [structured center/width follow-up](../studies/transfer/structured_center_width_followup/FINAL_REPORT.md). Against corrected M3, center-only magnitude worsens aggregate normalized RMSE by 0.62%; center-conditioned width improves 25g V1 by 4.81% (5/5 seeds) but worsens 25g V2 by 3.09%, does not replicate on 40g, and gains only 0.38% in aggregate. Neither seven-coefficient candidate passes the frozen gate. Outer truth remained blind. Center/width calibration expansion is closed; the next separately preregistered computational stage is the filtered random learning curve. Active learning remains unauthorized until that curve establishes a material label-scarcity regime.

Latest corrected stage: **`NO_CONTROLLED_LIGHTWEIGHT_CALIBRATION_HEADROOM`** in the [controlled lightweight transfer audit](../studies/transfer/controlled_lightweight_transfer_audit/FINAL_REPORT.md). The corrected Conditional-EA nested extensions match the historical baseline to `7.105e-15` while preserving `SSE/n`, the base-slope-to-one prior, intercept/EA penalties, source scales, and mass ratios. M2 magnitude, width, loading-solvent, and the six-coefficient center/width reparameterization all fail the frozen 3% aggregate plus 3-of-4 endpoint gate. No outer truth was read and no candidate was promoted. Calibration-variable expansion is formally closed for this tested family. The next separately preregistered computational stage is the filtered random learning curve; it was not started by this audit, and active learning remains unauthorized until label scarcity is demonstrated.

The preceding `NO_LIGHTWEIGHT_CALIBRATION_HEADROOM_IDENTIFIED` result in the [filtered transfer headroom audit](../studies/transfer/filtered_transfer_headroom_audit/FINAL_REPORT.md) is retained as preliminary historical evidence. Its M2 comparison was not strictly nested and its M3 coefficient count was misstated; the corrected study above owns the current calibration decision.

Latest completed stage: **`SIMPLE_CALIBRATION_REMAINS_COMPETITIVE`**. The [matched cross-column absolute-error benchmark](../studies/transfer/matched_rmse_benchmark/MATCHED_RMSE_REPORT.md) ranks eight strategies under one no-threshold protocol: final qualified 4g source SHA `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`, inherited 8g/25g/40g row and compound splits, five seeds, and 30/50/70/100 revealed-label budgets. RMSE/MAE in mL are primary; R² is a secondary trend metric.

The new paper-style current-V2 control freezes all 120 predictions before any new test evaluation. It does not satisfy the preregistered rule of at least 5% paired gains in both B=100 normalized RMSE and AULC, with at least 4/5 wins, versus scale-only, affine and local identity shrinkage in both 25g and 40g. Conditional EA and local identity shrinkage therefore remain the primary simple baseline family; target-head-only remains a useful 8g comparator. The full matched table, AULC, paired deltas and method-level caveats are in the benchmark report rather than repeated here.

Large-column absolute error is not explained away by high R²: at B=100, high-volume test tails contribute 60.4%–92.8% of target SSE on 25g and 58.9%–87.7% on 40g across methods/protocols. This is matched developmental evidence, not an independent confirmation set, because the reused source-anchored artifacts had historical test exposure. The legacy [paper-transfer reconstruction](../studies/transfer/paper_transfer_reproduction/REPRODUCTION_REPORT.md) remains `PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION` only: its filtering, larger label fraction, old source and different test population exclude it from the ranking.

No additional complex model, UQ change, or Active Learning study is authorized by this result. The next scientific need is independently collected/held-out compound and batch evidence with deliberate high-volume-tail coverage and crossed column specifications. That evidence must precede a new preregistered complexity intervention.

The preceding stage remains **`STRUCTURED_FAILURE_BUT_NO_MATERIAL_MODEL_GAIN`**. The [scaling failure audit](../studies/transfer/scaling_failure_audit/SCALING_FAILURE_AUDIT.md) finds reproducible EA/V1 structure in training-only evidence. One conditional-scaling experiment completed 120 frozen contexts. Its validation-selected policy improves 25g AULC versus scale/affine, but not materially versus matched additive/shrinkage controls across columns. Standalone 40g compound has a positive AULC signal, which does not meet the preregistered replication or high-budget accuracy gate. See the [complete model decision](../studies/transfer/scaling_failure_audit/NEXT_MODEL_DECISION.md). No B/C, readout, or additional model was appended after test evaluation.

Retained calibration interpretation: **`NO_ADDITIONAL_COMPLEXITY_JUSTIFIED_FOR_TESTED_CALIBRATION_EXTENSIONS`**. The historical decision below concerns the tested calibration extensions, not all nonlinear/shared models. The updated [research status and backlog](research/CROSS_COLUMN_TRANSFER_STATUS.md) distinguishes these historical results from the completed representation-level transfer stage.

The preceding [two controlled residual diagnostics](../studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md), based on cross-column commit `61f20c9`, conclude **`NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA`**. Monotone nonlinear calibration has no stable material benefit. Shared calibration improves compound portfolio AULC by 9.04% versus affine, but only 1.42% versus scale-only and 1.64% versus independent shrinkage. Neither family passes the preregistered cross-seed/cross-column gate. The remaining error cannot yet be attributed uniquely to readout or data limitations. No extra model is appended to obtain a positive result.

The representation stage completed ordinary full fine-tuning as a preregistered matched control. It does not start Active Learning, an adapter width sweep, Clean, or adaptive readout. The older cross-column report's `ACTIVE_CALIBRATION` proposal is historical, not the current execution instruction. Further mechanism identification would benefit from independent batches/compounds, replication, crossed mass/flow settings and tail coverage. Target-compound holdout is not source-unseen OOD.

The final standalone QGeoGNN-V2 passed exact R2-pruned equivalence and six-run 4g qualification: `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`. Predictor architecture is no longer the default research object.

The current quantile head is `CURRENT_HEAD_RETAINED_FOR_POINT_TRANSFER`. Compound-seed-525 V1 crossing is 15.22%, and nominal 80% intervals under-cover, especially on unseen compounds. Width remains positively associated with absolute error. `MONOTONIC_HEAD_CONTROL_REQUIRED_BEFORE_ACTIVE_TRANSFER` therefore applies to future UQ/active-transfer qualification; it does not block ordinary point transfer. No replacement head was trained.

The [final-source baseline report](../studies/transfer/4g_to_8g/TRANSFER_BASELINE_REPORT.md) and [machine-readable decision](../studies/transfer/4g_to_8g/decision.json) own the measured transfer ranking. The comparison uses one preregistered standalone source, five existing frozen target row partitions, four nested random-label budgets and five fixed adaptation families. This is developmental evidence: independent target/compound/column validation remains necessary.

Ordinary transfer and UQ are parallel workstreams. Independent transfer validation plus UQ qualification precede active transfer. No active acquisition, adapter sweep, Clean repair or predictor benchmark was executed. Historical T1/T1b/G0/S1 are `HISTORICAL_LEGACY_PREDICTOR_EVIDENCE`; their rankings are not imported into the new decision.

Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / HISTORICAL_NEGATIVE_RESULT`. Historical decision JSONs retain their original next-stage proposals to preserve provenance; this document supersedes those proposals, including the former requirement to finish a quantile-head experiment before ordinary transfer.
