# Physics column-conditioned transfer decision

**CURRENT_DATA_DO_NOT_IDENTIFY_COLUMN_PHYSICS_SUFFICIENTLY**

| method | endpoint | replicated | passing_contexts |
| --- | --- | --- | --- |
| packing_mass_physical_scale | aulc | False | [] |
| packing_mass_physical_scale | budget100 | False | [] |
| physics_scale_residual | aulc | False | [] |
| physics_scale_residual | budget100 | False | [['40g', 'row']] |
| raw_column_conditioned | aulc | False | [] |
| raw_column_conditioned | budget100 | False | [] |
| mass_normalized_column_conditioned | aulc | False | [] |
| mass_normalized_column_conditioned | budget100 | False | [] |

Retain Scale/Shrinkage; no new arm passes both replicated endpoints.

Do not add features, attention, LR/alpha/architecture sweeps, column IDs, readout or source anchoring after test. No AL launched. Prioritize metadata verification, crossed mass × flow, independent batches/repeats and source-unseen/tail coverage. A negative result applies to this source-initialized training recipe, not physical impossibility. A predictive gain cannot identify a causal mass or flow mechanism. No claim of TRANSFER_SOLVED; 40g absolute errors remain separately reported.

See RESULT_INTERPRETATION.md, paired_comparisons.csv and budget100_metrics.csv for quantitative evidence.
