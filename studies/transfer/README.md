# Cross-column transfer studies

Keep the low-label, unthresholded benchmark separate from filtered FULL-data studies. Their populations, budgets and normalization differ. These are developmental results, not independent external confirmation.

## Main evidence

| Question | Result to read |
| --- | --- |
| Which low-label strategy is competitive? | [Matched absolute-error benchmark](matched_rmse_benchmark/MATCHED_RMSE_REPORT.md) |
| What is the corrected structural FULL-data reference? | [Hierarchical Center/Width semantic repair](hier_cw_semantic_repair/FINAL_REPORT.md) |
| Does a converged staged neural recipe help? | [P0/P1 convergence baseline](traditional_transfer_converged_baseline_v1/FINAL_REPORT.md) |
| Do alternative losses or adaptive readout help? | [Loss screen](conditioned_source_readout/LOSS_SCREEN_REPORT.md), [readout result](conditioned_source_readout/FINAL_REPORT.md) |
| Does column-conditioned joint training help? | [Multitask/FiLM](column_conditioned_multitask/FINAL_REPORT.md) |
| Is gradient conflict the missing mechanism? | [PCGrad](column_conditioned_pcgrad/FINAL_REPORT.md) |

[Transfer synthesis](../../docs/research/CROSS_COLUMN_TRANSFER_STATUS.md) summarizes what was tested, what is closed and what data are still needed. [Current status](../../docs/NEXT_STAGE_DECISION.md) owns next steps.

## Completed supporting evidence

| Family | Records |
| --- | --- |
| Final-source baseline and cross-column schedules | [4g to 8g](4g_to_8g/TRANSFER_BASELINE_REPORT.md), [cross-column](cross_column/CROSS_COLUMN_TRANSFER_REPORT.md) |
| Calibration and source anchoring | [Residual diagnostics](residual_diagnostics/RESULT_INTERPRETATION.md), [scaling audit](scaling_failure_audit/NEXT_MODEL_DECISION.md), [source-anchored transfer](source_anchored_shared_transfer/NEXT_STAGE_DECISION.md) |
| Closed lightweight extensions | [Corrected calibration audit](controlled_lightweight_transfer_audit/FINAL_REPORT.md), [structured follow-up](structured_center_width_followup/FINAL_REPORT.md) |
| FULL-data development | [Filtered benchmark](filtered_full_data_benchmark/FINAL_REPORT.md), [baseline finalization](full_data_baseline_finalization/FINAL_REPORT.md), [architecture controls](full_data_architecture_headroom/FINAL_REPORT.md), [joint latent controls](full_data_joint_hierarchical_latent/FINAL_REPORT.md), [selective latent audit](row_column_selective_latent_audit/FINAL_REPORT.md) |
| Physics and metadata | [Physics transfer](physics_column_conditioned_transfer/NEXT_STAGE_DECISION.md), [identifiability audit](physical_metadata_identifiability_audit/FINAL_REPORT.md) |
| Earlier neural recipes | [Formal ROW](traditional_transfer_improvement/formal_row_5seed/FINAL_REPORT.md), [metric/convergence audit](stage2_transfer_audit/FINAL_REPORT.md) |
| Paper reconstruction | [Reproduction](paper_transfer_reproduction/REPRODUCTION_REPORT.md), [gap decomposition](paper_transfer_gap_decomposition/FINAL_REPORT.md); distinct protocol, not a matched ranking |

## Legacy evidence

[S1 source-target shift](source_target_shift/README.md), [T1 low-label adaptation](low_label_adaptation/README.md) and [T1b adapter capacity](adapter_capacity/README.md) use older predictor contracts. Their rankings do not establish current V2 performance.

Closed one-off scripts are represented by their retained reports and [recovery registry](../../docs/repository/RETIREMENTS.json). Shared implementations and any runners still imported by code or tests remain available.
