# Script index

Run from the repository root in the validated Conda `fish` environment. Read the linked protocol before any fitting or test evaluation. A runnable entry point is not authorization to start a new experiment.

## Current 4g work

| Entry in `studies/` | Role | Study / execution |
| --- | --- | --- |
| `run_qgeognn_v2_batch_adaptivity.py` | Formal matched-budget control | [Phase 1](../studies/active_learning/qgeognn_v2_batch_adaptivity/README.md); prepared run/resume, not an independent duplicate launch |
| `summarize_qgeognn_v2_adaptivity.py` | Reporting | Current adaptivity artifacts; requires complete frozen trajectories |
| `report_qgeognn_v2_efficiency.py` | Reporting only | [Phase 0](../studies/active_learning/qgeognn_v2_efficiency_review/REPORT.md); retained results, no retraining |

## Retained workflows

These are completed studies or reproduction entry points. Follow their frozen configs and use separate reproduction outputs.

| Family | Entries in `studies/` | Role / evidence |
| --- | --- | --- |
| Standalone predictor | `run_final_v2_engineering.py`, `run_final_4g_qualification.py`, `summarize_final_4g_qualification.py` | Engineering and formal [qualification](../studies/predictor/final_4g_qualification/README.md) |
| Current-V2 row AL | `run_qgeognn_v2_4g_row_lcmd_pilot.py`, `run_qgeognn_v2_4g_row_small_batch_benchmark.py`, `run_qgeognn_v2_row_hybrid_extension.py`, `run_qgeognn_v2_row_sequential_b32.py` | Completed formal/diagnostic [row studies](../studies/active_learning/README.md) |
| IVR | `run_qgeognn_v2_row_kernel_ivr_b32.py`, `run_ivr_mechanism_audit.py`, `report_ivr_mechanism_diagnostics.py` | Completed exploratory and mechanism studies; no automatic follow-up |
| Transfer baselines | `run_final_v2_transfer.py`, `run_cross_column_transfer.py`, `run_matched_rmse_benchmark.py` | Frozen [matched benchmark](../studies/transfer/matched_rmse_benchmark/README.md) |
| Source anchoring | `run_source_anchored_transfer.py`, `evaluate_source_anchored_transfer.py`, `summarize_source_anchored_transfer.py` | Blind fit, gated evaluation, report; [protocol](../studies/transfer/source_anchored_shared_transfer/README.md) |
| Physics control | `run_physics_column_transfer.py`, `evaluate_physics_column_transfer.py`, `summarize_physics_column_transfer.py`, `audit_column_physics.py`, `supplement_column_physics_audit.py` | Completed [physics study](../studies/transfer/physics_column_conditioned_transfer/README.md) |
| Filtered FULL-data controls | `run_filtered_full_data_benchmark.py`, `run_full_data_baseline_finalization.py`, `run_hier_cw_semantic_repair.py` | Shared record readers and corrected [structural baseline](../studies/transfer/hier_cw_semantic_repair/README.md) |
| Neural transfer | `run_traditional_transfer_recipe_pilot.py`, `run_traditional_transfer_converged_baseline_v1.py`, `run_conditioned_source_readout.py`, `run_column_conditioned_multitask.py`, `summarize_column_conditioned_multitask.py`, `run_column_conditioned_pcgrad.py` | Completed controlled studies; shared helpers and regression-test consumers remain |
| Paper reconstruction | `run_paper_transfer_reproduction_25g_40g.py`, `run_paper_transfer_gap_decomposition.py` | Historical distinct-protocol [reproduction](../studies/transfer/paper_transfer_reproduction/REPRODUCTION_REPORT.md) |

## Compatibility and reproduction dependencies

Some old runners still export functions consumed by retained workflows or tests. They stay at their original paths until those consumers and scientific hashes can be migrated together.

- Predictor controls: `studies/run_clean_4g_baseline_qualification.py`, `studies/run_point_predictor_regression_audit.py`, `studies/run_r2_pruned_requalification.py`.
- Transfer readers and frozen source contracts: `studies/run_next_transfer_diagnostics.py`, `studies/run_scaling_failure_audit.py`, `studies/run_conditional_scaling_audit_model.py`.
- Legacy study contracts: `studies/run_s1_source_target_shift.py`, `studies/run_t1_low_label_adaptation.py`, `studies/run_t1b1_adapter_capacity.py`, `studies/run_a1a_hybrid_batch_control.py`.
- Top-level Legacy dependencies: `run_e0_4g_baseline.py`, `run_e0_8g_controls.py`, `run_e0_8g_transfer.py`, `run_e1_signal_qualification.py`, `run_e2_4g_active_learning.py`, `run_e2_compound_failure_audit.py`, `run_e4_active_transfer.py`, `run_e4_a2a_formal.py`, and `run_g0_1*` through `run_g0_4*`.
- `al_engine.py`, `al_acquisition.py`, `qgeognn_graphs.py` are compatibility shims; `transfer_aware_acquisition.py` is a Legacy acquisition utility.

New scientific logic belongs in [the package](../src/qgeognn_al/README.md), not these runners.

## Retired one-off scripts

34 completed audit, smoke, qualification-report and closed-transfer scripts have been removed from the current checkout. Their reports, metrics, protocols and decisions remain at the original study paths. The complete mapping is [RETIREMENTS.json](../docs/repository/RETIREMENTS.json).

For example, `run_d45_oracle_marginal_utility.py` and `run_d46_oracle_utility_reliability.py` now have retained scientific records only. Their reusable diagnostic algorithms and tests remain.

Inspect any retired file at the recorded commit:

```bash
git show 0e7b2ee940e3f71c6dbc7400c420694855eec597:scripts/run_d45_oracle_marginal_utility.py
```

For full reproduction, use a separate checkout at that commit and restore the required data/checkpoint anchors. Do not paste an old runner into the current execution pipeline.

## Maintenance

- `audit_repository_hygiene.py`: read-only branch, retention, recovery and scientific-boundary audit.
- `audit_datasets.py`: dataset inspection; reads label values and is not a blind model preflight.
