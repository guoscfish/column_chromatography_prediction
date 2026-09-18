# Current study entry points

- `studies/run_qgeognn_v2_row_kernel_ivr_b32.py`: experiment
  `qgeognn_v2_row_kernel_ivr_b32`; diagnostic (exploratory matched extension),
  tests one conditional Kernel-IVR strategy against frozen Random and LCMD.
  May run directly after `--prepare --test-report PATH` seals the passed preflight;
  `--run-all` executes/resumes the fixed five-seed matrix and gated final reporting.
  See [the protocol](../studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/PROTOCOL.md).

Use `studies/run_final_v2_engineering.py`, `studies/run_final_4g_qualification.py`, `studies/summarize_final_4g_qualification.py` and `studies/run_final_v2_transfer.py` for the final standalone workflow. Read their study preregistrations before execution. Legacy, Clean and diagnostic runners below are historical reproduction tools.

The matched representation-transfer study uses
`studies/run_source_anchored_transfer.py` for blind, resumable fitting,
`studies/evaluate_source_anchored_transfer.py` for globally gated test evaluation,
and `studies/summarize_source_anchored_transfer.py` for scientific tables and
figures. Its [frozen study protocol](../studies/transfer/source_anchored_shared_transfer/README.md)
controls execution; these entry points do not authorize follow-up model tuning
or Active Learning.

# Scripts

Current R2-pruned study: `studies/run_r2_pruned_requalification.py` runs reachability/equivalence gates before the controlled retrain (`--gates-only` checks without retraining); `studies/summarize_r2_pruned_requalification.py` compares completed artifacts with R2. See [the study](../studies/predictor/r2_pruned_requalification/README.md).

The current matched absolute-error study has one thin entry point:
`studies/run_matched_rmse_benchmark.py`. Use `--prepare` to validate and freeze
the protocol, `--execute` only for the missing paper-style current-V2 fits, and
`--summarize` to rebuild reports solely from globally frozen predictions. Its
[study README](../studies/transfer/matched_rmse_benchmark/README.md) is the
authority for the no-test-tuning boundary; reusable calibration, adaptation,
evaluation and protocol APIs live in `src/qgeognn_al/transfer/`.

Reusable scientific code now lives in `src/qgeognn_al/`. `al_engine.py`, `al_acquisition.py`, and `qgeognn_graphs.py` are compatibility shims for historical imports.

Historical top-level `run_*.py` files are **historical / reproduction only**:

- E0/G0/D04: `run_e0_4g_baseline.py`, `run_e0_8g_controls.py`, `run_e0_8g_transfer.py`, `run_g0_1_quantile_monotonicity.py`, `run_g0_2_interval_calibration.py`, `run_g0_3_threshold_sensitivity.py`, `run_g0_4_paper_style_transfer.py`, `run_d04_conformer_selection.py`.
- D28/E1/E2: `run_d28_engineering_checks.py`, `run_e1_signal_qualification.py`, `run_e2_4g_active_learning.py`, `run_e2_compound_failure_audit.py`, `run_e2_random_smoke.py`.
- E4 family: `run_e4_active_transfer.py`, `run_e4_a2a_engineering_smoke.py`, `run_e4_a2a_formal.py`, `run_e4_a2a_low_budget.py`.
- Post-hoc diagnostics: `run_d42_e4_headroom_audit.py`, `run_d43_transfer_aware_qualification.py`, `run_d44_active_learning_suitability.py`, `run_d45_oracle_marginal_utility.py`, `run_d46_oracle_utility_reliability.py`.

No historical top-level runner authorizes a new experiment. Historical reproductions use a new output directory under `experiments/reproductions/`; runtime/checkpoints/history/progress are gitignored.

Current config-driven study-family runners live under `scripts/studies/`:

- `run_s1_source_target_shift.py`
- `run_a1a_hybrid_batch_control.py`
- `run_t1_low_label_adaptation.py`
- `run_t1b1_adapter_capacity.py`

These are current study infrastructure, not historical reproduction-only scripts. Their own frozen configs and authorization gates control which actions may run. T1a and T1b-1 have completed separately authorized formal runs. T1b-1 produced 180/180 Adapter fits and the retained compact result artifacts; its runtime checkpoints, histories, and prediction files remain gitignored.

Future work should use `scripts/run_experiment.py` or a small protocol-family runner with config/spec differences for partitions, budgets, transfer strategies, and acquisitions. Add a new runner only when the scientific protocol family changes; new code imports `src.qgeognn_al.*`, never another `run_*.py` for scientific core.
