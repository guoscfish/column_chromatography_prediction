# Scripts

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

These are current study infrastructure, not historical reproduction-only scripts. Their own frozen configs and authorization gates control which actions may run. T1a has completed its separately authorized formal run; future studies still require their own explicit authorization.

Future work should use `scripts/run_experiment.py` or a small protocol-family runner with config/spec differences for partitions, budgets, transfer strategies, and acquisitions. Add a new runner only when the scientific protocol family changes; new code imports `src.qgeognn_al.*`, never another `run_*.py` for scientific core.
