# Script map

## Core reusable

- `al_engine.py` — current reusable fit/predict, stable-ID state, save/resume, and source-initialization layer.
- `al_acquisition.py` — current reusable deterministic acquisition and representation utilities.
- `qgeognn_graphs.py` — current reusable molecular graph construction primitives.
- `audit_datasets.py` — repository-wide dataset integrity audit.

## Historical Gate0

- `run_e0_4g_baseline.py` — frozen 4g source baseline and canonical-data builder.
- `run_e0_8g_transfer.py` — original 4g→8g transfer baseline.
- `run_e0_8g_controls.py` — paired transfer controls and shared 8g graph-cache loader.
- `run_d04_conformer_selection.py` — historical conformer-policy diagnostic.
- `run_d28_engineering_checks.py` — active-learning engine engineering checks and partition generation.
- `run_g0_1_quantile_monotonicity.py` — monotonic quantile-head conversion/control.
- `run_g0_2_interval_calibration.py` — held-out interval calibration study.
- `run_g0_3_threshold_sensitivity.py` — authoritative 574-row no-threshold target dataset and sensitivity study.
- `run_g0_4_paper_style_transfer.py` — paper-style transfer comparison.

## E1

- `run_e1_signal_qualification.py` — source-member creation and acquisition-signal qualification.

## E2

- `run_e2_4g_active_learning.py` — formal source-free row/compound E2 runner.
- `run_e2_random_smoke.py` — bounded historical E2 Random smoke.

## Diagnostics

- `audit_e2_row_mechanisms.py` — post-hoc E2 row mechanism audit.
- `run_e2_compound_failure_audit.py` — corrected D38 post-hoc compound failure audit and compact query-history builder.

## E4 current

- `e4_preregistration_preflight.py` — partition and real-loading-path source compatibility audit.
- `run_e4_active_transfer.py` — E4 source-reinitialized active-transfer runner; currently used only for the bounded Protocol A engineering smoke.
- `run_e4_a2a_low_budget.py` — frozen E4-A2a nested partition generator and exact audit (42/22/8/486/58; seeds 42/525/1101).
- `run_e4_a2a_engineering_smoke.py` — bounded A2a engineering smoke; only Protocol A/seed 42 is accepted and formal training is not an available action.
