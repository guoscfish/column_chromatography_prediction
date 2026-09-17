# Center/Width Gradient-LCMD performance

Completed: CENTER_WIDTH_NO_GAIN, 2/5 wins. See FINAL_REPORT.md, paired_comparison.csv
and per_seed_metrics.csv. All 30 scoped tests and five actual-seed preflights passed.

Five development seeds, one B32 acquisition, five scratch evaluation fits.
See PROTOCOL.md for the frozen scientific definition and FINAL_REPORT.md for
the eventual result. Source benchmark and innovation runtimes are read-only.

Use the existing fish environment from this worktree:

```bash
python scripts/studies/run_qgeognn_v2_4g_cw_performance.py --preflight --artifact-repository /Users/fish/Documents/GitHub/column_chromatography_prediction --innovation-repository /Users/fish/Documents/GitHub/column_chromatography_prediction_al_innovation
python scripts/studies/run_qgeognn_v2_4g_cw_performance.py --execute --artifact-repository /Users/fish/Documents/GitHub/column_chromatography_prediction --innovation-repository /Users/fish/Documents/GitHub/column_chromatography_prediction_al_innovation
```

Execution requires passing tests recorded in tests.json and a sealed preflight.
Completed fits resume only under identical contracts. No seed/method/budget/
training overrides are exposed. Formal sequential studies remain unchanged.

The checked-in tests.json binds the successful JUnit test_results.xml and exact
tested code hashes. The executed test scope was:

```bash
python -m pytest -q tests/active_learning_v2/test_center_width_performance.py tests/active_learning_v2/test_innovation_screen.py tests/active_learning_v2/test_no_label_leakage.py tests/active_learning_v2/test_lcmd.py tests/active_learning_v2/test_gradient_sketch.py tests/active_learning_v2/test_row_protocol.py
```

acquisition_freeze.json binds the protocol, code and five selected batches.
pre_test_freeze.json binds all five fitted checkpoints and predictions before
test access. preflight_audit.json includes read-only source hashes and exact
historical selection regression evidence. execution_audit.csv preserves the
CW fit receipts outside gitignored runtime. final_validation.json records the
post-run consistency checks. Full-data gap values are intentionally absent.
