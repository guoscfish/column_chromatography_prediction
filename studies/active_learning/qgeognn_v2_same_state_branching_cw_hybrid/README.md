# Same-state CW / IVR / MaxDet pilot

This development pilot compares two-round acquisition returns at eight exact historical CW/Hybrid source states. Source trajectory `hybrid` remains a source label; it is not a primary branch strategy. Primary strategies are exact `center_width_lcmd`, `kernel_ivr`, and `gradient_maxdet`.

Seeds 157/6101 × two source trajectories × budgets 429/653 × three strategies × two batches = **24 branches, 48 planned fits**. The current task stops after selector audit; no pilot model has been trained and no test truth has been revealed.

Read `IMPLEMENTATION_AUDIT.md` for the scientific corrections, ordered-ID regression evidence, Hybrid artifact inventory, and cost estimate. `PROTOCOL.md` defines fixed L0 transforms and the test firewall. `decision.json` is the current readiness record. The old seal is invalid and archived under `runtime/deprecated_pre_exact_audit/`.

Selection-only maintenance commands (repository root; existing fish environment):

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --selector-audit
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python -m pytest -q tests/active_learning_v2/test_same_state_branching_cw_hybrid.py tests/active_learning_v2/test_ivr.py tests/active_learning_v2/test_maxdet.py tests/active_learning_v2/test_cw_lcmd_extension.py --junitxml=/tmp/cw_exact_tests.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --prepare /tmp/cw_exact_tests.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --validate
```

`--prepare` requires exact-selector evidence and dedicated tests, seals the current inputs, and runs selection-only smoke. It never trains. After a seal exists, code/protocol edits deliberately invalidate it; they require a reviewed new audit/seal, not an automatic override. The OpenMP environment flag follows the existing local repository environment; no dependency was changed by this audit.

The next command **only after separate authorization to train** is:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --execute-seed 157
```

Then require `--stage-check 157` before seed 6101. Global freeze and report are separate later actions. No performance, headroom, or controller recommendation can be inferred from selection overlap alone.
