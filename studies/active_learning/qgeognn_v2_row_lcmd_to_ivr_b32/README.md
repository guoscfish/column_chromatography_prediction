# LCMD-L653 to IVR continuation

Status: sealed, engineering checks passed, formal execution not started.

See [PROTOCOL.md](PROTOCOL.md) for the critical design review and frozen methods,
and [HANDOFF.md](HANDOFF.md) for provenance, validation, costs and exact commands.

This is an exploratory fixed-switch intervention on a previously evaluated
five-seed row cohort. It reuses the exact frozen LCMD prefix through 653 labels,
then applies conditional batch IVR on the current gradient kernel through 1005.
It does not replay training or acquisition from 333.

Source artifacts are read-only. All five checkpoints and gradient banks passed
reuse checks. Small tracked lineage records include exact source L/U IDs,
hashes and compatibility evidence; large future outputs belong in runtime/.
The source runtime checkpoints and gradients must remain available locally:
a repository clone alone does not supply these gitignored dependencies.

Preparation used 55 passing tests and five deterministic selection-only checks.
There are no new test metrics or formal continuation fits in this study yet.
The reporting implementation is validated with synthetic curves only.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/active_learning_v2/test_lcmd_to_ivr.py tests/active_learning_v2/test_ivr.py tests/active_learning_v2/test_ivr_study.py tests/active_learning_v2/test_sequential_b32.py --junitxml=/tmp/lcmd_to_ivr_preflight.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --prepare /tmp/lcmd_to_ivr_preflight.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --selection-smoke
```

The formal --run-all and --execute-seed actions require subsequent authorization.
Neither performs automatic test reporting. Explicit --report first verifies
all five complete continuations before any new test-truth access.
