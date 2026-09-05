# R2-pruned requalification

Status: `FUNCTION_PRESERVING_PARAMETER_CLEANUP_SUCCESS` / `POINT_PREDICTOR_CANDIDATE_BASELINE`.

See [comparison](R2_PRUNED_COMPARISON.md), [decision](decision.json), [protocol](protocol.json), [function gates](function_equivalence_audit.json), and the before/after reachability inventories.

Reproduce in the historical conda `fish` environment:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_r2_pruned_requalification.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/summarize_r2_pruned_requalification.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python -m pytest -q
```

The local historical R2 runtime best checkpoint is required and SHA-verified before P1. Its metadata stays tracked; runtime checkpoints remain local under the existing retention policy. Formal training refuses to proceed if any function/reachability gate fails. P0/P1 test-domain comparisons are equivalence-only engineering gates, not fitting or checkpoint selection.
