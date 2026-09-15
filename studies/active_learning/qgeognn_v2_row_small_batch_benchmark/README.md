# QGeoGNN-V2 Realistic Small-Batch Row Benchmark

Status: **PHASE 1 / PREREGISTERED / FORMAL EXPERIMENT NOT RUN**.

Commit A contains implementation, retrospective label-free audits, tests and
preregistration only. Stop after Commit A. Formal five-seed/ten-seed training,
formal test performance, FINAL_REPORT conclusions and Commit B are outside this
task. No formal decision or sequential-learning recommendation is available.

- [PREREGISTRATION.md](PREREGISTRATION.md): hypotheses, frozen cohorts, methods,
  primary/secondary endpoints and decision gates.
- [PROTOCOL.md](PROTOCOL.md): reproducible execution, label access, cache contracts
  and budget accounting.
- [AUDIT_INTERPRETATION.md](AUDIT_INTERPRETATION.md): retrospective **old B=333**
  input/gradient redundancy and label-free mechanism analysis.
- [PHASE1_VALIDATION.md](PHASE1_VALIDATION.md): tests and isolated smoke evidence.
- `protocol.json`, `splits/`, `artifact_manifest.json`: machine-readable freeze.
- `results/`: Phase 1 audits and budget accounting only.

The earlier `../qgeognn_v2_row_lcmd/` STRONG_POSITIVE study remains unchanged. It
is a **large-batch one-step mechanism proof**, not a realistic closed-loop study.
Its previously reported 31.69% NRMSE reduction is historical context supplied in
the handoff; its formal test performance was not re-read for this phase.

The new executable is
`scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py`. It is a thin CLI;
the implementation lives in `src/qgeognn_al/active_learning_v2/`.

Formal execution writes separate `formal_results/`, `secondary_results/`,
`formal_decision.json` and freeze records. This preserves the committed Phase 1
audit files and manifest. Such outputs are absent from Commit A. A later,
explicitly authorized Phase 2 must interpret the results, write FINAL_REPORT,
and create Commit B without changing this preregistration.
