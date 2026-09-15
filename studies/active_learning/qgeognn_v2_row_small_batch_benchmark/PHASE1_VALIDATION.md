# Phase 1 Validation

Status: PASS. The machine-readable execution counts and hashes are in
`results/phase1_validation.json` and `results/smoke_report.json`.

## Verification

- Full repository unit/integration suite: **489 passed**, zero failures, errors
  or skips. Includes **28 active_learning_v2 tests**. Existing dependency
  deprecation and graph node-count warnings remain; they did not fail tests.
- Real hidden-label mutation: canonical 4g feature/target fixture, 160 rows,
  fixed seed 29; original and independently rebuilt mutated target stores train
  actual current-V2 models. Hidden U0/test V1/V2 values are replaced by
  deterministic extreme values. L0/validation labels, all X and IDs are retained.
  Checkpoint semantic hash, gradient hash, latent hash, K=3 prediction hash and
  every ordered acquisition-ID list/hash are identical. Provenance/source and
  fit-contract hashes differ as intended. Reusing the original cache with the
  mutated store is rejected; changing authorized L0 training truth also rejects
  a completed fit cache.
- Full small fixture smoke: preprocessing, scrubbed graph cache, baseline,
  K=3 ensemble, gradients, representations, eight B32 arms, six B16 and six B64
  arms, label-free diagnostics, same-initialization retraining, artifact freeze,
  prediction identity verification and held-out evaluation all execute.
  Held-out targets are synthetic constants installed before numeric parsing.
  No formal test metric is used. Seed 29 and the two-epoch cap are fixed
  engineering fixtures, not candidate experimental hyperparameters.
- TP CoreSet semantics, stable uncertainty ranking and quantile-width exclusion,
  all exact batch sizes, independent deterministic Random controls, dual budget,
  fixed cohorts, shared split identities and initialization hashes are tested.
- Missing seed freeze, corrupted frozen prediction, mismatched or partial cache,
  pre-Commit-A execution, and premature secondary execution are rejected.
- Decision gates are tested using synthetic metric tables, including the exact
  2% endpoint boundary and incomplete cohort rejection. No formal metrics are
  consulted to test the gate.
- Old B=333 regression: all five ordered 333-ID sequences match cached prior
  selections. Old study tracked artifacts and the frozen gradient/LCMD source
  have no diff against the base commit. Retrospective source hashes are verified
  again at sealing.

## Scope of the Result

The formal runner's production fitting/acquisition/freeze/evaluation components
are exercised end to end on a real small feature fixture; its release guards
and decision rules are separately tested. This verifies executable plumbing and
label isolation. It does not establish full-scale convergence, formal wall time,
small-batch performance or scientific superiority. No 5-seed/10-seed formal
benchmark, confirmation training, formal FINAL_REPORT or scientific decision
has been generated.

An early test run passed 487 tests before two additional release-gate tests.
The final complete suite is the 489-test run recorded by the sealing command.
The first CLI smoke also passed; final smoke was rerun after engineering guard
and audit-output changes so its code hash matches the frozen implementation.
No method, seed, batch, representation or training parameter was changed based
on smoke metrics.

## Reproduction and Seal

```bash
conda run --no-capture-output -n fish python -m pytest -q --junitxml=/tmp/qgeognn_small_batch_phase1_tests.xml
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --smoke /tmp/qgeognn_v2_small_batch_phase1_smoke_final
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --seal-phase1 --test-report /tmp/qgeognn_small_batch_phase1_tests.xml --smoke-report /tmp/qgeognn_v2_small_batch_phase1_smoke_final/smoke_report.json
```

The seal is created only after preparing the final code hashes and completing
the audit. It checks passing test/smoke records and absence of formal outputs.
All scientific choices are frozen before Commit A; after Commit A the task stops.
