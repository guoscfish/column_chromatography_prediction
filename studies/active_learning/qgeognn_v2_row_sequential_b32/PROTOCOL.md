# Execution Protocol

## Phase boundary

Commit A consists of implementation, tests, preregistration, protocol, copied
split identities, and preflight evidence. It must stop before formal model
training. `runtime/` must not exist when Commit A is sealed.

Formal execution fits rounds 0 through 21 and reacquires after rounds 0 through
20. Every round stores ordered labeled/unlabeled state, incoming selected batch,
input contract, evaluation checkpoint, test predictions, fit audit, acquisition
artifacts, and a pre-test contract. Each contract binds seed, method, round,
ordered L_t/U_t hashes, initialization, training configuration, fixed endpoint
scales, acquisition configuration, code, checkpoint, and predictions.

## Sequential state transition

At round t, train from scratch on L_t with shared validation, create test
predictions without reading test truth, recompute the method's acquisition from
the current model and current U_t, select exactly 32 rows, freeze their IDs,
reveal only those labels, append them to L_t, remove them from U_t, and continue.
The implementation rejects duplicates, non-U_t selections, reordering drift,
partial batches, and any state change other than +32/-32.

Random uses its single original-U0 permutation. Hybrid recomputes K=3 current
U_t q50 disagreement and current L_t+U_t representations; member 0 is the
evaluation checkpoint. LCMD recomputes current L_t+U_t full-network q50 gradient
features. Hybrid and LCMD center counts must equal the current active-label
count, not 333.

## Label barrier

Canonical features are loaded with X-only columns. `RestrictedLabelStore`
authorizes L0 and validation for initial fitting and only frozen selected U0
IDs thereafter. Graphs used for prediction/acquisition retain zero-sentinel
labels. Test truth cannot be requested until acquisitions and predictions for
all 5 seeds x 3 methods x 22 curve points, plus all five full-data references,
are frozen in `global_pre_test_freeze.json`.

Only `--reveal-test-and-report` crosses that barrier. It verifies every global
entry and prediction identity first, reveals each seed's test labels, calculates
all metrics in one pass, writes the preregistered tables/figures, and applies the
frozen decision. No partial matrix can produce a decision.

## Resume and sharding

Completed graph, fit, feature, acquisition, round, and trajectory caches are
reused only when their full contracts and protected file hashes match. Partial
or mismatched caches raise an error. A resumed Hybrid trajectory restores the
member-1/member-2 fit audit rows so its audit is identical to uninterrupted
execution. Round wall time is provenance and is excluded from semantic resume
comparison.

Runs may be interrupted and resumed. Seeds can run independently in separate
processes because their runtime directories do not overlap. Do not run multiple
processes for different methods of the same seed.

## Commit A commands

Run from the repository root in the `fish` environment:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --prepare
conda run --no-capture-output -n fish python -m pytest -q --junitxml=/tmp/qgeognn_sequential_commit_a.xml
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --validate-preflight
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --seal-commit-a --test-report /tmp/qgeognn_sequential_commit_a.xml
```

Commit with the exact registered subject and stop. Preparation and validation
never construct `SequentialSeedContext` and never train a model.

## Formal commands after explicit authorization

Replace `COMMIT_A_SHA` with the resulting commit. A single process can run the
whole pre-test phase:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --execute-pre-test --preregistration-commit COMMIT_A_SHA
```

Alternatively, schedule the five seed commands independently:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --execute-seed 157 --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --execute-seed 887 --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --execute-seed 2357 --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --execute-seed 6101 --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --execute-seed 12203 --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --finalize-pre-test --preregistration-commit COMMIT_A_SHA
```

After the global freeze exists and has been independently checked:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_sequential_b32.py --reveal-test-and-report --preregistration-commit COMMIT_A_SHA
```

There is no CLI override for methods, B, rounds, ensemble size, training epochs,
patience, targets, decision thresholds, or test timing.

## Formal outputs

`results/` receives learning-curve, cohort-mean, normalized-AULC, labels-to-
target, label-saving, full-reference, endpoint, runtime, trajectory-integrity,
label-access, and initialization audit tables. `figures/` receives both NRMSE
curves, AULC, labels-to-target, saving, and endpoint-RMSE plots. `FINAL_REPORT.md`
and `decision.json` remain pending until the complete post-freeze evaluation.
