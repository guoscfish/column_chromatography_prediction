# Column-conditioned multi-task QGeoGNN

Preregistered comparison of A1 `MULTITASK_SEPARATE_HEADS` and A2
`MULTITASK_COLUMN_FILM`, jointly supervised by qualified 4g source-train,
25g gradient-train, and 40g gradient-train rows. See `PROTOCOL.md`,
`protocol.json`, and `IMPLEMENTATION_AUDIT.md` before execution.

Run from the repository root in the existing `fish` conda environment:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/test_column_conditioned_multitask.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_column_conditioned_multitask.py --action execute
```

The execute action verifies the source, completes the 50 COMPOUND inner fits,
and computes the locked continuation decision. A failed gate terminates before
outer predictions. A passed gate permits fixed-epoch COMPOUND refits and freezes
all predictions; scoring is always a separate action:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_column_conditioned_multitask.py --action score --protocol compound
```

After COMPOUND scoring, use the same `inner`, `aggregate`, `final`, `freeze`,
and `score` actions with `--protocol row` for the secondary confirmation.
Individual inner/final contexts take `--seed` from the five frozen outer seeds.
Completed contexts are reused only after their artifacts pass SHA256 checks.
An interrupted incomplete fit is restarted at its original deterministic seed;
completed fits are preserved. Runtime checkpoints are ignored by Git; aggregate
scientific evidence and split identities are retained in the study directory.

After a terminal negative decision, or after both eligible outer confirmations,
independently audit the inner evidence and generate the final report:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_column_conditioned_multitask_evidence.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/summarize_column_conditioned_multitask.py
```

The audit recomputes metrics from frozen inner predictions and authorized inner
labels, checks global molecule separation, and verifies unchanged early source
parameters and BN buffers. The report generator reads only saved evidence.
Both scripts reject incomplete or changed completion records.

The macOS OpenMP setting follows the existing project runtime convention.
Use CPU and one Torch thread as frozen in the protocol.
