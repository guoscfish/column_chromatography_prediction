# Controlled shared-backbone PCGrad study

One-candidate mechanism test of deterministic PCGrad applied only to the
shared late-backbone gradient scope diagnosed in completed A2 column FiLM.
Read `IMPLEMENTATION_AUDIT.md`, `PROTOCOL.md`, and `protocol.json` before
execution.

From the repository root in the `fish` environment:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/test_column_conditioned_multitask.py tests/test_column_conditioned_pcgrad.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_column_conditioned_pcgrad.py --action execute
```

The execution is resumable by seed/fold and stops automatically on either
inner gate. Completed fits are accepted only when protocol, implementation,
environment, and artifact hashes match. For parallel recovery, use `--action
inner --protocol {compound,row} --seed SEED`, then `--action aggregate`.

If both gates pass, `execute` creates all blind COMPOUND and ROW predictions
and one global freeze manifest, but never scores them. Score separately:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_column_conditioned_pcgrad.py --action score
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/summarize_column_conditioned_pcgrad.py
```

If a gate fails, run only the summarizer. It writes the terminal negative
report without fabricating outer artifacts. Runtime checkpoints are ignored by
Git; aggregate evidence, identities, hashes, decisions, and the report are
retained.
