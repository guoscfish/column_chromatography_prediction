# A1/A2 column-conditioned Row-aligned v1

2026-10-13 Phase 1 only: fixed outer seed 769539383, five inner Row folds, A1/A2 (ten actual fits). Historical Group-CV artifacts and outer identities are preserved. No other seeds, FULL-data refit or outer scoring are authorized or implemented here.

- [PROTOCOL.md](PROTOCOL.md): design review, fixed training recipe and label boundaries.
- [REPORT.md](REPORT.md): actual pilot results and continuation recommendation.
- [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md): tests, pairing, steps, BN, immutable historical evidence and checkpoint replay.
- ROW_SPLIT_MANIFEST.csv: actual pilot fold identities; OUTER_ROW_IDENTITIES.csv: unchanged five-seed outer identity authority.
- INNER_RESULTS.csv, PAIRED_COMPARISONS.csv, TRAINING_HISTORY.csv, EPOCH_SELECTION.csv: primary results and epoch selection, without refit.
- INPUT_*AUDIT.csv, CROSS_COLUMN_INPUT_AUDIT.csv: duplicate identities versus matching model inputs.
- GROUP_VS_ROW_*CSV files: historical versus new within-protocol paired gains; absolute errors cannot rank the CV protocols.
- a1_a2_row_cv_learning_curves.png and group_vs_row_cv_selection.png: development diagnostics.

Run from the repository root in the validated `fish` environment:

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python -m pytest -q studies/transfer/column_conditioned_row_aligned_v1/test_protocol.py
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/transfer/column_conditioned_row_aligned_v1/run_phase1.py
```

Completed runs are reused only after protocol, execution provenance and artifact hashes match. Incomplete runs restart from the deterministic seed; there is no optimizer-state continuation. Three isolated CPU workers use one thread each, with fixed RNG seeds and paired per-epoch schedules.

Reporting-only regeneration:

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/transfer/column_conditioned_row_aligned_v1/summarize_phase1.py
```

This includes a feature-only, zero-label replay of all saved target OOF predictions. Checkpoints and raw per-fit diagnostics are in ignored runtime/. Aggregate results, histories, plots, protocol and audit receipts are retained in this directory. No efficacy claim follows from one seed or from correlated CV folds.
