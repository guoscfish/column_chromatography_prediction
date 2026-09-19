# Repository cleanup status

Updated 2026-09-18. This page records repository maintenance; scientific results remain in their study reports.

## Completed file cleanup

- Retired 34 completed one-off runners with no remaining import consumers. The conditional-scaling runner remains because a frozen protocol still verifies its exact source hash.
- Consolidated 11 superseded project plans and duplicate navigation/policy pages.
- Rewrote the project and topic indexes around qualified V2, current 4g active learning, and completed transfer evidence.
- Preserved all scientific result trees, raw/canonical data, splits, protected anchors, source checkpoints and active runtime.
- Preserved the ongoing uncommitted batch-adaptivity/efficiency work and its code paths.

[RETIREMENTS.json](RETIREMENTS.json) is the exact file ledger: original path, SHA-256, size, line count, retained evidence and recovery commit. The source snapshot is `0e7b2ee940e3f71c6dbc7400c420694855eec597`. For reproduction, use the complete code at that revision and its recorded environment, not an isolated old script in a new pipeline.

The earlier paper-runtime cleanup remains recoverable at tag `archive/pre-matched-rmse-cleanup-2026-09-08`. Its 20-run scalar summary and compact artifact manifest remain in `studies/transfer/paper_transfer_reproduction/`.

## Branch disposition

| Branch | Disposition |
| --- | --- |
| `main` | Keep as integrated baseline |
| `codex/4g-evaluation-adaptivity` | Keep: current checkout with ongoing uncommitted work |
| `exp/qgeognn-v2-4g-row-al` | Deleted local duplicate of `0e7b2ee`; remote retains the published AL snapshot until integration |
| `research/matched-rmse-benchmark-cleanup` | Deleted local and remote refs at `ae4b6d4`, already contained in `origin/main` (0 unique commits; main was 31 commits ahead) |
| `exp/qgeognn-v2-4g-al-innovation-screen` | Keep: separate worktree and research commits outside current ancestry |
| `exp/qgeognn-v2-4g-cw-lcmd-performance` | Keep: separate worktree with additional CW evidence |

No history rewrite or forced merge is needed. The independent experiment branches should be integrated or archived through a separate evidence review, not deleted as duplicates.

## Checks

```bash
python3 scripts/audit_repository_hygiene.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
git diff --check
```

The dependency audit covers import consumers and literal code references, including uncommitted Python files. Retirement checks verify Git recovery hashes and retained records. Scientific-boundary checks verify the frozen source, schedule, data and existing result manifests.

Validation on 2026-09-18: the full suite ran 560 tests (557 initially passed). One retired-script link, a duplicate index header and a hash-dependent runner were corrected; all 40 relevant follow-up tests passed. Retirement recovery, 367 active-experiment sealed files, protected anchors and `git diff --check` passed. Remote inspection confirmed the merged cleanup ref was removed while main and the published AL snapshot remained.

Large ignored runtime is intentionally retained: the final source checkpoint and current AL study depend on it. This cleanup reduces maintained code and conflicting documentation; it does not claim to reclaim historical Git objects or all local training storage.
