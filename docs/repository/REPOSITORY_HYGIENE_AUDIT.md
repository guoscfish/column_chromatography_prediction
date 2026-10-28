# Repository cleanup status

Updated2026-10-28. [Current research status](../NEXT_STAGE_DECISION.md) owns decisions; the [five-seed Free / LCMD report](../../reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md) owns current curves and completion. Dated reports retain their original scope.

## Publication and consolidation

- Preserve the current code and compact experiment results on `main`: Free / LCMD continuations, dual-expert receipts, matched validation analyses, and completed transfer Phase2 / 3A / 3B results. Running experiments retain their original paths, lock ownership and frozen budgets.
- Move the current five-seed plotting implementation into `scripts/report_free_lcmd.py`. Keep the original monitor path as a compatibility entry. Offline `--from-snapshot` redraws retained curves without checkpoints; online collection verifies frozen local artifacts. `--require-complete` rejects any incomplete trajectory. The older two-seed report reader remains explicitly historical.
- Retire496 temporary presentation files (41,394,073bytes) across `docs/presentation_assets/`, `reports/ppt_202610_al/` and `reports/ppt_202610_transfer/`: PPT-specific documents, render previews, obsolete figures and duplicate source/data copies. Their original bytes are recoverable from commit`2c730ef6c6766e05ab5af39f74f913fb2936b7f2`. Each file has a hash and recovery entry in [RETIREMENTS.json](RETIREMENTS.json) and [PRUNE_MANIFEST.csv](../PRUNE_MANIFEST.csv).
- Retain the scientific B100/B200/B300/FULL transfer curve under its [study](../../studies/transfer/filtered_row_learning_curve_b200_20261013/README.md). Its reporting entry now reads canonical study summaries directly, eliminating presentation-only input copies. No training or scoring changed.
- Keep207,478,213bytes of new acquisition gradient caches local and ignored, alongside console logs, locks, runtime checkpoints and matplotlib caches. Compact contracts, selected IDs, label-access receipts, failed responses, costs and results remain versioned. Existing historical tracked scientific evidence is preserved.
- Update root/topic/script navigation and project conclusions. The old October13 snapshot, partial dual-expert trial and one-seed Row-CV pilot no longer stand in for current progress. Current numerical endpoints derive from a single generated report.

## Branches and worktrees

At publication, the only local branch is `main`, and `git ls-remote --heads origin` lists only`main`. No redundant local or origin branch remains to delete. `upstream` is unchanged. Two historical managed worktrees remain detached at`a66a753` and`8a8a917`; their directories and ignored experimental storage are retained.

The earlier retirement of37 completed one-off runners and11 superseded project documents remains recorded in the same registry. [October5 cleanup](CLEANUP_20261005.md) is a historical record, not a current branch inventory. No history rewriting or forced push is part of publication.

## Verification

- Canonical regression:1068 tests passed (165.05s). Warnings come from existing library/deprecation checks.
- Focused reporting/navigation/recovery checks:12 passed before the additional presentation retirement; final reporting/navigation/retention checks27 passed after the additional retirement.
- Snapshot-only redraw: all three retained JSON and three PNG outputs match the collected snapshot byte-for-byte. Offline rendering validates table consistency; it does not claim to re-audit absent runtime files.
- Free amendment, LCMD extension and recovery-epoch frozen source validations passed after reporting consolidation. No scientific source, prompt, protocol or retry budget was edited.
- Retirement recovery, current navigation, protected data/checkpoint anchors, staged whitespace and publication secret checks are verified before push.

Reproduce checks from the repository root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q
python scripts/audit_repository_hygiene.py
git diff --check
python scripts/report_free_lcmd.py --from-snapshot reports/free_llm_five_seeds_to1485_20261024/figures --output /tmp/free_lcmd_redraw
```

Runtime payloads remain local. A new clone contains reproducible report inputs and audit hashes, not every training checkpoint or full LLM request. Source/raw datasets, frozen partitions, scalers and authoritative checkpoint anchors are excluded from pruning.
