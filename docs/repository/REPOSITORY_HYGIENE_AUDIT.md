# Repository cleanup status

Updated 2026-10-11. Scientific measurements remain in study reports; the [latest validation snapshot](../../reports/active_learning_latest/REPORT.md) is the current results entrypoint.

## Current maintenance

- Integrate the existing CW branch and latest LLM/dual-expert results into `main`, with no new branch. The remote previously lacked 18 CW commits; publication includes them and the new records.
- Consolidate conflicting project status into `docs/NEXT_STAGE_DECISION.md`. Remove obsolete blocked/recovery descriptions from current navigation; retain original dated receipts.
- Update the Scientist V2, dual-expert, active-learning, script and code indexes. Batch/adaptivity is completed; V3.2 stopped at 365, and its original 525 target is not a result.
- Replace two untracked report-local plotting scripts with `scripts/report_active_learning_snapshot.py`. One entrypoint collects completed validation fits or redraws a retained snapshot without runtime.
- Untrack 132 historical runtime JSON receipts (2,000,963 bytes). Original local paths remain untouched. A verified byte-exact metadata archive, per-file SHA-256 manifest and 18 readable fit audits remain in the Scientist V2 study. See [manifest](../../studies/active_learning/qgeognn_v2_row_llm_scientist_v2/historical_runtime_manifest.json); the metadata recovery commit is `caa07e8b5317afe31934280efcab63d8e2b4ee95`.
- Ignore new console `.log` files under studies. Structured status, correction/cost receipts, measured feedback, protocols and result summaries remain versioned. Previously tracked logs are historical evidence and retain their paths.
- Preserve protected data/checkpoints, active runtime, sealed source hashes and all original scientific results. Frozen Scientist V2 lifecycle code and parameterized dual-expert lifecycle code deliberately coexist; changing their relationship during a sealed run would break provenance.

## Branches and worktrees

`main` owns integrated research and routine maintenance. The old local
`codex/cw-geometry-feedback` and `codex/llm-scientist-dual-expert-v1` refs can be
removed after integration; neither needs a new remote branch. The old dual-expert
worktree was removed after verifying that the current process runs in the canonical
checkout. Its differing legacy files were backed up outside this repository.

The V3/V3.2 worktrees remain attached to existing chats; their integrated branch
refs are kept until those checkouts are intentionally retired. No remote worktree
branch or Git history is rewritten.

## Historical cleanup and recovery

The earlier cleanup retired 34 completed one-off runners and consolidated 11
superseded plans/navigation pages. [RETIREMENTS.json](RETIREMENTS.json) records
original paths, hashes, retained results and recovery commit
`0e7b2ee940e3f71c6dbc7400c420694855eec597`.
The paper-runtime cleanup remains recoverable at
`archive/pre-matched-rmse-cleanup-2026-09-08`.
[The October 5 record](CLEANUP_20261005.md) describes that earlier integration;
its draft dual-expert status and worktree inventory are historical.

## Verification

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python -m pytest -q
python3 scripts/audit_repository_hygiene.py
git diff --check
```

The first targeted check found the 132 tracked runtime files despite the ignore
rule; consolidation fixes the repository contract while preserving their exact
metadata. Snapshot publication checks freeze/checkpoint hashes, conflicting
same-budget metrics, finite validation values and zero test labels used for fitting.

Validation on 2026-10-11: the full regression run passed 1053 tests and found one
pre-existing navigation-parser false positive on fenced SMILES. After fixing the
parser (without changing the frozen report), all 51 affected/navigation/hygiene
and snapshot tests passed. The snapshot verified 559 source hashes and reproduced
its JSON/report/plots without reading local runtime. Retirement violations are
zero, all 17 protected anchors exist, and tracked runtime/progress paths are zero.
