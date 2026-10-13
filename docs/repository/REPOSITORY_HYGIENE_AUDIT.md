# Repository cleanup status

Updated 2026-10-13. Scientific measurements remain in study reports; the [latest validation snapshot](../../reports/active_learning_latest/REPORT.md) is the current results entrypoint.

## Current maintenance

- Publish the October13 LLM/dual-expert continuations, A1/A2 evidence audit, Row-CV inner pilot, filtered Row FULL and B100/B200/B300 curves, and existing presentation figures on `main`. No new training, LLM request or experiment scoring is part of this cleanup.
- Refresh the canonical validation snapshot from completed prediction freezes. Defaults select the October13 Scientist continuation and the completed621-label dual-expert revision. Report endpoints and common-budget tables derive from retained data; obsolete493/525/653 summaries no longer overwrite current progress on regeneration.
- Retire three report-local plotting scripts: the two nearly identical model-switch copies and the four-arm progress plot. Preserve all original figures and numeric evidence. Each new entry in [RETIREMENTS.json](RETIREMENTS.json) has its own recovery commit and SHA-256; current plots use `scripts/report_active_learning_snapshot.py`.
- Ignore approximately204 MB of newly created full LLM requests and thirteen lock files. They remain at their original local paths for resume/provenance; compact protocols, request hashes, selected IDs, measured feedback, answer/cost receipts and scientific results stay versioned. Existing tracked request evidence is untouched.
- Preserve frozen scientific implementations and Phase1 archived sources, which have live imports or byte-hash consumers. This includes the similar historical and parameterized Scientist lifecycle modules and the B200/full-data runners. Protected datasets, checkpoints and training caches retain their paths.

## Branches and worktrees

`codex/llm-scientist-v3-2` (`a66a753`) and `codex/llm-scientist-v3-context` (`8a8a917`) have zero commits outside `main`. Both checkouts were clean and had no active repository process at inspection. Their local branch refs were removed; publication also removes the two matching merged origin refs. The original worktrees remain at their existing commits with detached HEAD, preserving chat workspaces and ignored runtime. `main` remains the integrated local/remote branch; `upstream` is unchanged.

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

The prior October11 regression passed1053 tests with one pre-existing navigation-parser false positive, then all51 affected/navigation/hygiene/snapshot tests passed after the parser repair. October13 verification is recorded after the cleanup below. Historical stopped/failed/paused receipts remain unchanged; current completion is derived from frozen fit audits, never from planned label budgets.

A separately authorized Row-CV Phase2 process started during cleanup; its newly generated files remain uncommitted for the owning task. Published Phase1 reports remain frozen and do not imply that Phase2 is complete.

Default pytest discovery is limited to `tests/`. Frozen study archives contain duplicate test basenames; collecting those archived copies as current tests caused an import mismatch. Study-specific synthetic audits remain executable explicitly, without altering archived source bytes.

Validation on2026-10-13: full canonical regression1065 passed (484.80s); A1/A2 label-free and Row-CV Phase1 synthetic audits22 passed. Verified615 snapshot/result source hashes and21 frozen transfer implementation hashes; no mismatch. Snapshot-only redraw reproduced all six JSON/report/PNG/SVG files byte-for-byte. Retirement recovery has zero violations; all17 protected anchors exist; tracked runtime/progress paths are zero. `git diff --check` passed.
