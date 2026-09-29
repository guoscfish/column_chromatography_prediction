# Repository cleanup — 2026-09-29

Fetched origin and upstream with prune before inspection. New development base: `8a8a917008f8f2e0e718491a68f93c6d9336aa41`.

## Ancestry and preservation

| Origin branch | Commit | Contained in V3.1 |
|---|---|---|
| main | d9b4518fde8745d7f3819a7607ec03052722919f | True |
| codex/llm-al-full-pool-feedback | 647271df406cb30f606fd58de97b182b87fe2fd5 | True |
| codex/llm-scientist-v2 | 04869e5dfcea0f362121240ac3f602e15c22a549 | True |
| codex/llm-scientist-v3-context | 8a8a917008f8f2e0e718491a68f93c6d9336aa41 | True |
| exp/qgeognn-v2-4g-cw-lcmd-performance | 6ac0f9e4cd8abe7c40b27cc56041b9db685df43a | False |

V1 → V2 → V3.1 is linear. CW performance diverges and owns one unique commit, 23 files / 1502 added lines, including runner, implementation, test, FINAL_REPORT, protocol, acquisition freeze, per-seed/aggregate metrics, selected batches, and figure. Its entire tree is retained by the archive tag; it is not merged into V3.2.

## Verified remote archive tags

- `archive/llm-full-pool-v1-2026-09-29` → `647271df406cb30f606fd58de97b182b87fe2fd5` (pushed and verified with ls-remote peeled tag).
- `archive/scientist-v2-2026-09-29` → `04869e5dfcea0f362121240ac3f602e15c22a549` (pushed and verified with ls-remote peeled tag).
- `archive/scientist-v3.1-pilot-2026-09-29` → `8a8a917008f8f2e0e718491a68f93c6d9336aa41` (pushed and verified with ls-remote peeled tag).
- `archive/cw-lcmd-performance-2026-09-29` → `6ac0f9e4cd8abe7c40b27cc56041b9db685df43a` (pushed and verified with ls-remote peeled tag).

Remote legacy branches will be removed only after the new branch is pushed and the archive targets are reverified. Local branches/worktrees belonging to other chats remain intact. Upstream/main is read-only and outside origin cleanup scope.

## Conservative layout policy

No old entrypoint met the safe deletion criterion: V1/V2 runners remain historical reproduction entrypoints or are used by tests/infrastructure. V3 depends on V2 stable training, split, gradient, and selector primitives. No model, split, trainer or gradient extractor is refactored. See scripts/studies/README.md. Existing V3.1 artifacts remain byte-for-byte preserved; new runtime exclusions do not untrack existing files.

## Hybrid audit (no execution)

V3 predictor.build_catalog calls CW-LCMD for 16 positions before building pending rows; stage passes selection_count=16; pending rows are excluded from legal candidates; freeze_batch concatenates pending IDs and 16 unique legal LLM IDs and requires exactly 32. Round-zero initialization inherits the frozen V2 CW16 catalog and verifies its clean L333/U0 provenance.

Next V3.2 hybrid should expose preselected_cw_experiments / fixed_batch_prefix and retain hard exclusion from selectable candidates. Prompt: “These 16 CW-selected experiments are already guaranteed members of the batch. Choose 16 additional experiments that complement them using chemistry, observed feedback, unresolved failures, hypothesis tests, and coverage gaps. Do not simply imitate CW.” This request registers and runs only Free-LLM32, seed157, round0; hybrid is not registered for execution.
