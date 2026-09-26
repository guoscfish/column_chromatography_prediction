# Full-pool response-feedback LLM: audited dialog variant

**Status: COMPLETE.** Both LLM trajectories reached L525, the unified test evaluation and post-evaluation diagnostics are written, and the pre-registered gate was not met. See [FINAL_REPORT.md](FINAL_REPORT.md).

See [PROTOCOL.md](PROTOCOL.md) and machine-readable `protocol.json` for the frozen design. This study uses existing Codex account authentication, with a separate projectless selector task per seed and a JSON-only catalog relay. It does not require an API key. This is an audited no-tool workflow, not a hard tool sandbox.

## Local execution

Use the existing `fish` environment, with `KMP_DUPLICATE_LIB_OK=TRUE` for this host's OpenMP runtime. From the repository root:

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_dialog.py validate
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_dialog.py execute --seed 157
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_dialog.py execute --seed 6101
```

`execute` resumes existing fits, selection freezes and label feedback. Its `fit_start` log is printed before the fit-cache check; use `fit_audit.json` to distinguish a reused fit from new training. Never run two `execute` processes for the same seed concurrently.

When stopped at `AWAITING_DIALOG`, relay the exact `dialog_prompt.txt` into the task bound to that seed. Before sending a subsequent round, run `bind --seed SEED --round ROUND --thread-id TASK_ID`. First-round binding uses `--first` after task creation. Run `process --seed SEED --round ROUND` after the answer completes. A `QUERY_RESULT` names the exact reply file to send, once. `SELECTION_ACCEPTED` permits resuming `execute`. Never manually repair choices or introduce a random fallback.

Formal selector tasks:

| seed | task ID |
|---|---|
| 157 | `01a0db84-bcec-7e71-943d-bd336cf2abba` |
| 6101 | `01a0db84-ca8a-7f20-9879-8dfaf658fa32` |

Seed 157 round 05 was recovered in independent task `01a0dcd5-66ac-72c2-9a78-f290f08d29a6` after the original task returned a 401 before producing a final answer. The recovery packet included the complete round 00–04 history; no labels were revealed during the handoff. See `round_05/dialog_rebind.json` and `execution_boundary_audit.json`.

Validation failures leave the batch unfrozen and unobserved. Archive the rejected JSON, return only technical schema/legality errors to the same task, and count the correction against the original 16-answer budget. Repeating already viewed legal IDs for schema repair adds no new candidate information. Keep these messages alongside the round's artifacts. A native tool call or model change invalidates the task and must not be treated as a format repair.

## Unified evaluation and diagnostics

Only after both six-round LLM trajectories complete:

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_dialog.py report
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/summarize_llm_dialog.py
```

The reporting command verifies the complete prediction grid before opening test truth. `summarize_llm_dialog.py` is a reporting-only addition and never participates in selection. It separates CW/random/LLM coverage and enumerates single-recorded-condition contrasts against previously measured records.

## Artifacts

- `selections/seed_*/cw16_llm16_full_pool/round_*/`: input, full feature-only candidate catalog, original measured records, actual query requests/results, model output/audit, acceptance, 32-ID freeze, reveal receipt and feedback.
- `runtime/`: local training checkpoints, fit audits and prediction freezes; ignored in Git under repository convention.
- `results/`: metrics, AULC, per-source coverage, matched condition contrasts, token/call counts and audits.
- `figures/`: learning curves.
- Raw Codex rollout files remain under the local Codex sessions directory. Their immutable prefixes are hashed and audited; do not publish hidden reasoning or system messages. Preserve these local files for exact acceptance verification after moving the study.

The two historical controls are reused. The original API study and old 128-candidate experiment remain unchanged. Only this dialog variant receives the results of the keyless run.
