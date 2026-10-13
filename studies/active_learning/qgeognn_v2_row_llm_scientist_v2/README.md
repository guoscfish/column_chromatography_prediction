# LLM Scientist V2 — full-catalog delivery

The original runner default revision is `astra_high_full_pool_20261001`. Both LLM arms
send every eligible candidate and every currently observed record in each round.
The default transport is streaming Responses through `https://token4research.cn/v1`,
using `gpt-6-astra` with `high` reasoning and `store=False`.

- `free_llm32_scientist_v2`: LLM directly selects all 32 experiments.
- `cw16_llm16_scientist_v2`: CW selects 16 pending experiments; the LLM sees those
  plus the complete remaining pool and selects 16 more.
- Random32 and CW32 reuse audited historical predictions for the same seed/budgets.

The full tables preserve all records and numeric precision. Molecular structures,
descriptors and field names are factored into dictionaries and column headers.
No retrieval, paging, 480-candidate view limit or numerical shortlist is used.
The model receives its own trajectory's previous feedback and updated predictions.
Candidate predictions are not measured evidence: only observed-table IDs may appear
in measured-evidence lists. Invalid JSON or invalid plans receive explicit correction
feedback, with at most three plan responses. Each correction still includes the
complete catalog; no model-generated rationale is silently rewritten. Network
failures have bounded retries and no automatic model or algorithmic fallback.

## Run six cycles and evaluate

From the repository root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python -u scripts/studies/run_qgeognn_v2_row_llm_scientist.py run --seed 157 --method free_llm32_scientist_v2 --status-interval 30
```

This runs six 32-label acquisitions: 333 → 365 → 397 → 429 → 461 → 493 → 525.
For each cycle it freezes all selected IDs and premeasurement predictions, reveals
only the selected labels, retrains from the same initialization, freezes new
predictions and supplies the resulting feedback to the next selection. It reuses
the L333 model and keeps the registered optimizer, validation checkpoint rule and
training budget. Restart the identical command to reuse completed artifacts.

A single-seed run automatically evaluates only after all six batches and all seven
prediction points pass their audits. Results go to
`revisions/astra_high_full_pool_20261001/reports/seed_157/free_llm32_scientist_v2/`:
`results.json`, `learning_curves.csv`, `aulc.csv`, and `learning_curves.png`.
The report compares this arm with same-seed Random32 and CW32. This is a one-seed
development comparison, not statistical confirmation. Test results never enter
selection prompts. The `trajectory-report --seed ... --method ...` action can
reproduce the post-freeze report separately.

Omitting both `--seed` and `--method` runs two seeds (157, 6101) × both LLM arms,
24 acquisition cycles in total. That full study retains the global 28-prediction
barrier before its `report` action. Use a new `--revision` when changing frozen
code, prompts, models or training settings; old artifacts remain immutable.

## Authentication and inspection

Set `SCIENTIST_API_KEY` or `SCIENTIST_API_KEY_FILE`. For token4research, the default
key file is `~/.config/qgeognn-scientist/token4research.api-key`. Credentials are
never written to study artifacts; global ChatGPT login tokens are not used.
The optional `codex_cli` backend retains its native-tool audit, but the tested path
for Astra is streaming Responses and does not require the CLI executable.

Actions: `prepare`, `validate`, `check-transport`, `dry-run`/`stage`, `select`,
`advance`, `run`, `report`, and `trajectory-report`. `stage` constructs a full packet
without calling the model or revealing new labels. `select` freezes the next batch;
`advance` reveals that batch and trains one model. `check-transport` only checks
local configuration and credentials. Runtime heartbeat events are recorded in
`execution_progress.jsonl`; selections, rejected attempts and feedback remain under
`selections/seed_*/METHOD/round_*/`.

## Historical trials

The September 30 query-based streaming trial saw 172 of 2,997 candidates and selected
32 after 11 queries. The standalone all-candidate trial sent all 2,997 candidates
and 333 observations in one request (455,139 input tokens, 486.8 seconds). It returned
32 legal IDs but failed measured-evidence validation for one hypothesis; no labels
or training were advanced. Original artifacts remain in their respective revision
directories. These snapshots require their frozen source versions for exact replay.

`run_qgeognn_v2_row_llm_all_candidates_trial.py` remains the independent one-shot
trial runner for a previously frozen catalog (`--live` enables the actual request).
It does not perform training and does not replace the formal six-cycle runner above.

## Current completed results (2026-10-11)

CW16+LLM16 completed both seeds at 813 labels (15 batches each). Validation
NRMSE is 0.322792/0.380292 for seeds 157/6101. Free completed 525/653 labels,
with validation NRMSE 0.645484/0.520703. The October 1 quota stop and October 9
stop-after-621 request are historical; later authorization resumed the hybrid
trajectory. These records do not authorize a new run.

See the [latest matched-budget snapshot](../../../reports/active_learning_latest/REPORT.md)
and [813-label comparison](../../../reports/llm_hybrid_complete_20261010/same_budget_metrics.json).
The completed extension result and compatibility/cost receipts remain under
`revisions/astra_high_full_pool_20261001/extension_authorized_to813/`.
Historical Free correction deviations remain part of interpretation; two seeds
and previously exposed splits provide development evidence, not confirmation.

132 old tracked runtime JSON receipts were consolidated into
`historical_runtime_records.tar.gz`; `historical_runtime_manifest.json` lists exact
paths, hashes and the Git recovery commit. `historical_fit_audits.json` retains 18
old fit audits for readable review. Local originals remain at their frozen paths.
This archive contains metadata only, without checkpoints or training payloads.
Restore metadata for historical replay from the repository root:

```bash
tar -xzf studies/active_learning/qgeognn_v2_row_llm_scientist_v2/historical_runtime_records.tar.gz
```

Historical replay still needs the original code revision and local checkpoints.

## Latest completed continuation (2026-10-13)

Free completed both seeds to813 in `free_sol_xhigh_continuation_20261012`; the later `sol_xhigh_hybrid_free_to1005_20261013` revision retains the inherited prefix and targets1005. Completed frozen endpoints are Hybrid845/813 and Free845/845 for seeds157/6101. The1005 target is incomplete; the pause receipt requires no recovery before2026-10-13 18:30 Asia/Shanghai. See the [current validation snapshot](../../../reports/active_learning_latest/REPORT.md) for same-budget comparisons.

Free changed from Astra/high to Sol/xhigh after525/653 respectively; Hybrid changed after813. These mixed-model development trajectories do not isolate a model-only effect. Full requests and locks stay local; protocols, selections, measured feedback, costs and hashes remain versioned. Historical source files and frozen revisions remain immutable.
