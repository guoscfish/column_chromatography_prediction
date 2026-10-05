# LLM Scientist V2 — full-catalog delivery

The current default revision is `astra_high_full_pool_20261001`. Both LLM arms
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

## Current six-cycle run status (2026-10-01)

The seed-157 full-pool run completed cycles 1–3 (365, 397 and 429 active labels),
including retraining and frozen predictions. Cycle 4 stopped after three provider
responses reported `429 USAGE_LIMIT_EXCEEDED / DAILY_LIMIT_EXCEEDED`. The provider
did not supply a reset time. No final test evaluation was performed. All completed
artifacts passed the post-stop audit. See `revisions/astra_high_full_pool_20261001/run_status.json`.
Once provider quota is restored, run
`revisions/astra_high_full_pool_20261001/resume_six_rounds.sh` to reuse the first three
cycles and complete the remaining cycles and final report.
