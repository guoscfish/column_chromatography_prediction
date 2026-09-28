# LLM Scientist V2

Independent two-seed development study: seeds 157 and 6101; budgets
333, 365, 397, 429, 461, 493, 525; batch size 32.

- `random32`: exact historical Random trajectory reuse.
- `center_width_lcmd`: exact historical CW32 trajectory reuse.
- `cw16_llm16_scientist_v2`: pending CW16 plus freely planned LLM16.
- `free_llm32_scientist_v2`: all 32 planned by the LLM; no pending CW or quotas.

`protocol.json` freezes code, source hashes, model/provider and training settings.
`control_reuse.json` audits 28 historical control prediction points without reading
historical performance metrics. `historical_artifacts.json` fingerprints the earlier
LLM studies. `selector_prompt.txt` is the complete scientific system prompt.

The active model-switch revision uses `codex exec` with `gpt-6-sol`, high reasoning,
and the provider/authentication from `~/.codex/config.toml` and `auth.json`.
Provider connectivity and model availability are tested by the live selector call.
The CLI backend has a no-native-tool audit, not a hard filesystem deny-read sandbox.
The direct Responses backend instead uses `SCIENTIST_API_KEY` and `store=False`.

Use `scripts/studies/run_qgeognn_v2_row_llm_scientist.py`:

1. `prepare`: freeze the protocol and reuse audit.
2. `dry-run --seed 157 --method free_llm32_scientist_v2`: prepare a real packet;
   no LLM request, new labels, validation/test truth or fitting.
3. `select --seed 157 --method free_llm32_scientist_v2 --round 0`: query and select;
   freeze 32 IDs and premeasurement predictions; stop before label reveal.
4. `advance --seed 157 --method free_llm32_scientist_v2 --round 0`: reveal that
   frozen batch and train exactly one next-budget model. No next selection starts.
5. Repeat explicit select/advance commands for rounds 1 through 5 for each arm/seed.
6. `report`: requires all four completed trajectories and 28 frozen new prediction
   points before revealing test truth. Reports all four methods on the same grid.

No fallback picks or silent retries. Failed/incomplete provider calls leave a
`.started.json` marker and fail closed. Do not delete this marker or rewrite frozen
protocol/artifacts to retry; investigate and register a separate revision/study.

Design and critical review:
`docs/research/4G_LLM_SCIENTIST_ACTIVE_LEARNING_V2_2026-09-26.md`.
