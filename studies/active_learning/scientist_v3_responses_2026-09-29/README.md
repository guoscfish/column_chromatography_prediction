# Clean Scientist V3 via Responses — 2026-09-29

The user authorized switching to direct Responses calls, using token4research,
`gpt-6-sol` and `high`, and starting the first formal experiment. This registration
is independent of the original CLI-prepared V3 study. Its root is the child
`qgeognn_v2_row_llm_scientist_v3/` directory. Always pass that root explicitly;
the runner's default root still identifies the preserved CLI preparation.

No Python source, scientific prompt, memory policy, query interface, predictor,
budget or selection strategy changed. The existing Responses backend is selected
at preparation. The new protocol is
`714d12d99954d5c8db18a155ee03278b38187e62b1b62e588ea69fb87771904d`.
It uses tools=[], store=False and max_retries=0, with a 600-second call timeout.
The authentication key is loaded from the private file
`~/.config/qgeognn-scientist/token4research.api-key` into the calling process's
SCIENTIST_API_KEY environment variable. No key is stored in this repository.
Personal Codex configuration and authentication are not modified by this backend.

Preserved evidence:

- `cli_prepared_artifact_manifest.json`: hashes of the entire prior CLI V3 study.
- `api_preflight.json`: successful minimal direct API connectivity, no experiment data.
- `prior_cli_preflight.json`: rejected CLI preflight, no experiment data.
- `registration.json`: scope and source provenance for this independent experiment.
- `label_safe_stage_receipt.json`: initial stage under guards rejecting all label-store
  and real transport calls; existing L333 records and round-zero predictions only.

The authorized operation is seed157 / free_llm32_scientist_v3 / round 0 selection
and freeze of 32 candidates. No automatic advance, label reveal, fit, report or
test evaluation. Any selector/provider/context/protocol failure stops the operation;
no retry, source repair or protocol revision inside this run.

Canonical run status comes from the immutable artifacts through `study.state`;
receipts here record individual operations, not a mutable current-state cache.

The equivalent CLI action, with SCIENTIST_API_KEY supplied securely to its process, is:

```sh
python scripts/studies/run_qgeognn_v3_row_llm_scientist.py select \
  --study-dir studies/active_learning/scientist_v3_responses_2026-09-29/qgeognn_v2_row_llm_scientist_v3 \
  --seed 157 --method free_llm32_scientist_v3 --round 0
```

## Recorded outcome

The single authorized operation ran from 02:19:52 to 02:31:13 UTC on
2026-09-29 (681 seconds), then stopped on request 18 with a transport failure
classified as `runtime_or_provider_failure`. The sanitized receipt does not
identify a more specific cause. There were 17 completed responses, 24 query
attempts and 128 distinct candidate views. Four responses failed JSON parsing;
six queries failed argument validation. These received the existing protocol's
structured error feedback without human intervention or source changes.

No selection or batch was frozen. Canonical state is
`STOP_failed_or_interrupted_selector`, with 333 active labels. There were zero
label-store calls, no training and no test evaluation. This attempt must not be
rerun using the example command above: `selector.started.json` preserves its
single-attempt boundary. No retry or protocol revision was performed.

The audit hash chain and protocol validation passed. Request sizes ranged from
31,232 to 75,395 characters, below the 100,000-character hard limit. Completed
responses reported 350,460 input and 10,472 output tokens; usage for the failed
request is unknown. All 727 preserved V2 files and all 30 prior CLI V3 files
retained their hashes. The full private API key was absent from these artifacts.

See `selection_operation_20260929.json` for the operation receipt and
`selection_verification_20260929.json` for the independently checked outcome.
