# Long-request transport repair, 2026-10-29

User authorization: “帮我处理一下”, following the proposed stream logging and
long-request transport repair. The existing authorization for five independent
NoCW seeds and persistent network retries remains in force.

The additive runner `scripts/studies/run_qgeognn_v2_row_llm_nocw_stream.py` imports
the frozen five-seed and persistent-retry runners. Neither frozen source nor
scientific config is edited. `stream_transport_amendment.json` binds its code and
offline tests to both previous protocol hashes.

## Changes

- Send the same Responses body: complete input, gpt-6.1-sol/xhigh, tools=[],
  store=false, max_output_tokens=32768, unchanged sampling parameters, stream=true.
- Parse SSE explicitly, as historical Free did. Direct connection, no environment
  proxy, no hidden HTTP retries, existing 600-second inactivity timeout, TCP
  keepalive enabled with a 60-second idle interval where supported.
- Save metadata for headers, created/queued/in-progress, event cursor/count and
  last line arrival. No prompt, candidate table, generated reasoning, or output
  text is written to stream progress files. A received line proves data arrived;
  it does not prove useful reasoning or a valid selection.
- Validate completed identity/status/tools and save the full answer in a local
  immutable completion receipt before closing the HTTP stream. Do not wait for
  trailing EOF. A later close error cannot discard a validated completed answer.
- If interrupted after local completion but before the attempt receipt, matching
  local completion can be used without another request. An unresolved started
  marker without a validated local completion still requires reconciliation.
- Continue the existing retry ledger without renumbering or deletion. Protocol,
  identity, quota/authentication and output-cap failures stop for reconciliation;
  transient connection, timeout and server failures retain persistent retries.

## Handover

Stop only the old supervisor process, leaving its in-flight workers untouched.
The new supervisor holds the same supervisor lease. For each persistent worker,
wait for a recorded failure and retry backoff. Briefly stop that process and
recheck both its recorded backoff state and all started markers before terminating
it. If a request started in the meantime, immediately resume the process and
wait. Retain the remaining scheduled backoff and then resume from the original
ledger under the existing seed lease. The old finite-retry worker can be taken
over after it exhausts transport retries and exits. There is no selection or
training interruption and no two requests for one seed.

## Background recovery remains disabled

Official documentation describes background generation and stream resumption:
https://developers.openai.com/api/docs/guides/background
https://developers.openai.com/api/docs/guides/streaming-responses

No public Token4Research documentation establishing compatibility was found in
the documentation search on 2026-10-29; the web reader could not access its home
page. This does not prove support or lack of support. No paid inference probe was
issued. This amendment does not set background=true, enable server storage,
retrieve responses, switch endpoint/model/key, or claim remote resume works.

TCP keepalive and explicit SSE processing can reduce avoidable transport loss;
they do not guarantee provider stability or shorten model reasoning time.

## Operations

Use the fish Python environment and KMP_DUPLICATE_LIB_OK=TRUE. CLI actions default
to read-only `status`; paid actions require all three explicit flags. Inspect
`progress/stream_supervisor.json`, `progress/stream_seed_*.json`, all attempt
receipts, seed locks, batch freezes and fit audits together. During migration,
an absent stream file means the old request is still running, not stalled by
the new adapter. Never infer useful progress from a stale SELECTING record.

At all five L1005 completions, run the new entry point `report --budget 1005`;
reporting still delegates to the existing audited persistent report generator.
