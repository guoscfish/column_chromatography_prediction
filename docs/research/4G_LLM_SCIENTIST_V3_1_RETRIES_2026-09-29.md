# Scientist V3.1: bounded Responses transport retries

The user authorized changing the stop-on-first-error behavior on 2026-09-29,
after the first V3 Responses run stopped on request 18. V3.1 defaults to direct
Responses calls with `gpt-6-sol`, reasoning effort `high`, and the previously
configured provider. Authentication still comes from `SCIENTIST_API_KEY`.

Each logical model request permits three extra transport attempts, for four
attempts total. The delays are 2, 4, and 8 seconds. Only connection failures,
timeouts, HTTP 408/409/429 and HTTP 500–599 qualify. Known exhausted-quota codes
are excluded even when the status is 429. Authentication, invalid parameters,
model provenance errors, native tool output, unknown runtime errors, protocol
drift and context or scientific budget failures stop immediately. CLI transport
does not gain automatic retries.

This distinction follows the retryable-versus-configuration-error guidance in
[official OpenAI documentation](https://developers.openai.com/api/docs/guides/error-codes).
The third-party provider may report errors differently; the implementation uses
the SDK's typed exceptions and HTTP status, without guessing from error text.

Retries use exactly the same messages, model and reasoning settings. They do not
repeat catalog queries or modify scientific working state. Each attempt checks
the protocol guard and appends an audit event. Failed attempts also receive an
immutable sanitized receipt with HTTP status, failure category, attempt count,
and retry decision. No exception text, provider response body, API key or
authorization header is written to these receipts. The SDK's own retry count
remains zero, so there are no hidden nested retries.

The existing 28 logical model-call, 24 query and 480 candidate-view limits remain.
With the default retry policy, at most 112 transport attempts are possible per
selection. A timed-out request may have been processed remotely; retries do not
guarantee exactly-once billing. Usage for failed requests is unknown. JSON and
query validation errors retain the existing bounded feedback path; completed
responses are not silently discarded and resampled by this transport policy.

The protocol version is `llm_scientist_v3_1`. New runs default to the independent
`studies/active_learning/qgeognn_v2_row_llm_scientist_v3_1` directory. The runner
defaults to Responses; `prepare --transport-retries 0|1|2|3` can choose a smaller
retry budget before registration. The chosen policy is fingerprinted and cannot
be overridden with `select` or another action.

The stopped V3 run and original CLI preparation are untouched. Their frozen
source fingerprints intentionally reject this new implementation; the original
implementation and its artifacts remain available at commit `1557358`. There is
no automatic restart of an already started selector, no in-place rewrite of its
protocol, and no new label access, training or live API call during this change.

Synthetic tests cover identical-request recovery, bounded exhaustion, secret
redaction, permanent failures, guard changes between attempts, disabled SDK
retries, protocol locking of the retry policy and successful batch freezing after
a transient failure. Existing V2 and selector regression tests are also run.

Validation on 2026-09-29: 111 tests passed across `test_scientist_v3.py`,
`test_scientist_v2.py`, `test_full_pool_selector.py`, and `test_dialog_bridge.py`.
All 727 preserved V2 files and 30 prior CLI V3 files matched their recorded
hashes; the stopped Responses study also had no changes from commit `1557358`.
