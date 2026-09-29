# Scientist V3.1 — first live Responses selection

Registered on 2026-09-29 after the user requested continuing the experiment with
the bounded transport retry policy. The model is `gpt-6-sol`, reasoning `high`,
through `https://token4research.cn`. Each logical request permits three extra
transport attempts, waiting 2, 4 and 8 seconds for qualifying transient errors.
The protocol SHA-256 is
`f1e176df98385a12975e1f03c818d9642ecc5031281512e7b10a8e268fcfd891`.

This is a clean L333/U0 initialization. The previous V3 Responses attempt remains
unchanged and is recorded by `execution/prior_responses_manifest.json`. No prior
choices, acquired outcomes or hypothesis ledger are imported. Preparation and
staging ran with guards forbidding label reads and live transport calls.

The operation is seed157 / free_llm32_scientist_v3 / round 0 selection and batch
freeze. It does not automatically reveal labels, train, advance or evaluate test
data. Retries are audited within a logical request. An exhausted retry budget or
other terminal failure stops this attempt without an in-place protocol change.

Operation and verification receipts are stored under `execution/`. Canonical
scientific state is derived by `study.state` from immutable phase artifacts.
