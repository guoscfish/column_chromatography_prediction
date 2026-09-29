# Scientist V3.2 execution report

Requested: six acquisitions, L333 -> L365 -> L397 -> L429 -> L461 -> L493 -> L525. Actual status: **STOPPED**, 1/6 completed, **L365**.

Frozen scientific source: `d277aa7aed8f47403db66dd19db33b61bdadc54c`. Protocol SHA-256: `ecd8f964d1bdb5cec26961ebdd8d10776a4599e81db492c0d727eba44ed99673`. Seed157, Free-LLM32, gpt-6-sol/high, token4research Responses streaming. Same initialization, fixed validation, zero test truth access. Validation metrics were not sent to the selector.

## Outcome

Controller diagnostic: STOP: response validation repair budget exceeded. The recovered response proposed 32 rows, but H0001 used `support` while changing its statement; the frozen validator requires explicit `revise` for a statement change. This was round 1's fifth validation error (three query errors plus two response errors), exceeding the shared allowance of four. No second batch was frozen, no second measurement was revealed, and no L397 checkpoint exists.

| Labels | Fixed-validation combined NRMSE |
|---:|---:|
| 333 | 0.853886566 |
| 365 | 0.812055654 |

One seed; checkpoint validation scores are not an independent test or a matched acquisition-strategy comparison.

## Interaction and concentration

| Round | Labels after | Queries | Invalid queries | Response repairs | Molecules | Largest molecule | Targeted / broad |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 365 | 24 | 0 | 3 | 28 | 3/32 | 12 / 20 |

Round 0 completed with 24 accepted queries and no invalid queries, compared with 18 successful / 6 invalid in the V3.1 pilot. Three response repairs remained. Its 32 rows covered 28 molecules, with at most three rows per molecule; role metadata indicated 12 targeted and 20 broader exploration choices. Twenty-one rows were molecule-novel relative to L333. These are descriptive diagnostics, without quotas or proof of strategy superiority.

Round 1 repeatedly passed experiment row IDs to get_hypotheses (three errors) and produced multiple top-level JSON objects once. The host rejected these without guessing. All four repair allowances were already used before the process interruption. The latest directly linked evidence correctly required reviews of H0001-H0003 in the round-1 packet; an unaccepted response is not a completed hypothesis review. Interface improvements reduced some pilot failures but have not eliminated protocol waste or provider instability.

Selected-point acquisition-time residuals and all 32 accepted IDs are in ROUND0_REPORT.md and compact/round_00. They are historical measurement-time errors, not measurements of the retrained model. No q10-q90 span is interpreted as epistemic uncertainty.

## Interruption and exact replay

The original controller disappeared during round 1 without a terminal receipt; its cause is unknown. Fourteen completed responses, 23 accepted queries, all four repair charges, and the original audit prefix were retained. The explicit recovery runner re-executed the frozen scientific selector against saved responses and verified every historical audit event, outgoing request hash and immutable artifact before making another call. It counted the unknown third transport attempt as consumed and allowed only attempt 4. The operational recovery runner was added after registration and is disclosed separately; it did not alter the scientific code fingerprint or reset budgets.

See execution/continuous_loop/interruption_observed.json and execution/process_recovery/result.json. A terminal failure remains a terminal failure; no future batch is fabricated, no reset is hidden, and no provider/model fallback occurs.

## Cost accounting

Formal trajectory: 32 logical requests, 32 complete responses, 35 HTTP attempts, 47 accepted queries, 3 invalid queries, 5 response-validation errors.

Earlier failed attempts: non-streaming 15 logical / 18 HTTP / 24 accepted queries; first streaming 4 logical / 4 HTTP / 6 accepted queries. Both revealed zero new labels. One failed diagnostic replay and three data-free capability probes are additional costs.

Total observed network attempts including these extras: **61**. Failed or interrupted HTTP attempts may be billed; unavailable token usage is not zero. Exact monetary cost was not retrieved.

## Verification and retention

Relevant scientific and recovery tests: 231 passed, 147 dependency warnings. The separate repository hygiene checks retain 3 documented baseline failures (one missing ignored historical checkpoint and two blanket assertions against 132 already-tracked V2 runtime records). No old evidence was deleted to conceal these failures.

V3.1 artifacts, V3/V2 source, and existing model/split/trainer/gradient code are unchanged. Hybrid 16+16 remains audit-only. Compact reports, protocol, results and manifests are committed; large catalogs, requests, audits and runtime checkpoints remain local and ignored. The Git checkout alone is not the complete runtime archive.

## Repository disposition

Origin retains main and codex/llm-scientist-v3-2. V1, V2, V3.1 and divergent CW performance tips are retained in verified remote archive tags before deletion of the four legacy remote branches. Local legacy refs/worktrees remain because they belong to other chats or retain existing state. See docs/research/REPOSITORY_CLEANUP_2026-09-29.md.

Implementation commits:

- `32224a12149d33109cfd961132725df5f1bd9e80` chore: clean active-learning repository layout
- `adfffcb2ec2c4499afe6dd8d390100d6e82f2501` feat: harden llm scientist v3.2 interaction protocol
- `a476d9009d9aa133a5c635e76d83656e785b758c` chore: record verified milestone archive cleanup
- `0e9776cfc9a65fda66e1d29cba159bde978ecb10` fix: stream Responses and register six-round scientist run
- `d277aa7aed8f47403db66dd19db33b61bdadc54c` fix: retry interrupted Responses streams within frozen budget

- `aa2bd1b75ef7b26ffb27c3c1e5e9ad6dfc96d6b7` fix: recover interrupted scientist requests with exact audited replay

## Requirements for a future continuation (not executed)

The frozen V3.2 trajectory ends at L365. Continuing after its terminal limit would require an explicitly registered new experiment version. A concrete follow-up would distinguish experiment-row IDs from hypothesis IDs in the wire schema, make action-specific hypothesis update contracts explicit (support/weaken/retain preserve statement; revise may change it), and regression-test the exact failed H0001 update. The host must not silently rewrite support into revise, choose among multiple JSON objects, or reset this trajectory's repair count.

A new version could start from the preserved L365 model/labels/ledger for five additional acquisitions, but this would be a segmented protocol rather than six uninterrupted V3.2 rounds. Alternatively it could start clean at L333. Neither follow-up was launched. Provider reliability still needs to be demonstrated separately; the completed schema probes did not guarantee reliable experiment responses.
