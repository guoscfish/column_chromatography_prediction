# Scientist V3 context design — 2026-09-28

## Audit completed before implementation

Baseline: `04869e5dfcea0f362121240ac3f602e15c22a549`, fetched and verified against
`origin/codex/llm-scientist-v2`. This document starts with a source/artifact audit,
not the stale V2 README status. Work takes place in an independent worktree on
`codex/llm-scientist-v3-context`. The original V2 execution lock is untouched.
All V2 files (including untracked historical artifacts) are byte-hashed in
`v2_preservation_manifest.json`; no V2 runner action is invoked.

### Actual V2 execution path

`scripts/studies/run_qgeognn_v2_row_llm_scientist.py` dispatches to
`scientist_study.py`. `stage` validates protocol, builds `Context` (reads only
initial L333 truth), reconstructs L/U/truth/history through `state`, obtains the
checkpoint through `prediction_freeze`, and calls `make_catalog`. This computes
current q10/q50/q90 predictions and CW distances; hybrid reserves 16 CW points,
free reserves none. `_observed` supplies initial truth and already-revealed
trajectory outcomes. `stage` persists packet/catalog/contract without LLM calls.

`select` stages, then `scientist_transport.run_selector` runs a bounded JSON
relay. Each query returns catalog records. Selection validation requires viewed,
legal unique candidate IDs and observed evidence references. `batch_freeze.json`
binds all 32 IDs, acquisition-time predictions and artifact hashes before reveal.
`advance` audits that freeze, reveals exactly the frozen batch, saves feedback
and label receipt, reads validation truth only for the fixed training rule,
and fits the next predictor from the same initialization. The next `stage`
rebuilds memory from these feedback artifacts. `report` is the separate global
test barrier; it is outside this task and will not be executed.

An AL round is a selection/freeze/measurement/retraining cycle (32 labels).
Several LLM calls within that round are query/decision turns, not AL rounds.
Cross-round memory is derived from feedback; same-round transcript is the
query conversation. V2 conflates neither in storage, but repeatedly sends both.
`messages += [assistant answer, user query_results]` demonstrably resends SYSTEM
+ entire packet + every earlier query/result on every subsequent call.

### Historical protocol and failure evidence

Read-only inspection of revision receipts found model switch to gpt-6-sol,
CLI event casing/config-warning adaptations, nested filters, execution wrapper
hardening, custom model-catalog tool removal, and request_user_input removal.
The round-3 rejected-call receipts record native tool attempts, not merely a
source-code error. CLI exit 1 alone does not establish a code defect.

`query_error_feedback_revision.json` changed frozen code registration after
completed batches, then `query_error_replay_revision.json` restored the exact
legacy successful-feedback shape to prevent request-hash drift. The active
protocol now has hash `919be72cc74af0363113507faba958473956a830942e531011abbd43f9ca5f47`.
V3 must not permit this repair-and-continue pattern after a scientific freeze.
Execution wrappers/catalogs also need provenance, not just Python source hashes.

Actual seed157 Free-LLM32 has freezes and feedback for rounds 00–05 and runtime
round_06 (525 labels). `training_readiness.json` still describes round 0 / 365.
No performance/test result was needed to establish this state drift. V2
`state()` already reconstructs labels from immutable evidence; readiness is a
stale parallel status copy. V3 will expose derived status only.

### V2 packet size: seed157 Free-LLM32 round 03

Counts below use `len(json.dumps(value, ensure_ascii=False))`; field-name and
separator overhead explains the difference between sums and enclosing objects.

| Packet field | Characters |
| --- | ---: |
| study_version | 18 |
| seed | 3 |
| method | 25 |
| round | 1 |
| active_label_count | 3 |
| selection_count | 2 |
| objective | 56 |
| memory | 25,750 |
| pending_experiments | 2 |
| candidate_overview | 530 |
| initial_cards | 16,675 |
| observed_examples | 6,768 |
| observed_count | 3 |
| observed_record_access | 30 |
| validation_test_records | 1 |
| packet_sha256 | 66 |
| **Whole packet** | **50,244** |

| Memory field | Characters |
| --- | ---: |
| previous_hypotheses | 3,685 |
| previous_feedback_interpretation | 674 |
| previous_batch_rationale | 758 |
| previous_selected_ids | 2,304 |
| recent_outcomes | 14,518 |
| high_error_measured_examples | 3,510 |
| memory_policy | 107 |

The 96 previous IDs already belong to observed catalog. Recent outcomes repeat
5,000 characters of per-point reasons; high-error and recent records overlap in
one ID here. Rationale/interpretation/hypotheses overlap semantically. Raw cards
repeat q50 as pred_V1/2_ml and have derived features; query responses can repeat
initial/observed examples. Top-error is max absolute raw mL, favoring V2 scale.
Previous interpretation was written BEFORE latest measurements, not about them.

Round-3 query artifacts contain 154,797 / 70,035 / 58,438 / 3,176 characters
(8 / 4 / 8 / 4 queries). Reported per-call provider input usage is 35,164 /
96,803 / 127,169 / 152,273 / 153,541 tokens. These are provider-reported CLI
usage, not a tokenizer count of the locally serialized messages; the CLI adds
its own packaging. Final before/after audit will report both separately.

## Planned implementation and scientific principles

Database stores facts. Hypothesis ledger stores beliefs. Audit log stores complete
history. Decision context contains the information needed for the current choice.
No model, acquisition function, predictor training rule or scientific batch quota
changes. No test/validation performance in selector context. No real selection,
scientific freeze, reveal, advance, report or real model calls in this task.

1. Independent `active_learning_v3` package and CLI; reuse established pure catalog
   filters/training primitives without changing V2 globals or files.
2. Requests rebuilt as SYSTEM + compact context + bounded working state + latest
   results. Every query answer must carry a validated <=12,000-character state.
   Limit query fanout per answer as well as per-query rows; otherwise a single
   24-query answer defeats compaction. Hard context overflow fails with component
   sizes, never truncates evidence. Approximate tokens are explicitly estimates.
3. Ledger H0001… IDs are host-issued, trajectory-local; deterministic update events
   preserve identity/creation round/full history. Current belief snapshots enter
   context, revision history stays in immutable ledger artifacts. Bounded active
   beliefs plus queryable retired beliefs avoid unbounded history in every prompt.
4. Memory = current ledger, latest measurement digest, balanced replay (up to 12),
   open questions, and explicitly named prior interpretation before latest results.
   Normalize signed acquisition residuals by immutable L333 target scales;
   combined error is RMS of the two normalized residuals. Replay has stable ties,
   distinct IDs and balanced failures/successes/contradictions/recent cases.
   These are evidence presentation rules, never batch selection quotas.
5. Read-only condition series, Morgan-r2 analogs, matched loading/solvent contrasts;
   all paths share query and candidate-view accounting, including references.
   Pool aggregate uses U features/predictions only. q90-q10 is named predicted span.
6. Append-only hashed audit event stream stores requests, responses, queries,
   results, working states, final plan, usage and freeze/provenance. It is not the
   run-state database. Immutable linked round artifacts are the canonical source.
7. Prepare freezes configuration and source fingerprint. First scientific freeze
   locks protocol; missing lock with an existing freeze also fails closed. No
   revision command or automatic repair/retry. Lock binds source, prompt, schemas,
   budgets, memory policy, transport/model/effort and backend provenance. Stop and
   open V3.1/V4 if a bug occurs after lock.
8. CLI/provider/runtime failure: sanitized structured receipt, request hash,
   model/effort/version/exit/category; no raw stderr or credentials; stop operation.
   Invalid JSON/query syntax may receive bounded structured feedback under the
   frozen rules. Query errors consume query attempts in V3 (unlike V2).

## Validation plan and stopping boundary

Synthetic/mock tests cover A–K from the request: permission boundary, compaction,
audit completeness, schemas/IDs, persistent identity, balanced normalized replay,
budgets, protocol lock, transport stop, V2 regressions and unchanged artifacts.
A read-only V2 round-3 reference audit compares actual V2 request serialization
with an explicitly labeled V3 counterfactual projection, not a V3 trajectory.
Prepare and round-zero dry-run may reuse already-authorized L333 and predictor
artifacts. Stop after tests, commit and push; print but do not execute formal select.

## Future storage migration

- SQLite for canonical run state and transactional transitions.
- Parquet for large candidate/prediction tables.
- JSON/JSONL for immutable audit.

This change does not attempt that database migration.

## Implemented flow and final decisions

V3 lives in `src/qgeognn_al/active_learning_v3/`. V2 sources, prompt, studies and
frozen artifacts remain unchanged. The entry point is
`scripts/studies/run_qgeognn_v3_row_llm_scientist.py`; version is `llm_scientist_v3`.

`prepare` imports only clean initial L333 observations and initial predictor outputs,
with partition/ID/checkpoint/catalog hash verification. It imports **no V2 selection,
scientific memory, acquired labels or hypothesis ledger**. It freezes all source
hashes, numerical/training configuration, initial artifact hashes, schemas and
transport configuration. This reuses identical initial scientific conditions without
continuing the V2 trajectory. The seed6101 active V2 directories lack round-zero
contracts; matching archived contracts bind the exact same current catalog bytes.
Their explicit archive paths/hashes are pinned. No missing V2 file is recreated.

`stage` builds a fresh V3 packet and ledger-before snapshot and seals their contract.
Round zero needs no label store, inference, fitting or model service: authorized
initial rows/predictions suffice. Later stages reconstruct only this V3 trajectory's
observations and use its newly trained predictor, the same existing CW gradient
features and QGeoGNN prediction primitives. Hybrid reserves CW16, free reserves none.
The full candidate pool is queryable; no numerical shortlist/acquisition formula is added.

`select` is explicit and executes stateless bounded JSON relay calls. Working state
is validated **before** executing its accompanying queries. Every request consists
of two messages: SYSTEM and a JSON object containing `compact_round_context`,
`working_state`, `latest_query_results`. Successful or failed old results never get
appended to future messages. Query responses contain their own query so the latest
result has sufficient provenance. Invalid JSON/response/query syntax receives
structured correction feedback under the frozen call/query budget. Selection writes
the plan and deterministic ledger-after snapshot, then protocol lock and batch freeze.
No labels are revealed by selection.

The selection `audit.jsonl` is hash-chained and append-only, with full requests/raw
answer text/provenance/usage/query results/state evolution. It is sealed by the batch
freeze. A separate `freeze_audit/audit.jsonl` records the resulting actual freeze hash;
`freeze_receipt.json` binds that event stream without a circular self-hash. Turn and
context audit files record input characters, usage when available, query counts and
candidate views. A started but interrupted selector cannot silently restart or recover
from a mutable personal CLI history. A scientific failure requires stopping and a new
version, not patch-and-continue.

Future explicit `advance` verifies the batch, lock, artifacts, query replay and ledger,
reveals the frozen batch, persists a measurement plus label-access receipt, and trains
one next predictor under the existing initialization/optimizer/validation rule.
Validation truth stays inside the trainer and never enters a selector packet. Training
completion seals the predictor artifacts against that measurement. Tests exercise this
path with a synthetic Store and fake fit only. No real advance was executed.
There is deliberately no `report`/test-evaluation CLI action in this V3 change.

`status` derives round, active labels and next action from immutable initial,
stage/freeze/measurement/training artifacts. It checks contiguous rounds, cross-round
ledger identity, exact query replay, residual arithmetic and source hashes. It never
reads a readiness cache or JSONL as canonical run state. The displayed dry-run audit
is a historical receipt, not a mutable current-status file.

### Frozen budgets and replay meaning

- 24 query attempts, including invalid queries; 480 distinct candidate views.
- At most 2 queries per answer, 24 rows per query and 28 calls per round.
- 20 initial candidate cards, 6 initial observed examples; all facts remain queryable.
- Working state <=12,000 serialized characters; full request <=100,000 characters.
  Hard overflow records all component sizes and stops without truncating evidence.
- At most 12 active beliefs; 4 new host-issued IDs available per round. Rejected beliefs
  stay in the persistent ledger and are queryable, but their full content is not
  automatically repeated. Full revision history stays outside context.
- Balanced replay <=12 unique rows: up to two contradiction, lowest-error (`success`),
  high-normalized-error, overprediction, underprediction, recent-representative rows,
  in that deterministic priority, followed by deterministic backfill. A `success`
  category means relatively low historical error, **not an absolute accuracy claim**.
  Categories are evidence presentation rules, never selected-batch quotas. A row can
  scientifically satisfy several categories but appears once under the first slot.
  Empty/overlapping categories cannot guarantee every category has a distinct exemplar.
- Stable ties use ID. Recent representatives span the recent error ordering. Combined
  normalized error is `sqrt((e1/s1)^2 + (e2/s2)^2)/sqrt(2)` with fixed L333 target scales.
  Initial rows have no acquisition residual. Contradiction examples may still include
  initial observed rows. No host rule generates a chemical explanation.

### Actual working state and ledger schemas

`schema.py` validates exact keys, types, per-field lengths, global length, scientific
role enums, viewed candidate IDs, measured evidence IDs and existing hypothesis IDs.
The working state is exactly:

```json
{
  "focus_questions": [],
  "active_hypothesis_ids": [],
  "shortlist": [{"id": "candidate ID", "why_still_interesting": "<=180 chars", "scientific_role": "hypothesis_test"}],
  "key_evidence": [{"observed_id": "measured ID", "relevance": "<=240 chars", "direction": "supports"}],
  "rejected_directions": [],
  "open_questions": [],
  "next_query_intent": "<=600 chars"
}
```

Question/direction arrays have <=6 entries of <=400 characters. Shortlist <=48,
key evidence <=16, active IDs <=12. Directions are supports/contradicts/control.

The persistent ledger is trajectory-local, with host-owned allocation metadata:

```json
{
  "seed": 157,
  "method": "free_llm32_scientist_v3",
  "next_id": 2,
  "hypotheses": {
    "H0001": {
      "id": "H0001",
      "statement": "...",
      "created_round": 0,
      "last_updated_round": 1,
      "status": "supported",
      "confidence": "low",
      "basis": "mixed",
      "supporting_observed_ids": [],
      "contradicting_observed_ids": [],
      "why_it_matters_for_model_learning": "...",
      "next_discriminating_question": "...",
      "history": [{"round": 0, "action": "create", "reason": "...", "belief": {}}]
    }
  }
}
```

Each history `belief` contains the complete eight belief fields (statement, status,
confidence, basis, both evidence lists, learning relevance, next question), not a
partial patch. Context has the current non-rejected entries without `history`.
Final plans contain updates `{id, action, reason, belief}`. `retain` requires null
belief; revise/weaken/support/reject/create require a complete validated belief.
Status actions must match their status; changing a statement requires `revise`.
Unmentioned identities persist; ID reuse/reassignment is rejected. Semantic continuity
of a revised statement still requires scientific review—it cannot be proved by syntax.

The memory object is exactly `hypothesis_ledger`, `latest_measurements_digest`,
`balanced_replay_bank`, `open_scientific_questions`, and
`prior_interpretation_before_latest_measurements`. The digest contains measurement
count, residual count/timing, normalized error quantiles, combined-error quantiles,
per-target over/under/equal counts and at most three key IDs. No long prior reasons or
all-selected-ID list is sent.

Each replay entry has this actual structure:

```json
{
  "id": "observed ID",
  "category": "high_normalized_error",
  "residuals": {
    "id": "observed ID",
    "raw_error_V1_ml": 8.0,
    "raw_error_V2_ml": 10.0,
    "normalized_error_V1": 0.8,
    "normalized_error_V2": 0.1,
    "combined_normalized_error": 0.570087712549569
  },
  "observation": {
    "id": "observed ID",
    "smiles": "...",
    "conditions": {},
    "descriptors": {},
    "V1_ml": 1.0,
    "V2_ml": 10.0,
    "record_kind": "acquired_with_premeasurement_prediction"
  }
}
```

The illustrative observation above abbreviates catalog X: the actual card has the
five conditions, recomputed descriptors, loading/novelty/similarity evidence, and
acquisition prediction/error fields where available. It also exposes the five raw/
normalized residual fields for observed sorting. Residuals are null for an initial
observation used as contradiction evidence. Full exact reference entries are saved in
`context_audit/v3_reference_context.json`; the synthetic values above are not findings.

### Read-only query API

| API | Arguments / behavior |
| --- | --- |
| get/search_candidates, get/search_observed, get/search_pending | V2 filters, pagination, stable salted ordering; V3 spans and normalized errors replace legacy widths/raw max-error |
| get_condition_series | reference_id, offset, limit; same molecule across all three roles; PE fraction then loading ordering |
| get_nearest_analogs | reference_id, k or limit, min_tanimoto, offset; radius-2 Morgan/2048-bit Tanimoto, decreasing similarity then ID |
| get_matched_loading_contrasts | reference_id, solvent_tolerance (absolute PE fraction difference, default 0), offset, limit |
| get_matched_solvent_contrasts | reference_id, loading_relative_tolerance (difference/max absolute loading, default 0), offset, limit |
| get_hypotheses | at most two stable IDs; current belief, including rejected beliefs; full history stays in audit |

Matched contrasts require same molecule, same carrier solvent and carrier volume,
and differing loading or differing PE fraction. They are reference-relative views,
not an algorithm asserting causal identification. Exact conditions are returned.
Every candidate row returned via any route consumes the same distinct view budget;
references included as rows also count. All operations, including hypothesis lookup
and failed query attempts, consume the query budget. Unknown fields and truth/error
sorts for unobserved rows fail closed. Aggregate summaries use U_t features/predictions,
with L_t molecular features only for novelty/similarity. Molecule condition counts are
counts of experiment rows per molecule in U_t; repeated conditions count as rows.

### Protocol and transport lock

The prepare-time fingerprint contains all Python sources in `src/qgeognn_al`, the V3
entry point, full prompt, query/selection/working-state schemas, memory policy, all
budgets, unchanged training config, package versions and transport config. The first
scientific freeze writes `protocol.lock.json` under an exclusive operation lock **before**
the batch file. It binds the protocol file SHA256 and full fingerprint SHA256. Every
subsequent operation checks both, all pinned inputs and the CLI binary hash. A freeze
with a deleted/missing lock fails. Changing code and also rewriting protocol to its
new hash still fails against the original lock. No revision command exists. These are
application-level immutable artifacts, not signed/WORM protection against a person
rewriting every hash. A mid-write interruption fails closed.

The prepared configuration is `codex_cli`, `gpt-6-sol`, `high`, endpoint
`https://token4research.cn`, pinned CLI `0.158.0-alpha.2.1` and executable SHA256.
Only `codex --version` was executed, never `codex exec` during this task. Actual calls
use temporary cwd and fresh CODEX_HOME, allowlisted endpoint/auth config and copied
authentication only. Private history/MCP config/project instructions are not imported.
Native tools are disabled and successful output is allowlist audited; read-only is
still not an OS deny-read sandbox. Any native/unknown/provenance failure stops.
CLI service-side storage cannot be guaranteed by this adapter; the Responses backend
explicitly uses tools=[] / store=False / max_retries=0. Backend cannot change mid-study.

A transport/CLI/provider/runtime failure writes request hash, frozen model/effort,
CLI version when applicable, exit code when available, error category and a constant
sanitized diagnostic. No raw stderr, exception body, credential, API key or header is
persisted. It never modifies source/protocol, retries or reuses a partial response.
The runner only has explicit scientific operations, with no source-repair action.

## Before / after context audit

Read-only seed157 Free-LLM32 round 03. Every reconstructed V2 request hash matches its
saved turn receipt. No re-selection, new truth access, LLM call or second-LLM summary.
V3 is a **counterfactual representation**, not measured V3 model behavior or proof of
equivalent decisions. Its synthetic bounded working state preserves viewed IDs and
prior evidence references. The last V2 query group has four queries; all four results
are retained here for a conservative comparison although real V3 permits two per turn.

| Component | V2 chars / rough tokens | V3 chars / rough tokens | Reduction |
| --- | ---: | ---: | ---: |
| SYSTEM | 8,085 / 2,022 | 7,239 / 1,810 | 10.46% |
| Packet / compact context | 50,244 / 12,561 | 41,730 / 10,433 | 16.95% |
| Accumulated conversation / latest result | 310,224 / 77,556 | 2,974 / 744 | 99.04% |
| Working state | 0 | 5,705 / 1,427 | new |
| All previous selected IDs (packet subset) | 2,304 / 576 | 0 | 100% |
| Recent per-point reason text (packet subset) | 5,000 / 1,250 | 0 | 100% |
| Previous batch rationale (packet subset) | 758 / 190 | 0 | 100% |
| Hypotheses / current ledger (packet subset) | 3,685 / 922 | 3,812 / 953 | -3.45% |
| Top-error examples / balanced replay (packet subset) | 3,510 / 878 | 14,772 / 3,693 | -320.85% |
| **Whole serialized request** | **372,451 / 93,113** | **62,193 / 15,549** | **83.30%** |

Subsets must not be added again to totals. Counts include each implementation's actual
JSON serialization; whole-request escaping adds overhead. Estimates are ceil(chars/4),
not tokenizer measurements. V2 reported provider usage for the final call is 153,541
input tokens, substantially above its local character estimate due to tokenization/
CLI packaging. V3 has no live provider token count. At the full 12,000-character working
state bound this reference is approximately 68,488 serialized characters. The benefit
comes mainly from removing accumulated transcript, while replay evidence is **larger**.
No guarantee is made that every possible query group fits; hard overflow stops visibly.

Machine-readable files: `context_audit/context_size_comparison.json` and the explicit
counterfactual `context_audit/v3_reference_context.json`. They are analysis artifacts
only and are not inputs to the clean V3 study.

## Validation results and stop point

- **87 passed**: 46 V3 tests and 41 existing Scientist V2/full-pool/dialog regression tests.
  118 existing dependency/deprecation warnings; no failures in the final run.
- Real data prepare/validate and all four seed/method round-zero stage/dry-runs passed
  with `RestrictedLabelStore.reveal` and both real transports replaced by immediate
  failure guards. Every label store call, including initial truth, was prohibited;
  staging reused already-authorized L333 catalogs instead.
- Stage request sizes: seed157 hybrid 45,632 chars, free 31,232; seed6101 hybrid
  45,557 chars, free 31,222. These first requests have no latest query results yet.
- Canonical state is round 0 / 333 labels / next action select for all four arms.
  Protocol is prepared and valid, **unlocked**, because no scientific batch is frozen.
- Real LLM calls 0; new labels 0; validation/test truth accesses 0; fits 0;
  scientific freezes 0; real advance/report/test evaluation 0.
- Complete V2 manifest: 727 files, unchanged hashes and no additions/deletions.
  V2 code/entry point/tracked studies/documentation diff is empty against the baseline.
- Python compilation and `git diff --check` passed.

Remaining risks: the V3 CLI adapter has synthetic coverage but no live provider check,
so its first real attempt may fail closed on CLI format/tool exposure or provider
compatibility. Character estimates are not calibrated token counts. Hard evidence
size limits can stop a scientifically legitimate large query. Working-state semantic
compression and hypothesis revisions remain LLM judgments; structural validation cannot
prove retention of every scientific insight. Balanced replay is a fixed presentation
policy that still introduces attention bias, although it removes raw-V2-only ranking.
Absolute pinned input/binary paths make this prepared study specific to this checkout;
a clean relocation requires preparation in a new study before any scientific freeze.
Application/CLI updates or any source edits invalidate the prepared protocol. Real
retraining integration was tested with mocks, not by running a new experiment.

The next formal command, **not executed**, from this V3 worktree is:

```sh
/Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v3_row_llm_scientist.py select --seed 157 --method free_llm32_scientist_v3 --round 0
```

This starts only the first clean V3 selection and freezes its batch; it does not reveal,
train, evaluate test data or continue V2. The current task stops before that command.

## File map

Exact file list: `studies/active_learning/qgeognn_v2_row_llm_scientist_v3/changed_files.txt`.
All entries are new files; no V2 implementation or historical file is edited.

| File/group | Responsibility |
| --- | --- |
| active_learning_v3/__init__.py | independent package |
| active_learning_v3/schema.py | bounded working state and shared constants |
| active_learning_v3/memory.py | persistent deterministic ledger, digest and balanced replay |
| active_learning_v3/catalog.py | permission-preserving query views and aggregate summary |
| active_learning_v3/selector.py | scientific prompt, packet and final-plan schema |
| active_learning_v3/artifacts.py | write-once files, operation lock, append-only hashed audit |
| active_learning_v3/transport.py | compact relay, CLI/Responses adapter, sizes/usage/failure receipts |
| active_learning_v3/protocol.py | source/schema/config fingerprints and protocol lock |
| active_learning_v3/study.py | independent prepare/stage/select/advance and canonical derived state |
| active_learning_v3/predictor.py | lazy unchanged training/prediction integration |
| active_learning_v3/context_audit.py | hash-verified V2 read-only context comparison |
| scripts/studies/run_qgeognn_v3_row_llm_scientist.py | explicit V3 CLI actions |
| tests/active_learning_v3/test_scientist_v3.py | 46 synthetic unit/integration tests |
| this document | preimplementation audit, implementation, schemas, evidence and risks |
| V3 study protocol.json / selector_prompt.txt | prepared protocol and exact prompt |
| V3 study initial/ and selections/ round_00 | four clean initial datasets, staged inputs/catalogs/ledger-before/contracts |
| V3 study context_audit/ | before/after numbers and explicit counterfactual payload |
| V3 study dry_run_audit.json / verification_results.json | final label-safe and test/preservation receipts |
| V3 study v2_preservation_manifest.json | 727 historical V2 file hashes |
| V3 study .gitignore / changed_files.txt | ignore transient operation lock; exact delivery inventory |
