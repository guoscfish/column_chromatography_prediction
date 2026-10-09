# CW geometry and LLM feedback — preregistered development study

2026-10-08; starting commit `89c42c164aab2d873ccb65f409cfd0417233bd17` (origin/main verified). Branch `codex/cw-geometry-feedback`. No applicable AGENTS.md found; repository conventions and retention policy apply. Initial working tree clean, no local training worker; other worktrees preserved.

## Questions and interpretation

A: Are measured-feedback-driven LLM selection and CW geometry complementary? B: Does CW geometry improve LCMD, MaxDet and IVR, and is the benefit output correlation correction rather than unique center/width semantics? Task remains 4g Row overall V1/V2. No architecture, transfer or controller expansion. Historical CW/IVR portfolio uses CW-LCMD and Raw-IVR; it is not CW-IVR. Historical innovation whitening is selection-only unless full retraining provenance proves otherwise.

## Frozen matrix and limits

Development seeds 157, 6101; L0=333, B=32; budgets 333,365,397,429,461,493,525. Raw/CW × LCMD/MaxDet/IVR plus Random. Reuse audited Raw-LCMD, Raw-MaxDet, Raw-IVR, CW-LCMD, Random only on exact data/IDs/split/L333/initialization/training/scales/sketch/objective and trajectory match. New CW-MaxDet and CW-IVR: 24 fits. Whitening-LCMD: 12 fits. Maximum 36 new formal fits; shared L333 reused. No extra control fits on reuse failure. Complete all six batches per seed regardless of early performance; prioritize two CW arms if resources block. Rotation and metric-matched CW only diagnostic, no formal fits.

All arms start shared L333 and independently consume only their own acquired labels. Scratch retrain with identical seed-specific initialization; existing QGeoGNN-V2 six-head model, Adam LR .001, weight decay 0, batch 2048, maximum 1000 epochs, patience 100, deterministic epoch shuffle, equal V1/V2 loss, validation combined NRMSE checkpoint. Fixed L333 target scales for original V1/V2 evaluation. Freeze batch IDs and premeasurement predictions before revealing labels. No new test evaluation before complete global matrix budget/checkpoint/prediction freeze. Historical exposed test is never a new blind test. Development validation comparison is explicitly exploratory.

## P0 geometry and compatibility

L333-only population scales: Raw D=diag(1/sV1,1/sV2); Rotation QD (45-degree orthogonal, no rescaling); CW rows (.5/sC,.5/sC),(-1/sW,1/sW); existing population whitening with absolute floor 1e-8 and relative floor 1e-6; H=sqrt(Tcw.T Tcw). Freeze transforms once. Small uncompressed gradient Gram audits Raw/Rotation and CW/H, no full dataset Jacobian saved. Projection seeds 17,101,1009; total dimensions 256,512,1024. Existing production 512D concatenated-output CountSketch with seed outer_seed+4000037 stays unchanged. Fixed finite sketch need not preserve rotation. Optional shared-projection two-block diagnostic retains total dimension and tests exact orthogonal invariance.

Use existing selectors: MaxDet current-L feature RMS; IVR fixed L+U universe RMS; LCMD historical distance geometry. Same-selector Raw/CW policies match; cross-selector differences cannot be attributed only to objective. Verify direct logdet and explicit covariance IVR, duplicates, canonical tie, L/U ordering, model state, historical Raw/CW selection regression. CW-IVR is a CW gradient row-kernel IVR proxy, not an exact multi-output posterior risk.

## P2 offline scope and future ablations

Keep historical CW16 pending + LLM16 and Full prompt frozen. Future comparisons: CW32, CW16+Random16, Full, No-narrative, No-direct-response, No-chemistry. Unified schema/allowlisted filters must preserve candidate/pending eligibility and frozen batch count. No-narrative retains all measured numeric responses/errors/time but removes free text, self-explanations and hypotheses. No-direct-response removes measured endpoints, errors, before/after changes, response-derived summaries and narrative; retains current predictions, geometry, IDs, conditions and chemistry. Trainer still uses labels: this is not no feedback. No-chemistry keeps numeric observations/predictions, conditions, stable anonymous groups and predefined numeric geometry, removes structure/names/descriptors and leaking text. Remaining model signals can have chemical origins. Coupled field removal limits causal attribution.

This round: offline fixture only, paid LLM calls=0. Future independent trajectories from L333, not replayed Full curves; two development seeds to525. Freeze provider/model/reasoning from Full (gpt-6-astra/high), context/prompt limits, retries, invalid-response policy and request/response model IDs; no automatic fallback or manual ID repair. Future run config must explicitly authorize finite cost/call ceilings; quota stops resumably. Pending-aware and deterministic condition controls deferred. If Full does not robustly beat CW32, do not claim overall label efficiency gains. Confirm with new measurement/test region or second system; selected-row error reduction is not generalization.

## Primary analysis and continuation rules

Primary metric frozen now: normalized trapezoidal validation AULC333–525. Secondary NRMSE@525, V1/V2 RMSE/MAE/R², paired seed differences. delta_s=AULC(CW,s)-AULC(Raw,s). Cross-selector differences are developmental interactions, no significance claims with two seeds. Report selection overlap/Jaccard and norms/rank/distances/structure-condition coverage; distinguish same-state diagnostics from separate trajectories.

Candidate threshold: mean AULC improvement >=2% relative within selector AND both seeds improve. One new selector passes: selector-specific confirmation eligibility; both new selectors pass: cross-selector candidate eligibility. Threshold is resource allocation, not statistical significance. Do not substitute favorable endpoint or test. If whitening matches/beats CW, describe output geometry correction, not CW-specific semantics.

Next stage planned only: freeze one candidate and strong Raw/CW/whitening controls; audit and preregister five unused split seeds before running L333→525. Extend to813/1005 only under subsequent prior budget. No rolling seed substitution. New splits are robustness, not external validation. Later add genuinely multi-output second system; single-output HPLC cannot validate CW multi-output geometry.

## Status and artifact owner

Study owner: `studies/active_learning/qgeognn_v2_row_cw_selector_factorial/`.

| Stage | Actual state | Evidence / next step |
|---|---|---|
| inventory | completed | 70 reused points: checkpoint, predictions, exact training/preprocessing/initialization and ordered trajectory checked |
| P0 | completed | 166 geometry records, four historical first-batch replays, 31 final regression tests passed |
| P1a | completed | 36/36 total completed fits; see RUN_STATE.json and RESULTS.md |
| P1b | completed | 36/36 total completed fits; see RUN_STATE.json and RESULTS.md |
| P2 offline | completed | four variants, numeric/text boundary and freeze tests; zero paid calls |

Required outputs: PROTOCOL.md, protocol.json, REUSE_MANIFEST.json, RUN_STATE.json, MECHANISM_AUDIT.md, RESULTS.md, per-seed CSV/PNG, costs and executable recovery commands. Every stage transition updates this table and state. Blockers record evidence and recovery commands, never masquerade as completion. Stop after P0/P1a/P1b/P2 offline; preserve historical files and ignored checkpoint policy. Commit plan before formal runs and commit code/results after verification; no force push.

## Completed outcome — 2026-10-09

P0, P1a, P1b and P2 offline completed. Exactly36 new formal fits; six independent trajectories each acquired192 labels over six batches. Seventy historical control points reused; all16 seven-point validation curves available. No extra/retry fits, no paid LLM requests by this study, no new test reveals. Delivery audit verifies every new checkpoint, prediction, initialization and premeasurement batch freeze. Final regression suite:31 passed.

CW relative mean validation AULC improvement: LCMD +6.9275% (both seeds improve), MaxDet −4.8558% (opposite directions), IVR +0.6881% (opposite directions). Neither new selector qualifies for confirmation under the registered gate. No new seed confirmation is launched. Whitening-LCMD minus CW-LCMD AULC is +0.056486/+0.020482 for seeds157/6101; CW is better on both here, but neither semantic exclusivity nor cross-selector universality follows. Report the observed LCMD-specific boundary with finite-sketch and normalization caveats.

Measured aggregate training wall time:5018.1s; acquisition:1680.9s. See resource_costs.csv, environment.json and RESULTS.md. All new training is stopped. P2 live independent ablations, five-unused-seed confirmation if separately justified, and second-system validation remain planned, not completed.

Next exact command is a verification-only handoff:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_delivery.py
```

Gate demonstration (expected blocked; does not train):

```sh
python scripts/studies/prepare_cw_confirmation.py --candidate cw_ivr --seeds 14653 18013 22349 27329 32719
```

The listed seeds are proposals, not a frozen eligible confirmation cohort: the candidate gate fails before unused-seed validation. P2's paid limits remain zero in P2_CONFIG.json and P2_FUTURE_RUN_TEMPLATE.json; a future live run needs its common adapted prompt/schema, retry/context limits and explicit positive bounded budget frozen first. This turn does not resume historical LLM jobs.

## Exploratory extension authorized — 2026-10-09

After seeing the completed 333–525 results, the user asked to run several more rounds. Freeze a bounded four-round continuation for ALL three new arms and both seeds: budgets557,589,621,653;24 additional fits maximum, separate from the completed36-fit primary study. Continue each own exact L525 selected set/checkpoint, then scratch retrain from the same L333 initialization. L333 output transforms, production sketch, selector normalization, optimizer/checkpoint rules and label firewall remain unchanged.

Extension primary descriptive metric is normalized validation AULC525–653, fixed before additional outcomes; secondary AULC333–653, NRMSE653 and per-target metrics. The original AULC333–525 result and failed new-selector confirmation gates remain unchanged. This is post-development exploration, not independent confirmation or a replacement success criterion. Complete all24 regardless of early performance and hard-stop at653. No paid LLM calls or new test reveals. Do not roll the budget forward to813/1005 without a new instruction and freeze.

Owner: `studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_653/`. Forty additional historical control budget points are available for exact protocol/checkpoint/ordered-lineage reuse; controls are not refitted. Source six L525 states and original global matrix freeze are hash-bound. Original study artifacts remain immutable. Current extension status: running; protocol and implementation committed as e31ff6e before training. Acquisition boundary regression passed. RUN_STATE.json and fit_ledger.json track the24-fit allowance; all four rounds per trajectory remain mandatory.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py run
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py report
```

## User-authorized extension to1005 — 2026-10-09

The user explicitly requested “都跑到1005把” while the653 continuation was running. All eight arms × two seeds will reach1005. Preserve the existing653 run/protocol and finish its24 fits; then run11 more batches for each of the six new trajectories (66 more fits), bounded90 additional fits from525,126 including the original36. Five historical control arms already reach1005;110 further budget points pass training, preprocessing, initialization, checkpoint, prediction and ordered-lineage reuse audits. No control refits. No paid LLM calls or new test reveals.

Owner: `studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/`. Status: planned, waiting for653. The frozen primary descriptive extension metric is validation normalized AULC525–1005; secondary AULC653–1005, AULC333–1005, NRMSE1005 and per-target metrics. This supersedes653 as the user-requested final budget, not the previously frozen scientific conclusions or intermediate metric. This remains post-development exploration. Complete all trajectories regardless of effect; hard stop1005. No automatic new-seed confirmation.

The653 stage retains its single-process execution. The subsequent stage permits two independent fits concurrently, two CPU threads each; seeds, model/loss/training/checkpoint policies, L333 transforms and production sketches are unchanged. All six653 anchors are hash-bound only after their full freeze and delivery audit. Per-trajectory fit ledger caps11, detects interrupted reservations and refuses silent retries.

```sh
# Supervises the existing653 process, then reports/audits and runs1005.
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/continue_cw_factorial_to1005.py
# Verification only after completion:
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_to1005.py
```
