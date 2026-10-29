# 研究记录：qgeognn_v2_row_small_batch_benchmark

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# QGeoGNN-V2 Realistic Small-Batch Row Benchmark

Status: **PHASE 1 / PREREGISTERED / FORMAL EXPERIMENT NOT RUN**.

Commit A contains implementation, retrospective label-free audits, tests and
preregistration only. Stop after Commit A. Formal five-seed/ten-seed training,
formal test performance, FINAL_REPORT conclusions and Commit B are outside this
task. No formal decision or sequential-learning recommendation is available.

- [PREREGISTRATION.md](README.md#记录-4): hypotheses, frozen cohorts, methods,
  primary/secondary endpoints and decision gates.
- [PROTOCOL.md](README.md#记录-5): reproducible execution, label access, cache contracts
  and budget accounting.
- [AUDIT_INTERPRETATION.md](README.md#记录-2): retrospective **old B=333**
  input/gradient redundancy and label-free mechanism analysis.
- [PHASE1_VALIDATION.md](README.md#记录-3): tests and isolated smoke evidence.
- `protocol.json`, `splits/`, `artifact_manifest.json`: machine-readable freeze.
- `results/`: Phase 1 audits and budget accounting only.

The earlier `../qgeognn_v2_row_lcmd/` STRONG_POSITIVE study remains unchanged. It
is a **large-batch one-step mechanism proof**, not a realistic closed-loop study.
Its previously reported 31.69% NRMSE reduction is historical context supplied in
the handoff; its formal test performance was not re-read for this phase.

The new executable is
`scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py`. It is a thin CLI;
the implementation lives in `src/qgeognn_al/active_learning_v2/`.

Formal execution writes separate `formal_results/`, `secondary_results/`,
`formal_decision.json` and freeze records. This preserves the committed Phase 1
audit files and manifest. Such outputs are absent from Commit A. A later,
explicitly authorized Phase 2 must interpret the results, write FINAL_REPORT,
and create Commit B without changing this preregistration.

<a id="记录-2"></a>

## 记录 2：`AUDIT_INTERPRETATION.md`

# Retrospective Label-Free Audit

Population: the existing B=333 study's five development outer-training sets.
Only canonical X, label-scrubbed graphs, frozen L0 checkpoints, gradient sketches
and selected ID tables were read. No formal test metric or test-target access
was needed. No acquisition algorithm or hyperparameter was changed.

## Exact-X and Sketch Duplicates

| Seed | Outer rows | Unique model inputs | Duplicate groups | Excess duplicate rows | Mixed-X sketch groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| 73 | 3330 | 3112 | 209 | 218 | 0 |
| 311 | 3330 | 3113 | 204 | 217 | 0 |
| 1297 | 3330 | 3103 | 210 | 227 | 0 |
| 4093 | 3330 | 3104 | 209 | 226 | 0 |
| 8191 | 3330 | 3110 | 206 | 220 | 0 |

All exact gradient duplicates are explained by identical actual model-input
tensors. Their group counts and excess-row counts match exactly. There is no
observed different-X/same-sketch group requiring a CountSketch-collision or
local-symmetry/saturation explanation. This is evidence about exact collapse in
these five finite datasets, not proof that approximate collisions cannot occur.

Across the five LCMD batches, redundancy relative to L0 or an earlier selected
row is 0/333 in every seed. Random controls select 3-12 such redundant rows per
333 (0.90%-3.60%). Membership in any X-duplicate group is a different measure:
LCMD selects some members but does not acquire multiple already represented
copies. Avoiding redundant inputs is a plausible contributor, not by itself a
quantitative explanation of the previously reported performance gain.

The frozen B=333 LCMD algorithm was rerun on cached features for all five seeds.
All five ordered 333-ID selections match the existing artifacts exactly. Only
selection was replayed; no predictor was retrained and no formal metric read.

## Norm Mechanisms

Mean of the five within-U0 Spearman correlations (descriptive):

| Variable | Mean rho | Seed range |
| --- | ---: | --- |
| Predicted V2 q50 | 0.8931 | 0.8786 to 0.9024 |
| Predicted V1 q50 | 0.7668 | 0.6925 to 0.8467 |
| Nearest-L0 gradient distance | 0.6397 | 0.5847 to 0.6759 |
| Eluent logP | 0.5203 | 0.2603 to 0.6988 |
| Atom count | 0.1937 | 0.0552 to 0.2707 |
| Bond count | 0.2085 | 0.0594 to 0.3254 |
| MolWt | -0.1480 | -0.2346 to -0.0604 |
| Nearest-L0 condition distance | 0.0119 | -0.0456 to 0.0720 |

Across selected rows (five LCMD arms versus 25 Random controls), means are:

| Covariate | LCMD | Random |
| --- | ---: | ---: |
| Sketch gradient norm | 206.5137 | 119.9898 |
| Atom count | 12.0553 | 11.4489 |
| MolWt | 178.6460 | 182.4353 |
| Predicted V1 q50 | 11.9644 | 8.2478 |
| Predicted V2 q50 | 26.2335 | 17.3757 |
| Nearest-L0 condition distance | 0.2302 | 0.1986 |
| Nearest-L0 gradient distance | 125.9293 | 60.4803 |

The strongest label-free pattern is joint model sensitivity/gradient-space
novelty and **predicted retention-tail targeting**, with an eluent-condition
association. A simple "larger molecular weight" explanation is not supported:
MolWt correlates negatively and its selected mean is lower for LCMD. Atom/bond
count effects are modest. Direct condition novelty has near-zero norm
correlation even though its selected mean is somewhat higher for LCMD.

These are associations from model predictions and the sketch norm. They do not
establish true high retention, baseline error, causal mechanism, or small-batch
superiority. True-label/error diagnostics remain deferred. Correlated eluent
descriptor columns are not independent mechanisms; constant hydrogen-donor
values have undefined rho and are explicitly flagged. No diagnostic here is a
selection threshold or tuning signal.

Source file hashes and the no-test-access statement are recorded in
`results/retrospective_audit_provenance.json`. Full group and correlation tables
are the three requested audit CSVs, with selection profiles in a fourth CSV.

<a id="记录-3"></a>

## 记录 3：`PHASE1_VALIDATION.md`

# Phase 1 Validation

Status: PASS. The machine-readable execution counts and hashes are in
`results/phase1_validation.json` and `results/smoke_report.json`.

## Verification

- Full repository unit/integration suite: **489 passed**, zero failures, errors
  or skips. Includes **28 active_learning_v2 tests**. Existing dependency
  deprecation and graph node-count warnings remain; they did not fail tests.
- Real hidden-label mutation: canonical 4g feature/target fixture, 160 rows,
  fixed seed 29; original and independently rebuilt mutated target stores train
  actual current-V2 models. Hidden U0/test V1/V2 values are replaced by
  deterministic extreme values. L0/validation labels, all X and IDs are retained.
  Checkpoint semantic hash, gradient hash, latent hash, K=3 prediction hash and
  every ordered acquisition-ID list/hash are identical. Provenance/source and
  fit-contract hashes differ as intended. Reusing the original cache with the
  mutated store is rejected; changing authorized L0 training truth also rejects
  a completed fit cache.
- Full small fixture smoke: preprocessing, scrubbed graph cache, baseline,
  K=3 ensemble, gradients, representations, eight B32 arms, six B16 and six B64
  arms, label-free diagnostics, same-initialization retraining, artifact freeze,
  prediction identity verification and held-out evaluation all execute.
  Held-out targets are synthetic constants installed before numeric parsing.
  No formal test metric is used. Seed 29 and the two-epoch cap are fixed
  engineering fixtures, not candidate experimental hyperparameters.
- TP CoreSet semantics, stable uncertainty ranking and quantile-width exclusion,
  all exact batch sizes, independent deterministic Random controls, dual budget,
  fixed cohorts, shared split identities and initialization hashes are tested.
- Missing seed freeze, corrupted frozen prediction, mismatched or partial cache,
  pre-Commit-A execution, and premature secondary execution are rejected.
- Decision gates are tested using synthetic metric tables, including the exact
  2% endpoint boundary and incomplete cohort rejection. No formal metrics are
  consulted to test the gate.
- Old B=333 regression: all five ordered 333-ID sequences match cached prior
  selections. Old study tracked artifacts and the frozen gradient/LCMD source
  have no diff against the base commit. Retrospective source hashes are verified
  again at sealing.

## Scope of the Result

The formal runner's production fitting/acquisition/freeze/evaluation components
are exercised end to end on a real small feature fixture; its release guards
and decision rules are separately tested. This verifies executable plumbing and
label isolation. It does not establish full-scale convergence, formal wall time,
small-batch performance or scientific superiority. No 5-seed/10-seed formal
benchmark, confirmation training, formal FINAL_REPORT or scientific decision
has been generated.

An early test run passed 487 tests before two additional release-gate tests.
The final complete suite is the 489-test run recorded by the sealing command.
The first CLI smoke also passed; final smoke was rerun after engineering guard
and audit-output changes so its code hash matches the frozen implementation.
No method, seed, batch, representation or training parameter was changed based
on smoke metrics.

## Reproduction and Seal

```bash
conda run --no-capture-output -n fish python -m pytest -q --junitxml=/tmp/qgeognn_small_batch_phase1_tests.xml
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --smoke /tmp/qgeognn_v2_small_batch_phase1_smoke_final
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --seal-phase1 --test-report /tmp/qgeognn_small_batch_phase1_tests.xml --smoke-report /tmp/qgeognn_v2_small_batch_phase1_smoke_final/smoke_report.json
```

The seal is created only after preparing the final code hashes and completing
the audit. It checks passing test/smoke records and absence of formal outputs.
All scientific choices are frozen before Commit A; after Commit A the task stops.

<a id="记录-4"></a>

## 记录 4：`PREREGISTRATION.md`

# Preregistration

## Status and Scope

Frozen on branch `exp/qgeognn-v2-4g-row-al`, starting from
`8081e8653fd0539554ca42b01482f1493a0dc3fa`. Commit A has the exact subject:

`study: preregister realistic-batch V2 row active learning benchmark`

This is Phase 1. No formal small-batch experiment or formal test performance is
available. The future command requires the actual Commit A SHA and checks its
committed artifact manifest, code, source data, graph cache and split hashes.
The manifest cannot contain its own commit SHA; Git supplies that binding.

Questions:

1. Does Gradient-LCMD retain an advantage over Random at B=32?
2. At matched predictor, split, initial labels, batch, training and evaluation,
   does it beat epistemic uncertainty and latent-space CoreSet?

This study is one-step acquisition. It cannot establish the benefit of repeated
sequential adaptation. A future sequential study must retrain and recompute
acquisition features at each step, not slice a single initial ranking.

## Existing Evidence and Independence

The completed B=333 result is preserved as a **large-batch one-step mechanism
proof**. Its handoff reports STRONG_POSITIVE, 5/5 outer-seed wins, 25/25 comparisons
against Random controls and 31.69% mean combined NRMSE reduction. No files in that
study are changed, and its formal metrics are not inputs to this implementation.

Development/matched seeds: **73, 311, 1297, 4093, 8191**.

Untouched confirmation seeds: **157, 887, 2357, 6101, 12203**.

All ten seed identities and partitions are committed before formal execution.
Confirmation is the primary decision cohort; development is a separately
reported methodological comparison. The old five seeds informed method choice
and must not be described as independent unseen-seed confirmation. Untouched
refers to these small-batch benchmark runs: no confirmation model fitting,
acquisition performance or test outcomes are inspected in Phase 1. This does not
claim that the underlying canonical dataset has never been studied.

Within each cohort, the outer seed is the unit of comparison. Five Random
controls are conditional replicates, not 25 independent outer seeds. Cohorts are
not pooled to change a decision. Overlapping canonical rows across splits also
preclude interpreting the seeds as independent new laboratory datasets.

Future execution trains confirmation seeds first, then development. All ten
primary seed acquisitions, checkpoints and predictions must freeze before the
first primary test-truth reveal. If interrupted or resource-limited, resume the
same matrix; incomplete matrices receive no formal decision. No replacement
seeds or reduced-control fallback are permitted.

## Split, Labels and Costs

Canonical 4g: 4,163 rows. Fixed row split for each seed:

| Role | Rows |
| --- | ---: |
| Outer training | 3,330 |
| Initial L0 | 333 |
| Initially hidden U0 | 2,997 |
| Shared validation | 416 |
| Test | 417 |

Use the existing `make_row_protocol` integer rules and RNG, preserving canonical
row order. All arms within a seed share the same partition and initial L0.
Validation belongs to the observed budget but never to acquired training rows.
This is a row split: compounds may overlap across roles. Conclusions concern
row-level prediction under this protocol, not unseen-compound generalization.

| Stage | Active training labels | Shared validation | Total observed non-test |
| --- | ---: | ---: | ---: |
| L0 | 333 | 416 | 749 |
| L0+B16 | 349 | 416 | 765 |
| L0+B32 | 365 | 416 | 781 |
| L0+B64 | 397 | 416 | 813 |

Every arm metric and learning-curve-compatible table carries
`active_label_count`, `active_label_fraction_of_outer_train`,
`shared_validation_label_count`, `total_observed_label_count`,
`total_observed_fraction_of_full_dataset`. Fractions use 3,330 and 4,163,
respectively. Per-arm costs are counterfactual experimental budgets; the union
of labels simulated across competing arms is not the cost of one method.

Do not claim "only 10% of experimental data" without the validation cost. Shared
validation preserves fairness between methods but consumes 416 observed labels.

## Batch Sizes

Primary: **B=32**, exactly 32 distinct U0 rows for every arm.

"B=32 is used as a realistic small-batch proxy pending direct platform throughput
logs." It is an engineering proxy, not a claim of exactly 32 experiments per day.
The old 333-row batch was 10% of outer training, not throughput-derived; such a
large acquisition leaves less opportunity for adaptation between retraining
steps. This motivates B=32 without establishing a daily platform capacity.

Secondary: **B=16 and B=64**, Random versus LCMD only, with five controls and the
same ten seeds. Execute only after all primary results are frozen. Each is a
fresh one-step acquisition from the same L0, not a continuation from B=32.
No secondary result can replace the primary B or alter any method. Relative
advantages across B describe batch sensitivity; even a B16>B32>B64 pattern alone
does not prove sequential adaptivity because this experiment has only one step.

## Matched Methods

1. **Random**: five deterministic, independently seeded uniform samples without
   replacement from all U0 rows. Controls may overlap each other.
2. **Ensemble uncertainty**: K=3 independently initialized current QGeoGNN-V2
   models trained on the same L0, with shared validation for checkpointing.
   Member 0 reuses the baseline checkpoint; members 1 and 2 are independent
   initializations. For endpoint population standard deviations (ddof=0),
   `u = sqrt((std(V1_q50)/s_V1)^2 + (std(V2_q50)/s_V2)^2)`.
   Select the highest B scores. Scales are derived from L0 labels only. No
   quantile-width term, shortlist or legacy ensemble is used.
3. **Current-V2 CoreSet**: `model.extract_representation` is the authoritative
   128D vector after fixed sum pooling plus the current condition residual and
   before the prediction head. No normalization or whitening is added. Install
   every L0 vector as a center, then greedy farthest-first Euclidean k-center
   over the complete U0, adding each new point as a center. No uncertainty
   shortlist is allowed.
4. **Frozen Gradient-LCMD**: retain the existing full-network q50 Jacobian of
   `V1_q50/s_V1` and `V2_q50/s_V2`, concatenated and projected into 512D by the
   same deterministic CountSketch. Preserve squared Euclidean geometry,
   corrected LCMD-TP cluster score (sum of squared nearest-center distances),
   existing L0 centers, maximum-distance point in the largest-score cluster,
   and canonical-order tie resolution. No gradient clipping/normalization,
   sketch dimension change, norm filter or feature reweighting is introduced.

All acquisition functions receive only label-free arrays and fixed parameters.
Stable ties use the first candidate in canonical U0 order. Selected batch order
is retained in training identities. Five-seed B=333 regression checks verify the
existing LCMD selected ID sequences are unchanged.

## Frozen Predictor and Randomness

Standalone QGeoGNN-V2, CPU, two torch threads. Model architecture, condition
branch, head, six outputs and training loss remain as qualified previously.
Adam: lr=0.001, weight decay=0; training batch=2048; maximum epochs=1000;
patience=100. Same validation combined normalized RMSE checkpoint rule.
No test during training, predictor tuning or warm-start evaluation is allowed.

For outer seed s and ensemble member j in {0,1,2}:

- Initialization: `s + 2_000_003 + j * 1_000_033`.
- Training shuffle seed: `s + 3_000_017 + j * 1_000_033`.
- Epoch order: existing `default_rng(training_seed * 10000 + epoch)`.
- Sketch seed: `s + 4_000_037`.
- Random control c in {0,...,4}: `s * 1_000_003 + 50_000 + c`.

Every evaluated baseline/after-acquisition arm uses j=0 and restarts from the
same initial parameter state and RNG seed. Ensemble members are acquisition
models only. Evaluation never uses ensemble averaging. Preprocessing is fitted
once to outer-training X; L0 labels alone determine endpoint scales. L0 target
scales remain fixed after acquisition for both checkpointing and evaluation.

## Metrics and Decision

Primary test metric:
`combined_normalized_RMSE = sqrt(0.5*((RMSE_V1/s_V1)^2+(RMSE_V2/s_V2)^2))`.

Secondary metrics: V1/V2 RMSE, MAE and R2, all from q50 predictions. Gain is
baseline NRMSE minus after-batch NRMSE. Report each non-Random method's gain minus
the within-seed Random mean gain and Random median gain, directional wins, and
LCMD pairwise comparisons with uncertainty and CoreSet. Random mean aggregates
five controls per seed, then gives all five outer seeds equal weight. Ties are
not wins. No p-value threshold or post-hoc significance gate is introduced.

Apply these gates separately to each complete five-seed cohort. Only the
confirmation decision supports the study's primary classification.

**BEST_CURRENT_ROW_ACQUISITION** requires all of:

- LCMD mean after-batch NRMSE is strictly below Random mean, uncertainty mean
  and CoreSet mean.
- LCMD beats the within-seed Random median in at least 4/5 seeds.
- LCMD beats each of uncertainty and CoreSet in at least 3/5 seeds.
- For each endpoint separately, LCMD mean RMSE is at most 1.02 times the lowest
  competitor mean endpoint RMSE (Random controls collapsed to per-seed means,
  uncertainty, or CoreSet). This conservative endpoint definition is fixed now.

Otherwise, **LCMD_BEATS_RANDOM_NOT_OTHER_BASELINES** requires LCMD mean NRMSE
below Random mean and at least 4/5 wins against Random medians. This category
does not certify superiority over the deterministic baselines or the endpoint
guard. If this Random gate fails, classify **LARGE_BATCH_ONLY_SIGNAL**. The name
describes the evidence boundary; it does not prove benefit is impossible at
every other small batch. Missing/invalid artifacts block a decision.

## Audits and Interpretation

Phase 1 audit population is the old five development outer-training sets only.
Read existing L0 checkpoints, cached sketches and selections; use canonical X
with graph labels scrubbed. Do not load formal test metrics or targets. Compute
exact-X/model-tensor duplicate groups, sketch duplicate groups and explanation
counts, and acquisition redundancy relative to L0 plus earlier batch rows.
For any different-X/same-sketch group, compare unsketched full Jacobian hashes
to separate projection collision from identical local Jacobians. Identical
Jacobians alone cannot distinguish saturation from architectural symmetry.

Mechanism variables: sketch L2 norm, atom/bond counts, MolWt, predicted q50
V1/V2, all eluent/condition variables, separate density/sample volume and
nearest-L0 condition/gradient distances. Spearman correlations are calculated
over U0 separately per seed; report selection profiles for all old arms.
Solvents use indicator contrasts, not an ordinal interpretation of codes.
Constant variables produce missing correlations with an explicit flag.
P-values are descriptive, uncorrected and not independent-row causal evidence.
The same label-free audit runs before label reveal in the future benchmark.

True V1/V2 and baseline absolute-error correlations are deferred to explicitly
authorized post-hoc Phase 2 analysis after relevant acquisitions/predictions
are frozen. They may not change selection, seeds, thresholds or parameters.
Association with predicted retention is not evidence of high *true* retention
or acquisition benefit. Correlation and redundancy avoidance do not establish
which mechanism causes test improvement.

Historical E2/A1a/E4/A2a strategies used different predictors, seeds, batches
and protocols; their numbers are not matched superiority comparisons. The
future report must discuss that limitation. Only this current-V2 matched
benchmark addresses representative uncertainty versus coverage versus gradient
principles directly.

## Stop Rules and Later Work

Phase 1 smoke uses 160 deterministic canonical feature rows, separate seed 29,
two epochs and synthetic held-out targets. It checks execution, not performance;
do not tune any method from its metric output. Hidden-label mutation tests may
perturb U0/test target stores in fixtures without inspecting formal performance.

Commit A, then STOP. No automatic launch, automation, FINAL_REPORT conclusion,
scientific decision or Commit B is authorized in Phase 1. No failure to achieve
a desirable future outcome permits changes to B, seeds, K, uncertainty formula,
representation, CountSketch, predictor or early stopping. A genuine bug requires
a recorded description, affected artifacts and explicit invalidation/restart
scope in Git before any further formal run.

A future NEXT_STAGE_RECOMMENDATION may be created only if the confirmation
Random gate passes. It must propose genuinely sequential B=32 recomputation
(333 -> 365 -> 397 -> 429 -> ...), without executing it in this study.

<a id="记录-5"></a>

## 记录 5：`PROTOCOL.md`

# Execution Protocol

## Phase 1 Boundary

Run preparation, retrospective label-free audit, unit/integration tests and
isolated smoke. Commit with the preregistered subject, then stop. The Phase 1
manifest includes docs, protocol, ten splits, retrospective outputs and test
evidence. No formal result tables, decision, FINAL_REPORT or Commit B are
generated in this phase.

The authoritative scientific choices are in PREREGISTRATION.md and protocol.json.
Seed split hashes and L0/U0/validation/test identity hashes are in
`splits/split_manifest.json`. Existing B=333 study artifacts are read-only.

## Module Boundaries

| Module in active_learning_v2 | Responsibility |
| --- | --- |
| benchmark_protocol.py | Constants, seed schedules, budgets, preparation, Commit A guard |
| benchmark.py | Across-seed barriers, primary/secondary orchestration, isolated smoke |
| benchmark_seed.py | Source/X setup, scrubbed graphs, L0/ensemble fits, acquisition freeze, arm fits |
| acquisition.py | One four-method interface with exact batch validation |
| uncertainty.py | K=3 q50 disagreement ranking |
| coverage.py | Current representation extraction and TP k-center |
| gradient_features.py, lcmd.py | Existing frozen Jacobian sketch and LCMD implementations |
| cache.py | Array semantics and cache contract/content validation |
| runner.py | Shared preprocessing, authorized-label installation and same-initialization training |
| diagnostics.py | X/sketch duplicates and label-free mechanism audits |
| benchmark_reporting.py | Post-freeze metrics, paired comparison, decision and plotting |
| phase1.py | Validate Phase 1 evidence and seal the committed artifact manifest |

The new runner is a thin CLI. The old 1,000+ line pilot is neither extended nor
imported as the new orchestration layer. The old gradient and LCMD algorithms
remain untouched. Existing training cache receipts intentionally become
incompatible if they lack the new, stronger fit contract; old result artifacts
are not migrated or regenerated.

## Label Access and Ordering

1. Read canonical CSV with an explicit X-only `usecols` list: sample_id,
   canonical_smiles, density, sample volume, loading solvent, loading-solvent
   volume and PE/EA. Validate row identity and molecule order against the split.
2. `RestrictedLabelStore` parses only requested, authorized targets. Read L0 and
   validation targets for initial fitting; U0 and test targets remain hidden.
3. Fit graph-descriptor and eluent min/max scalers on outer-training X only.
   Loading amount/volume ranges also use outer-training X. Compute V1/V2
   population standard deviations on L0 truth alone. Freeze these scales.
4. Build model graphs with y=[0,0] for every row. Identity positions never become
   predictors. Only clones at authorized train/validation positions receive
   target values. Cached graphs must retain all-zero target sentinels.
5. Train baseline j=0. Extract gradients over canonical L0 followed by canonical
   U0; extract current-V2 representations in the same order. Train j=1 and j=2
   on identical L0/validation. Ensemble pool predictions must align by sample ID.
6. Freeze every within-seed acquisition batch and ordered selected IDs before
   revealing selected U0 targets. Diagnostic computations are label-free.
7. Train each evaluation arm from j=0 initialization on L0 plus its own selected
   labels. Enforce common L0/validation/test identity hashes and identical
   evaluation initialization hashes. Freeze checkpoints and test predictions.
8. Across all ten primary seeds, verify every receipt before writing the global
   pre-test freeze. Only then can a new restricted store unlock test targets.
   Verify source provenance and prediction identity before metric computation.
9. Aggregate confirmation and development separately. Freeze primary outputs.
   Secondary execution requires that complete primary freeze and matching
   Commit A SHA. No secondary acquisition function accepts test metrics.

The five Random controls and three non-Random methods therefore produce eight
after-batch arms at B=32. Baseline plus these arms share one evaluation
initialization. Two additional ensemble fits use distinct initializations solely
for acquisition. There are eleven fits per primary seed (baseline + 2 ensemble
+ 8 arms); each secondary batch adds six arm fits, reusing identical L0 caches.

## Cache and Freeze Contracts

- Raw graph cache is the canonical frozen 4g cache and is SHA256-verified.
- Scrubbed graph cache binds source/store SHA256, canonical ordered IDs,
  partition, code hashes, package versions, preprocessing, normalization and
  smoke/formal mode. Partial or incompatible graph/gradient caches are rejected.
- Completed fit caches bind context hash, arm, ordered train/validation IDs and
  indices, their authorized target array hashes, prediction IDs/order,
  initialization seed/state hash, training config, normalization and
  preprocessing. Verify checkpoint and prediction file SHA256 before reuse.
- Gradient caches additionally bind baseline checkpoint file hash, ordered outer
  IDs, target scales, sketch seed, dimension=512 and feature-file SHA256.
- Model `checkpoint_state_hash` identifies tensor names, types, shapes and
  values. Checkpoint file SHA256 also covers serialization/provenance. A hidden
  target-store mutation must change provenance/fit contracts but cannot change
  the trained semantic state, sketches or selected IDs.
- Files existing alone are not evidence of compatible completion. No silent
  cache reuse or fallback to a different model/representation is permitted.
- `assert_formal_authorized` requires Commit A subject, ancestry, committed
  manifest equality and frozen file hashes; current protocol, packages, Python,
  source data, graph cache and code must match the preregistered record.
- Primary execution refuses to overwrite an already frozen primary result.
  An interrupted run can resume compatible completed fits under the same code
  and protocol. Genuine incompatible changes require documented invalidation.

Execution is deliberately serial with CPU threads=2. This avoids resource races
while retaining deterministic cache reuse. No concurrency or training-semantics
change is required to perform this benchmark.

## Input-Equivalence Audit

The exact model-input key hashes typed atom x, atom/bond edge indices and edge
attributes, bond-angle edge indices and edge attributes (including scaled
descriptors). Labels, sample IDs and canonical positions are excluded. Separate
component hashes cover canonical molecular graph and raw descriptors; raw X
identities retain canonical_smiles, PE/EA, solvent, density, sample volume and
loading volume. Density and sample volume are recorded independently, although
the model consumes their product. PE/EA representations are the actual six
eluent descriptors and their fitted transforms. All nine intended condition
inputs, including categorical solvent and continuous completion-branch inputs,
are represented in the tensors.

`exact_X_excess_rows` counts rows beyond one representative per tensor identity.
`gradient_excess_rows` uses exact numerical equality of the 512D feature rows.
`selected_redundant_fraction` counts a selected row when its identity already
occurs in L0 or earlier in that ordered acquisition. A separate fraction counts
membership in *any* duplicate group, even when its peers were never selected.
These quantities must not be conflated. Mixed-X sketch groups trigger a
full-Jacobian check only for representatives of distinct input identities.

The recorded gradient norm is **the L2 norm of the 512D scaled CountSketch**, not
the norm of the unprojected network Jacobian. The old study used this convention;
it is retained and explicitly labeled. Nearest-L0 gradient distance uses the
same sketch geometry. Condition distance standardizes the frozen 9D condition
matrix on outer-training X with population mean/std (constant dimensions get
scale 1), then uses Euclidean distance to L0. It is an audit covariate, never an
acquisition input for these four methods.

## Commands

Use the existing `fish` environment. The local verified interpreter is
`/Users/fish/miniforge3/envs/fish/bin/python` (Python and package versions are
recorded in environment.json). Commands below run from the repository root.

Phase 1 only:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --prepare
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --audit-existing
conda run --no-capture-output -n fish python -m pytest -q
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --smoke /tmp/qgeognn_v2_small_batch_phase1_smoke
```

The standalone smoke and integration tests use the same SeedContext,
acquisition, fit, freeze-validation and evaluator code as the formal runner.
The fixed fixture takes 160 evenly spaced canonical X rows and uses seed 29.
It replaces held-out target strings with synthetic values before float parsing;
formal test performance is never read. Full architecture, 512D sketches, K=3,
exact B=32/16/64 and five controls remain in force, with only training capped at
two epochs. Smoke metrics are plumbing output in an isolated directory, not
scientific evidence, and are not copied into formal result tables.

Future Phase 2 only, after explicit authorization (replace COMMIT_A_SHA):

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --execute-primary --preregistration-commit COMMIT_A_SHA
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py --execute-secondary --preregistration-commit COMMIT_A_SHA
```

No formal per-seed shortcut or configurable seed/batch/K/epoch override is
exposed. A missing Commit A argument is rejected. The secondary command rejects
missing or changed primary freezes. Run primary to completion first.

## Outputs and Review

Phase 1 `results/` contains input_redundancy_audit.csv,
gradient_duplicate_audit.csv, gradient_mechanism_correlations.csv,
gradient_mechanism_selection_profiles.csv, lcmd_333_regression.csv, provenance,
label_budget_accounting.csv and validation evidence. These refer to the old
development L0 models or the explicitly identified fixture, not formal B=32.

Future primary metrics and diagnostics are written to `formal_results/` with
per-arm and per-seed identifiers and B32 suffixes. Checkpoint, initialization,
label-access and runtime audits are retained per seed and aggregated. Primary
ranking plots go to `figures/`. `formal_decision.json` explicitly separates the
confirmation primary decision from descriptive development outcomes. Secondary
metrics go to `secondary_results/`; sensitivity includes per-seed B16/B32/B64
Random means, LCMD NRMSE, relative improvements and median-based wins.

Later scientific reporting must explain endpoint tradeoffs, small-batch versus
old large-batch evidence, validation-inclusive label costs, uncertainty and
coverage comparisons, mechanism limitations and historical comparability. It
must not infer causal mechanisms from correlations, make a legacy numerical
superiority claim or generate a sequential recommendation before its gate.
