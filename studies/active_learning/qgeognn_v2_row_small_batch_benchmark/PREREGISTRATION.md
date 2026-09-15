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
