# LCMD-L653 to Kernel-IVR: fixed-switch development study

## Critical design decision, before implementation or new outcomes

Use option C: a single fixed-653 proof of concept, followed only by separately
authorized research on validation-only switching. The scientific question is
whether IVR improves the remaining trajectory CONDITIONAL ON the exact state
created by LCMD, compared with continuing LCMD from that same state. Pure IVR
reaches 653 with a different selected set and different predictor; its better
endpoint does not establish that this intervention will work.

This is a post-hoc development hypothesis. The published LCMD test plateau and
N80 crossing motivated 653. Freezing the switch now prevents additional tuning,
but cannot undo test-driven design. The IVR AULC is 0.500070 versus LCMD 0.479539
(4.28% worse; only 1/5 paired wins), while its endpoint is 0.395913 versus
0.408614 and its cohort curve sustains N90 from 973. Five highly overlapping
row splits of one chemical dataset do not prove a general early/late law.
New seeds alone cannot supply independent confirmation. A future confirmation
requires untouched experimental/time-held-out data aligned with deployment.

Validation-only switching would avoid direct online test access but introduces
an additional stopping-rule search and repeated use of the same 416 validation
labels. Do not tune slope windows, thresholds, switch points, or policy variants
in this study. No RL, bandit training, or hidden-pool-label diagnostics.

## Frozen design

Seeds: 157, 887, 2357, 6101, 12203, exactly matched to sequential and pure IVR.
Source is frozen Gradient-LCMD, active budgets 333, 365, ..., 1005. Obtain the
switch round by locating 653 in the protocol schedule and replaying every
frozen source selection through its round input arrays. It is round 10;
its predictor is already fitted on 653 labels, and its outgoing batch produces
round 11 at 685. Eleven B32 acquisitions/fits remain, ending at round 21 / 1005.
The first 320 source-acquired labels are restored, never selected again.

Use FixedSwitchPolicy(before=gradient_lcmd, after=kernel_ivr,
switch_active_labels=653). The strategy state has no test metrics or hidden
labels. Validation and optional label-free diagnostics have explicit fields;
the fixed policy only consults label count. Acquisition receives only gradient
features and index partitions. No policy training is implemented.

## Provenance and reuse

All old studies are read-only. Save exact ordered L653/U653 IDs and indices,
membership/order hashes, dataset and graph hashes, split, scaler, preprocessing,
L0 target scales, initialization, training truth hash, full fit contract,
checkpoint/model hash, gradient hash and code compatibility in lineage/seed_*.json.

First try exact checkpoint plus gradient reuse. Otherwise reextract from the
verified checkpoint, or scratch fit only the exact L653 state. Never retrain
333..621. A missing or incompatible cache is an explicit recorded fallback;
corrupt ID lineage or incompatible predictor/source protocol fails closed.
Reuse decisions are sealed: a later mutation aborts, never silently changes plan.

The historical gradient module has appended alternative extractors. Compatibility
requires an unchanged original AST prefix (except importing typing.Mapping),
verified against the historical commit and its recorded SHA256. Only the two
reviewed appended function names are allowed. Existing predictor/data/training
and application primitives must match historical IVR code hashes exactly.
Current context must match historical context after this explicit code audit;
scaler bytes must also match. Recompute the original fit-contract hash with
exact known labels, validation labels, ordered IDs, initialization and config.
Validate the checkpoint schema, preprocessing, training config and state hash.

Historical round-10 features are ordered L653 then U653. Verify this order and
reindex by canonical sample indices into fixed R=L0+U0 order. Never assume a
row-10 matrix is in round-zero order. Preserve original U0 order in the remaining
candidate pool, including deterministic ties. Freeze current implementation,
numerical package versions, source lineage, baseline metrics and passing tests.

## IVR mathematics

Use the unchanged current full-network q50 endpoint-scaled 512D CountSketch.
The q50 heads use the existing MSE training objective. Let s^2=mean_R ||phi||^2,
X=phi/s, C=(I+X_L'X_L)^-1 and M=X_R'X_R/3330. Score each candidate by
(x'C M C x)/(1+x'C x). Use unit prior precision and unit observation variance;
perform 32 conditional greedy rank-one updates without any new labels.
R is the original 3330-row outer-training universe, uniformly weighted and
including labeled rows; validation/test inputs are excluded. This scalar row
kernel is a risk surrogate, not an exact joint V1/V2 posterior or guaranteed
RMSE reduction. Conditioning accepts any valid L/U partition, including LCMD's.

After every frozen selected batch reveal only those 32 labels, scratch retrain
with the original initialization/config and all current L, and extract new
gradients. No warm start. Validation cost remains an additional 416 labels.

## Evaluation, frozen before continuation

Primary contrast: paired late normalized partial AULC (653..1005) and endpoint
combined NRMSE against LCMD. Report effect sizes and all five paired directions;
no independent-seed significance claim or automatic success/follow-up rule.
Full AULC is secondary because the prefix is identical by construction.
Early (333..525), middle (525..653), late (653..1005) boundaries follow existing
row phase definitions. AULC means trapezoidal area divided by budget span,
without dividing by the initial error or full-data gap.

One summary includes Random, LCMD, Hybrid, pure IVR and LCMD->IVR for this exact
five-seed cohort. Exclude the two-seed MaxDet/Fusion development studies. Report
per-budget combined NRMSE, V1/V2 RMSE, NRMSE using fixed per-seed L0 scales, R2;
full/partial AULC; T_R30=Random@1005 and T80/T90/T95 using matched full-data gap;
first interpolated, first observed and sustained-through-final crossings;
1005 endpoint metrics and uncapped gap closure. Censor beyond 1005, never
extrapolate. Separate per-seed crossings and crossings of the cohort mean.
Sustained means only through this observed endpoint, not future persistence.

All five new continuations and their predictions must pass recursive content
and trajectory checks before explicit --report can read test truth. --run-all
does not report automatically. Reused metrics are historical development data.
Account separately for reused prefix work and new fits/epochs/training time,
gradient extraction and IVR time. Historical machine timing is descriptive.

## Current authorization and commands

This change authorizes only tests, provenance inspection and selection-only
smoke checks. No formal fit, full seed, or new test evaluation is run in setup.
The following are ready for a subsequent explicitly authorized execution:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --validate
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py --run-all
```

--execute-seed SEED resumes one shard, starting at 653. Seeds run sequentially
with two CPU threads each. With all five source models reusable: 55 new fits,
50 new gradient extractions and 55 IVR batches. A missing source model adds one
L653 fit and one extraction for that seed. Reporting is a separate later action.
