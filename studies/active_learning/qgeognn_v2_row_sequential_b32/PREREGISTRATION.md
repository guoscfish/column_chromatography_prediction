# Preregistration

## Scope and question

This is a **sequential extension on the established confirmation cohort**, not
a new untouched-seed confirmation. It asks how many active training labels are
needed to reach the same QGeoGNN-V2 prediction level and how many new column
chromatography experiments Hybrid or Gradient-LCMD save relative to Random.

The study starts from repository commit
`4ec81c9fadb6617fd57c6729d81645752c8d7668` on branch
`exp/qgeognn-v2-4g-row-al`. Commit A has the exact subject:

`study: preregister sequential B32 active learning efficiency`

Commit A contains no formal trajectory fit and no formal test metric. The
formal command requires the actual Commit A SHA and validates its committed
manifest, protocol, code, data, graph cache, packages, and split hashes.

## Cohort, split, and budget

Only outer seeds **157, 887, 2357, 6101, 12203** are used. Each keeps the
established canonical 4g row split: 3,330 outer-training rows, 416 shared
validation rows, and 417 test rows. Outer training starts as 333 labeled L0
rows and 2,997 hidden U0 candidates. No development-cohort run is permitted.

Batch size is exactly 32. The 22 registered learning-curve budgets are:

`333, 365, 397, 429, 461, 493, 525, 557, 589, 621, 653, 685, 717, 749, 781, 813, 845, 877, 909, 941, 973, 1005`.

There are 21 complete acquisition rounds and no partial batch. This ceiling is
frozen before test reveal, and test performance cannot stop or extend it. Every
output records active labels, active fraction of 3,330, shared validation count,
total observed non-test labels, and total observed fraction of 4,163. Thus the
initial and final total observed counts are 749 and 1,421.

## Compared methods

Only these three methods are allowed:

1. **Random** uses one deterministic permutation of original U0 per outer seed,
   seeded by `outer_seed * 1_000_003 + 900_001`. Round r takes the next 32-row
   slice. It is nested and is never chosen from the previous five one-step
   controls.
2. **V2-Hybrid** trains a current K=3 ensemble on every L_t. Evaluation member
   0 is reused for acquisition; members 1 and 2 use the established independent
   deterministic initialization seeds. Current U_t uncertainty is population
   standard deviation of q50, combined as
   `sqrt((u_V1/s_V1)^2 + (u_V2/s_V2)^2)`. The exact ceil(25%) highest-current-
   uncertainty shortlist is passed to current-V2 128D latent farthest-first,
   with every current L_t representation installed as an existing center.
3. **Gradient-LCMD** recomputes the full-network q50 Jacobian for current L_t
   and U_t every round, uses L0-fixed endpoint-scale normalization, the frozen
   deterministic 512D CountSketch, squared Euclidean geometry, and corrected
   LCMD-TP. Every current L_t gradient vector is an existing center.

No acquisition receives U_t truth or test truth. No old ranking is sliced for
Hybrid or LCMD. No additional method, adaptive mixture, transfer learning, or
predictor change is allowed.

## Predictor and randomness

Every method and curve point trains a standalone QGeoGNN-V2 evaluation model
from scratch. Architecture, six-output head, optimizer, learning rate, batch
size, 1,000 maximum epochs, patience 100, deterministic epoch shuffle, and
validation combined-normalized-RMSE checkpoint selection match the completed
B=32 study. For seed s and member j:

- evaluation/ensemble initialization: `s + 2_000_003 + j * 1_000_033`;
- training shuffle seed: `s + 3_000_017 + j * 1_000_033`;
- gradient sketch seed: `s + 4_000_037`.

All evaluated models use j=0. Ensemble averaging is prohibited for evaluation.
Covariate preprocessing is fixed per outer split. Endpoint scales are L0
population standard deviations (`ddof=0`) and remain unchanged through the
entire trajectory. Validation is shared and test is never used for training,
early stopping, checkpoint selection, acquisition, or budget selection.

One matched full-data reference per seed trains on all 3,330 outer-training
labels with the same predictor, member-0 initialization, validation protocol,
and original L0 endpoint scales. It never participates in acquisition.

## Primary metrics

For each seed, method, and budget, define `E_s,m(N)` as test combined normalized
RMSE. Test metrics are computed only after the global pre-test barrier.

**Normalized AULC** uses trapezoidal integration from N=333 through N=1005:
`integral(E(N), dN) / 672`. Lower is better. Report per-seed values and method
mean, median, and population standard deviation, plus paired differences
Hybrid-Random, LCMD-Random, and LCMD-Hybrid.

**Primary target T_R30** is cohort-mean Random NRMSE at N=1005. Random is
defined to require 1005 labels. For Hybrid and LCMD, report the first crossing
on the cohort-mean curve, using linear interpolation only between adjacent
registered points. If absent, report `>1005 / censored`; never extrapolate.

For reached methods, total saving is `(1005-N_target)/1005`. The primary
incremental new-experiment saving treats L0 as sunk cost:
`(1005-N_target)/672`. Report both the saved experiment count and percentage.

**Secondary target T_80** is
`E_full + 0.20 * (E0 - E_full)` on cohort means. The same first-crossing and
censoring rules apply. Endpoint RMSE, MAE, and R2 are secondary diagnostics and
cannot determine active-learning superiority.

## Frozen decision logic

Classify `SEQUENTIAL_LCMD_BEST` only when LCMD mean normalized AULC is below
both Random and Hybrid, LCMD incremental saving is positive, LCMD labels-to-
T_R30 is no greater than Hybrid's, and LCMD AULC beats Random in at least 4/5
paired seeds.

Classify `SEQUENTIAL_HYBRID_BEST` when Hybrid AULC is below LCMD and Hybrid
labels-to-T_R30 is lower than LCMD (including an LCMD-censored result), provided
there is a preregistered active-learning gain over Random.

Classify `SEQUENTIAL_LCMD_HYBRID_COMPARABLE` when their mean AULC relative
difference is strictly below 2% and labels-to-T_R30 differ by at most one batch
(32 labels), provided there is a preregistered active-learning gain.

If neither active strategy improves Random in AULC or has positive label saving,
or no preceding positive classification applies, classify
`NO_CLEAR_SEQUENTIAL_AL_GAIN`. An incomplete formal matrix cannot be interpreted;
resource exhaustion is reported as `BLOCKED_RESOURCE_LIMIT` without Commit B.

## Evidence boundary and reuse

Each outer seed has one Random trajectory, so within-seed Random stochasticity
is less fully estimated than in the prior five-control one-step benchmark. The
five outer seeds still provide paired trajectory robustness. Prior Round-1
results may be background evidence but are not joined onto these curves.

The prior B=32 and Hybrid artifacts were audited. They are not imported because
they lack this study's Commit-A, ordered current-L_t/current-U_t, per-round, and
resume contracts. Formal Round 1 is recomputed for every arm. Existing study
files remain read-only.
