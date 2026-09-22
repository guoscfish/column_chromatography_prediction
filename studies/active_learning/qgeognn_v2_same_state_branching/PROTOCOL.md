# Same-state short-rollout branching protocol

## Question and evidence class

This is a development/mechanism study on seeds 157 and 6101, not independent
confirmation. It asks whether acquisition mechanisms have different two-batch
future returns from the exact same state, and whether their ranking changes with
budget or source trajectory. No controller, threshold search, reinforcement
learning, contextual bandit, ensemble uncertainty, or fixed 653 switch is fit.

## Frozen factorial design

- Source trajectories: frozen Gradient-LCMD and frozen Kernel-IVR.
- Anchor budgets: 429 and 653, located from the source schedule (rounds 3 and 10).
- Candidate strategies: Gradient-LCMD, Kernel-IVR, Gradient-MaxDet.
- Rollout: acquire B32, reveal only that batch, scratch retrain; acquire B32 from
  the branch-specific updated checkpoint/state, reveal only that batch, scratch
  retrain. Endpoints are 429/461/493 and 653/685/717.
- Total new fits: 2 seeds x 2 sources x 2 budgets x 3 strategies x 2 = 48.

For a given seed/source/budget the ordered L, ordered U, checkpoint, gradient
bank, candidate pool, preprocessing, scales, initialization/config, split and
label-scrubbed graph cache are identical across all three first-round branches.
Every identity and content hash is recorded. Historical artifacts are read-only.

The shared representation is the current model's full-network endpoint-scaled
q50 gradient 512D CountSketch. Method definitions remain historical rather than
being silently homogenized: LCMD uses raw squared Euclidean geometry; IVR uses a
single RMS scale over all 3330 outer rows with unit prior/noise; MaxDet uses the
current-L RMS scale and conditional D-optimal log-determinant gains. Thus the
representation is controlled while method-defining normalization is preserved.

## Leakage barrier

Acquisition modules receive only the feature matrix and L/U indices. They have
no label argument. Each branch owns a separate RestrictedLabelStore. A selected
batch is content-frozen before its labels can be revealed. No comparator batch,
comparator future performance, hidden-U labels, or test labels are available.
All 24 branches and all 48 test-X predictions must be recursively hash-frozen
before the reporting action can authorize test truth. Validation is used by the
predictor checkpoint selection and is therefore a development diagnostic, not
an independent generalization estimate.

## Test-blind diagnostics

At each anchor: label count; current validation NRMSE; last-step delta; linear
slope over the last two and three source-trajectory intervals (3 and 4 points);
gradient norm mean/std/p50/p90/p95/max; participation-ratio effective rank,
`(sum lambda)^2 / sum(lambda^2)` for `X'X`; current-U nearest-current-L gradient
distance mean/median/p90/max; IVR integrated variance before, predicted greedy
B32 reduction and relative reduction; MaxDet total B32 marginal logdet gain;
and all first-batch pairwise intersections, fractions and Jaccards.

For each new fit record the best validation checkpoint's train loss, V1/V2
validation RMSE/R2, best epoch, epochs run and training time. At each actual
acquisition state record the current feature diagnostics and the strategy's own
objective. The n+64 endpoint intentionally performs no unregistered third
acquisition; it therefore has fit/validation diagnostics but no selector score.

## Post-freeze evaluation

After the global freeze, reveal each seed's test labels once per isolated
branch store and score the frozen anchor, n+32 and n+64 predictions. Report V1
RMSE/R2, V2 RMSE/R2 and combined NRMSE. Short AULC is the trapezoidal integral
over 64 labels divided by 64, equivalently `(E0 + 2*E32 + E64)/4`. Positive
delta is `NRMSE_anchor - NRMSE_future`.

Analysis is descriptive and paired within each anchor. Winner matrices examine
budget, source-trajectory and seed changes. Exploratory Spearman correlations
use only eight anchors, have no p-values, train no predictive model, and cannot
support statistical generalization claims.

## Staged execution

Seed 157's 24 fits run first. A test-blind stage check verifies branch divergence,
nondegenerate overlaps, exact endpoints, complete artifacts and zero test access.
It never removes a method based on validation ranking. Only then may seed 6101's
24 fits run. The final report is a separate explicit action after global freeze.
