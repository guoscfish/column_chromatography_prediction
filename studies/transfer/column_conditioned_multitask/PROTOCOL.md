# Preregistered protocol

Frozen before any new fit on 2026-09-14. Only A1 separate heads and A2 categorical
column FiLM are eligible. The historical exclusion audit precedes implementation.

## Architecture and training

Both models load the SHA-verified qualified standalone QGeoGNNV2 source.
Backbone width 128, five node blocks, four effective edge updates, fixed sum
pooling, condition completion, and six-output quantile semantics are retained.
All three heads are exact independent copies of the qualified source head.
Train node blocks 3/4, the edge/angle update at index 3 feeding the last block,
condition completion, and all heads. Freeze early parameters and their BN
statistics. Late trainable BN sees one mixed equal-task batch per step.

A2 alone adds a 3-by-16 categorical task embedding and two Linear(16,256)
FiLM generators, applied to the node GIN output before the existing activation
at blocks 3 and 4. The generators start at zero, so gamma and beta are zero
for every task and A2 equals A1 at epoch zero. Column identity has no physical
meaning and consumes none of the legacy geometry or mass/flow descriptors.

Use Adam, weight decay 1e-5, pretrained shared parameters/source head lr 1e-4,
copied new target heads and FiLM/embedding lr 3e-4. The maximum is 500 epochs,
patience 80. No learning-rate, dimension, width, or loss search is permitted.
Each task loss is the current raw q10 pinball + q50 MSE + q90 pinball + adjacent
crossing penalties summed over V1/V2. Optimize the arithmetic mean of the three
raw task losses. Gradient diagnostics never modify this objective.

Per-task batch size is min(682, largest target training population). An epoch
has ceil(largest target population / per-task size) steps. Each task supplies
the same number of samples in each step using independent deterministic
epoch/task-seeded permutation cycles. Small tasks wrap; 4g is subsampled.
The largest target is traversed each epoch, while 4g volume never dominates
epoch length. The sampler and A1/A2 use paired seeds and identical sample draws.

Frozen early-block outputs are cached from the qualified source in eval mode
using features only. The executable cache equivalence test compares both arms'
training predictions, gradients, and optimizer updates against full forwards.
This is an execution optimization and does not change the frozen model recipe.

## Identity and label boundaries

Copy the exact filtered FULL-data outer manifest, including all five seeds and
both protocols. Fit on source row-seed-42 source_train and target gradient_train
only. Target outer validation/test graphs carry zero endpoint placeholders.
Source validation is diagnostic only and cannot select an epoch.

Construct 5-fold GroupKFold globally on canonical SMILES across joined 25g and
40g gradient_train rows. A molecule present in both target tasks always has the
same inner role. Hash every actual training/validation identity and record the
full inner schedule. Source molecules may overlap target holdouts intentionally.

Every fold fits target endpoint standard deviations on its own inner training
rows (population standard deviation, ddof=0). Select one joint checkpoint by
the arithmetic mean of the two columns' combined normalized RMSE. Ties retain
the earliest epoch. Qualified source input preprocessing is reused without any
target fit. Fixed refits use the round-half-up median of the five joint best
epochs separately for each arm/outer seed, with no validation selector.

## Metrics and decision

Report endpoint RMSE/MAE in mL, combined normalized RMSE, R2, Center RMSE,
Width RMSE, and Center/Width error variance/covariance. Tail thresholds are the
per-endpoint 80th percentiles of outer gradient_train labels, exactly as requested.
They are diagnostic/evaluation definitions and never select checkpoints. Inner
tail errors are pooled across OOF folds within each seed before averaging seed
tail RMSEs; this handles folds with no tail rows without inventing a zero RMSE.
Report tail SSE, total SSE, their fraction, and tail counts. Low/mid/high regimes
use outer gradient_train center tertiles, for descriptive diagnostics only.

The COMPOUND gate compares A2 to A1. BOTH columns must have >=3% arithmetic
mean paired fold-relative NRMSE improvement, >=3/5 wins comparing each seed's
mean fold NRMSE, >=13/25 individual fold wins, no mean endpoint RMSE increase
above 2%, and no mean seed-pooled tail endpoint RMSE increase above 2%.
Every one of the 50 fits and 100 target fold observations must be complete.
On failure, stop before new outer predictions and report the negative result.

On passing, refit both arms and freeze ALL outer validation/test prediction
files across both columns and five seeds before a separate score action reads
test truth. Score COMPOUND first, then run ROW inner selection/refit/freeze/score
with the same models and recipe. Every outer result is DEVELOPMENTAL CONFIRMATION.

Frozen references are ID-aligned and rescored with each context's target
gradient_train scales. Use paper_style_current_v2 and corrected HIER shared
lambda for both protocols; converged P0 is available only for ROW. Its absent
COMPOUND artifact is reported, never retrained or replaced by another P0 recipe.
OLD_HIER_REFERENCE is excluded as authority. M3 is optional secondary history.

REPRESENTATION_SIGNAL requires robust A2 superiority to A1. PROJECT_TRANSFER_GAIN
additionally requires improvement in both columns versus every available
legitimate reference (a conservative strongest-reference guard), >=3/5 paired
seed wins in each, no mean endpoint/tail RMSE deterioration >2%, and no ROW
mean NRMSE deterioration >2%. Missing compound P0 limits the scope of any claim.

## Diagnostics and restrictions

Record raw task loss and per-task validation metrics every epoch. Compute
shared-parameter gradient norms and cosines for 4g/25g, 4g/40g, and 25g/40g on
the actual balanced training batch at epoch 1 and every 10 epochs. This fixed
diagnostic cadence reduces extra backward computation without changing fitting.
Interpret alignment, magnitude imbalance, and persistent conflict descriptively.

No 8g, PCGrad, GradNorm, domain-specific BN, calibration, physical scaling,
architecture/loss/dimension sweep, additional candidate, ensemble, or Active
Learning may be added. Final reporting answers all 13 user questions and may
recommend a future experiment, but does not execute it.
