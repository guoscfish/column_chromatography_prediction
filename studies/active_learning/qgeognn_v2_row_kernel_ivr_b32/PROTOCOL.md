# Kernel-IVR matched sequential extension

Frozen before new strategy training; the timestamp is recorded in `seal.json`.
This is exploratory work on
an already evaluated cohort. Historical test outcomes motivated the hypothesis;
the cohort cannot supply a fresh confirmatory claim.

## Single acquisition hypothesis

Preserve the existing full-network endpoint-scaled concatenated 512D CountSketch.
At each round extract features from the current scratch-trained predictor for
the fixed original outer-training universe R = L0 union U0 (3330 rows).
Use no validation or test inputs in this reference distribution.

Let s^2 = mean_R ||phi(x)||^2 and X = phi/s. This is a single global scaling,
not per-row normalization or whitening. Set prior precision = 1 and homoscedastic
observation noise variance = 1. These dimensionless values are frozen without
an output-dependent fit or sweep; this is an explicit surrogate assumption.
Equivalently, in raw feature coordinates the ridge is mean_R ||phi||^2.

C = (I + X_L^T X_L)^(-1); M = X_R^T X_R / 3330.
Score candidate x by (x^T C M C x)/(1 + x^T C x).
Greedily select 32 rows. After each pick update C by Sherman-Morrison, then
recompute scores for remaining candidates. Exact ties use original U0 order.
The selected row is an entire experiment with both endpoint labels.
This scalar row-kernel surrogate is not an exact multi-output posterior.

## Matched experiment

Seeds: 157, 887, 2357, 6101, 12203. L0=333, U0=2997, validation=416, test=417.
Budgets: 333, 365, ..., 1005, giving 22 points and 21 acquisitions of 32.
Identical scratch initialization, shuffle, optimizer, loss, early stopping,
validation checkpoint criterion, fixed L0 target scales, and graph preprocessing
as `qgeognn_v2_row_sequential_b32`. No warm start and no additional ensemble.

Read-only reuse: completed Random and LCMD trajectories and matched full-data
references. Reuse LCMD's round-zero predictor, predictions and gradient matrix
after exact preprocessing/training/split/checkpoint checks. No later model or
gradient reuse across diverging acquisitions. There are 105 new scratch fits
and 100 new full-reference gradient extractions, plus 105 IVR batches.

All 110 new-strategy points, selections, checkpoints, and predictions must be
content-frozen before test truth is revealed for IVR. Resumes verify code,
environment, source, baseline, state, model, and acquisition hashes. Failed
partial fits restart from the same initialization with old attempts preserved.
The historical context class is reused only as an exact preprocessing adapter;
new study contracts and its independent seal govern all new fits and selections.
At most two seeds run concurrently, with two CPU threads each.

## Endpoints and decision

Primary: normalized AULC over the whole 333..1005 grid, paired seed differences
against LCMD and Random. Report every seed and mean; overlapping splits preclude
an independent-seed significance claim. Smaller values are better.

Targets: Random@1005, and N80/N90/N95 relative to the matched full reference,
T_p = E_full + (1-p)(E0-E_full). Report per-seed and cohort-mean-curve crossings
separately. Show first adjacent interpolated count, first observed grid budget,
and first budget that stays at/below target through the final point. A rebound
must remain visible. Never extrapolate beyond 1005. Random's empirical first
crossing of its final error is reported, not forcibly assigned 1005 labels.
All methods additionally use the shared 416 validation labels.

Call the extension promising only if all hold: mean AULC at least 2% lower than
LCMD, at least 4/5 paired AULC wins, cohort-mean first observed N80 no later than
LCMD (both reached), and neither final endpoint RMSE more than 2% worse than
LCMD. These are exploratory practical gates, not a statistical significance
claim. Otherwise report no clear improvement and the actual effect sizes.

Cost: fit count, epochs, training seconds, gradient seconds and IVR seconds.
Separate historical/reused versus newly incurred work. Historical load differed,
so timing is descriptive. No variants, seed selection, test tuning, automatic
budget extension, or automatic follow-up are authorized by this protocol.
