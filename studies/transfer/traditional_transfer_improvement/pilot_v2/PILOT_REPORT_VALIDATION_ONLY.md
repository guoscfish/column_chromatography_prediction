# Validation-only pilot v2

| method | stage | combined_nrmse | best_epoch | epochs_run | trainable_parameters |
| --- | --- | --- | --- | --- | --- |
| P0 | B | 0.840365846635055 | 40 | 40 | 36387 |
| P1 | A | 2.140926782828947 | 40 | 40 | 774 |
| P1 | B | 0.6595846266682924 | 40 | 40 | 36387 |
| P2 | A | 2.140926782828947 | 40 | 40 | 774 |
| P2 | B | 0.5437980912168753 | 29 | 40 | 36387 |
| P3 | A | 0.3428672825962158 | 4 | 14 | 774 |
| P3 | B | 0.30106213953215 | 5 | 15 | 36387 |

| method | bn_buffer_drift | bn_affine_parameter_drift | crossing_count |
| --- | --- | --- | --- |
| P0 | 40.14940658620113 | 0.0604601198128918 | 5 |
| P1 | 40.15142035626344 | 0.0590884209164819 | 5 |
| P2 | 0.0 | 0.0417513197736932 | 5 |
| P3 | 0.0 | 0.0382431492517725 | 0 |

## Contract audit

Baseline HEAD: b1be9685ae6465d310793959bf1445e9226b4ec8, equal to fetched origin/main.
The old 19-row, 15/1/3, one-epoch pilot is INVALID_PILOT_SUPERSEDED.
Its four invalidation reasons are incomplete graph-cache population, non-frozen
random target split, mismatched shallow scope, and preprocessing not derived
from the qualified source checkpoint. Its results were retained, not overwritten.

The correct 25g filtered population has 408 rows. Expected and graph-covered
sample ID sets are equal, with zero missing/extra rows. The exact frozen ROW
context uses target outer seed 769539383: 320 gradient_train, 5 validation,
83 test. All 320 training rows are fitted. Only the 325 train/validation rows
are passed to the fit API; test endpoint cells and retention-time proxies are
excluded from fitting input. Source seed 42 identifies the qualified 4g model
and is not the target outer seed.

The source checkpoint SHA256 is
fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0.
The descriptor/eluent scaler is read directly from its preprocessing payload.
Source preprocessing SHA256 is
4930fd303f5a52288524f1d074bf06dccbcfce191ce1cbc0f1dab08c5ed3388a;
scaler SHA256 is
5aacda00b600eb6d8f031bff52575c5fa9a4386697e7358e26579af32792f96d.
Target train-only scales are V1=9.965594410763911 and V2=18.08300749095768.

Historical shallow trains backbone.convs.4 (33,281 parameters),
condition_branch (2,332), and head (774): total 36,387. The source-anchored
wrapper's target_head maps exactly to the current V2 head. This is checked
against the original wrapper inventory and historical training_audit.csv.
All separate angle/geometry components and earlier atom convolution blocks
remain frozen. BN affine parameters in the selected block may update;
BN running buffers obey their separately configured policy.
The head starts as a source head copy; it is not randomly initialized.
P0 and all Stage-B inventories are identical.

## Interpretation

Stage A -> B reduces validation combined NRMSE:
P1 2.140927 -> 0.659585 (69.19%); P2 2.140927 -> 0.543798 (74.60%);
P3 0.342867 -> 0.301062 (12.19%). A regression test deliberately makes
Stage-A final worse than best, verifies every restored state tensor, and
compares Stage-A best predictions with Stage-B initial predictions.

P1 improves over direct P0 (0.840366), but uses an additional head-adaptation
stage, hence additional optimization budget. This does not isolate staging
from extra compute. P1 and P2 have identical head-only Stage A: CURRENT_BN
already freezes BN with frozen affine parameters. Their Stage-B difference
therefore reflects the specified BN policy under this pilot.

SOURCE_BN_STATS improves Stage-B validation NRMSE by 17.55% versus P1.
P2 and P3 show exactly zero BN buffer drift; affine parameters do update.
This is a favorable single-context signal, not evidence of cross-seed stability.

P3 completes with finite training losses and validation predictions. Stage-A
best is step 4 of 14, Stage-B best step 5 of 15, with early stopping. L-BFGS
is worth retaining as a candidate. Its raw loss, learning rate, iteration
budget and lack of weight decay differ from the Adam recipe; this result is
not a pure optimizer causal comparison. Strong-Wolfe closure evaluations can
exceed the nominal max_iter count, so epochs are outer optimizer steps.

P0/P1/P2 have crossing in all five final validation rows; P3 has none. The
existing six-output loss is a soft crossing penalty and does not guarantee
ordered quantiles. No post-hoc sorting was applied. Finite-output and loss
contracts pass independently of crossing incidence; P0/P1/P2 should not be
claimed to provide calibrated, noncrossing uncertainty intervals.

P0, P1 Stage A/B and P2 Stage A hit the 40-epoch budget with their best at
epoch 40. P2 Stage B selects epoch 29 of 40. Consequently Adam has not been
shown to converge; do not interpret the pilot's ranking as a final ranking.
Only five frozen validation rows are available, so no statistical superiority
claim is justified and no test metric was computed.

## Next-round recommendation (not executed)

Retain P0 as the direct shallow control and P1 as the staging/CURRENT_BN
control; prioritize P2 and P3 as the promising recipes for a separately
authorized 25g + 40g x five frozen ROW seeds. None of P0-P3 should be discarded
solely on this small pilot. Freeze convergence budgets and selection rules
before that run; allow sufficient Adam convergence and keep optimizer recipe
hyperparameters explicit. No additional method, full FT, last2 sweep, loss
candidate, compound protocol or active learning is warranted in that round.

## Execution and recovery

Environment: conda fish, Python 3.11, CPU, one torch/OpenMP/MKL thread;
KMP_DUPLICATE_LIB_OK=TRUE matches the historical project environment.
Training completed once. CSVs and prediction arrays were saved before optional
pandas Markdown export failed because tabulate was unavailable. The dependency
was removed and export recovered with --finalize-only; no refit or candidate
search was performed. The exact executed runner/adaptation snapshots and
original execution log are preserved. Final source additionally corrects
source-versus-target preprocessing bookkeeping and restores the unrelated
legacy last2 scope; neither changes the executed pilot's numerical path.
The pilot restores best states in memory; persistent model checkpoints were
not exported, so this artifact set is for validation audit, not deployment.

## Final readiness

**TRAINING_READY_FOR_FORMAL_5SEED_ROW**. All A-M gates pass. Full repository pytest: 377 passed, 195 warnings, 85.48 seconds. No formal training started.
