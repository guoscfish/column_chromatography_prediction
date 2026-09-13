# Transfer research roadmap — 2026-09-12

## Scope and non-comparability

Two different questions exist in this repository and their values must not be put into one ranking.

1. **Low-label matched benchmark:** B=30/50/70/100, no-threshold, COMPOUND and ROW, five seeds. B is a target-label budget, never an epoch budget. It asks which method is effective when the target column has very few labels. Conditional-EA, shrinkage, and low-dimensional calibration are materially stronger than naive neural fine-tuning in the 25g/40g compound evidence.
2. **Filtered FULL-data neural-transfer study:** roughly 320–360 gradient-train rows in the filtered operational domain, five seeds, with ROW as the current formal neural setting. It asks whether neural parameter transfer itself can be trained and selected well when target labels are relatively abundant.

RMSE, R2, and NRMSE from these studies are not interchangeable rankings: populations, partitions, label budgets, and denominators differ. The current outer test identities have historical exposure, so new ROW scores are **DEVELOPMENTAL CONFIRMATION**, not pristine external validation.

## Evidence carried forward

- The repository implements real neural parameter transfer: source parameters are loaded and an explicit trainable scope is optimized. This is distinct from affine/scale post-hoc calibration.
- No completed evidence establishes that full/shallow neural fine-tuning stably exceeds strong low-dimensional transfer across both columns.
- P3's single-context pilot advantage did not reproduce as a stable formal five-seed effect.
- Source BN statistics have partial 25g benefit and negative 40g behavior; they are not a default.
- L-BFGS is not shown superior to Adam: Adam had a censored 150-epoch budget and P3 changed multiple factors.
- The simplest retained neural hypothesis is staged P1: head-only warm-up followed by historical-shallow fine-tuning.
- The qualified 4g ROW source has substantial predictive signal. Capacity expansion is not the leading explanation for the transfer gap.

The first suspected bottlenecks are convergence, unstable small validation selection, loss/evaluation mismatch, catastrophic forgetting, and column-domain heterogeneity—not hidden-size shortage.

## Stage N1: converged P0/P1 baseline

`traditional_transfer_converged_baseline_v1` is the isolated N1 protocol. It compares only P0 and P1 with current BN, Adam, raw `quantile_target_loss`, learning rate 1e-4, weight decay 1e-5, batch 2048, maximum 500 epochs and patience 80. It uses a five-fold `GroupKFold` within each outer `gradient_train`, grouped by `canonical_smiles`. Inner train scales use only that fold's training labels. Outer validation has zero selection role and test truth has zero fit/selection role.

P0 refits for the round-half-up median of the five Stage-B best epochs. P1 separately takes the Stage-A and Stage-B medians, then fixed-epoch refits A followed by B with B inheriting A's true final state. Epoch 150/300/500 diagnostics, slopes, drifts, crossing counts, and run ceilings are retained. An adequacy warning is `STILL_BUDGET_CENSORED` when at least 40% of folds select epoch 500 and at least 40% retain a negative late validation slope. A global prediction freeze precedes every test read.

P1 can be called the preferred neural transfer baseline only if it improves mean shared train-NRMSE on both columns, wins at least 3/5 seeds per column, has no endpoint RMSE/MAE mean deterioration above 2%, has no clear systematic R2 reversal, and is not seriously budget-censored. Otherwise the correct result is `NO_UNIVERSAL_STAGED_TRANSFER_GAIN`.

**Completed N1 result:** `NO_UNIVERSAL_STAGED_TRANSFER_GAIN`. P0 Stage B has
0/25 (25g) and 1/25 (40g) ceiling selections; its late slopes are not a severe
censoring signal. P1 Stage B is likewise not severe, but head-only Stage A is
still censored in 25/25 folds per column with a negative late slope. P1 loses
25g shared NRMSE by 1.10% (2/5 wins) and improves 40g by 0.57% (3/5 wins); its
25g V1 RMSE worsens 2.20%. P1 is not promoted and the Stage-A budget is not
automatically extended. N2 should compare raw and endpoint-normalized loss on
the converged P0 baseline.

## Neural branch after N1

Run one controlled change at a time, always with inner-CV selection.

1. **N2:** raw quantile loss versus endpoint-normalized quantile loss.
2. **N3:** quantile multitask versus q50 standardized MSE and q50 MSE with weak q10/q90 auxiliary loss.
3. **N4:** normalized L2-SP. The existing raw sum-of-squares penalty must become parameter-count/layer normalized before lambdas 0, 1e-6, 1e-5, 1e-4, 1e-3 can be compared.
4. **N5:** adaptation scope plus discriminative learning rates centered at head/late/early rates 1e-3/1e-4/1e-5, with only 0.3x/1x/3x ratios.
5. **N6:** hard ordered quantile head: q50=m, q10=m-softplus(d_low), q90=m+softplus(d_high).
6. **N7:** Center/Width neural head: C=(V1+V2)/2 and W=softplus(raw_W), restoring V1=C-W/2 and V2=C+W/2.

## Structured/domain branch

1. **S1:** HIER/Center-Width strong baseline.
2. **S2:** column-specific shrinkage with exact fallback. For `Vhat_c=V_H,c + alpha_c(V_J,c-V_H,c)`, alpha=0 must exactly return H and alpha=1 J; alpha is chosen only with outer-train OOF evidence.
3. **S3:** measured 4g anchor: predicted, exact measured, matched-measured-with-prediction-fallback, and delta-learning.
4. **S4:** delta/Center-Width cross-column transfer.
5. **S5:** explicitly separate source-known-compound transfer from source-unseen-compound prediction.

HIER/FULL128's opposite 25g/40g behavior is a reason to make this branch controlled, not a license to select alpha after test inspection.

## Paused until a gate is passed

Arbitrary affine/scale variants, uncontrolled latent Ridge/PLS, random tiny adapters, descriptor sweeps, source BN as default, L-BFGS as default, immediate source replay, hidden-size expansion, MAML, DANN, CORAL, large molecular pretraining, and Active Learning are paused. Existing evidence either lacks replicated cross-column benefit, confounds multiple changes, or remains budget/selection-limited. These are not generic claims that a method "does not work."

## COMPOUND and Active Learning gates

Only a stable neural or structured baseline may enter a separately preregistered COMPOUND confirmation, then the B=30/50/70/100 learning curve. Active Learning requires stable compound point metrics, acceptable coverage/crossing behavior, a demonstrated scarcity regime, and AULC—not merely a favorable B=100 endpoint.
