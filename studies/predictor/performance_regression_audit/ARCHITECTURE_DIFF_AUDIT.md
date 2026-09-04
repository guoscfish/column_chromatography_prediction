# Legacy/V2/Clean architecture-difference audit

Status: `DEVELOPMENTAL_REGRESSION_DIAGNOSTIC`. This is a code and mathematics audit of the frozen variants; no architecture was changed.

## Shared controlled boundary

R0–R3 used the same 4,163-row historical E0 dataset, byte-identical frozen split (`9a758e…b198`), graph/conformer cache, train-only molecular/eluent scaling, seed 42, batch size 2,048, learning rate 0.001, maximum 1,000 epochs, patience 100, and post-selection test evaluation. R1–R3 used identical loss weights, deterministic epoch shuffling, zero weight decay, and validation combined normalized RMSE selection. R1 and R2 had function-identical initialization: R2 wrapped the same-seed random R1 Legacy backbone with a zero-output condition residual.

## Architecture summary

| Property | Legacy R0/R1 | Condition Completion R2 | Clean R3 |
|---|---|---|---|
| Nominal parameters | 775,476 | 777,808 | 413,732 |
| Graph representation | 128D sum-pooled | Same 128D Legacy path | 128D sum-pooled, then 64D projection |
| Conditions entering message passing | Eluent ExactMolWt, TPSA, nRotB, HBD | Same four Legacy inputs | None |
| Other conditions | Forward-unreachable in Legacy | Typed residual after pooling | Typed separate condition encoder |
| Fusion | Conditions affect edge/message representations from layer 1 | Early Legacy interaction plus additive post-pool residual | Concatenate 64D molecule and 64D condition |
| Prediction head | Linear(128,6) + ReLU | Same Legacy head | Linear(128,6), then componentwise softplus monotonic parameterization |

## A. Condition–molecule interaction

The Legacy data builder places bond length, six eluent descriptors, loading-solvent code, loading mass, and loading volume after the three categorical bond fields. `application/QGeoGNN.py` passes the continuous tail to `BondFloatRBF`, whose five configured inputs consume bond length followed by the first four eluent dimensions: ExactMolWt, TPSA, nRotB, and HBD. These edge representations enter every Legacy node-message layer. Thus the executed Legacy representation can learn molecule–condition interactions during graph message passing.

R2 retains this path and adds a zero-output-initialized typed residual after sum pooling for HBA, LogP, loading solvent, loading mass, and loading volume. Its high performance shows that completing the missing conditions is not itself the regression.

Clean removes all experimental conditions from molecular message passing. It computes `z_mol` and `z_cond` independently, concatenates them, then applies one linear head. For each raw head logit the operation is exactly:

`raw_j = W_m,j z_mol + W_c,j z_cond + b_j`.

Softplus subsequently transforms individual logits, but there are no learned bilinear, product, hidden-MLP, modulation, or message-passing cross terms before the output parameterization. `ADDITIVE_LATE_FUSION_WITHOUT_EXPLICIT_INTERACTION` is therefore confirmed as a structural fact. Because R1 and R2 retain early condition interaction and remain strong while R3 removes it and collapses, loss of early interaction is a `SUPPORTED` regression mechanism, but it is not individually isolated from the other simultaneous Clean changes.

## B. 128D → 64D molecular projection

Legacy sends the 128D sum-pooled representation directly to its prediction head. Clean inserts `Linear(128,64)` (8,256 parameters) and `LayerNorm(64)` (128 parameters), then uses only the 64D result. No performance experiment supported choosing 64 dimensions before this audit. It is an `UNVALIDATED_DESIGN_CHOICE` and a plausible information bottleneck.

The R3 test-set mean L2 norms were 75.363 before projection and 73.028 after the linear projection but before LayerNorm. Thus the linear map does not simply collapse vector magnitude. This audit does not measure retained information rank or sufficiency, so the bottleneck remains `PLAUSIBLE`, not confirmed.

## C. LayerNorm after sum pooling

R3 test mean L2 changes from 73.028 before LayerNorm to 11.045 after LayerNorm. Train/validation show the same pattern (72.081→10.956 and 69.045→10.830). LayerNorm intentionally removes per-sample location/scale information and gives the latent a constrained magnitude, so it can plausibly discard information encoded in sum-pooling magnitude. The observed norm change proves normalization is active, not that it causes the R² loss. Classification: `PLAUSIBLE_REGRESSION_MECHANISM_REQUIRING_CONTROLLED_TEST`.

## D. Loss

- R0: `V1_loss + 0.5 × V2_loss`.
- R1/R2/R3: `V1_loss + V2_loss`.

R1 and R2 remain at or above historical performance under the current loss. Therefore the loss-weight change is `NOT_SUPPORTED` as the primary regression mechanism. It may still alter target trade-offs inside another architecture.

## E. Shuffle

- R0: `shuffle=False`.
- R1/R2/R3: deterministic reshuffle each epoch.

R1 retains V1 performance and improves V2 under reshuffling. Shuffle is `NOT_SUPPORTED` as the regression source.

## F. Weight decay

- R0: `1e-5`.
- R1/R2/R3: `0`.

R1/R2 remain strong with zero weight decay. This change is `NOT_SUPPORTED` as the regression source.

## G. Checkpoint selection

- R0: minimum validation `V1_RMSE² + 0.5 × V2_RMSE²`.
- R1/R2/R3: minimum validation combined normalized RMSE using train-only target scales.

R1/R2 remain strong with the normalized selection metric. Checkpoint selection is `NOT_SUPPORTED` as the regression source.

## H. Monotonic softplus head and output compression

On R3 test, prediction/truth q50 standard-deviation ratios were 0.909 for V1 and 0.895 for V2; neither met the prespecified diagnostic flag of below 0.1. q50 clamp-to-zero was exactly zero for both targets. Therefore global low-variance compression and q50 clamping are `NOT_SUPPORTED` as primary explanations.

There is nevertheless a local tail symptom: 8.87% of V1 and 7.91% of V2 test q50 predictions were at or below `1e-6`, produced by strongly negative raw median logits before softplus. Across all six quantiles, 7.59% were clamped to zero, principally lower quantiles. Predicted maxima were also lower than truth maxima. The head–latent interaction remains `PLAUSIBLE`, but the present ladder does not isolate it.

## Evidence classification

- `CONFIRMED`: R2→R3 Clean architecture/forward-path package regression; R3 training-set underfit relative to R2; additive pre-softplus late-fusion form.
- `SUPPORTED`: removal of Legacy early molecule–condition interaction plus replacement by additive late fusion is the leading mechanism.
- `PLAUSIBLE`: unvalidated 128→64 bottleneck; LayerNorm removal of pooled magnitude; monotonic softplus head interaction with Clean latents.
- `NOT_SUPPORTED`: split difference, current training protocol, V2 condition completion, global q50 variance collapse, or q50 evaluation clamp as the primary cause.

These levels deliberately do not claim that any single Clean subcomponent has been causally isolated.
