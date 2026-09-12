# Preregistration: converged P0/P1 neural-transfer baseline v1

## Question

With an adequate Adam budget and train-only epoch selection, what do the basic
neural parameter-transfer recipes P0 and P1 achieve? This is not a model
search.

## Frozen population and methods

Use the filtered FULL-data 25g/40g ROW populations, their frozen five outer
seeds, and the SHA256-locked qualified 4g checkpoint. Compare only P0
(historical-shallow) and P1 (head-only then historical-shallow), both using
current BN, Adam, raw quantile loss, 1e-4 learning rate, 1e-5 weight decay,
batch 2048, maximum 500 epochs, and patience 80. No P2/P3, source BN, L-BFGS,
L2-SP, normalized loss, replay, expanded scope, or other ablation is allowed.

## Selection and data authority

For each outer gradient-train set, make a five-fold GroupKFold by canonical
SMILES. Inner train and validation groups must be disjoint. Inner-train labels
alone fit endpoint scales. P0 selects the round-half-up median Stage-B epoch;
P1 separately selects Stage-A/B medians. The full-gradient refit accepts no
validation/test indices. P1 Stage B begins from Stage A's actual final state.

Outer validation has no hyperparameter or epoch-selection role. Outer test is
not read until all 20 final prediction files, final checkpoints, selected
epochs, source checkpoint and split identities are hash-locked.

## Evaluation and promotion

Score RMSE/MAE/R2 for V1/V2 and the common outer-gradient-train ddof=0
combined NRMSE; P0/P1 share an identical denominator per outer context.
Source-scale NRMSE is reported separately. Test results are developmental
confirmation only.

P1 is promoted only with both-column mean NRMSE improvement, at least 3/5
paired wins per column, no endpoint RMSE/MAE mean deterioration over 2%, no
systematic opposite R2 direction, and no serious inner-CV budget censoring.
