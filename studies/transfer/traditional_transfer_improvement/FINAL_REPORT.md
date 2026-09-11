# Traditional transfer improvement: implementation report

Date: 2026-09-11  
Repository HEAD: `f52cdd0` (`Update transfer model and add row-column selective latent audit`)

## Scope and status

This turn completed the reusable implementation and froze the filtered FULL-data
ROW protocol. No new target test result was read, and no compound or active
learning run was started. Existing historical tables are cited only as
developmental references.

## Findings from the audit

The current adaptation path used one Adam learning rate and the mixed six-output
loss. It exposed `head_only`, `last2`, and `full`, but had no staged LP->FT
operation, no train-fold target scale helper, no discriminative parameter
groups, and no source-distance regularizer. The current quantile contract is
q10/q90 pinball plus q50 MSE and crossing penalties; therefore normalization is
implemented as a complete endpoint loss weight, preserving output semantics.

The validation set in the inherited compound protocol is only one compound in
some contexts, so early stopping is noisy. This is why the protocol orders
controls before combinations and requires a future train-only inner-CV choice
if a full rerun is launched. Historical row audits already show column-specific
behavior and do not justify an unbounded backbone or readout rewrite.

## Implemented controls

- `last1` trainable scope, in addition to the existing scopes.
- `fit_target_scales`, which reads labels from explicit training indices only.
- `scaled_quantile_target_loss`, weighting all endpoint terms by `1/s_e^2`.
- `build_discriminative_optimizer` with named head/late/early LR groups.
- `l2_sp_penalty` and `parameter_drift` against an immutable source state.
- `train_staged_target_adaptation` for head warmup followed by last1/last2/full.
- Best validation state is restored into the in-memory model after fitting.

Adaptive readout, adapters/PEFT, source measured-value anchors, compound runs,
and test-guided hyperparameter expansion remain explicitly deferred.

## Verification

`KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography pytest -q tests/test_transfer_adaptation_primitives.py tests/test_final_v2_transfer.py`

Result: **8 passed**. Python compilation and `git diff --check` also pass. The
base environment lacks PyTorch; the project Conda environment is required for
model tests.

## Next executable step

Implement a study runner that consumes the inherited ROW split manifest and
source checkpoint, runs T0 through T4 in the fixed order, writes per-seed
validation histories and drift, freezes checkpoints, and only then evaluates
the five-seed test table. The runner must call `fit_target_scales` on each
outer training fold and never pass test indices to adaptation.
