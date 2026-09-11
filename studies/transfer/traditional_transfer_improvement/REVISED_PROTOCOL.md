# Revised Protocol: Traditional Transfer Recipe Pilot

## Question

Within the already-qualified shallow parameter scope, can the transfer recipe improve 4g→25g/40g transfer? This is a recipe question, not a layer-depth search.

## Frozen shallow scope

Use the historical `standard_shallow_finetune` scope: `backbone.convs.4`, condition branch, and target head. Current V2 aliases are accepted only where they select these same parameters. No last1/last2 candidate sweep is introduced.

## Phases

**A. Training recipe**

`A0` is direct shallow fine-tuning from source initialization. `A1` is head-only linear probing followed by the exact same shallow scope. Stage B must restore Stage A's validation-best checkpoint before fine-tuning.

**B. BN policy**

`B0 CURRENT_BN` preserves current training behavior. `B1 SOURCE_BN_STATS` freezes `running_mean`, `running_var`, and `num_batches_tracked` from the source checkpoint; affine gamma/beta remain trainable when their parameters are in the shallow scope. BN buffer drift is recorded.

**C. Optimizer pilot**

`C0` uses the existing Adam baseline. `C1` uses full-batch PyTorch L-BFGS with an explicit closure, small fixed `max_iter`, strong-Wolfe line search, and iteration/function-evaluation diagnostics. No full-model L-BFGS is allowed.

Pilot methods: P0=A0+B0+Adam, P1=A1+B0+Adam, P2=A1+B1+Adam, P3=A1+B1+L-BFGS. Only 25g/row and the first frozen outer seed are run. Validation labels alone control fitting and selection; test truth is not read.

## Scaling and controls

Endpoint scales come only from target train rows. Pinball/crossing terms use `1/s`; q50 MSE uses `1/s^2`. Report both stage checkpoints, validation trajectory, trainable count, parameter drift, BN drift, optimizer diagnostics, checkpoint hash, and finite/crossing checks. Train and validation identities must be disjoint, and no adaptation API accepts test indices.

Adaptive readout, adapters, Fisher/BWC, source replay, full fine-tuning, compound protocols, and active learning are deferred.
