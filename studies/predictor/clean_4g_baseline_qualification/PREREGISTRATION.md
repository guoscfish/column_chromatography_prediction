# Clean-QGeoGNN 4g baseline qualification preregistration

Status: `FROZEN_BEFORE_FORMAL_TRAINING`.

## Scope

The sole trained model is `qgeognn_clean_fusion_v1`, preflight revision 2. Architecture and input contracts are frozen: five GNN layers; molecular hidden 128; molecular latent 64; condition hidden 32; condition latent 64; solvent embedding 4; sum pooling; molecular ReLU; condition GELU; first-embedded geometry; official-code nRotB descriptor schema; typed sample-level conditions; current monotonic quantile head; equal V1/V2 loss weights; no dropout; no column input; `paper_method_equivalent=false`.

Legacy is a `FROZEN_HISTORICAL_REFERENCE`; Condition Completion V2 is an `ENGINEERING_QUALIFIED_ABLATION / ARCHIVED_FOR_REFERENCE`; Clean-contract MLP is an `OPTIONAL_FUTURE_ARCHITECTURE_ABLATION`; Paper ANN is not run. No architecture or loss change is allowed in response to these results.

## Data and estimands

The source is `dataset/dataset_4g.csv`, SHA-verified against the audited source. The fixed continuity domain is `V1 <= 60`, `V2 <= 120`, expected retained rows 4,163. Exact row IDs and canonical SMILES are frozen before training. A count mismatch stops execution.

`ROW_INTERPOLATION` uses row-level 80/10/10 splits and permits the same compound across partitions. It estimates prediction of other chromatography conditions for seen/similar compounds and is never called unseen-molecule generalization.

`COMPOUND_GENERALIZATION` shuffles unique `canonical_smiles` and assigns approximately 80/10/10 of compounds. Train/validation/test compound intersections must all be empty.

Both use seeds 42, 525, and 1101, producing exactly six formal runs. No five-fold expansion is allowed.

## Training and leakage boundary

Adam, learning rate 0.001, batch size 2048, maximum 1,000 epochs, patience 100, no scheduler, validation-only best-checkpoint selection, V1 weight 1.0, V2 weight 1.0. Test predictions are computed exactly once after the validation-best checkpoint is frozen and reloaded; no test metric appears in training history.

Condition normalization, molecular-descriptor min/max, eluent min/max, and normalized-target scales are fit only on that run's training rows. Records must show zero validation, test, and 8g rows used. Deterministic molecular-only graph/conformer artifacts may be reused.

## Metrics and diagnostics

Point metrics: V1/V2 RMSE, MAE, R², and combined normalized RMSE using training-set target standard deviations.

UQ metrics: V1/V2 q10–q90 coverage, mean interval width, mean pinball loss, within-target crossing rate, q50 V1>V2 rate, and V1 q90>V2 q10 rate. No hard cross-target constraint is added.

Condition-use diagnostics compare full predictions against a fixed within-test-set condition permutation, condition-disabled/molecule-only mode, and optional condition-only mode. Report RMSE, MAE, and R² changes without imposing an arbitrary effect threshold.

Each validation-best model also reports test molecular/condition latent L2 means and validation-based molecular-projection/condition-encoder gradient norms for V1 and V2. Diagnostics explain behavior and cannot tune the model.

## Decision framework

- Gate A: 6/6 completion, finite loss/predictions, reloadable checkpoints, correct shape, maintained within-target order.
- Gate B: describe row/compound performance and seed stability, explicitly flag catastrophic learning failures without a Legacy-superiority cutoff.
- Gate C: describe coverage/width/pinball/crossing; point-baseline readiness may coexist with `UQ_REQUIRES_FURTHER_QUALIFICATION`.
- Gate D: distinguish structural reachability from learned reliance. Negligible permutation/disabled effect requires `CONDITION_USAGE_CONCERN` or `CONDITION_REACHABLE_BUT_PREDICTIVE_USE_WEAK`.

After the report, stop. Recommend exactly one next gate: UQ qualification, 4g→8g transfer qualification, or predictor diagnosis. Do not execute it.
