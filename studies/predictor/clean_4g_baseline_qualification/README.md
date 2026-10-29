# 研究记录：clean_4g_baseline_qualification

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# Clean-QGeoGNN 4g baseline qualification

Status: `4G_ENGINEERING_QUALIFICATION_COMPLETE / FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE / UQ_PAUSED`.

Current mainline: [R2-pruned candidate baseline](../r2_pruned_requalification/README.md). Clean engineering reachability PASS; point performance FAIL. The next separate controlled test changes only the R2-pruned quantile head; it has not been run. All measured Clean results below are historical provenance.

This study established numerically valid, interpretable 4g engineering and input-contract behavior for `qgeognn_clean_fusion_v1` preflight revision 2. A later controlled E0 regression audit found that its point performance is far below the reproducible Legacy/V2 controls, so this study no longer qualifies Clean as a point-prediction baseline. It does not test statistical superiority over Legacy and trains no other model.

The primary data contract is `LEGACY_THRESHOLD_DOMAIN`: `V1_ml <= 60`, `V2_ml <= 120`, exactly 4,163 rows. This is a `PROJECT_CONTINUITY_DECISION`, not a scientific or instrument-limit claim. Row interpolation and compound generalization each use fixed 80/10/10 manifests at seeds 42, 525, and 1101.

Formal checkpoints and transient state belong under gitignored `runtime/`. Only compact configs, histories, predictions, metrics, diagnostics, checkpoint metadata, and artifact manifests belong under `results/`.

All six formal runs completed and passed the numerical artifact audit. Row-interpolation combined normalized RMSE was 0.736 mean (0.066 sample SD); compound-generalization was 0.839 (0.142). Condition permutation and disabling each worsened both-target RMSE in 6/6 runs, supporting learned condition reliance. These results remain engineering/input-contract evidence, not current point-baseline qualification. The regression audit and R2-pruned cleanup are complete; the next separate gate is R2-pruned quantile-head qualification; UQ, transfer, and active-learning work are paused. No 8g or active-learning work was executed.

See [`QUALIFICATION_REPORT.md`](QUALIFICATION_REPORT.md), [`PREREGISTRATION.md`](README.md#记录-2), `protocol.json`, `data_usage.json`, `decision.json`, and the compact machine-readable artifacts under `results/`.

<a id="记录-2"></a>

## 记录 2：`PREREGISTRATION.md`

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
