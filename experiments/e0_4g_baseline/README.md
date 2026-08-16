# E0-2 4g baseline reproduction

Status: completed for one fixed row-level seed (`seed=42`).

## Frozen protocol

- Source: `dataset/dataset_4g.csv` only; SHA256 is recorded in `data_metadata.json`.
- Current-code volume rule: `V1=t1*flow/1200<=60 mL`, `V2=t2*flow/1200<=120 mL`.
- Repeated experiments: retained.
- Current-code label anomalies: retained and flagged (`1` negative-label row, `8` rows with `t1>t2`).
- RDKit/graph audit: all `217` unique retained structures built successfully; `216` MMFF and `1` UFF fallback.
- Split: deterministic row-level 80/10/10 using seed 42; train/valid/test = 3330/416/417.
- A compound-group comparison split is also frozen, but it does not replace this row-level baseline.
- Scalers: fitted on the training split only.
- Training: QGeoGNN architecture and quantile loss from `application/QGeoGNN.py`; lr=1e-3, batch=2048, max 1000 epochs, validation patience=100.
- Selection: best checkpoint by validation score only; test evaluated once after training.

## Result

Best checkpoint: epoch 91. Early stopping ended at epoch 191.

| Split | Metric | V1 | V2 |
|---|---|---:|---:|
| Validation | MAE | 1.4582 | 2.7171 |
| Validation | RMSE | 2.9028 | 5.5019 |
| Validation | R2 | 0.8468 | 0.8705 |
| Test | MAE | 1.6040 | 2.8682 |
| Test | RMSE | 2.9713 | 4.9691 |
| Test | R2 | 0.8669 | 0.9029 |
| Test | Mean pinball loss | 0.6736 | 1.0551 |
| Test | Nominal 80% interval coverage | 0.8321 | 0.6163 |

The test result is close to the target cited in the experiment plan (R2(V1)=0.859, R2(V2)=0.913). V1 is above that reference; V2 is lower by about 0.010. This local result is the baseline for later paired experiments because it is tied to the current repository CSV, regenerated deterministic conformers, a leakage-free training-only scaler, and the recorded environment.

The row-level split is retained only for the planned paper-comparable baseline. It places 210 compounds in more than one subset, and 24 of the 52 repeated-condition groups cross subsets. Therefore these scores are not treated as leakage-resistant generalization evidence. The saved compound-group split has zero compounds crossing subsets and will be used for the strict comparison.

The test quantile crossing rate is 0.0504. V2's nominal 80% interval covers only 61.63% of test labels. Therefore E0-4 still needs explicit monotonic-output and calibration treatment/checks before the active-learning interface is accepted.

## Main artifacts

- `canonical_4g.csv`: frozen 4163-row baseline dataset.
- `sample_decisions_4g.csv`: every source row and its keep/drop decision.
- `graph_audit_4g.csv`: per-structure RDKit/conformer/graph status.
- `split_seed_42.csv`: frozen split indices.
- `compound_group_split_seed_42.csv`: leakage-resistant comparison split for later experiments.
- `scaler.json`: train-only feature scaling statistics.
- `checkpoints/best.pt`: best validation checkpoint.
- `training_history.csv` and `metrics.json`: training trace and final metrics.
- `environment.json` and `artifact_manifest.json`: runtime versions and SHA256 checksums.

Reproduce in the requested environment:

```bash
conda run --no-capture-output -n fish python scripts/run_e0_4g_baseline.py --reuse-cache --epochs 1000 --batch-size 2048 --patience 100 --seed 42
```
