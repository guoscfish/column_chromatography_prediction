# Convergence Audit

## Provenance

This is a retrospective train-only audit of the pre-existing local formal ROW runtime. It reads only `training_history.csv` and `stage_metrics.csv` from the frozen P0-P3 run. It does not read outer test truth, retrain a model, select a test checkpoint, or modify the formal study.

## Evidence

- Adam stages audited: 50.
- Best checkpoint at the configured maximum epoch: 41/50.
- Adam budget in the formal protocol: 150 epochs, patience 40.
- Last-20-epoch validation slope remained negative below -1e-4 in 43/50 stages.
- Stage A head-only runs for P1/P2 reached epoch 150 in all 20 contexts. P0/P1/P2 Stage B also frequently reached the ceiling, especially for 40g.

The raw stage-level evidence is in `train_only_history_audit.csv`; grouped counts are in `convergence_summary.csv`.

## Assessment

The formal Adam budget is not an adequate convergence basis for stable optimizer or recipe ranking. A best epoch equal to the maximum is a censoring signal, not evidence that the maximum is optimal. The observed negative late validation slopes are train-only evidence that some trajectories were still improving when stopped.

This audit does not establish that a longer budget will improve outer-test performance. It establishes only that the current formal result cannot answer that question.

## Independent follow-up protocol

Create a new train-only developmental convergence study with the same frozen populations, source checkpoint, trainable scope, BN policies, and seeds. Freeze the budget rule before any outer-test read. The minimal candidate budget is maximum epochs 500 with patience 80, with 150/300/500 recorded for diagnosis; selection must use inner validation or a predeclared plateau rule. Any later outer-test score is developmental confirmation because the current outer test has already been exposed in existing artifacts.
