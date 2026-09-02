# T1 — Low-label Transfer Adaptation Benchmark

T1a is preregistered as a row-protocol benchmark of adaptation capacity under fixed Random target labels. Each outer seed uses one fixed 10% test partition, eight fixed validation labels, and a deterministic nested gradient-training ranking for total budgets 30, 50, 70, and 100.

Primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. `target_head_only` means QGeoGNN `mode="head_only"`: graph pooling remains fixed sum pooling and only `graph_pred_linear` is trained. Full fine-tuning is disabled as an optional diagnostic; `paper_style`, active acquisition, Protocol B, and advanced diversity methods are excluded.

This directory contains preregistration and compact engineering audits. Runtime checkpoints and histories are gitignored. `formal_authorized` remains false, so `--run` must refuse. Engineering smoke results are not scientific evidence and select no winner.

## Frozen protocol

The master seed is 20260902 and its five NumPy `SeedSequence` children are frozen in `config.json`. Partition and ranking construction uses only `sample_id`, positional `canonical_index`, and `canonical_smiles`; it does not read V1/V2. Each seed has 58 test rows, eight validation rows, and a deterministic ranking of the remaining rows. Gradient-training counts are therefore 22, 42, 62, and 92. All label sets are nested and all methods use the same role IDs within a seed/budget.

Zero-shot is the mean q50 prediction of frozen source members 42/525/1101. Affine fits separate V1/V2 linear corrections using gradient-training labels only. Condition-Ridge predicts the two-dimensional source residual from source V1/V2 plus the frozen 9D condition representation; alpha is selected from 0.01/0.1/1/10/100 by compound GroupKFold within gradient-training labels. Insufficient groups trigger the recorded deterministic alpha=1 fallback and never validation/test fitting.

Each neural source member is adapted separately using the source scaler, monotonic-softplus quantile head, equal target weights, learning rate 1e-4, weight decay 1e-5, batch size 2048, 500 epochs, patience 100, validation-best selection, and no test-based early stopping. Final neural predictions use the K=3 q50 mean. Primary metrics are V1 RMSE, V2 RMSE, and combined NRMSE normalized by source-train-only scales (`ddof=0`); MAE, R², paired effects versus `current_last2_head`, across-seed summaries, parameter counts, and convergence/failure rate are secondary. Learning curves/AULC belong only to a separately authorized formal run.

The formal design requires 5 × 4 × 3 × 3 = 180 neural fits. Full fine-tuning remains disabled, `paper_style` is excluded because it confounds capacity with input structure, and Track C active transfer remains deferred.
