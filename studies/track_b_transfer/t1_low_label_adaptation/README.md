# T1 — Low-label Transfer Adaptation Benchmark

T1a is preregistered as a row-protocol benchmark of adaptation capacity under fixed Random target labels. Each outer seed uses one fixed 10% test partition, eight fixed validation labels, and a deterministic nested gradient-training ranking for total budgets 30, 50, 70, and 100.

Primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. `target_head_only` means QGeoGNN `mode="head_only"`: graph pooling remains fixed sum pooling and only `graph_pred_linear` is trained. Full fine-tuning is disabled as an optional diagnostic; `paper_style`, active acquisition, Protocol B, and advanced diversity methods are excluded.

Historical planning notes used the name `target_readout_only`; in this repository that candidate is implemented and reported as `target_head_only`. It does **not** add or train a learnable graph-readout module: sum pooling is fixed, and the 774 trainable parameters are only those of `graph_pred_linear`. A genuinely learnable target readout is a separate T1b proposal and is not part of T1a.

This directory now contains the preregistration, compact engineering audits, and the completed formal result artifacts. The run completed 180/180 neural fits with no failures or missing fits and all 120 evaluation contexts passed the completion gate. Runtime checkpoints, histories, predictions, progress, and failure records remain gitignored. Engineering smoke results are not scientific evidence; see `FORMAL_RESULTS.md` for the formal analysis.

## Frozen protocol

The master seed is 20260902 and its five NumPy `SeedSequence` children are frozen in `config.json`. Partition and ranking construction uses only `sample_id`, positional `canonical_index`, and `canonical_smiles`; it does not read V1/V2. Each seed has 58 test rows, eight validation rows, and a deterministic ranking of the remaining rows. Gradient-training counts are therefore 22, 42, 62, and 92. Validation consumes 26.67%, 16.00%, 11.43%, and 8.00% of total budgets 30, 50, 70, and 100. All label sets are nested and all methods use the same role IDs within a seed/budget.

Zero-shot is the mean q50 prediction of frozen source members 42/525/1101. Affine fits separate V1/V2 linear corrections using gradient-training labels only. Condition-Ridge predicts the two-dimensional source residual from source V1/V2 plus the frozen 9D condition representation; alpha is selected from 0.01/0.1/1/10/100 by compound GroupKFold within gradient-training labels. Insufficient groups trigger the recorded deterministic alpha=1 fallback and never validation/test fitting.

Each neural source member is adapted separately using the source scaler, monotonic-softplus quantile head, equal target weights, learning rate 1e-4, weight decay 1e-5, batch size 2048, 500 epochs, patience 100, validation-best selection, and no test-based early stopping. Final neural predictions use the K=3 q50 mean. Primary metrics are V1 RMSE, V2 RMSE, and combined NRMSE normalized by source-train-only scales (`ddof=0`); MAE, R², paired effects versus `current_last2_head`, across-seed summaries, parameter counts, and convergence/failure rate are secondary. Learning curves/AULC belong only to a separately authorized formal run.

Because every gradient-training set is smaller than batch size 2048, each epoch is one full-batch optimizer step. Best-epoch and stopping counts must be interpreted as optimizer-step counts rather than ordinary large-data epochs. The shared learning rate, weight decay, and early-stopping contract support a controlled capacity comparison; they do not establish each method's optimal optimization regime.

The eight-row validation set may produce noisy checkpoint selection. Affine and Ridge fit and tune within gradient-training labels and do not consume the eight validation rows, whereas neural methods reserve those labels for early stopping. The protocol is paired and controlled, but not fully algorithm-aware label-efficiency fair. Future proposals, not retrospective changes, may compare cross-validation/cross-fitting followed by all-label refit, explicit method-specific validation overhead, or a preregistered shared-validation design with simple-method all-label refit while maintaining test isolation.

The formal design requires 5 × 4 × 3 × 3 = 180 neural fits. Full fine-tuning remains disabled, `paper_style` is excluded because it confounds capacity with input structure, and Track C active transfer remains deferred.

## Frozen formal analysis and decision gate

Every neural fit has the stable key `seed_<outer>/budget_<budget>/<method>/member_<source>`. Resume reuses a fit only when `best.pt`, `history.csv`, `fit_result.json`, and `formal_contract.json` all exist and the formal config, partition, schedule, source checkpoint, label-ID, checkpoint, adaptation-config, and parameter-count contracts match. Partial or stale single-fit directories are quarantined and only that fit is rerun. A numerical/runtime failure receives at most one identical-contract retry; poor scientific performance is retained and is never a retry reason.

Formal evaluation reports V1/V2 MAE, RMSE, and R² plus combined source-normalized NRMSE. For each seed/method, `AULC_30_100` is trapezoidal integration across 30/50/70/100 revealed labels; `mean_NRMSE_over_budget_interval = AULC_30_100 / 70` is the primary overall score. Paired delta is candidate minus `current_last2_head`, so negative favors the candidate.

A candidate may be described as a stable low-label improvement only when all three preregistered conditions hold: mean paired delta is below zero, median paired delta is below zero, and it wins at least 4/5 outer seeds. Capacity crossover is analyzed separately for `target_head_only → last1_head → current_last2_head` at every budget and remains descriptive. No final decision is generated unless all 120 evaluation contexts and all 180 neural fits pass the completion gate without unresolved failure.

## Formal outcome

The completion gate passed, but no candidate passed the stable-improvement gate against `current_last2_head`. `target_head_only` had the best mean normalized AULC (0.6577 versus 0.7634 for the reference) and best mean combined NRMSE at every budget, but won only 3/5 paired AULC seeds. This favors a low-capacity hypothesis without establishing a stable winner. A genuine learnable-readout comparison is recommended only as a separately preregistered T1b; it is not implemented or authorized, and active transfer remains deferred.
