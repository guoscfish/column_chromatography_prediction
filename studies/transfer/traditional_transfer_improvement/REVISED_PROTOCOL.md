# Revised protocol: matched traditional transfer pilot v2

This task stops after a validation-only 25g ROW pilot. No formal five-seed run is authorized.
The original pilot is INVALID_PILOT_SUPERSEDED; its numbers are not evidence of readiness.
Baseline HEAD: b1be9685ae6465d310793959bf1445e9226b4ec8 (equal to fetched origin/main).

## Population and roles

Use filtered_full_data_benchmark/protocol.json and split_manifest.csv directly:
first target outer seed 769539383; 408 rows, 320 gradient_train, 5 validation,
83 test. All 320 training rows are used, without label budgets or new splits.
Merge SOURCE_GRAPH_CACHE with graph_cache_25g_only.pt and fail on any missing
sample ID. Frozen canonical hash, exact population identity, unique IDs, role
union, and role disjointness are asserted before fitting.

Only feature columns needed by build_model_data are projected from the feature
artifact; retention times and endpoint labels are excluded. The existing
_read_authorized_truth reader parses label cells only for train/validation IDs.
Only those 325 rows are constructed as labeled graphs and passed to fit APIs.
Test IDs are recorded for identity auditing; no test graph, truth, or score is
passed to training, normalization, BN, early stopping, or reporting.

## Source and historical scope

Qualified source: final_4g_qualification/runtime/row/seed_42/best.pt.
Source checkpoint seed 42 is distinct from target outer seed 769539383.
The hash must match the frozen benchmark. Descriptor/eluent scaler comes directly
from its preprocessing payload. Preserve full source preprocessing and scaler
hashes; target endpoint standard deviations use only the 320 gradient_train rows.

The sole shallow reference is SourceAnchoredTransfer(source, 'shallow') in
src/qgeognn_al/transfer/source_anchored.py. Its exact inventory is mapped to V2:
backbone.convs.4.*, condition_branch.*, and target_head.* -> head.*.
Total trainable parameters: 36,387, matching historical training_audit.csv.
BN affine parameters within those components train. Earlier atom blocks and all
separate bond/angle/geometry components stay frozen. No last1/last2 substitution.
The head is copied from the source; this is source-initialized head adaptation,
followed by shallow representation fine-tuning, not random-head probing.

## Four fixed recipes

| Method | Stages | BN policy | Optimizer |
|---|---|---|---|
| P0 | Direct historical shallow | CURRENT_BN | Adam |
| P1 | Head-only -> same historical shallow | CURRENT_BN | Adam |
| P2 | Head-only -> same historical shallow | SOURCE_BN_STATS | Adam |
| P3 | Head-only -> same historical shallow | SOURCE_BN_STATS | L-BFGS |

Adam: learning rate 1e-4, weight decay 1e-5, maximum 40 epochs per stage,
patience 15. Historical shallow training best epochs can be around 48 (the
first historical audit row); 40 epochs is a bounded trajectory pilot, not a
convergence claim. L-BFGS: learning rate 0.1, maximum 20 steps per stage,
patience 10, max_iter 5, history_size 10, strong-Wolfe line search, no weight
decay. Batch size 2048 covers the full training set. No hyperparameter search.
Both optimizers use the original raw quantile loss. Normalized-loss scale-one
and scale-invariance tests exercise the implementation, without adding a loss
candidate. L-BFGS is evaluated as a fixed recipe; weight decay is not matched
with Adam, so results do not isolate optimizer choice alone.

Stage A restores its validation-best state before Stage B. An explicit test
forces the final Stage-A epoch to be worse than the first, checks all inherited
state tensors, and compares Stage-A best versus Stage-B initial predictions.
Each stage selects its own validation best. Stage B need not outperform Stage A;
report both rather than silently selecting the better stage.

CURRENT_BN retains the existing set_training_mode behavior: BN with frozen affine
parameters is eval, trainable BN is train. SOURCE_BN_STATS additionally freezes
all running_mean, running_var, and num_batches_tracked buffers after each mode
transition. Report restored-best buffer drift and BN affine drift separately.

History reported_step_loss is the optimizer-returned/pre-update batch loss;
post_step_train_loss (also train_loss) is recomputed after the step in eval mode,
avoiding additional running-stat mutations. These differ under CURRENT_BN as
well as after an optimizer step. Record finite six-output validation predictions
and crossing counts; the quantile loss penalizes crossing but does not enforce
hard monotonic ordering.

## Gate and artifacts

pilot_v2 contains the frozen protocol, exact split manifest, full parameter
inventory, stage metrics/history, BN drift, validation prediction arrays, report,
input/output hashes, and complete pytest log. Source scaler provenance, graph
coverage, scope, stage inheritance, BN, L-BFGS, loss invariance, no-test fitting,
finite prediction and quantile contracts must pass before readiness is declared.
A single seed with five validation rows supports only a direction signal.
No test scoring, compound split, active learning, full FT, new adapter,
regularization candidate, normalized-loss candidate, or formal run is included.
