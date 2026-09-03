# T1a Formal Low-label Adaptation Results

T1a completed the frozen row-protocol experiment on 2026-09-03. The completion gate passed: 180/180 neural fits completed, 0 failed, 0 were missing, and all 120 method × seed × budget evaluation contexts were present. Test truth was first read only after every neural fit and all 120 prediction files were frozen. These are formal results; the earlier engineering smoke remains non-scientific.

## Primary AULC result

Lower is better. `mean score` is `AULC_30_100 / 70`, the mean combined NRMSE over the budget interval.

| Method | Mean AULC | Mean score | Median score | SD across seeds |
|---|---:|---:|---:|---:|
| `target_head_only` | 46.0365 | 0.6577 | 0.5978 | 0.1951 |
| `last1_head` | 48.2616 | 0.6895 | 0.6099 | 0.2328 |
| `affine` | 49.9622 | 0.7137 | 0.7299 | 0.2387 |
| `current_last2_head` | 53.4398 | 0.7634 | 0.8514 | 0.2610 |
| `condition_ridge_residual` | 55.9664 | 0.7995 | 0.9281 | 0.2398 |
| `zero_shot` | 102.7799 | 1.4683 | 1.6007 | 0.3437 |

## Preregistered paired gate

Delta is candidate minus `current_last2_head` on each seed's normalized AULC; negative favors the candidate. Passing requires mean < 0, median < 0, and at least 4/5 seed wins.

| Candidate | Mean delta | Median delta | SD | Seed wins | Gate |
|---|---:|---:|---:|---:|---|
| `target_head_only` | -0.1058 | -0.0455 | 0.1330 | 3/5 | fail |
| `last1_head` | -0.0740 | -0.0441 | 0.1152 | 3/5 | fail |
| `affine` | -0.0497 | +0.0108 | 0.1956 | 2/5 | fail |
| `condition_ridge_residual` | +0.0361 | +0.0618 | 0.1005 | 1/5 | fail |
| `zero_shot` | +0.7049 | +0.7493 | 0.1472 | 0/5 | fail |

No candidate is a stable low-label improvement over `current_last2_head` under the frozen decision rule. `target_head_only` and `last1_head` have favorable mean and median effects, but each wins only 3/5 seeds.

## Results by revealed-label budget

Cells are mean / median combined NRMSE, with the number of best-method wins among all six methods in parentheses.

| Budget | zero-shot | affine | condition-Ridge | target-head-only | last1-head | current-last2-head |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1.4683 / 1.6007 (0) | 0.7248 / 0.7905 (1) | 0.7471 / 0.7667 (1) | **0.6648 / 0.5934 (3)** | 0.7103 / 0.6274 (0) | 0.7985 / 0.8133 (0) |
| 50 | 1.4683 / 1.6007 (0) | 0.7639 / 0.7940 (1) | 0.8648 / 0.8695 (0) | **0.6818 / 0.6537 (1)** | 0.6962 / **0.6056** (1) | 0.7802 / 0.7434 (2) |
| 70 | 1.4683 / 1.6007 (0) | 0.6976 / 0.6976 (1) | 0.8002 / 0.9495 (0) | **0.6503 / 0.5996 (1)** | 0.6756 / 0.6100 (1) | 0.7264 / 0.7934 (2) |
| 100 | 1.4683 / 1.6007 (0) | 0.6664 / 0.6580 (3) | 0.7462 / 0.8517 (0) | **0.6330 / 0.5928 (1)** | 0.6897 / 0.6543 (0) | 0.7794 / 0.8927 (1) |

Among only the three neural capacities, `target_head_only` has the best mean at every budget. Its neural-only seed wins are 5/5, 2/5, 2/5, and 4/5 at budgets 30, 50, 70, and 100. Median capacity is less uniform: `last1_head` is best at budget 50, while `target_head_only` is best at the other budgets. This is descriptive capacity evidence, not a passed stability gate or a causal/significance claim.

The simple baselines remain consequential. Affine is the best of all six methods on 1/5, 1/5, 1/5, and 3/5 seeds across the four budgets. At budget 100 it wins more individual seeds than any neural method, although `target_head_only` has the lower across-seed mean and median. Condition-Ridge does not improve reliably over affine and has positive paired mean/median AULC deltas versus the historical neural reference.

## Convergence and integrity

| Neural method | Fits | Mean best epoch | Mean epochs run | Hit 500 epochs | Early stopped |
|---|---:|---:|---:|---:|---:|
| `target_head_only` | 60 | 411.0 | 470.5 | 31 | 29 |
| `last1_head` | 60 | 97.8 | 193.4 | 4 | 56 |
| `current_last2_head` | 60 | 82.3 | 179.6 | 3 | 57 |

All methods shared the same gradient-training, validation, and test ID hashes within every seed/budget context. Gradient labels alone fit affine and Ridge and select Ridge alpha by compound GroupKFold. Validation labels alone select neural checkpoints. Source-only V1/V2 scales (7.7755/15.9773 mL, `ddof=0`) normalize metrics. There were no failed, missing, stale, partial, reused, or retried fits in this first formal execution. The 20 Ridge contexts all used five-fold gradient-only selection; no small-group fallback was needed.

## Scientific decision and next step

T1a does not establish a stable winner. It does show a coherent low-capacity signal: fixed-sum `target_head_only` is best by mean AULC and mean combined NRMSE at every budget, while deeper adaptation is not consistently better. Between-seed heterogeneity is substantial, the protocol is row-level only, and five-seed comparisons are descriptive without a significance claim.

Historical `target_readout_only` terminology must not be interpreted as a tested learnable readout: T1a fixed sum pooling and trained only the 774-parameter `graph_pred_linear` head. The recommended next isolated study is a separately preregistered T1b comparison adding a genuine learnable graph-readout module against this fixed-pooling head baseline. T1b is not implemented or authorized here. Track C remains deferred, and Track A is not restarted.

Machine-readable evidence is in `per_context_metrics.csv`, `aulc_by_seed.csv`, `paired_aulc_effects.csv`, `capacity_by_budget.csv`, `simple_vs_neural_by_budget.csv`, `convergence_audit.csv`, `ridge_selection_audit.csv`, `formal_label_hash_audit.csv`, `resume_audit.json`, and `formal_run_audit.json`.
