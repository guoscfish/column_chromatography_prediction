# T1b-1 Formal Results — Adapter Capacity Sweep

## Question and method

T1a showed that deeper adaptation was not advantageous in this low-label row protocol, but it did not establish that 774 trainable parameters were optimal. T1b-1 tested three low-capacity graph adapters between the 774-parameter prediction head and the 93,454-parameter Last1 endpoint; it did not sample the whole interval.

The frozen intervention was a residual graph adapter after fixed sum pooling and before the monotonic prediction head:

`h'_G = h_G + W_up(ReLU(W_down(h_G)))`

The 128D graph representation was detected from the real model. `W_up` was zero-initialized, giving a maximum initial prediction difference of 0.0. The GNN remained frozen; only the Adapter and 774-parameter head trained. Widths 8/16/32 yielded 2,958/5,014/9,126 trainable parameters. Head, Last1, and Last2 results were reused from frozen T1a artifacts and were not retrained.

This is distinct from `condition_ridge_residual`: Ridge learns a linear V1/V2 residual in prediction space from chromatography conditions, while the Residual Graph Adapter learns a nonlinear residual correction in the 128D latent graph representation.

## Protocol and integrity

Five outer seeds, budgets 30/50/70/100, eight fixed validation labels, and three source members produced 180 new Adapter fits. Training used learning rate 1e-4, weight decay 1e-5, 500 epochs, patience 100, batch size 2048, equal V1/V2 weights, the source scaler, the monotonic-softplus quantile head, and validation normalized MSE checkpoint selection. All six capacity methods shared the frozen T1a gradient-train, validation, and test IDs within each seed/budget context.

Gradient-training counts were 22/42/62/92, so every epoch was one full-batch optimizer step. The shared learning rate, weight decay, and early-stopping rule support a controlled capacity comparison, not a claim that every method used its own optimal optimization regime. The fixed eight validation rows account for 26.67%/16.00%/11.43%/8.00% of the respective total label budgets and may yield noisy checkpoint selection. Simple T1 baselines did not consume these validation rows, while the neural methods did; the reused protocol is paired and controlled but not fully algorithm-aware label-efficiency fair.

The preflight verified frozen T1a and dataset hashes, source checkpoints, exact parameter counts, 180 unique run keys, no duplicates, and 9/9 initialization identity checks without reading test truth. Formal completion was 180/180 Adapter fits, 0 failed, 0 missing, 0 reused, and 0 rerun. All Adapter test predictions were frozen before test truth was read; 120/120 six-method contexts were evaluated.

The complete repository suite passed before execution at the freeze commit (122 passed, 0 failed). After results, documentation, and lifecycle-test maintenance, it passed again (123 passed, 0 failed).

## Overall capacity result

Lower normalized AULC is better.

| Rank | Method | Trainable parameters | Mean | Median | Std |
|---:|---|---:|---:|---:|---:|
| 1 | `target_head_only` | 774 | 0.657664 | 0.597794 | 0.195098 |
| 2 | `graph_adapter_r32` | 9,126 | 0.657770 | 0.612891 | 0.199657 |
| 3 | `graph_adapter_r8` | 2,958 | 0.658312 | 0.591027 | 0.200018 |
| 4 | `graph_adapter_r16` | 5,014 | 0.658661 | 0.599191 | 0.197983 |
| 5 | `last1_head` | 93,454 | 0.689451 | 0.609926 | 0.232778 |
| 6 | `current_last2_head` | 186,134 | 0.763426 | 0.851390 | 0.260980 |

The frozen primary delta is Adapter minus Head, so negative is favorable. r8 had mean/median delta +0.000647/+0.003452 and 2/5 wins; r16 had +0.000997/+0.001397 and 2/5 wins; r32 had +0.000105/-0.004337 and 3/5 wins. Stable improvement required negative mean, negative median, and at least 4/5 wins. All three failed.

## Budget results

Each cell is mean / median / standard deviation combined NRMSE; the final number is best-by-seed count.

| Budget | Head | r8 | r16 | r32 | Last1 | Last2 |
|---:|---|---|---|---|---|---|
| 30 | 0.6648 / 0.5934 / 0.2062; 2 | 0.6593 / 0.5934 / 0.2048; 1 | 0.6587 / 0.6019 / 0.2015; 0 | 0.6608 / 0.6163 / 0.1985; 2 | 0.7103 / 0.6274 / 0.2148; 0 | 0.7985 / 0.8133 / 0.2702; 0 |
| 50 | 0.6818 / 0.6537 / 0.1863; 0 | 0.6747 / 0.6402 / 0.1929; 2 | 0.6732 / 0.6284 / 0.1914; 0 | 0.6730 / 0.6113 / 0.1913; 1 | 0.6962 / 0.6056 / 0.2418; 0 | 0.7802 / 0.7434 / 0.3361; 2 |
| 70 | 0.6503 / 0.5996 / 0.1928; 1 | 0.6526 / 0.5906 / 0.2017; 1 | 0.6539 / 0.5991 / 0.1991; 0 | 0.6497 / 0.6126 / 0.2044; 0 | 0.6756 / 0.6100 / 0.2544; 1 | 0.7264 / 0.7934 / 0.2602; 2 |
| 100 | 0.6330 / 0.5928 / 0.2120; 2 | 0.6453 / 0.5921 / 0.2084; 1 | 0.6471 / 0.5995 / 0.2067; 0 | 0.6490 / 0.6133 / 0.2075; 1 | 0.6897 / 0.6543 / 0.2169; 0 | 0.7794 / 0.8927 / 0.2308; 1 |

## Convergence behavior

| Method | Fits | Mean best epoch | Mean epochs run | Early stopped | Hit 500 |
|---|---:|---:|---:|---:|---:|
| Head | 60 | 411.0 | 470.5 | 29 | 31 |
| r8 | 60 | 169.5 | 267.5 | 58 | 2 |
| r16 | 60 | 129.6 | 229.6 | 60 | 0 |
| r32 | 60 | 109.1 | 207.5 | 59 | 1 |
| Last1 | 60 | 97.8 | 193.4 | 56 | 4 |
| Last2 | 60 | 82.3 | 179.6 | 57 | 3 |

Validation best epochs generally occurred earlier as trainable capacity increased, most clearly from Head through r32. This describes optimization behavior only: earlier stopping is not evidence of better generalization, and the AULC ordering did not improve with capacity.

## Scientific decision

No tested low-capacity graph-adapter benefit was found. Head-only retained the best mean normalized AULC, and no 2,958/5,014/9,126-parameter Adapter produced a stable improvement. This does not establish the absence of a sweet spot across the full 774-to-93,454 interval: the 9,126-to-93,454 gap, including possible 17k/34k/67k regions, was not tested. T1a's narrower observation should be retained as “deeper adaptation was not advantageous in the tested low-label regime,” not generalized to “fewer parameters are always better.” These results favor shallow output remapping over this particular tested latent correction only.

T1b-1 reuses the already-consumed T1a row protocol. It is developmental / hypothesis-testing evidence, not an independent pristine confirmatory study. Current compound splits are target-label holdouts, not source-plus-target molecule OOD. Future source-aware molecule or scaffold holdouts require separate preregistration. Additional widths, T1b-2, and active transfer remain unauthorized.

Machine-readable evidence is retained in `per_context_metrics.csv`, `capacity_curve.csv`, `capacity_aulc_summary.csv`, `paired_aulc_effects.csv`, `convergence_audit.csv`, `formal_fit_resume_details.csv`, `resume_audit.json`, `formal_run_audit.json`, `decision.json`, and `t1b1_capacity_curve.png`. Runtime checkpoints, histories, predictions, and quarantine data remain gitignored under the artifact-retention policy.
