# Stage 2 Transfer Audit: Metric, Convergence, and Decision

## Provenance

Stage 1 did not newly train 40 models. It audited and summarized a complete pre-existing local formal ROW run whose ignored runtime artifacts cover 25g/40g x five outer seeds x P0-P3. The local runtime hashes, context freezes, global prediction freeze, and one-shot score manifest were verified. This Stage 2 report uses those frozen predictions and train-only histories; it does not claim a new fit.

## Problem -> audit

The old report compared formal P0-P3 combined NRMSE with historical reference combined NRMSE while using different denominators. Formal P0-P3 used each `column x outer seed` gradient-train target standard deviation. Historical reference methods used fixed qualified 4g source-train scales `[7.879659, 16.076510]`. The ROW test IDs were matched, but normalization was not.

The independent [metric comparability audit](../row_metric_harmonization/METRIC_COMPARABILITY_AUDIT.md) re-scored existing frozen predictions under one contract: each outer split's gradient-train population standard deviation, shared by every method, with no test-driven selection.

The [convergence audit](../traditional_transfer_convergence_audit/CONVERGENCE_AUDIT.md) reads only formal train/validation histories. It finds 41/50 Adam stages selecting their best checkpoint at epoch 150 and 43/50 with a negative last-20 validation slope below the preregistered threshold.

## Unified evidence

Mean values across five seeds. Combined NRMSE is the train-only harmonized value.

| column | method | V1 RMSE | V1 MAE | V1 R2 | V2 RMSE | V2 MAE | V2 R2 | combined NRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25g | P0 | 7.278 | 4.397 | 0.377 | 10.895 | 6.891 | 0.546 | 0.666 |
| 25g | P1 | 7.359 | 4.466 | 0.358 | 10.708 | 6.916 | 0.560 | 0.666 |
| 25g | P2 | 7.137 | 4.457 | 0.418 | 10.355 | 6.918 | 0.593 | 0.646 |
| 25g | P3 | 7.330 | 4.607 | 0.380 | 11.158 | 7.124 | 0.516 | 0.677 |
| 25g | paper-style | 7.409 | 4.462 | 0.349 | 11.111 | 6.967 | 0.525 | 0.679 |
| 25g | HIER shared | 8.009 | 4.521 | 0.191 | 11.031 | 7.113 | 0.530 | 0.709 |
| 25g | BALANCED FULL128 | 7.694 | 4.536 | 0.284 | 10.997 | 7.045 | 0.534 | 0.691 |
| 40g | P0 | 15.533 | 10.832 | 0.613 | 18.356 | 13.254 | 0.730 | 0.537 |
| 40g | P1 | 13.944 | 8.948 | 0.686 | 16.606 | 11.326 | 0.779 | 0.484 |
| 40g | P2 | 15.617 | 10.159 | 0.610 | 18.713 | 13.305 | 0.720 | 0.544 |
| 40g | P3 | 14.064 | 8.878 | 0.676 | 16.940 | 11.448 | 0.764 | 0.490 |
| 40g | paper-style | 13.109 | 7.987 | 0.721 | 15.897 | 10.643 | 0.797 | 0.459 |
| 40g | HIER shared | 14.348 | 8.938 | 0.664 | 17.435 | 12.556 | 0.757 | 0.503 |
| 40g | BALANCED FULL128 | 13.425 | 8.165 | 0.703 | 16.559 | 11.444 | 0.780 | 0.474 |

## Scientific answers

1. **P0-P3 ranking:** On harmonized combined NRMSE, 25g is P2, P1, P0, P3; 40g is paper-style, P1, P3, P0, P2. Endpoint rankings are not identical to combined-NRMSE rankings, so no single scalar should hide endpoint tradeoffs.
2. **Historical comparability:** Yes, the old combined-NRMSE comparison was inconsistent. The old cross-method normalized ranking is withdrawn. Raw RMSE, MAE, and R2 remain directly comparable on the matched test IDs; unified values now live in `row_metric_harmonization/`.
3. **P3 pilot:** The pilot advantage is not stable. Formal P3 loses to P0 on 25g combined NRMSE in 4/5 seeds and wins on 40g in 5/5. The pilot does not establish a transferable P3 effect.
4. **Source BN statistics:** Not universal. P2 vs P1 is -0.0202 on 25g and +0.0595 on 40g, with 3/5 and 0/5 wins respectively. Zero BN drift proves implementation of the policy, not performance benefit.
5. **L-BFGS vs Adam:** Not identified. P3 changes optimizer and staged recipe together, while Adam is frequently budget-censored. The evidence cannot support an optimizer-causal claim.
6. **P1 head warm-up:** It is the most credible simple formal candidate for further train-only confirmation. Relative to P0, harmonized combined delta is -0.0006 on 25g with 3/5 wins and -0.0533 on 40g with 5/5 wins. It is not promoted yet because convergence is unresolved and paper-style remains stronger on 40g.
7. **Adam budget:** The 150-epoch budget is inadequate for stable ranking evidence. A new budget audit is required before interpreting optimizer or recipe differences.
8. **Strongest methods:** 25g formal P2 has the best combined NRMSE and V1/V2 RMSE/R2 among P0-P3, while P0 has the lowest V1 MAE; paper-style has the lowest 25g V2 MAE. On 40g, paper-style is strongest across RMSE, MAE, R2, and unified NRMSE among the compared methods.
9. **Universal neural recipe:** None is established. P1 improves the formal combined score in both columns versus P0, but its 25g endpoint tradeoff and unresolved convergence prevent promotion, and it does not beat paper-style on 40g.
10. **COMPOUND eligibility:** No. Metric harmonization exposed a prior comparison error, and convergence is not adequate. No compound run is authorized by this audit.
11. **Current bottleneck:** The immediate bottleneck is not model capacity. It is a valid, converged, train-only-selected transfer baseline under one metric contract, followed by independent batch/compound evidence.
12. **Active Learning:** No. The baseline is not stable enough to support acquisition claims; current quantile crossings and exposed outer-test development history add further qualification requirements.

## Decision

`STOP_BEFORE_COMPOUND_AND_ACTIVE_LEARNING`. First run the independent train-only convergence protocol in `traditional_transfer_convergence_audit/PROTOCOL.json`, with a pre-frozen longer Adam budget such as 500 epochs and patience 80, recording 150/300/500 diagnostics. After that, any outer-test score is developmental confirmation, not pristine external confirmation. Do not choose P2 for 25g and P1 for 40g from the exposed outer test.

## Model taxonomy

- Architecture: QGeoGNN and its output head.
- Adaptation: head-only warm-up, shallow fine-tuning, source BN statistics.
- Optimizer: Adam or L-BFGS.
- Post-hoc/calibration: scale, conditional EA, Center/Width, HIER_CW.

Changing an optimizer is not changing the GNN architecture. The prior reports must preserve this distinction.

## Not run in Stage 2

No new formal fit, no new outer test comparison, no COMPOUND, no Active Learning, no ordered quantile head, no Center/Width neural head, no descriptor sweep, no latent/PLS/Ridge expansion, and no arbitrary adapter expansion were run.
