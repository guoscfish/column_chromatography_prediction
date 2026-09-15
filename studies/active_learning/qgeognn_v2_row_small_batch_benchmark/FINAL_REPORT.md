# Final report — realistic-batch V2 row active learning

## Protocol and completion

Commit A `a030ed1972615ad1ddfa4a1e4601eb67cce33726` froze the protocol.  All ten B=32 primary seed matrices were frozen before the single global test-label reveal; the global pre-test freeze is retained in `formal_results/global_pre_test_freeze_b32.json`.  The B=16 Random-versus-LCMD sensitivity matrix was likewise frozen before its global test evaluation.  B=64 was **not run**: the user explicitly cancelled it after B=16 completed, before any B=64 artifact was created.

The confirmation cohort is the preregistered decision cohort.  Development-cohort numbers remain descriptive.

## Primary B=32 confirmation result

| Method | Mean combined NRMSE |
| --- | ---: |
| Random | 0.694920 |
| Ensemble uncertainty | 0.675275 |
| CoreSet | 0.673834 |
| Gradient-LCMD | **0.663146** |

Gradient-LCMD won 4/5 seeds against the within-seed Random median, 3/5 against uncertainty, and 3/5 against CoreSet.  Its mean V1 RMSE was 4.907886 and V2 RMSE was 8.894684, each below the best competing mean endpoint RMSE.  The preregistered confirmation decision is `BEST_CURRENT_ROW_ACQUISITION`.

## Batch sensitivity

Mean LCMD improvement relative to Random is -0.57% at B=16 and +4.81% at B=32 in the confirmation cohort.  Thus the prior 333-label strong result cannot be generalized to a monotonic small-batch effect: at B=16 LCMD does not improve over Random on average, while B=32 meets the preregistered primary gate.  B=64 is `NOT_RUN_USER_CANCELLED`, not zero effect and not evidence.

## Diagnostics and interpretation

The preregistered label-free input redundancy, gradient duplicate, and mechanism-correlation outputs are preserved in the formal result directories.  They describe candidate geometry and duplicate structure but do not alter acquisition.  Exact-X redundancy can explain part of an advantage only as a mechanism hypothesis; it is not a causal decomposition.  Validation cost is included in every metric row: B=32 has 365 active labels plus 416 shared validation labels, totaling 781 observed non-test labels (18.76% of the 4,163-row dataset).

## Next study

Because the confirmation Random gate passes, a genuinely sequential B=32 study is eligible.  It must recompute models and acquisitions at every step; this one-step study does not establish a sequential learning curve.
