# 4g active-learning studies

Current work uses the qualified standalone QGeoGNN-V2. These row experiments concern source-domain label efficiency; cross-column active transfer remains deferred.

| Study | Status and interpretation |
| --- | --- |
| [Phase 1 batch/adaptivity](qgeognn_v2_batch_adaptivity/README.md) | Current execution: Static LCMD / Adaptive B32x10 / nested Random at 333+320 labels |
| [Phase 0 efficiency review](qgeognn_v2_efficiency_review/REPORT.md) | Completed reporting-only review of endpoints, AULC, target crossings and compute cost |
| [Sequential B32](qgeognn_v2_row_sequential_b32/FINAL_REPORT.md) | Completed; LCMD and Hybrid improve mean AULC by about 23% over Random, 5/5 wins each |
| [Kernel-IVR](qgeognn_v2_row_kernel_ivr_b32/FINAL_REPORT.md) | Completed exploratory extension; no clear gain over LCMD |
| [IVR mechanism audit](qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md) | Stopped before Gate C; no additional training or predictive-effectiveness claim |
| [Small-batch benchmark](qgeognn_v2_row_small_batch_benchmark/FINAL_REPORT.md) | Completed B32/B16 controls; batch-size claims require matched total budgets |
| [Hybrid extension](qgeognn_v2_row_hybrid_extension/README.md) | Completed matched one-step comparison |
| [LCMD pilot](qgeognn_v2_row_lcmd/FINAL_REPORT.md) | Completed large-batch pilot; subsequent sequential evidence is above |

The sequential decision label `NO_CLEAR_SEQUENTIAL_AL_GAIN` does not negate the measured gains over Random. Its frozen classification failed to rank LCMD versus Hybrid; both have `active_gain=true`. Preserve the original classification and explain both facts.

The independent CW/innovation studies remain on their separate worktree branches; they are not silently merged into these results. The [research review](../../docs/research/4G_ACTIVE_LEARNING_REVIEW_2026-09-17.md) discusses their scope.

## Historical and deferred

- [Legacy A1a hybrid batch control](4g/hybrid_batch_control/README.md): earlier predictor and design, not pooled with current V2.
- [Active transfer](transfer/README.md): requires a stable independently validated transfer baseline and an adequate uncertainty contract.

Use [current status](../../docs/NEXT_STAGE_DECISION.md) for the next action and [script index](../../scripts/README.md) for execution.
