# 4g active-learning studies

Current work uses the qualified standalone QGeoGNN-V2. These row experiments concern source-domain label efficiency; cross-column active transfer remains deferred.

| Study | Status and interpretation |
| --- | --- |
| [Latest validation snapshot](../../reports/active_learning_latest/REPORT.md) | Completed-budget curves and matched comparisons; refresh without training |
| [LLM Scientist V2](qgeognn_v2_row_llm_scientist_v2/README.md) | CW16+LLM16: seed157845 / seed6101813; Free: both seeds845; 1005 extension incomplete |
| [Dual expert](qgeognn_v2_row_llm_dual_expert_v1/README.md) | Seed157 completed 621; validation NRMSE0.597021; no new test evaluation |
| [CW extension to1005](qgeognn_v2_row_cw_selector_extension_to_1005/RESULTS.md) | Completed: seven arms × two seeds1005; Whitening stopped525; 60 new fits |
| [Phase 1 batch/adaptivity](qgeognn_v2_batch_adaptivity/FINAL_REPORT.md) | Completed matched653-label comparison; endpoint benefit without AULC benefit |
| [Phase 0 efficiency review](qgeognn_v2_efficiency_review/REPORT.md) | Completed reporting-only review of endpoints, AULC, target crossings and compute cost |
| [Sequential B32](qgeognn_v2_row_sequential_b32/FINAL_REPORT.md) | Completed; LCMD and Hybrid improve mean AULC by about 23% over Random, 5/5 wins each |
| [Kernel-IVR](qgeognn_v2_row_kernel_ivr_b32/FINAL_REPORT.md) | Completed exploratory extension; no clear gain over LCMD |
| [IVR mechanism audit](qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md) | Stopped before Gate C; no additional training or predictive-effectiveness claim |
| [Small-batch benchmark](qgeognn_v2_row_small_batch_benchmark/FINAL_REPORT.md) | Completed B32/B16 controls; batch-size claims require matched total budgets |
| [Hybrid extension](qgeognn_v2_row_hybrid_extension/README.md) | Completed matched one-step comparison |
| [LCMD pilot](qgeognn_v2_row_lcmd/FINAL_REPORT.md) | Completed large-batch pilot; subsequent sequential evidence is above |
| [Innovation screen](qgeognn_v2_row_innovation_screen/README.md) | Completed selection-only comparison of fixed gradient and latent geometries; no test labels or retraining |

The sequential decision label `NO_CLEAR_SEQUENTIAL_AL_GAIN` does not negate the measured gains over Random. Its frozen classification failed to rank LCMD versus Hybrid; both have `active_gain=true`. Preserve the original classification and explain both facts.

The CW and innovation studies are separate evidence records and are not pooled with the sequential
results. The innovation-screen implementation is now part of the integrated `main` code line; the
[research review](../../docs/research/4G_ACTIVE_LEARNING_REVIEW_2026-09-17.md) discusses its scope.

## Historical and deferred

- [Legacy A1a hybrid batch control](4g/hybrid_batch_control/README.md): earlier predictor and design, not pooled with current V2.
- [Active transfer](transfer/README.md): requires a stable independently validated transfer baseline and an adequate uncertainty contract.

Use [current status](../../docs/NEXT_STAGE_DECISION.md) for the next action and [script index](../../scripts/README.md) for execution.

- [CW selector factorial and LLM feedback boundaries](qgeognn_v2_row_cw_selector_factorial/README.md): two-seed development, frozen validation AULC333–525, max36 new fits; P2 offline only.

- [CW factorial extension to653](qgeognn_v2_row_cw_selector_extension_to_653/README.md): post-development exploration, four additional rounds on all three new arms and both seeds;24-fit hard cap. Original333–525 gate unchanged.

- [CW factorial extension to1005](qgeognn_v2_row_cw_selector_extension_to_1005/README.md): completed seven-arm/two-seed1005 extension; Whitening stopped525 by user request,60 new fits, no paid LLM requests or new test evaluation.
