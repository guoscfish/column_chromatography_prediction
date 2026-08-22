# D42 — E4 Protocol A Headroom & Acquisition-Shock Audit

Offline, post-hoc descriptive audit only (`n=3` outer splits). No QGeoGNN was trained, no acquisition/predictor was changed, and Protocol B / E4-A2 were not run. The frozen Protocol A primary conclusion remains **active evidence = null**.

## Headroom hypothesis

Initial recovery at L0=50 was `0.742` (seed42), `0.930` (seed525), and `0.912` (seed1101). Seed42 had more headroom and all four active strategies beat Random there; seeds525/1101 were already above 90% recovery and all four active strategies lost on AULC. This supports the **headroom hypothesis descriptively**; it does not prove it.

`labels_to_90=50` means the split was already at 90% before any active query, not that acquisition was effective. Historical `label_efficiency.csv` is unchanged. D42 calls `E_full` the **full-data reference**, not a ceiling, because partial-label models can outperform it and recovery can exceed 1.

## First-round and mechanism audit

Largest 50→60 NRMSE degradations were `[{'outer_seed': 525, 'strategy': 'pretrained_quantile_width', 'delta_NRMSE_50_to_60': 0.3172614641899685}, {'outer_seed': 525, 'strategy': 'pretrained_ensemble', 'delta_NRMSE_50_to_60': 0.2794066231915231}, {'outer_seed': 1101, 'strategy': 'pretrained_ensemble', 'delta_NRMSE_50_to_60': 0.22966256305290633}, {'outer_seed': 525, 'strategy': 'pretrained_coverage', 'delta_NRMSE_50_to_60': 0.22044511351496165}, {'outer_seed': 525, 'strategy': 'pretrained_hybrid', 'delta_NRMSE_50_to_60': 0.19566244796577426}]`. Source-residual association: `True`; label-extremeness association: `True`; optimization-difficulty clue: `True`. D42 source residual uses truth only after historical reveal and is strictly a post-hoc mechanism diagnostic; it must never be used as an acquisition score. The `queried_union_top_decile_*` thresholds use the union of Round1 queried samples within a seed, not the complete U0 pool. These are descriptive associations, not causal explanations or method-selection evidence.

At seed525 all four active strategies degraded on both V1 and V2. At seed1101 Ensemble, Hybrid, and Quantile Width degraded on both targets; Coverage improved V1 slightly but degraded V2 enough for a positive NRMSE shock. Random improved NRMSE at both high-saturation seeds. Active first batches were consistently more uncertain, more distant, and more diverse than Random in all three seeds, yet seed42 Coverage/Ensemble improved after the first batch. Diversity therefore does not by itself explain the shock. Source residual and label extremeness show the most consistent directional association with shock. Round1 best-epoch shifts are higher on average for active strategies in the high-saturation seeds, but individual strategies are mixed (for example seed1101 Coverage is easier than Random), so optimization is only a weak contributing clue. Normalized validation scores are not compared as absolute cross-strategy mechanism evidence because their target-variance denominators come from each strategy/round's current gradient-training labels. Raw nine-dimensional condition summaries are retained; no single condition-shift explanation is established.

## Decision

`low_budget_A2_warranted=True` and `protocol_b_warranted_now=false`. D42 does not alter the Protocol A null result. Any recommended E4-A2 would be a separate low-budget sensitivity study and was not executed here.
