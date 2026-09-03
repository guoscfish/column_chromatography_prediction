# Experiment index

| experiment | stage | status | authoritative? | scientific role | superseded_by |
|---|---|---|---|---|---|
| e0_4g_baseline | Gate0 | frozen | yes | 4g source anchor | — |
| data_audit | pre-Gate0 | frozen | yes | source-data audit | — |
| d04_conformer_selection | D04 | completed | yes | conformer sensitivity aggregate | — |
| e0_8g_transfer | Gate0 | historical | no | initial transfer baseline | g0_3_threshold_sensitivity / g0_4_paper_style_transfer |
| e0_3b_controls | Gate0 | frozen | yes | paired control evidence | — |
| e0_3c_loss_controls | Gate0 | frozen | yes | loss-control evidence | — |
| g0_1_quantile_monotonicity | Gate0 | frozen | yes | monotonic-head decision | — |
| g0_2_interval_calibration | Gate0 | frozen | yes | calibration evidence | — |
| g0_3_threshold_sensitivity | Gate0 | frozen | yes | authoritative 574-row no-threshold target | — |
| g0_4_paper_style_transfer | Gate0 | frozen | yes | frozen transfer-mode decision | — |
| g0_4_paper_style_transfer_random_init_diagnostic | Gate0 diagnostic | historical | no | superseded random-init adapter diagnosis | g0_4_paper_style_transfer |
| e1_signal_qualification | E1 | frozen | yes | source-member and signal qualification | — |
| e2_4g_active_learning | E2 row | frozen | yes | primary row-split AL result | — |
| e2_4g_compound_preflight | E2 compound | historical | no | bounded preflight | e2_4g_compound_active_learning |
| e2_4g_compound_active_learning | E2 compound | frozen | yes | suggestive compound-split AL result | — |
| e2_compound_failure_audit | D38R | corrected | yes | post-hoc descriptive failure audit | — |
| e2_random_smoke | E2 smoke | engineering only | no | historical pipeline smoke | e2_4g_active_learning |
| d28_al_engineering | D28 | frozen | yes | reusable engine/partition engineering evidence | — |
| e4_active_transfer_preregistration | E4 preflight | passed | yes | protocol, partitions, source compatibility | — |
| e4_protocol_a_engineering_smoke | D40 | engineering only | yes | pipeline smoke; no scientific conclusion | — |
| e4_a2a_low_budget_preregistration | E4-A2a | preregistered; formal completed | yes | low-initial-label sensitivity design | e4_a2a_low_budget_formal |
| e4_a2a_engineering_smoke | E4-A2a | passed | yes | bounded engineering audit; no scientific conclusion | e4_a2a_low_budget_formal |
| e4_a2a_low_budget_formal | E4-A2a | completed; evidence null | yes | preregistered L0=30 formal active-transfer sensitivity | — |
| e4_protocol_a_formal | E4 Protocol A | completed | yes | three-seed formal pilot; active acquisition null, transfer benefit retained | — |
| e4_protocol_a_headroom_audit | D42 | completed | yes | post-hoc descriptive headroom, first-round shock, and queried-label mechanism audit; primary null unchanged | — |
| e4_transfer_aware_acquisition_qualification | D43 | completed; gate failed | yes | unlabeled transfer-aware ranking/batch qualification; no performance evidence; low-L0 smoke not run | — |
| e4_active_learning_suitability_diagnosis | D44 | completed; T3R gate failed | yes | dataset/shift/model-update diagnosis plus no-truth soft-T3R qualification; historical performance links are post-hoc only | — |
| d45_oracle_marginal_utility | D45 | bounded diagnostic completed | yes | post-hoc single-label marginal utility audit; test truth consumed; no confirmatory evidence | — |
| d46_oracle_utility_reliability | D46-A | bounded diagnostic completed | yes | deterministic-by-construction, post-selection test-row diagnostic; 3/18 unique intervals exclude zero | — |
| reproductions | support | gitignored runtime target | no | non-authoritative reproductions | — |

Every current experiment directory has exactly one navigation row above.

## Current studies outside `experiments/`

- `studies/track_b_transfer/t1_low_label_adaptation`: T1a formal scientific run complete (180/180 fits, 120/120 contexts); no candidate passed the stable-improvement gate. See `FORMAL_RESULTS.md` and the machine-readable result artifacts.
- `studies/track_b_transfer/t1b1_adapter_capacity`: post-T1a developmental capacity sweep; engineering/preregistration and 9-fit smoke complete, 180-fit formal run not authorized.
