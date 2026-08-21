# Experiment index

| experiment | stage | status | authoritative? | scientific role | superseded_by |
|---|---|---|---|---|---|
| e0_4g_baseline | Gate0 | frozen | yes | 4g source anchor | — |
| e0_8g_transfer | Gate0 | historical | no | initial transfer baseline | g0_3_threshold_sensitivity / g0_4_paper_style_transfer |
| e0_3b_controls | Gate0 | frozen | yes | paired control evidence | — |
| e0_3c_loss_controls | Gate0 | frozen | yes | loss-control evidence | — |
| g0_1_quantile_monotonicity | Gate0 | frozen | yes | monotonic-head decision | — |
| g0_2_interval_calibration | Gate0 | frozen | yes | calibration evidence | — |
| g0_3_threshold_sensitivity | Gate0 | frozen | yes | authoritative 574-row no-threshold target | — |
| g0_4_paper_style_transfer | Gate0 | frozen | yes | frozen transfer-mode decision | — |
| e1_signal_qualification | E1 | frozen | yes | source-member and signal qualification | — |
| e2_4g_active_learning | E2 row | frozen | yes | primary row-split AL result | — |
| e2_4g_compound_preflight | E2 compound | historical | no | bounded preflight | e2_4g_compound_active_learning |
| e2_4g_compound_active_learning | E2 compound | frozen | yes | suggestive compound-split AL result | — |
| e2_compound_failure_audit | D38R | corrected | yes | post-hoc descriptive failure audit | — |
| d28_al_engineering | D28 | frozen | yes | reusable engine/partition engineering evidence | — |
| e4_active_transfer_preregistration | E4 preflight | passed | yes | protocol, partitions, source compatibility | — |
| e4_protocol_a_engineering_smoke | D40 | engineering only | yes | pipeline smoke; no scientific conclusion | — |
| e4_protocol_a_formal | E4 Protocol A | completed | yes | three-seed formal pilot; active acquisition null, transfer benefit retained | — |

Directories not listed remain historical support artifacts and are not silently deleted.
