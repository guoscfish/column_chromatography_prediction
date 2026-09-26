# Audited Dialog LLM Active Learning: Final Report

The six-round, 4g Row-split study completed for seeds 157 and 6101. Each round used CW16 plus a full-pool LLM16 supplement, froze all 32 IDs and premeasurement predictions before revealing responses, and then retrained from scratch at the next budget. The final evaluation opened test truth only after both trajectories reached L525.

## Label-AULC and L525

| seed | method | AULC 333-525 | combined NRMSE @525 |
|---:|---|---:|---:|
| 157 | CW-LCMD | 0.628014 | 0.518775 |
| 157 | CW + LLM feedback | 0.658775 | 0.575881 |
| 157 | CW + random full pool | 0.702193 | 0.612030 |
| 6101 | CW-LCMD | 0.826984 | 0.709563 |
| 6101 | CW + LLM feedback | 0.822272 | 0.705976 |
| 6101 | CW + random full pool | 0.897479 | 0.819090 |
| mean | CW-LCMD | 0.727499 | 0.614169 |
| mean | CW + LLM feedback | 0.740524 | 0.640929 |
| mean | CW + random full pool | 0.799836 | 0.715560 |

LLM minus CW AULC was +0.030761 for seed 157 and -0.004712 for seed 6101, with mean +0.013025. LLM minus random was -0.043417 and -0.075208, respectively. Thus the LLM supplement improved on random full-pool supplementation in both development seeds, but did not improve on the stronger CW-LCMD control under the preregistered two-seed gate.

## Audit

- 12 dialog rounds and 384 premeasurement error records passed the execution-boundary audit.
- No forbidden validation/test records or unobserved responses were exposed; native tool calls in accepted rollouts were zero.
- Seed 157 round 05 used a new independent recovery task after the original task failed with an invalid API key before producing a final answer. No labels were revealed during that recovery; the binding and reason are recorded in `round_05/dialog_rebind.json`.
- The report and diagnostic tables are in `results/learning_curves.csv`, `results/aulc.csv`, `results/mean_aulc.csv`, `results/source_coverage.csv`, `results/matched_condition_contrasts.csv`, and `results/dialog_calls.csv`.

This is developmental evidence on two seeds and does not support a statistical-significance claim or automatic continuation to a larger seed set.
