# CW-LCMD 429 to 525 developmental continuation

This extension continues only the frozen Center/Width-LCMD trajectory from 429 to 461, 493, and 525 for seeds 157 and 6101. No comparator was retrained. The LCMD-to-IVR study remains sealed with engineering checks passed and formal execution not started.

## Primary AULC 429-525

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Gradient-LCMD | Kernel-IVR |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.559942 | 0.605466 | 0.610593 | 0.653408 | 0.582817 |
| 6101 | 0.740139 | 0.767675 | 0.754740 | 0.805052 | 0.750381 |

Decision: **MIXED** against strongest baseline **hybrid**. Paired AULC deltas (CW minus strongest baseline) are {'157': -0.05065175469297778, '6101': -0.014600504249945523}; @525 deltas are {'157': -0.08425733878492125, '6101': 0.005421743710936178}. No continuation to 653 and no MaxDet->CW switch was implemented.

The marginal-improvement table separates 397->429 from 429->461, 461->493, and 493->525. A fast final drop at 397->429 is not treated as evidence by itself; the primary decision uses the complete 429->525 AULC and both seed directions.

All new batches were selected from current U_t using the current CW checkpoint and all current L_t as LCMD centers. Selected IDs were frozen before label reveal. Six new checkpoint/prediction artifacts were frozen before the single post-freeze test evaluation.
