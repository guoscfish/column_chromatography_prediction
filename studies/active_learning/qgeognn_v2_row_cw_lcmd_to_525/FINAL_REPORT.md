# CW-LCMD 429 to 525 developmental continuation

This extension continued only the frozen Center/Width-LCMD trajectory from 429 to 461, 493, and 525 for seeds 157 and 6101. This reporting correction reused only frozen metrics and artifacts; it performed no training, acquisition, label reveal, or historical artifact rerun. No comparator was retrained.

## Full AULC 333-525

AULC 333-525 measures overall label efficiency from the initial active-learning budget. Every seed/method integral uses exactly 333, 365, 397, 429, 461, 493, and 525 active labels.

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Gradient-LCMD | Kernel-IVR |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.628014 | 0.646002 | 0.654246 | 0.689157 | 0.658425 |
| 6101 | 0.826984 | 0.805782 | 0.818599 | 0.864861 | 0.789651 |
| Mean | 0.727499 | 0.725892 | 0.736422 | 0.777009 | 0.724038 |

## Mid-stage AULC 429-525

AULC 429-525 measures local efficiency during the CW continuation window. It is distinct from the full AULC above and uses exactly 429, 461, 493, and 525 active labels.

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Gradient-LCMD | Kernel-IVR |
|---:|---:|---:|---:|---:|---:|
| 157 | 0.559942 | 0.605466 | 0.610593 | 0.653408 | 0.582817 |
| 6101 | 0.740139 | 0.767675 | 0.754740 | 0.805052 | 0.750381 |
| Mean | 0.650040 | 0.686570 | 0.682667 | 0.729230 | 0.666599 |

Decision: **PROMISING** against strongest baseline **hybrid**. Paired AULC deltas (CW minus strongest baseline) are {'157': -0.05065175469297778, '6101': -0.014600504249945523}; @525 deltas are {'157': -0.08425733878492125, '6101': 0.005421743710936178}. Both seeds improve mid-stage AULC, but both endpoints do not improve: seed 6101 is worse by 0.005422. The mean endpoint delta is -0.039418, which is below the pre-existing +0.03 deterioration tolerance. This supports PROMISING, not STRONG_MIDSTAGE_SIGNAL. No continuation to 653 is recommended under this completed developmental protocol, and no MaxDet-to-CW switch was implemented.

The marginal-improvement table separates 397->429 from 429->461, 461->493, and 493->525. A fast final drop at 397->429 is not treated as evidence by itself; the primary decision uses the complete 429->525 AULC and both seed directions.

All new batches were selected from current U_t using the current CW checkpoint and all current L_t as LCMD centers. Selected IDs were frozen before label reveal. Six new checkpoint/prediction artifacts were frozen before the single post-freeze test evaluation. Their hashes were verified unchanged by this report-only regeneration.

The separate LCMD-to-IVR two-seed development run has completed. It showed mixed late-AULC behavior and was stopped before the remaining three seeds. That separate study does not alter the CW decision here.
