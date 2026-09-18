# Kernel-IVR matched sequential experiment

Decision: `NO_CLEAR_IVR_IMPROVEMENT`.

Exploratory extension on the previously evaluated cohort; no independent confirmation.

| Method | Mean AULC | NRMSE at 1005 |
|---|---:|---:|
| random | 0.624260 | 0.543301 |
| lcmd | 0.479539 | 0.408614 |
| kernel_ivr | 0.500070 | 0.395913 |

IVR paired AULC wins versus LCMD: 1/5; relative mean gain: -4.28%.

| Method | Target | Interpolated labels | First observed budget | Sustained through final |
|---|---|---:|---:|---:|
| kernel_ivr | Random@1005 | 490.72 | 493.00 | 493.00 |
| kernel_ivr | N80 | 814.85 | 845.00 | 845.00 |
| kernel_ivr | N90 | 952.98 | 973.00 | 973.00 |
| kernel_ivr | N95 | >1005 | >1005 | >1005 |
| lcmd | Random@1005 | 472.33 | 493.00 | 493.00 |
| lcmd | N80 | 651.19 | 653.00 | 653.00 |
| lcmd | N90 | >1005 | >1005 | >1005 |
| lcmd | N95 | >1005 | >1005 | >1005 |
| random | Random@1005 | 940.31 | 941.00 | 941.00 |
| random | N80 | >1005 | >1005 | >1005 |
| random | N90 | >1005 | >1005 | >1005 |
| random | N95 | >1005 | >1005 | >1005 |

Targets use the cohort-mean curve; per-seed crossings are reported separately.
Censored counts are never extrapolated. Random uses empirical first crossing, not a fixed 1005 target count.
Validation adds 416 labels to every active budget. Overlapping splits prevent an independent-seed significance claim.

## Cost

Timing is descriptive because historical machine load differs.
Runtime tables separate training, gradients, IVR, and newly incurred versus reused computation.
The IVR arm reuses five round-zero models and gradient matrices; 105 new scratch fits were required.

## Interpretation

The scalar row-kernel surrogate is not an exact two-output Bayesian posterior.
The frozen unit regularization is one hypothesis, not a tuned optimum. No automatic variants or budget extension follow this result.
