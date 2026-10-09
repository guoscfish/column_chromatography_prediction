# CW factorial: continuation to1005

Completed 14 curves through1005. 60 extra fits after525;150 additional historical control points reused. Original36 fits preserved. No paid LLM calls or new test reveals. Whitening remains at525 under the user stop amendment.

The user extended the budget after seeing the333–525 development results. The frozen extension metric is validation AULC525–1005. This is post-development exploration; the original AULC333–525 and failed new-selector confirmation gate remain unchanged. Two seeds do not establish statistical significance.

| Seed | Arm | AULC525–1005 | AULC333–1005 | NRMSE1005 |
|---|---|---:|---:|---:|
| 157 | cw_ivr | 0.405619 | 0.473035 | 0.333646 |
| 157 | cw_lcmd | 0.394227 | 0.457694 | 0.335051 |
| 157 | cw_maxdet | 0.464462 | 0.540051 | 0.361984 |
| 157 | random | 0.691885 | 0.734385 | 0.672336 |
| 157 | raw_ivr | 0.417250 | 0.485866 | 0.338977 |
| 157 | raw_lcmd | 0.396713 | 0.475150 | 0.332293 |
| 157 | raw_maxdet | 0.422132 | 0.485423 | 0.355451 |
| 6101 | cw_ivr | 0.469458 | 0.566807 | 0.436860 |
| 6101 | cw_lcmd | 0.447972 | 0.542363 | 0.401997 |
| 6101 | cw_maxdet | 0.531335 | 0.609614 | 0.492369 |
| 6101 | random | 0.652176 | 0.718690 | 0.579732 |
| 6101 | raw_ivr | 0.516838 | 0.599001 | 0.473583 |
| 6101 | raw_lcmd | 0.456163 | 0.562194 | 0.388837 |
| 6101 | raw_maxdet | 0.544572 | 0.623161 | 0.484668 |

## Paired effects

Positive relative improvement favors the method; negative deltas favor the method.

| Seed | Method / reference | AULC525–1005 delta | Relative improvement | NRMSE1005 delta |
|---|---|---:|---:|---:|
| 157 | cw_lcmd / raw_lcmd | -0.002486 | +0.63% | +0.002758 |
| 157 | cw_maxdet / raw_maxdet | +0.042330 | -10.03% | +0.006533 |
| 157 | cw_ivr / raw_ivr | -0.011631 | +2.79% | -0.005331 |
| 6101 | cw_lcmd / raw_lcmd | -0.008191 | +1.80% | +0.013161 |
| 6101 | cw_maxdet / raw_maxdet | -0.013237 | +2.43% | +0.007701 |
| 6101 | cw_ivr / raw_ivr | -0.047380 | +9.17% | -0.036723 |

## Interpretation of the extended window

cw_lcmd: mean paired relative AULC improvement +1.21%; CW improves in 2/2 seeds. This is a descriptive trajectory comparison, not a revised confirmation gate.
cw_maxdet: mean paired relative AULC improvement -3.80%; CW improves in 1/2 seeds. This is a descriptive trajectory comparison, not a revised confirmation gate.
cw_ivr: mean paired relative AULC improvement +5.98%; CW improves in 2/2 seeds. This is a descriptive trajectory comparison, not a revised confirmation gate.
Whitening was stopped at525 by explicit user instruction. No matched late-window Whitening/CW comparison is available; do not extrapolate the earlier geometry-control result to1005.
The observed CW effect is not uniform across selectors and seeds; retain selector-specific applicability boundaries.

V1/V2 RMSE, MAE and R²: endpoint_metrics.csv. Each seed curve: validation_curves.csv/PNG. Supplemental AULC653–1005: validation_summary.csv. Selector normalization and finite sketch behavior remain part of the comparison; cross-selector differences are not exclusively objective-function effects.

Additional525–1005 training wall-seconds summed over fits: 28279.7; acquisition: 3874.5. Two independent fits may run concurrently after653; summed wall time is not elapsed makespan. See resource_costs_525_1005.csv.

## Stop and recover

All training stops at1005. Reporting and verification only:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_to1005.py report
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_to1005.py
```

P2 paid independent ablations remain unrun. No new-seed confirmation is automatically authorized by a favorable extension endpoint. Any next study needs a fixed candidate/control set and independently audited unused seeds. Historical exposed test data is not a new blind test.
