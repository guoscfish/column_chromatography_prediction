# Quantile audit

Decision: `CURRENT_HEAD_RETAINED_FOR_POINT_TRANSFER`.

The audit is descriptive and non-blocking for ordinary point transfer. No alternative head or calibration was trained. Signed q90-q10 width and negative-width rates are retained; Spearman and the exactly ceil(20%)-sized top subset use width clipped at zero. Ties are resolved stably. Full train/validation/test audits are in the CSV.

Interval warning flags: `{'any_test_crossing_above_5_percent': True, 'any_test_negative_width_above_5_percent': False, 'any_test_uncertainty_anti_associated_with_error': False}`. Follow-up: `MONOTONIC_HEAD_CONTROL_REQUIRED_BEFORE_ACTIVE_TRANSFER`.

| mode | seed | split | target | rows | top_uncertainty_rows | negative_width_rate | crossing_rate | q10_pinball_loss | q50_rmse | q50_mae | q90_pinball_loss | empirical_coverage | interval_width | nonnegative_interval_width | uncertainty_error_spearman | top_20pct_uncertainty_error_enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| row | 42 | test | V1 | 417 | 84 | 0 | 0.00959232614 | 0.27987986 | 2.97305217 | 1.36419367 | 0.465431672 | 0.784172662 | 3.80037836 | 3.80037836 | 0.384308678 | 2.41356993 |
| row | 42 | test | V2 | 417 | 84 | 0 | 0.00239808153 | 0.62758885 | 4.91012215 | 2.33261675 | 0.779588012 | 0.714628297 | 5.38069881 | 5.38069881 | 0.326311282 | 2.55139827 |
| row | 525 | test | V1 | 417 | 84 | 0 | 0.0383693046 | 0.292796681 | 2.96536292 | 1.34991435 | 0.492508454 | 0.64028777 | 2.74748115 | 2.74748115 | 0.33763909 | 2.17966814 |
| row | 525 | test | V2 | 417 | 84 | 0 | 0.0071942446 | 0.492260219 | 5.60666255 | 2.22746474 | 0.88288116 | 0.705035971 | 4.5312024 | 4.5312024 | 0.345436387 | 2.18363511 |
| row | 1101 | test | V1 | 417 | 84 | 0 | 0.0239808153 | 0.338693733 | 2.81969201 | 1.53848221 | 0.467895326 | 0.741007194 | 4.06672975 | 4.06672975 | 0.422506905 | 2.61827663 |
| row | 1101 | test | V2 | 417 | 84 | 0 | 0.00239808153 | 0.70834652 | 6.05836155 | 2.96882242 | 0.853430882 | 0.652278177 | 5.58053965 | 5.58053965 | 0.479764221 | 2.66824493 |
| compound | 42 | test | V1 | 373 | 75 | 0 | 0.00804289544 | 0.599767732 | 6.74492577 | 3.35824017 | 1.60493463 | 0.605898123 | 3.9011502 | 3.9011502 | 0.545640691 | 2.77139336 |
| compound | 42 | test | V2 | 373 | 75 | 0 | 0.00804289544 | 1.39674089 | 12.634711 | 6.21231328 | 3.17953638 | 0.455764075 | 5.19245501 | 5.19245501 | 0.492302342 | 2.74279331 |
| compound | 525 | test | V1 | 414 | 83 | 0.0217391304 | 0.152173913 | 0.381481611 | 4.91336045 | 2.46258939 | 0.985471231 | 0.70531401 | 4.18233116 | 4.19418687 | 0.350519906 | 2.77290806 |
| compound | 525 | test | V2 | 414 | 83 | 0 | 0.0217391304 | 1.12701988 | 9.86576582 | 4.86694809 | 2.09168603 | 0.475845411 | 5.73810489 | 5.73810489 | 0.470958474 | 2.68713625 |
| compound | 1101 | test | V1 | 458 | 92 | 0 | 0 | 0.803129719 | 5.16679821 | 2.78428629 | 1.02891719 | 0.602620087 | 4.07482831 | 4.07482831 | 0.477344504 | 2.56461464 |
| compound | 1101 | test | V2 | 458 | 92 | 0 | 0 | 1.96326315 | 10.4385832 | 5.80524957 | 2.00341929 | 0.458515284 | 6.15395157 | 6.15395157 | 0.580878261 | 2.67356368 |

## Test aggregate across three seeds

| mode | target | crossing_rate / mean | crossing_rate / std | empirical_coverage / mean | empirical_coverage / std | interval_width / mean | interval_width / std | uncertainty_error_spearman / mean | uncertainty_error_spearman / std | top_20pct_uncertainty_error_enrichment / mean | top_20pct_uncertainty_error_enrichment / std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compound | V1 | 0.0534056028 | 0.0856303472 | 0.637944073 | 0.0583670936 | 4.05276989 | 0.141882391 | 0.457835034 | 0.0990125991 | 2.70297202 | 0.119823398 |
| compound | V2 | 0.00992734196 | 0.0109913967 | 0.463374923 | 0.0108870143 | 5.69483716 | 0.482206368 | 0.514713026 | 0.0582860955 | 2.70116441 | 0.0366848401 |
| row | V1 | 0.0239808153 | 0.0143884892 | 0.721822542 | 0.0738359919 | 3.53819642 | 0.697609307 | 0.381484891 | 0.0425043154 | 2.40383823 | 0.219466127 |
| row | V2 | 0.00399680256 | 0.00276906604 | 0.690647482 | 0.0335731415 | 5.16414696 | 0.55717878 | 0.383837297 | 0.083623702 | 2.46775944 | 0.252899702 |
