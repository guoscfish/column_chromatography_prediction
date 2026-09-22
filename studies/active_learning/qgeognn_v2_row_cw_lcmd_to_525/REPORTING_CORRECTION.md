# Reporting correction provenance

- Original reporting code SHA-256: `d607b1c5fdfa7c04edc1c36a0686743a8dff6fa322f94c22200d720f25462463`.
- Corrected reporting code SHA-256: `ac3fa88e3f702dff7f3aa69979e55da088cfe5523a147011d9f0b001d83f7140`.
- Frozen experiment protocol: unchanged.
- Checkpoints, predictions, selected-batch files/IDs, and test split: unchanged; 18 protected artifacts were verified byte-for-byte before and after regeneration.
- Test truth was not read again and was not used for method design. This pass read only previously frozen metric/reporting artifacts.
- Training, acquisition, label reveal, and historical artifact reruns: none.

The former `AULC_333_525` calculation assembled only the 429, 461, 493, and 525 points and divided that partial trapezoidal area by `525 - 333`. It omitted the 333, 365, and 397 points, so it was not an integral over the claimed interval. The corrected calculation requires exactly seven points—333, 365, 397, 429, 461, 493, and 525—for every seed and method, and fails loudly on any missing or duplicate point.

The former PROMISING rule used `abs(mean endpoint delta) <= 0.03`. Because endpoint delta is CW minus the strongest baseline, a negative value is an improvement; taking the absolute value incorrectly rejected sufficiently large improvements as though they were deterioration. The corrected no-deterioration rule is `mean endpoint delta <= +0.03`.

The canonical decision changed from **MIXED** to **PROMISING**. It did not change to STRONG_MIDSTAGE_SIGNAL: both seeds have better AULC 429-525 than the same strongest baseline, but seed 6101 does not have a better endpoint. This correction does not authorize continuation to 653.

The formerly misnamed `per_seed_cw_continuation_101.png` is retained only under `figures/legacy/reporting_error/` and is deprecated. The canonical seed figures are `per_seed_cw_continuation_157.png` and `per_seed_cw_continuation_6101.png`.
