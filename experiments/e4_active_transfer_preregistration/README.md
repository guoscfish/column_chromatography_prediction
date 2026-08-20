# E4 Active Transfer Preregistration

**Status: preregistered design and preflight only. No formal E4 model training has been run.**

The single scientific question is whether a frozen 4g QGeoGNN source prior can reach full-data 8g transfer performance with fewer expensive target labels than random target-label selection.

Protocol A is shared-chemistry calibration (D28 frozen row partitions). Protocol B is canonical-compound-held-out generalization (D28 frozen compound partitions); Bemis–Murcko scaffold OOD is deferred to E5. Both use `canonical_8g_no_threshold.csv` (574 rows), retaining the historical V1>60 or V2>120 tail.

The transfer contract is Gate 0 `last2 + head`, monotonic quantile head, Adam `lr=1e-4`, source scaler and validation-selected checkpoints. E2 was source-free random initialization; E4 loads a complete 4g prior and reinitializes target adaptation from the same source anchor at every round. It never warm-starts from the previous round.

Pilot: 3 outer seeds (42, 525, 1101), K=3 source members, L0=50 (42 train + 8 validation), B=10, 15 rounds (50 through 200). Acquisition methods are Pretrained+Random, +Coverage, +Ensemble, +Hybrid, and legacy +Quantile Width. Controls are 4g zero-shot, 8g scratch+Random, pretrained+Random, pretrained full-data ceiling, and optional scratch ceiling.

Primary metric is normalized target NRMSE and its trapezoidal AULC (lower is better). Recovery is `(E_baseline-E_t)/(E_baseline-E_full)` with zero-shot as the primary baseline; if the denominator is non-positive it is reported undefined, never forced. Labels-to-90%/95% are the first budgets with recovery at or above the threshold.

Protocol B records post-hoc test-to-labeled and selected-to-test fixed distances, compound HHI, duplicate concentration, and tail/common error. Test rows are never used for acquisition, scaler fitting, early stopping, checkpoint choice, or protocol changes.

Expansion to >=5 seeds x K=5 is allowed only after pilot paired variance, effect size, and measured compute are reviewed. Failure audit and E4 design do not reopen E2 or the frozen predictor.
