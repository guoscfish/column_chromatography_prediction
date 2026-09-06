# Transfer studies

[Current residual diagnostics](residual_diagnostics/RESULT_INTERPRETATION.md) conclude `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA` after train-only monotone calibration and matched-budget shared-column calibration. No further model or active-learning study follows automatically. The [preceding cross-column validation](cross_column/CROSS_COLUMN_TRANSFER_REPORT.md) remains unchanged as the frozen evidence base.

[Current final-source baseline](4g_to_8g/TRANSFER_BASELINE_REPORT.md) uses standalone QGeoGNN-V2. See its protocol and decision for the frozen design and measured ranking.

The following are `HISTORICAL_LEGACY_PREDICTOR_EVIDENCE`, retained for design reuse and provenance:

- [Source-target shift / S1](source_target_shift/README.md)
- [Low-label adaptation / T1](low_label_adaptation/README.md)
- [Adapter capacity / T1b](adapter_capacity/README.md)

Their measured rankings must not be presented as results for the corrected source. [Current transfer roadmap](../../docs/roadmap/TRANSFER_4G_TO_8G.md).
