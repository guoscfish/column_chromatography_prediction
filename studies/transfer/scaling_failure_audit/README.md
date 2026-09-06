# Scaling failure audit

Decision: **`STRUCTURED_FAILURE_BUT_NO_MATERIAL_MODEL_GAIN`**.

Training-only evidence identifies reproducible EA/V1 structure. One low-capacity conditional-scaling direction was run across 8g/25g/40g, row/target-compound splits, five seeds, and four frozen budgets (120 contexts). It does not achieve the preregistered replicated material improvement beyond strong controls. No further method was appended after test evaluation.

- [Scientific audit, A–E answers and data requirements](SCALING_FAILURE_AUDIT.md)
- [Model decision, full comparisons and standalone/policy distinction](NEXT_MODEL_DECISION.md)
- [Initial audit preregistration](PREREGISTRATION.md), [training-only direction decision](training_direction_decision.json), [subsequent model preregistration](MODEL_PREREGISTRATION.md)
- [Prior method coverage and 11-item research backlog](../../../docs/research/CROSS_COLUMN_TRANSFER_STATUS.md)
- [Per-seed point metrics](model_all_metrics.csv), [aggregates](model_aggregate_metrics.csv), [AULC](model_aulc_by_seed.csv), [paired AULC](model_paired_aulc.csv), [paired budget-100 metrics](model_paired_budget100.csv)
- [Pair identities](pair_identity_audit.csv), [coverage](pair_coverage.csv), [scale stability](scale_stability_by_seed.csv), [condition evidence](condition_evidence_summary.csv), [molecular evidence](molecule_evidence_summary.csv)
- [Verification](verification.json), [complete execution accounting](execution_audit.json), [artifact hashes](artifact_manifest.json)

Eight figures are provided as PNG and PDF in `plots/`. Per-context `models/` contains blind predictions, all candidate validation scores, label IDs, coefficients and frozen hashes. The qualified predictor and preceding experiment results are unchanged. Target-compound holdout is not source-unseen OOD.

Run the five commands in the scientific audit **sequentially**. A parallel invocation of model evaluation and descriptive reporting initially caused one protocol-read JSON error; the descriptive phase succeeded on a sequential retry with the identical code and contract. This did not change models or selection. The execution audit distinguishes this resolved reporting-stage failure from model failures.

The research-status document is the pre-audit snapshot, hash-locked before direction selection. The reports here and the project next-stage decision own the subsequent results. Future hypotheses remain hypotheses; no automatic follow-on training is authorized by this study's decision.
