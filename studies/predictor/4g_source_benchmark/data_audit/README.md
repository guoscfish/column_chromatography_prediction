# 4g source threshold audit

Status: `4G_THRESHOLD_AUDIT_COMPLETE / QUALIFICATION_BLOCKER_CLOSED_BY_PROJECT_CONTINUITY_DECISION`.

This source-only audit starts from `dataset/dataset_4g.csv` (SHA-256 `d485d3d46a96458d1baac11b2a21cd33374b1be42ba2303e7ae5823cc8ee553a`) and reproduces the repository volume conversion `V_ml = t_raw * Flow_mL_min / 1200`. A valid-label row has finite numeric t1, t2, and flow-derived V1/V2 and does not use an explicit `-1` label sentinel.

## Counts

| Quantity | Count |
|---|---:|
| Raw rows | 4,243 |
| Valid-label rows | 4,229 |
| Unique valid-label compounds (`canonical_smiles`) | 217 |
| V1 > 60 mL | 60 |
| V2 > 120 mL | 49 |
| Union affected by either threshold | 66 |
| Affected unique compounds | 29 |
| Compounds completely removed | 0 |
| Rows retained by the legacy thresholds | 4,163 |

All 66 affected labels are finite numeric observations under the repository conversion: none contains NaN, a `-1` sentinel, or a non-finite value. This is not proof that they are uncensored or scientifically valid. One negative V2 value occurs in the retained population, not in the threshold-affected set, and remains a separate label-quality issue.

## Distribution and concentration

Before filtering, V1 ranges from 0.1333 to 165.2 mL (median 5.8) and V2 from -0.1 to 247.3333 mL (median 12.625). After filtering, maxima are 59.5833 and 119.425 mL; medians are 5.7667 and 12.525 mL. Full quantiles, means, standard deviations, and condition distributions are in `threshold_audit.json`.

Removal is not uniform. DCM loading solvent accounts for 61/66 affected rows (2.074% of its 2,941 valid rows), PE for 5/66 (0.398% of 1,255), and EA for none. The affected eluent ratios are concentrated at 10/1 (24 rows), 50/1 (21), 20/1 (18), and 5/1 (3). At the combined condition level, DCM with 50/1 has 20/319 affected rows, DCM with 10/1 has 23/424, and DCM with 20/1 has 15/371. Twenty-nine compounds are affected, but every one retains at least one valid row under the legacy filter.

## Rationale trace and policy

The official released `application/QGeoGNN.py::Construct_dataset` hard-codes `V1 <= 60` and `V2 <= 120`. The paper Methods and the official/current repository README/docs inspected for this audit do not provide a physical, instrument-range, censoring, or measurement-validity justification. Conclusion: `NO_CONFIRMED_PAPER_LEVEL_RATIONALE_FOUND`.

This audit does **not** conclude `thresholds_are_wrong` or `thresholds_are_correct`. For the current Clean 4g baseline qualification, the project fixes the historical `V1 <= 60`, `V2 <= 120` domain solely to preserve continuity with prior predictor, transfer, and active-learning work. This is a `PROJECT_CONTINUITY_DECISION`, not a physical, instrument-limit, optimality, or scientific-truth claim. No 4,163-versus-4,229 model-performance comparison is authorized.

## Artifacts

- `threshold_audit.json`: counts, label/condition distributions, concentration tables, and rationale trace.
- `threshold_affected_rows.csv`: all 66 affected source rows with derived volumes and validity flags.
- `data_contract.json`: source identity, label/compound definitions, threshold status, and data-use boundary.
- Generator: [`../../../../scripts/studies/run_4g_threshold_audit.py`](../../../../scripts/studies/run_4g_threshold_audit.py).
