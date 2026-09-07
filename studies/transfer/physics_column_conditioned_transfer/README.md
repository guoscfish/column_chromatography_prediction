# Physics column-conditioned transfer

This directory contains the nominal-mass/flow metadata audit and a frozen 120-context experiment. Consult execution_audit.json for actual completion state; planned counts are not results.

The historical qualified predictor, data, splits and reference artifacts remain unchanged. New architecture lives in `src/qgeognn_al/transfer/column_conditioned.py`; physical schema and training-only normalization in `column_physics.py`.

Execution order (repository root, conda fish):

```bash
python scripts/studies/audit_column_physics.py
python scripts/studies/supplement_column_physics_audit.py
python -m pytest tests/test_physics_column_transfer.py tests/test_final_v2_predictor.py -q
python scripts/studies/run_physics_column_transfer.py --execute --workers 3
python scripts/studies/evaluate_physics_column_transfer.py
python scripts/studies/summarize_physics_column_transfer.py
python -m pytest tests/test_physics_column_transfer.py -q
```

Audit generation intentionally refuses overwriting its freeze. Do not rerun first-time audit generation on an existing frozen study. Model execution resumes identical contracts and skips verified completed contexts; code/data/protocol drift stops reuse. Only after the global freeze does the separate evaluator read test truth. Checkpoints, optimizer state, per-epoch loss history and logs stay in ignored runtime; blind predictions and identity/hash ledgers remain retained research artifacts.

Audit interpretation: nominal packing mass is an empirical engineering proxy, not verified bed or void volume. Old geometry constants exist but are not verified measurements. 4g flow is not constant. 8g metadata says 4g+4g. Audit outcome comparisons use frozen first-seed purchased training labels; those rows may occur in other seeds' heldout roles, so this is developmentally exposed evidence, not globally unseen test data. No audit labels enter another context's optimization or normalization. “Predictions frozen before test” specifically denotes this stage's model-evaluation boundary, not absence of historical/audit exposure.

Interpretation details: relative_difference in scale_physics_comparison.csv is `abs(frozen_scale - packing_mass_ratio) / packing_mass_ratio`. Loading mass follows the inherited density × sample-volume contract and is not additionally purity corrected; experimental confirmation of stock/sample composition is needed before interpreting it as exact analyte mass. Post-freeze prediction_physical_sanity.csv records negative volumes and reversed predicted windows without applying an unregistered clipping or head replacement.
