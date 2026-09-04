# Repository cleanup candidates

Status: `AUDIT_ONLY / NO_DELETION_PERFORMED`.

This classification reflects the point-predictor regression diagnostic. It is not a deletion authorization. Reproducibility anchors and artifacts must remain available until the regression is fixed, reviewed, and merged.

## KEEP_ACTIVE

These files are required by the current diagnostic or the likely next controlled predictor work:

- `application/QGeoGNN.py`: executed Legacy implementation required for R0/R1/R2 reproduction.
- `src/qgeognn_al/data.py`, `model.py`, and `resources.py`: shared data/model/resource contracts.
- `src/qgeognn_al/models/clean_fusion.py` and `schemas/clean.py`: current R3 implementation under diagnosis.
- `src/qgeognn_al/condition_complete_v2.py`: R2 control that localizes the regression.
- `scripts/run_e0_4g_baseline.py`: historical E0 contract and reproduction reference.
- `scripts/studies/run_point_predictor_regression_audit.py` and `summarize_point_predictor_regression_audit.py`: current controlled runner and artifact audit.
- `tests/test_point_predictor_regression_audit.py` plus core Legacy/V2/Clean compatibility tests.
- `experiments/e0_4g_baseline/` frozen split, canonical data, scaler, graph cache, checkpoint, metrics, and provenance artifacts.
- `studies/predictor/performance_regression_audit/` protocol, results, reports, and decisions.

## KEEP_AS_REFERENCE

These retain scientific or reproducibility value but are not the active production predictor path:

- `studies/predictor/clean_4g_baseline_qualification/`: measured six-run Clean evidence, now reinterpreted as engineering/input-contract evidence with point qualification reopened.
- `studies/predictor/clean_qgeognn/preflight/`: Clean engineering preflight provenance.
- `studies/predictor/condition_completion/` and `studies/track_b_transfer/predictor_v2_preflight/`: V2 design and reachability evidence.
- `studies/predictor/semantic_input_audit/`: reason the redesign began.
- `studies/predictor/4g_source_benchmark/`: superseded multi-model preregistration and threshold audit provenance.
- Historical E1/E2/A1a/G0/S1/T1/T1b/E4 results and runners: retain as explicitly labeled Legacy-contract or prior-study evidence; do not treat them as current Clean validation.
- `docs/model/PAPER_CODE_CONTRACT_AUDIT.md` and historical protocol/decision documents.

## ARCHIVE_CANDIDATE

These are candidates for a later archive namespace or release bundle after references are mapped and tests are migrated. They are not deletion candidates now:

- `scripts/studies/run_clean_4g_baseline_qualification.py` and `summarize_clean_4g_baseline_qualification.py`: completed one-off qualification execution; still needed to reproduce retained results.
- `scripts/studies/run_predictor_v2_preflight.py`: completed one-off engineering preflight.
- `scripts/studies/run_i0_predictor_semantic_audit.py` and `run_4g_threshold_audit.py`: completed audit runners.
- `tests/test_clean_4g_baseline_qualification.py`, `test_clean_qgeognn_preflight.py`, `test_predictor_v2_preflight.py`, and `test_predictor_v2_preregistration.py`: old-stage infrastructure that should be archived only after durable contract tests cover the same behavior.
- Superseded draft benchmark materials after all provenance links are preserved.

Archiving should mean moving to a documented historical/reproduction area, not dropping it from Git history or silently breaking artifact paths.

## DELETE_CANDIDATE

None are authorized or currently proven safe. A future item may enter this category only after demonstrating all of the following: exact duplication, zero live references, no scientific reproducibility value, recovery from Git history, and no effect on checkpoint or experiment reproduction.

## Branch cleanup recommendation

Current development history is effectively linear:

`main -> model/clean-qgeognn -> model/clean-qgeognn-prebenchmark -> study/clean-4g-baseline-qualification`.

After a controlled Clean regression fix is accepted and the final predictor line is merged, delete only phase branches whose commits are fully contained in the merged mainline. Do not delete them during this audit. Future branches should correspond to independent scientific questions—such as `study/uq-calibration`, `study/4g-to-8g-transfer`, and `study/active-transfer`—rather than every implementation/preflight/qualification phase.
