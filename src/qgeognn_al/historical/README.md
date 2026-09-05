# Historical implementations

Clean is a failed point-performance architecture experiment / historical negative result. `clean_fusion.py` and `clean_schema.py` are retained exclusively for historical reproduction and regression tests. They are not imported by the standalone predictor API or new source/transfer runners.

`condition_complete_v2.py` and `condition_complete_v2_pruned.py` at the package root are deprecated diagnostic/checkpoint-migration controls. Their paths remain stable to preserve recorded code hashes and historical tests. Likewise, the Clean, regression-ladder, R2-pruning and preflight scripts under `scripts/studies/` are reproduction-only. New scientific work imports `qgeognn_al.models.build_predictor` and `load_predictor_checkpoint`.

No historical measured results, reports, manifests or protected checkpoints were removed. `conversion.py` is an explicit migration utility, never called by the active builder.
