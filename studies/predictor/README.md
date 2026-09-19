# Predictor studies

The current predictor is **standalone QGeoGNN-V2**, qualified for point-transfer studies. Clean is a failed historical architecture experiment.

| Read first | What it establishes |
| --- | --- |
| [Standalone engineering](final_v2_engineering/README.md) | 458,952 effective parameters; exact six-output equivalence to R2-pruned |
| [Final 4g qualification](final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) | Six frozen row/compound runs; current point-prediction evidence |
| [Quantile audit](final_4g_qualification/QUANTILE_AUDIT.md) | Current head retained for point prediction; UQ needs further qualification |
| [Checkpoint and reproduction](final_4g_qualification/README.md) | Required local runtime anchors and commands |

Implementation starts at [models/qgeognn_v2.py](../../src/qgeognn_al/models/qgeognn_v2.py).

## Historical development

Legacy -> condition completion -> R2 pruning -> standalone V2. The [historical index](historical/README.md) covers Clean, the regression ladder and pruning. [Semantic input audit](semantic_input_audit/README.md), [condition completion](condition_completion/README.md) and [the superseded benchmark](4g_source_benchmark/README.md) preserve earlier reasoning.

Original decisions remain frozen. Their old pending gates do not supersede [current research status](../../docs/NEXT_STAGE_DECISION.md).
