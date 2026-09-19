# Scientific code map

Start with the current predictor, then follow its data, training and evaluation paths. Study orchestration lives in [scripts/studies](../../scripts/studies); scientific code must not import those runners.

## Current predictor

| Module | Responsibility |
| --- | --- |
| [models/qgeognn_v2.py](models/qgeognn_v2.py) | `build_predictor`, `load_predictor_checkpoint`, forward prediction and representation extraction |
| [schemas/conditions.py](schemas/conditions.py) | Typed condition contract |
| [data.py](data.py), [input_schema.py](input_schema.py) | Canonical records, graph construction and input semantics |
| [training/predictor.py](training/predictor.py) | Source training, preprocessing and checkpoint selection |
| [evaluation/point.py](evaluation/point.py), [evaluation/reporting.py](evaluation/reporting.py) | Point metrics and qualification reports |

## Current 4g active learning

Read `active_learning_v2/` in this order:

1. `protocol.py`, `sequential_protocol.py`: row identities, label accounting, budgets and frozen-source contracts.
2. `gradient_features.py`, `gradient_bank.py`, `lcmd.py`: gradient representation and selection.
3. `sequential_acquisition.py`, `sequential_runner.py`: per-round selection, retraining, freeze and evaluation.
4. `batch_adaptivity.py`: current fixed-budget Static/Adaptive comparison.
5. `sequential_reporting.py`, `efficiency_reporting.py`: learning curves, label targets and cost reporting.

`benchmark*` and `phase1.py` implement the earlier one-step benchmark; `phase1.py` is not the September batch-adaptivity study. `ivr*` and `block_ivr.py` retain the completed IVR implementation and diagnostics.

The selection-only innovation screen uses `gradient_transforms.py` and `innovation_screen.py`.
It compares fixed representations and selectors on frozen L0 inputs; it does not retrain a model
or inspect test labels. Keep this mechanism screen separate from sequential performance claims.

## Transfer

| Area | Modules under `transfer/` |
| --- | --- |
| Label boundaries, splits and metrics | `protocol.py`, `evaluation.py`, `baseline.py` |
| Neural adaptation | `adaptation.py`, `source_anchored.py`, `full_data.py` |
| Strong calibration controls | `calibration.py`, `matched_calibration.py`, `conditional_scaling.py` |
| Corrected structural control | `hierarchical_cw_corrected.py` |
| Completed controlled hypotheses | `adaptive_readout.py`, `source_augmentation.py`, `column_conditioned_multitask.py`, `pcgrad.py` |
| Historical calibration/architecture diagnostics | Remaining Center/Width, latent, scaling and physics modules |

The broad `transfer/__init__.py` exports are a compatibility surface, not a ranking of preferred models. Use explicit modules for new work and consult [the transfer index](../../studies/transfer/README.md) before selecting a baseline.

## Shared and historical code

- `partitions.py`, `metrics.py`, `resources.py`, `artifacts.py`: shared infrastructure.
- `model.py`, `engine.py`, `acquisition.py`, `t1_formal.py`, `t1b1.py`: retained Legacy scientific implementation; they are not the standalone predictor entry point.
- `condition_complete_v2.py`, `condition_complete_v2_pruned.py`: equivalence and reproduction controls.
- `historical/`: failed Clean architecture and conversion controls.
- `diagnostics/`, `uncertainty/`: diagnostic and quantile utilities; presence does not imply uncertainty qualification.

These files have live import, regression-test or frozen-hash consumers. Directory cleanup does not justify changing model semantics or relocating them during an active sealed experiment.
