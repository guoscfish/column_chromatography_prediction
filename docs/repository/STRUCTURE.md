# Repository structure

## Ownership

- `src/qgeognn_al/`: reusable scientific logic. New model and schema code must import from this package, never from a historical `scripts/run_*.py` file.
- `scripts/`: historical runners, current thin runners, and compatibility shims.
- `studies/`: semantic navigation plus compact current-study records.
- `experiments/`: frozen historical provenance store. Do not bulk move, rename, delete, or cosmetically reorder it.
- `docs/`: current contracts and retained historical documentation.
- `tests/`: fast implementation, provenance, and scientific-boundary checks; never a substitute for a formal experiment.
- `dataset/`, `application/`, `automation/`: retained data and legacy product/data-collection surfaces.

## Study navigation policy

New work is organized by scientific topic: `predictor/`, `transfer/`, and `active_learning/`. Experiment IDs may remain in metadata but are not the primary navigation scheme.

Existing I0, Predictor V2, S1, T1, T1b-1, A1a, and Track C directories were not moved. Current scripts and tests contain their paths, and some tests protect result files by SHA-256. The semantic study directories therefore link to those authoritative locations. This preserves provenance without forcing future work into `track_b_transfer/`.

## Runtime boundary

Large checkpoints, histories, per-fit predictions, state files, and engineering caches belong in gitignored `runtime/` directories unless explicitly registered as protected anchors. Compact configs, decisions, hashes, audits, and a small set of plots form the scientific record.
