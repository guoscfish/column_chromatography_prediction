# Project structure

Start with the root `README.md`, then `studies/README.md` for current work and `experiments/INDEX.md` for frozen history.

- `src/qgeognn_al/` is the reusable scientific core for data, models, training, acquisition, metrics, partitions, artifacts, and diagnostics.
- `scripts/al_engine.py`, `scripts/al_acquisition.py`, and `scripts/qgeognn_graphs.py` are historical compatibility shims, not reusable-core ownership boundaries.
- `scripts/studies/` contains all new study runners. New runners import `src.qgeognn_al.*`, never historical `scripts.run_*` modules.
- `experiments/` is a frozen historical record store. Do not add new top-level experiment directories or move existing history.
- `studies/` contains current and future research, organized by Track A/B/C. Runtime lives under `studies/**/runtime/` and is never committed.
- `tests/` contains fast contract and regression tests; it does not rerun frozen scientific experiments.
- `dataset/` contains source data anchors and is not reorganized during the E4 transition.
- `application/` and `automation/` are legacy product/data-collection surfaces and remain in place.

Do not infer authority from directory age. Use `experiments/INDEX.md` for history and `studies/README.md` for active work. Large per-fit checkpoints, histories, predictions, progress, and states are runtime artifacts unless explicitly protected.
