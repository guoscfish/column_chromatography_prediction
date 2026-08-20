# Project structure

Start with the root `README.md`, then `experiments/RESEARCH_ROADMAP.md` and `experiments/INDEX.md`. The code-to-result map is in `scripts/README.md`.

- `scripts/` contains reusable modeling/acquisition components plus stage runners. The reusable entry points are `al_engine.py`, `al_acquisition.py`, and `qgeognn_graphs.py`.
- `experiments/` contains frozen scientific aggregates, decisions, provenance, and bounded engineering outputs. Historical directories are retained as provenance even when superseded.
- `tests/` contains fast contract and regression tests; it does not rerun frozen scientific experiments.
- `dataset/` contains source data anchors and is not reorganized during the E4 transition.
- `application/` and `automation/` are legacy product/data-collection surfaces and remain in place.

Do not infer authority from directory age. Use `experiments/INDEX.md`, and treat experiment READMEs plus decision JSON/CSV aggregates as the scientific record. Large per-round checkpoints, histories, predictions, and states are runtime artifacts unless a README identifies a checkpoint as a required Gate0/E1 source anchor.
