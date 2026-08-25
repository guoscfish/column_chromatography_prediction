# Codebase and experiment documentation conventions

## Documentation ownership

### `README.md`

The top-level README contains only the current research question, current main conclusions, repository structure, primary entry points, and a concise environment guide. Detailed experiment results belong in their experiment directories.

### `EXPERIMENT_PLAN.md`

This file owns the scientific stage, preregistrations, frozen protocols, gates, and the current next step. A completed experiment must not remain described as pending.

### `experiments/INDEX.md`

The index contains exactly one row per experiment key with the fields `experiment`, `stage`, `status`, `authoritative?`, `scientific role`, and `superseded_by`. Add or update the one authoritative row whenever experiment state changes. Never register the same key twice.

### `experiments/METHOD_DECISION_REGISTER.md`

The register records method decisions, rationale, supporting evidence, excluded interpretations, and unresolved questions. It is not an execution log.

### `experiments/RESEARCH_ROADMAP.md`

The roadmap contains deferred hypotheses, conditional experiments, future directions, and their trigger conditions. Completed work may appear only as concise background and must not remain a future task.

### `scripts/README.md`

This is the executable map. Every new `run_*.py` entry records its experiment ID, purpose, scientific role (`engineering`, `diagnostic`, or `formal`), and whether it may be run directly.

## Experiment directory contract

Every new experiment directory contains at least:

- `README.md`
- `config.json`
- `environment.json`
- `decision.json`

Its README contains these sections: A. Scientific question; B. Why this experiment exists; C. Inputs / frozen dependencies; D. Dataset and split; E. What truth is visible at each stage; F. Method; G. Metrics; H. Exact commands; I. Outputs; J. Result; K. Interpretation; L. Limitations; M. Next decision.

Diagnostic experiments must explicitly identify post-hoc or test-truth use, state that they are not confirmatory evidence, and describe the contamination consequence for future validation. Large reproducible fit artifacts belong in gitignored runtime directories; compact metrics, hashes, audits, configuration, and figures are retained.
