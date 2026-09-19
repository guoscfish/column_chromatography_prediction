# Documentation

Use these pages in order; study reports own the detailed measurements.

| Purpose | Canonical page |
| --- | --- |
| Current conclusions and next step | [Research status](NEXT_STAGE_DECISION.md) |
| Results by topic | [Study index](../studies/README.md) |
| Read the implementation | [Code map](../src/qgeognn_al/README.md) |
| Run or reproduce a study | [Script index](../scripts/README.md) |
| Directory ownership and branches | [Repository structure](repository/STRUCTURE.md) |
| Write and maintain a study | [Conventions](CODEBASE_CONVENTIONS.md) |
| Keep or retire artifacts | [Retention policy](ARTIFACT_RETENTION_POLICY.md) |
| Data exposure and split semantics | [Data usage](protocols/DATA_USAGE_REGISTER.md) |

## Research context

- [Transfer evidence and remaining questions](research/CROSS_COLUMN_TRANSFER_STATUS.md)
- [4g active-learning review](research/4G_ACTIVE_LEARNING_REVIEW_2026-09-17.md)
- [Current matched-budget experiment and Phase 2 plan](research/4G_ACTIVE_LEARNING_NEXT_EXPERIMENT_2026-09-18.md)
- [Model implementation variants](QGEOGNN_IMPLEMENTATION_VARIANTS.md) and [input schema](model/INPUT_SCHEMA.md)

Dated reviews provide rationale; the current status and frozen study protocols determine what is complete or still pending. Legacy/Clean design documents under `model/` and the benchmark under `protocols/` are historical contracts, not alternate current roadmaps.

Superseded navigation and plans have been consolidated. [Cleanup status](repository/REPOSITORY_HYGIENE_AUDIT.md) explains retention and branch decisions; [RETIREMENTS.json](repository/RETIREMENTS.json) records the exact recovery commit for each removed file.
