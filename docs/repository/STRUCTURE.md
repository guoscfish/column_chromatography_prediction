# Repository structure

## Read in this order

[Project README](../../README.md) -> [current status](../NEXT_STAGE_DECISION.md) -> [study index](../../studies/README.md) or [code map](../../src/qgeognn_al/README.md).

| Directory | Owns | Reading rule |
| --- | --- | --- |
| `src/qgeognn_al/` | Scientific implementation | Start with `models/qgeognn_v2.py`; use the code map for shared and historical modules |
| `scripts/studies/` | Retained study runners | [Script index](../../scripts/README.md) distinguishes current work from reproduction dependencies |
| `scripts/` | Maintenance, compatibility, retained Legacy dependencies | Do not add a new top-level experiment runner |
| `studies/` | Predictor, active-learning and transfer protocols/results | Topic READMEs select the authoritative reports |
| `experiments/` | Early frozen evidence and shared source anchors | Historical store; directory age does not establish current authority |
| `tests/` | Implementation, leakage and provenance checks | Fast checks, not formal scientific reruns |
| `docs/` | Current status, contracts and maintenance | One page per responsibility; study-level detail belongs beside results |
| `dataset/` | Source data | Preserve original identities and provenance |
| `application/`, `automation/` | Legacy application and collection code | Some Legacy model code remains a dependency |

The old `track_a_4g_al`, `track_b_transfer`, and `track_c_active_transfer` directories remain at their recorded paths. Current topic indexes link to them. Moving frozen splits, checkpoints, or hashed protocols would break provenance.

`NEXT_TRANSFER_MODEL_AUDIT.md` remains at the root because its exact bytes are hashed by the residual-diagnostics protocol.

## Branches

- `main`: integrated research baseline.
- `codex/4g-evaluation-adaptivity`: current working branch; includes ongoing, uncommitted Phase 1 work.
- `exp/qgeognn-v2-4g-al-innovation-screen` and `exp/qgeognn-v2-4g-cw-lcmd-performance`: separate worktrees with research commits not integrated into the current branch.
- The local alias of the current tip and the already-integrated matched-RMSE branch were removed; details are in the [cleanup status](REPOSITORY_HYGIENE_AUDIT.md).

Use one branch per independent change. When its commits are integrated and no worktree uses it, delete the redundant branch. Retain an immutable tag when a result still needs a distinct recovery point. Do not merge experimental branches solely to reduce their count.

## Runtime and retirement

Large fits, histories, predictions and caches belong under ignored `runtime/`. Compact protocols, decisions, tables and audit records remain versioned.

Closed one-off scripts with no remaining code consumers may be retired while retaining their results. [RETIREMENTS.json](RETIREMENTS.json) records paths, hashes, evidence and recovery commits. Shared helpers, active experiment code, frozen source files and user work in progress stay in place. See the [retention policy](../ARTIFACT_RETENTION_POLICY.md).
