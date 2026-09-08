# Repository hygiene audit

Status: `AUDIT_COMPLETE / PAPER_RUNTIME_CLEANUP_COMPLETE / BRANCH_CLEANUP_PENDING_MAIN_INTEGRATION`

This is a read-only evidence record plus a disposition of the explicitly
authorized paper-transfer cleanup. It does not authorize deletion of other
historical studies, protected artifacts, local branches, or remote branches.
The dependency-free checker can be rerun with:

```text
python scripts/audit_repository_hygiene.py
```

## Audit snapshot and frozen anchors

- Paper-transfer snapshot: `cf219034b02b9252e5eea10f2d741304efbfb51e`
  on `codex/reproduce-paper-transfer-25g-40g-rmse`.
- Archive tag: `archive/pre-matched-rmse-cleanup-2026-09-08`, verified to
  resolve to that snapshot and contain the paper-transfer tip.
- Current transfer source:
  `studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt`.
  SHA-256: `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`.
- Cross-column schedule SHA-256:
  `b52b937aa750586ec8f2efb2f10f93dfddd2252bb03afaf304556e16b4fab5ee`.
- All 15 entries in `docs/PROTECTED_ARTIFACTS.json` existed during the
  audit. None was a cleanup target.

The qualified source checkpoint is intentionally an ignored local runtime
artifact; the final-qualification README documents the required exact restore
or reproduction step for a fresh clone. Its hash is the transfer contract.

## Branch ancestry

`git merge-base --is-ancestor`, `git branch --contains`, and bidirectional
`git rev-list --count` establish a single linear research chain. In every
adjacent pair, the ancestor-only count is zero and the descendant adds one
commit:

| ancestor | descendant | ancestor-only commits | descendant-added commits |
| --- | --- | ---: | ---: |
| `study/4g-to-8g-transfer` | `study/cross-column-transfer-validation` | 0 | 1 |
| `study/cross-column-transfer-validation` | `codex/study-transfer-residual-diagnostics` | 0 | 1 |
| `codex/study-transfer-residual-diagnostics` | `codex/study-scaling-failure-audit` | 0 | 1 |
| `codex/study-scaling-failure-audit` | `codex/study-source-anchored-shared-transfer` | 0 | 1 |
| `codex/study-source-anchored-shared-transfer` | `codex/study-physics-column-conditioned-transfer` | 0 | 1 |
| `codex/study-physics-column-conditioned-transfer` | `codex/reproduce-paper-transfer-25g-40g-rmse` | 0 | 1 |

The preceding six research refs are therefore
`SUPERSEDED_LINEAR_RESEARCH_BRANCH` candidates. The paper tip is currently
six commits ahead of `main`; it has not yet been integrated. Its remote and
all superseded remote branches remain in place. Do not delete any of them
until the current research branch has merged into `main`, the archive tag and
main have been pushed, tests pass, and scientific records remain accessible.

## Paper-transfer runtime disposition

At the snapshot, the paper-reproduction study had 107 indexed paths. Eighty-one
were reproducible runtime objects totaling 71,461,351 bytes:

| class | count | disposition |
| --- | ---: | --- |
| `best.pt` checkpoints | 20 | moved to ignored `runtime/runs/` |
| fit `history.csv` files | 20 | moved to ignored `runtime/runs/` |
| per-run and aggregate predictions | 21 | moved to ignored `runtime/` |
| per-run `result.json` files | 20 | moved to ignored `runtime/runs/`; compact fields retained in `run_summary.csv` |

The current index has zero paper-transfer runtime candidates. The local
ignored runtime still contains all 81 migrated objects, so this cleanup does
not discard the immediately available reconstruction cache. The historical
runner now writes all such outputs to `runtime/` automatically. Its compact
root manifest excludes ignored runtime paths.

`run_summary.csv` retains 20 column/protocol/seed/method records with:

- best epoch and epochs run;
- trainable and total parameter counts;
- source checkpoint and SHA-256;
- validation selection score and normalized validation/test RMSE;
- V1/V2 validation and test RMSE, MAE, and R2; and
- protocol, method, seed, and column-spec marker.

This preserves the scientifically useful scalar provenance while keeping
checkpoints, histories, and per-sample predictions reproducible rather than
tracked.

## Redundancy and dependency mapping

The dependency scan covered code, tests, docs, and manifests before removal.
No consumer outside the old paper-study manifest referenced the removed paths.
That manifest was regenerated from retained compact records only.

| former path(s) | classification and deterministic basis | retained replacement / disposition |
| --- | --- | --- |
| `runs/**/{best.pt,history.csv,predictions.csv.gz,result.json}` | reproducible runtime from the historical runner | ignored `runtime/runs/`; scalar metadata in `run_summary.csv` |
| root `predictions.csv.gz` | concatenation of the per-run prediction tables | ignored `runtime/predictions.csv.gz`; aggregate metrics remain in `all_metrics.csv` and `PAPER_TRANSFER_RMSE_SUMMARY.csv` |
| `canonical_25g_legacy_filtered.csv`, `canonical_40g_legacy_filtered.csv` | deterministic filter of the frozen cross-column canonical target using the threshold and raw-data hashes recorded in `protocol.json` | regenerated in memory by `read_target_data`; no duplicate tracked copy |
| ten `splits/*_row_seed_*.csv` files | deterministic `RandomState(seed)` row partition over that named filtered canonical order | one retained compact `split_manifest.csv`, plus protocol, seeds, and `make_row_split` implementation |
| `artifact_manifest.json` | formerly included runtime and redundant paths | regenerated to hash only 14 retained compact study records |

The retained root record contains protocol/config/environment, implementation
and provenance audits, filtering caveat, compact split ledger, all scalar
metrics, summary tables, R2 comparison, reports, and `run_summary.csv`.
The paper reproduction remains
`PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION`, uses `legacy_filtered`, and uses
the old E0 source SHA
`7b9e3d0d4c8036c738ef220802e7ee46bc6ab8261cc541fb7d194e8c17044323`.
It is a historical reference and is never eligible for a matched-strategy
ranking.

## Protocol and repository guards

The audit validates that the frozen cross-column protocol retains
`target_threshold: null`, validation-only neural selection,
gradient-train-only simple fitting, `test_tuning: false`, the expected source
hash, schedule hash, and a 120-context schedule with no target truth columns
or role/budget-ledger violations.

The hygiene tests additionally guard public scientific code against imports
from historical top-level `scripts/run_*.py`, ensure runtime/progress paths
are ignored and untracked, verify protected entries and hashes, and require
the paper study's compact manifest/summary boundary. The matched-benchmark
protocol guard verifies the inherited no-threshold source, schedule, labels,
columns, seeds, budgets, and no-test-selection declarations once that study is
materialized.

The wider study scan still reports historical runtime-like file names in
retained historical result trees. That inventory is diagnostic only, not a
bulk-deletion list: `experiments/` and unrelated retained study evidence are
protected by the artifact policy and were not moved or rewritten here.

## Required final checks

```text
python scripts/audit_repository_hygiene.py
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/test_repository_hygiene_contracts.py
git diff --check
git status --short --branch
```

After main integration, separately verify the pushed tag and mainline, then
delete superseded local and remote research branches only if the branch
containment audit still shows no ancestor-only commits.
