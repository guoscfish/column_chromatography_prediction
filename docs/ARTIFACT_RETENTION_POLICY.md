# Artifact Retention Policy

The repository retains evidence, not every file produced by a run. Git history remains the recovery path for pruned historical runtime; history rewriting is out of scope.

## A. PROTECTED_ANCHOR

Source/raw and canonical datasets, frozen partitions and scalers, authoritative source checkpoints and hashes, and graph caches loaded by active code are protected. The machine-readable allowlist is `docs/PROTECTED_ARTIFACTS.json`. A path may enter that file only after a dependency and scientific-role review.

## B. COMPACT_SCIENTIFIC_RECORD

A formal or diagnostic experiment retains `README.md`, `config.json`, `environment.json`, `decision.json`, core aggregate/audit CSVs, a small number of publication plots, and any result that cannot be reconstructed from retained evidence. E2 row and compound aggregate records are Track A evidence and are protected from “legacy” pruning.

## C. REPRODUCIBLE_RUNTIME

Checkpoints, `history.csv`, `fit_result.json`, state snapshots, progress/resume files, per-fit predictions, per-candidate checkpoints, and engineering caches belong under a gitignored `runtime/`. They are not committed unless explicitly registered as a protected anchor. Future runners should write into runtime and call `src.qgeognn_al.artifacts.finalize_experiment` with an explicit compact allowlist.

## D. SUPERSEDED_OR_REDUNDANT

Identical partial/final pairs, smoke/preflight checkpoints, completed comparison checkpoints, duplicate histories, and redundant fit records are removed when aggregate evidence exists, current code has no dependency, and the object is recoverable from Git history.

## Deletion contract

Every artifact candidate is listed in `docs/PRUNE_MANIFEST.csv`. `DELETE` requires `referenced_by_current_code=false` and is forbidden for source data, canonical data, partitions, scalers, authoritative checkpoints, or required graph caches. `REVIEW` is the default when dependency or scientific uniqueness is uncertain. Hygiene tests enforce the contract after every change.

## Completed code and superseded documents

Closed one-off runners may be removed from the working tree when their scientific records remain, no retained code imports or executes them, and the original implementation is recoverable from Git. Keep reusable scientific modules and their regression tests even if an old runner is retired.

Record each retired source/document in `docs/repository/RETIREMENTS.json` with its original path, recovery commit, SHA-256 and retained result or replacement page. This is the source/document equivalent of the artifact prune manifest. Historical report commands refer to the recorded revision. Frozen protocol inputs and user work in progress are not retirement candidates.

Consolidate mutable project summaries into their canonical pages rather than creating another archive of duplicate roadmaps. Do not rewrite frozen scientific results, hashes or original stage decisions to match the current summary. Active runtime and sealed source files keep their existing paths during execution.
