# Study entrypoints

The [script index](../README.md) owns the runnable workflow list. The [current research status](../../docs/NEXT_STAGE_DECISION.md) and [study index](../../studies/README.md) own completion and next actions; this directory does not maintain a second progress list.

Scientific algorithms belong in `src/qgeognn_al/`. Runners orchestrate frozen protocols, manifests, training and reporting. Some historical runners remain shared dependencies or byte-hashed protocol inputs and must retain their original paths. Closed entrypoints without consumers are recorded in [RETIREMENTS.json](../../docs/repository/RETIREMENTS.json).

Active Free / LCMD continuation uses its frozen report-local supervisors. Do not independently launch a second runner from this directory. A reproduction uses a separate output directory and requires the original protocol, environment and local checkpoint anchors.

New studies keep protocols, selected IDs, request hashes, accepted/rejected receipts and aggregate results in Git. Checkpoints, feature caches, full requests, process locks and console logs stay local under the repository retention rules. Existing tracked historical evidence remains preserved.
