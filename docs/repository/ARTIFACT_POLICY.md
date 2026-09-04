# Artifact policy

Scientific provenance takes priority over directory appearance. `experiments/` is a frozen historical store: existing artifacts are not moved, renamed, deleted, or rewritten to simulate compatibility with a new layout.

## Retention classes

- **Protected anchors:** source/canonical datasets, frozen partitions and scalers, authoritative checkpoints and hashes, and graph caches consumed by active code.
- **Compact scientific records:** README, config, environment, decision, aggregate/audit tables, hashes, and selected plots needed to interpret a study.
- **Reproducible runtime:** checkpoints, histories, predictions, progress state, and caches that can be reconstructed. These belong under ignored `runtime/` paths.
- **Superseded or redundant files:** removable only after a dependency and scientific-uniqueness audit and only through the existing prune policy.

The detailed historical policy and protected allowlist remain authoritative at [`../ARTIFACT_RETENTION_POLICY.md`](../ARTIFACT_RETENTION_POLICY.md) and [`../PROTECTED_ARTIFACTS.json`](../PROTECTED_ARTIFACTS.json). Git history is a recovery mechanism, not permission to rewrite scientific records.
