# Repository conventions

## One owner per document

| Information | Owner |
| --- | --- |
| Project introduction and starting links | Root `README.md` |
| Current conclusions and next action | `docs/NEXT_STAGE_DECISION.md` |
| Topic navigation | `studies/README.md` and its three topic READMEs |
| Executable entry points | `scripts/README.md` |
| Code responsibilities | `src/qgeognn_al/README.md` |
| A study's question, protocol and measured outcome | Its own `studies/<topic>/<study>/` directory |
| Historical experiment inventory | `experiments/INDEX.md` |
| Artifact retention and retirement | `docs/ARTIFACT_RETENTION_POLICY.md` |

Update these owners in place. Do not add another project roadmap, dated status dump or duplicate next-stage plan. A substantial literature review can remain separate; link to it from the owner instead of copying its conclusions across pages. Completed work must not remain listed as pending.

## Code

- Reusable model, training, acquisition and evaluation logic belongs in `src/qgeognn_al/`.
- New runners belong in `scripts/studies/`, import the scientific package, and handle arguments, protocol checks and execution.
- Do not copy a full runner for a parameter variant or import a runner as new scientific core.
- Existing runner-to-runner imports are compatibility debt. Preserve them until their consumers and frozen source hashes have been audited.
- Add implementation tests for scientific behavior and leakage boundaries. Formal experiments are not unit tests.

## Study records

New studies retain a concise README, frozen config/protocol, environment, decision, compact aggregate metrics and artifact hashes. The README covers the question, inputs, data/split, label visibility, method, commands, result, limitations and next decision.

A completed closed study may retain only its scientific record and a recovery reference for the original implementation. Do not change historical numerical results or regenerate a freeze solely to match a directory cleanup.

Diagnostic or post-hoc use of test truth must be explicit. Distinguish row interpolation, target-compound holdout and source-unseen OOD; do not combine rankings across filtered/unfiltered populations or incompatible budgets.

## Completion

Update the existing result and current-status pages, run the relevant checks, and remove abandoned entry points only after dependency review. Keep checkpoints, histories and resume state under ignored `runtime/`. Follow the [retention policy](ARTIFACT_RETENTION_POLICY.md) for recovery records.
