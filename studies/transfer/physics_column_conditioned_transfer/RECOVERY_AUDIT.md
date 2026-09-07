# Interrupted-run recovery

On 2026-09-07 the study was resumed on the existing
`codex/study-physics-column-conditioned-transfer` branch, based on
`76be41f` (the source-anchored study). The first inspection found 83/120
frozen contexts and three simultaneous schedulers: PID 5744 (fish, six
workers), PID 8664 and PID 10326 (chromatography, eight workers each).
Several contexts had simultaneous writers to the same runtime directory.

The two later schedulers and their observed remaining workers were stopped
with SIGTERM. The original fish scheduler and its six workers were retained.
The existing post-freeze finalizer was retained. No new model, feature,
hyperparameter, label ledger or evaluation criterion was introduced.

A subsequent check of all 85 then-frozen contexts verified each manifest
against its prediction/label/fit artifacts and each neural best checkpoint
against the SHA256 recorded in fit_audit.json. No discrepancy was found.
The frozen protocol and physical-audit hashes also passed verification.
This is an integrity check, not proof that concurrent execution never
affected an earlier optimization trajectory. The overlap is retained as
an execution limitation; these remain developmental results.

Final verification must repeat manifest/checkpoint checks for all 120
contexts, verify finite predictions and exact label roles, and establish
that each context freeze predates the stage's test-evaluation event.
Runtime progress snapshots are not final scientific results.
