# Protocol

The study inherits the qualified 4g checkpoint, filtered 25g/40g populations,
outer identities, source scaler, label accounting, protocols, and five seeds
from `full_data_baseline_finalization`. Compound is primary; row is secondary.

The global/shared coefficient penalty remains 0.1. Only column-specific Center
and Width deviations use the preregistered grid `[0, 0.01, 0.1, 1, 10, 100]`.
Shared and separate candidates use identical inner folds and scores. Compound
inner CV groups canonical SMILES; row inner CV uses the inherited shuffled-row
protocol. Selection, normalization, fitting, and prediction occur without
outer test truth. Predictions and audits are hash-frozen before scoring.

Primary evaluation stops on checkpoint drift, split drift, label-accounting
drift, legacy nesting failure, corrected identity failure, corrected joint
nesting failure, or premature test-truth access. No grid extension or
test-driven selection is permitted.
