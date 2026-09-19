# Data usage register

The original machine-readable exposure snapshot is [`../data_consumption_register.json`](../data_consumption_register.json), with its [historical interpretation](../DATA_CONSUMPTION_REGISTER.md). Later per-study protocols, prediction freezes and label ledgers extend that snapshot; the old 25g/40g audit-only entries are not the current exposure state.

## Current boundaries

- 4g and 8g outcomes have already been used for training, selection, evaluation, and diagnostics. Neither domain contains a pristine untouched confirmatory test under the current repository history.
- The historical 4g test may be reused for legacy comparability only with explicit disclosure.
- Source normalization uses the declared 4g source-train rows only. Target normalization and method selection follow each study's train/validation contract; test labels do not select models or acquisition strategies.
- 25g and 40g have since been used for transfer fitting, selection and held-out scoring. See the [transfer index](../../studies/transfer/README.md) and each study's label ledger. A later inner-only experiment may keep its outer boundary closed without making the historically exposed population pristine again.
- C18, CN, NH2 and DCM have at least repository audit exposure and are not automatically pristine.
- Current 4g active-learning evaluation follows its trajectory-freeze protocol. Freeze-time zero test-access counts do not imply the final reports have never evaluated held-out labels.

Future benchmark and transfer preregistrations must record exact row identities, outcome visibility by stage, split hashes, normalization fit IDs, and whether each evaluation is developmental, comparative, or confirmatory.
