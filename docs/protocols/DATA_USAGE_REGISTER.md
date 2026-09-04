# Data usage register

The machine-readable authority remains [`../data_consumption_register.json`](../data_consumption_register.json); the detailed historical interpretation remains [`../DATA_CONSUMPTION_REGISTER.md`](../DATA_CONSUMPTION_REGISTER.md).

## Current boundaries

- 4g and 8g outcomes have already been used for training, selection, evaluation, and diagnostics. Neither domain contains a pristine untouched confirmatory test under the current repository history.
- The historical 4g test may be reused for legacy comparability only with explicit disclosure.
- Clean-QGeoGNN normalization may use 4g source-train rows only. Validation rows used: 0. Test rows used: 0. 8g rows used: 0.
- Clean-QGeoGNN implementation preflight may use labels solely for a tiny pipeline/loss smoke check; any such result is not scientific performance evidence.
- 8g truth must not select the Clean architecture, input schema, normalization, or 4g benchmark policy.
- 25g, 40g, C18, CN, NH2, and DCM data have audit exposure and are developmental rather than automatically pristine.

Future benchmark and transfer preregistrations must record exact row identities, outcome visibility by stage, split hashes, normalization fit IDs, and whether each evaluation is developmental, comparative, or confirmatory.
