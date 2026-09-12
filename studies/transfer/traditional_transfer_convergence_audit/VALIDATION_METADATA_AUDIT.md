# Validation metadata audit

The frozen historical `formal_row_5seed` preprocessing metadata contains
`validation_rows_used = 0`. That wording is not semantically correct: outer
validation labels were read to select checkpoints, although they were never
used for gradient updates and outer-test truth was not used for fit or
selection.

This audit does not modify frozen historical artifacts. New and future runners
must record separately:

- `gradient_fit_rows`
- `validation_selection_rows`
- `test_truth_rows_used_for_fit = 0`
- `test_truth_rows_used_for_selection = 0`

The implementation for a historical rerun now writes those fields, and the
converged-baseline study records `validation_selection_rows = 0` because it
uses inner folds rather than outer validation for epoch selection.
