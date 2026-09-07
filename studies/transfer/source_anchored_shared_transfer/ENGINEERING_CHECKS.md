# Engineering verification before formal comparison

- New real-graph/synthetic contract tests: 13 passed.
- Complete repository regression suite: 265 passed (178 library warnings),
  120.22 seconds in conda fish / torch 2.10.0, before new target-test evaluation.
- Initial wrapper construction caught a non-leaf RBF deepcopy incompatibility.
  Fixed by constructing the exact qualified V2 and loading its full state;
  no RBF, architecture or function changes. The corrected version passed
  bitwise zero-step six-output equivalence for both shallow and full scopes.
- Independent source/target heads, frozen source head, exact trainable masks,
  source-loss backbone gradients, fixed source-pass BN statistics, and frozen
  parameter invariance checked on real graph inputs.
- Deterministic two-epoch shallow and full anchored fits reproduce every
  checkpoint tensor; completed-fit reuse validates contract/checkpoint hashes.
- All 120 ledgers checked for exact focal-column coverage, disjoint roles,
  nested training IDs, fixed validation/test IDs and actual budget accounting.
- Poisoned non-purchased label cells are not parsed by the selected-truth
  reader. Fitting data loading is instrumented to allow only purchased focal
  labels and checkpoint-source training IDs.
- Missing/modified prediction freezes reject evaluation before the truth
  reader is called. Macro, relative-error floor, strata and replicated
  all-reference material-gain logic have direct tests.
- Fixed full-length smoke `8g/row/769539383/30` completed all four arms:
  N1 148 epochs / 6.26 seconds; N2 157 / 8.62; M1 196 / 15.26;
  M2 152 / 17.41. All source-head L2 drifts exactly zero. These are execution
  checks, not test-score or model-selection gates. No test truth was read.

Regression command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/source-anchor-mpl conda run --no-capture-output -n fish python -m pytest -q tests --tb=short
```

Source construction and training code are hash-locked in `protocol.json`;
evaluation definitions/code are locked independently before formal test read
in `evaluation_protocol.json`.

Post-lock checks: the corrected evaluator passed all 13 contract tests again.
An additional interrupted-full-fit regression passed: restored checkpoint
tensors and replay IDs/draws exactly match uninterrupted training.
