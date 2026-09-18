# Kernel-IVR sequential B32 extension

## A. Scientific question
Does conditional batch integrated variance reduction improve label efficiency
over the existing Raw Gradient LCMD strategy?

## B. Why this experiment exists
LCMD improves substantially over Random but the matched curve remains short of
the full-data reference. This experiment changes only the acquisition criterion.

## C. Inputs / frozen dependencies
`../qgeognn_v2_row_sequential_b32/`, the current QGeoGNN predictor, canonical
4g data and graph cache. `seal.json` records source, code and baseline hashes.

## D. Dataset and split
Five existing seeds; 3330 outer train, 416 validation, 417 test. See `PROTOCOL.md`.

## E. What truth is visible at each stage
Training sees L0, fixed validation, and only previously frozen selected labels.
Acquisition uses features for original outer-training inputs only. IVR test truth
remains gated until every new trajectory and prediction is frozen. Existing
cohort results are already known, making this an exploratory extension.

## F. Method
Existing 512D raw gradient sketch, global RMS scaling, unit prior/noise, current
L conditioning, uniform original training reference, sequential rank-one batch
updates. One candidate definition; no tuning. Historical baselines are reused.

## G. Metrics
Full-curve AULC, paired differences, Random@1005 and N80/N90/N95 crossings,
endpoint errors, epochs and runtime. Actual budgets and interpolations are separate.

## H. Exact commands
Run from repository root using the `fish` environment:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/active_learning_v2/test_ivr.py tests/active_learning_v2/test_ivr_study.py tests/active_learning_v2/test_sequential_b32.py --junitxml=/tmp/ivr_preflight.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_kernel_ivr_b32.py --prepare --test-report /tmp/ivr_preflight.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_kernel_ivr_b32.py --run-all
```

`--execute-seed SEED` resumes a single authorized shard. `--report` requires all
five complete trajectories; `--run-all` automatically performs final reporting.

## I. Outputs
`config.json`, `environment.json`, `seal.json`, `preflight_tests.xml`,
`global_pre_test_freeze.json`, `results/`, `figures/`, `FINAL_REPORT.md` and
`decision.json`. Large models, gradients and logs are ignored under `runtime/`.

## J. Result
The full run started on 2026-09-18 after 36 passing tests and five successful
real-data round-zero compatibility checks. IVR selection took approximately
0.2 seconds per batch in that engineering check; this is not accuracy evidence.
`decision.json` is the machine-readable status authority. After completion,
`FINAL_REPORT.md` contains the full comparison, including negative outcomes.

## K. Interpretation
Only a complete matched curve supports an effectiveness conclusion. Selection
overlap, changed IDs or a single B32 point do not establish a gain.

## L. Limitations
Previously evaluated cohort, overlapping splits, scalar kernel approximation,
one untuned regularization assumption, historical uncontrolled machine load.

## M. Next decision
Apply the frozen practical gate after global evaluation. No automatic variants
or budget extension. See `PROTOCOL.md` for the exact criteria.
