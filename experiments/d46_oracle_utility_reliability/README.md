# D46-A — Oracle Utility Reliability Audit

## A. Scientific question
Is D45 single-label oracle marginal utility a stable candidate property, or is it substantially contaminated by target fine-tuning stochasticity and finite-test sampling?

## B. Why this experiment exists
D45 found candidate heterogeneity but weak score alignment. Oracle reliability must be audited before any future acquisition work.

## C. Inputs / frozen dependencies
Clean D45 preflight, frozen A2a seed42 partition, K=3 source checkpoints and hashes, source scaler, last2+head engine, and the shared E4 metric implementation.

## D. Dataset and split
The unchanged split has 22 gradient L0 rows, 8 fixed validation rows, 486 U0 rows, and 58 test rows. Bounded D46 will use 18 D45 representative candidates, six per utility stratum.

## E. What truth is visible at each stage
Candidate selection explicitly uses D45 post-hoc oracle truth. Candidate/test truth never enters acquisition. D46 remains post-hoc and is ineligible for direct method tuning.

## F. Method
Smoke passed with one candidate per stratum, repetitions 4601/4602, K=3, and 20/10 epochs/patience. Bounded is pending and freezes 18 candidates, repetitions 4601/4602/4603, K=3, and formal 500/100.

## G. Metrics
Sign consistency, pairwise ranking Spearman, one-way random-effects variance/ICC, paired 2000-resample test-row bootstrap, D45/D46 consistency, and strata summaries.

## H. Exact commands
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d46_oracle_utility_reliability.py --mode smoke --output experiments/reproductions/d46_oracle_reliability_smoke
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d46_oracle_utility_reliability.py --mode bounded
```

## I. Outputs
Compact metrics/audits, per-test-row predictions, reliability tables, variance JSON, plots, and hash-guarded gitignored runtime progress.

## J. Result
Engineering smoke passed: 3 candidates × 2 repetitions × K=3 plus paired baselines, 24 member fits. No scientific conclusion. Bounded pending.

## K. Interpretation
The bounded result may support stable candidate utility, optimization-noise-dominated utility, or stable but low-magnitude utility. No automatic gate chooses among them.

## L. Limitations
This is post-hoc and uses D45/test truth. Test-row bootstrap intervals describe sensitivity to resampling the frozen 58 rows, not future-population confidence intervals.

## M. Next decision
Run only the frozen bounded design after smoke and tests pass; then stop for manual review.
