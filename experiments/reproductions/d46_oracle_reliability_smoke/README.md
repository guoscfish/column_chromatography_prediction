# D46-A — Oracle Utility Reliability Audit

## A. Scientific question
Is D45 single-label oracle marginal utility a stable candidate property, or is it substantially contaminated by target fine-tuning stochasticity and finite-test sampling?

## B. Why this experiment exists
D45 found candidate heterogeneity but weak score alignment. Reliability of the oracle target must be audited before any future acquisition work.

## C. Inputs / frozen dependencies
Clean D45 preflight, frozen A2a seed42 partition, K=3 source checkpoints and hashes, source scaler, last2+head engine, and E4 metric helper.

## D. Dataset and split
The unchanged split has 22 gradient L0 rows, 8 fixed validation rows, 486 U0 rows, and 58 test rows. D46 uses 18 D45 representative candidates (6 per post-hoc utility stratum).

## E. What truth is visible at each stage
`candidate_selection_uses_D45_oracle_truth=true`. Candidate truth is used only after selection for its one-label fit; test truth is used for post-hoc utility and paired row bootstrap. No D45/D46 truth enters acquisition.

## F. Method
Mode `smoke`. Bounded uses target optimization seeds 4601/4602/4603, and each repetition independently fits its paired L0 baseline and every L0+candidate model from all three frozen sources at formal 500/100 epochs/patience. Smoke uses three candidates, two repetitions, and 20/10.

## G. Metrics
Sign consistency, all pairwise ranking Spearman values, balanced one-way random-effects variance/ICC(1,1), paired 2000-resample test-row bootstrap, D45/D46 Spearman/MAE/sign agreement, and strata summaries.

## H. Exact commands
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d46_oracle_utility_reliability.py --mode smoke --output experiments/reproductions/d46_oracle_reliability_smoke
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d46_oracle_utility_reliability.py --mode bounded
```

## I. Outputs
Compact metrics/audits, per-test-row predictions, reliability tables, variance JSON, five plots, and a hash-guarded gitignored runtime/progress tree.

## J. Result
Engineering smoke passed; no scientific conclusion.

## K. Interpretation
Three possibilities remain valid before results: stable candidate property; optimization-noise-dominated utility; or stable but too-small utility/headroom. No automatic gate selects among them.

## L. Limitations
This is a post-hoc diagnostic on 18 D45-truth-selected candidates and one outer partition. Bootstrap intervals measure sensitivity to resampling the frozen 58 test rows, not uncertainty for a future 8g population.

## M. Next decision
Manual review required. This experiment cannot open a new acquisition method, Protocol B, or D45-B automatically.
