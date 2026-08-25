# D45 — Oracle Marginal Utility Audit

## A. Scientific question
At frozen seed-42 L0=30, do individual U0 labels have heterogeneous marginal training utility, and do current legal acquisition scores identify high-utility candidates?

## B. Why this experiment exists
E4 Protocol A and E4-A2a both returned null active-acquisition evidence. D45 diagnoses candidate utility; it does not open another method.

## C. Inputs / frozen dependencies
Frozen 8g target data, A2a seed-42 partition, K=3 source checkpoints (42/525/1101), 4g scaler, QGeoGNN engine, E4 scores, and E4 metrics.

## D. Dataset and split
`l0_train=22`, fixed `l0_validation=8`, `u0=486`, `test=58`; the partition is unchanged.

## E. What truth is visible at each stage
Subset construction sees only IDs and unlabeled Round0 scores. Candidate truth is revealed only for that candidate's gradient fit. Frozen test truth defines oracle utility. `test_truth_used_for_oracle_utility=true`.

## F. Method
Every member resets from its frozen 4g source with the same candidate-independent fit seed. A candidate adds exactly one U0 row to gradient training, never validation. Mode `bounded` uses K=3. Smoke is 12 candidates at 20/10 epochs/patience and is engineering-only. Bounded uses random 48 plus deduplicated top-8 challenge candidates at formal 500/100.

## G. Metrics
Baseline-minus-after V1 NRMSE, V2 NRMSE, and frozen E4 combined NRMSE; bounded adds fixed-seed bootstrap Spearman 95% CIs, representative-subset AUROC/enrichment, and challenge summaries.

## H. Exact commands
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d45_oracle_marginal_utility.py --mode smoke
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d45_oracle_marginal_utility.py --mode bounded
```

## I. Outputs
Config/environment/decision JSON, candidate/baseline/utility/audit CSVs, bounded analysis tables, and plots. Runtime checkpoints are gitignored and removed after compact extraction.

## J. Result
Representative combined utility: median `0.003328`, IQR `0.016709`, P10 `-0.013582`, P90 `0.023010`, positive fraction `0.583`. Spearman rho: Ensemble `-0.037`, QWidth `0.110`, Coverage `0.041`; all bootstrap 95% CIs cross zero.

## K. Interpretation
`experiment_role=post_hoc_diagnostic`, `confirmatory_evidence=false`, `historical_E4_conclusion_changed=false`, and `eligible_for_direct_method_tuning=false`.

## L. Limitations
Bounded evidence is 48 representative rows from one outer seed. This test partition is consumed as oracle truth. Any D45-informed method needs a new preregistered confirmatory partition/protocol.

## M. Next decision
Manual review only. Do not automatically open D46, a new method, or a full oracle trajectory.
