# E2 Compound Failure Audit

This directory is a **purely post-hoc diagnostic**. It does not modify the E2 primary result and did not start formal E4 training. Existing selected IDs, partitions, Round-0 member-0 fixed reference, predictions, labels, and persisted AL states were reused.

## Findings

- Seed42 final test-to-labeled relative coverage gain: {"coverage": 0.040007707004080295, "ensemble": 0.02404033896883906, "hybrid": 0.030340732061525877, "random": 0.056320363358239645}. Positive values mean the held-out test is closer to the labeled set; this is not an acquisition objective.
- `gradient_train_rows` excludes the fixed validation rows. Validation remains part of label budget, but is excluded here because this diagnostic measures the geometry of rows actually used for gradient updates.
- Selected-to-test relevance uses fixed latent distance, Morgan radius-2/2048-bit maximum Tanimoto, and Euclidean distance in standardized 9D condition space. Normalization is fit only on fixed L0_train; neither U0 nor test contributes.
- `seed42_compound_paired_effects.csv` pairs each active strategy with Random on the same held-out compound, using the mean of normalized RMSE and normalized MAE. Concentration statistics are {"coverage": {"compounds": 21, "fraction_improved": 0.3333333333333333, "fraction_worsened": 0.6666666666666666, "improved": 7, "top1_worst_contribution": 0.5040419808608411, "top3_worst_contribution": 0.7770153756054404, "total_positive_excess_error": 2.256036370286127, "worsened": 14, "worst20pct_contribution": 0.8751428691679977}, "ensemble": {"compounds": 21, "fraction_improved": 0.42857142857142855, "fraction_worsened": 0.5714285714285714, "improved": 9, "top1_worst_contribution": 0.4041562401059924, "top3_worst_contribution": 0.6481579043000171, "total_positive_excess_error": 3.1975120491819444, "worsened": 12, "worst20pct_contribution": 0.8631535059076325}, "hybrid": {"compounds": 21, "fraction_improved": 0.23809523809523808, "fraction_worsened": 0.7619047619047619, "improved": 5, "top1_worst_contribution": 0.4849584836177675, "top3_worst_contribution": 0.7651970825125636, "total_positive_excess_error": 1.8054803566835602, "worsened": 16, "worst20pct_contribution": 0.8971526119469404}}. Localization remains inconclusive; no favorable localization cutoff was introduced.
- `strategy_distribution_shift.csv` separates nearest-distance coverage from the actual gradient-train-centroid-to-test-centroid distance. Both are post-hoc descriptive diagnostics.
- Historical `label_efficiency.csv` remains unchanged; its Random-final threshold interpretation is degenerate when Round-0 is already better than Random final and must not be cited as compound label-saving evidence.

## Audit boundary

`implementation_bug_found=False`, `data_leakage_found=False`, `evaluation_bug_found=False`. No implementation/leakage/evaluation bug was found. Global diversity, risk ranking, held-out-test geometric relevance and OOD utility are distinct quantities. The seed42 failure remains partly split-composition dependent and is not fully explained by one post-hoc diagnostic. All test relevance calculations are post-hoc only; test data never enters acquisition or E2 reruns.

Fresh-clone boundary: the aggregate E2 scientific result and compact query history are tracked and self-contained. Regenerating the full historical trajectory additionally requires runtime-only per-round checkpoints and test predictions; their hashes/provenance are recorded in `audit_input_manifest.json`, and they are intentionally not claimed as fresh-clone inputs.

## Decision

`safe_to_proceed_to_e4_preregistration=True`. The current failure hypothesis is descriptive only and does not justify changing the predictor or E2 strategy. Formal E4 training, E2 Quantile curve, causal controls, representation ablations, and advanced AL remain deferred.
