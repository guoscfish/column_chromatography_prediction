# E2 Compound Failure Audit

This directory is a **purely post-hoc diagnostic**. It does not modify the E2 primary result and did not start formal E4 training. Existing selected IDs, partitions, Round-0 member-0 fixed reference, predictions, labels, and persisted AL states were reused.

## Findings

- Seed42 final test-to-labeled relative coverage gain: {"coverage": 0.05413659366354684, "ensemble": 0.03445608839985518, "hybrid": 0.05946300069126009, "random": 0.05895638869099432}. Positive values mean the held-out test is closer to the labeled set; this is not an acquisition objective.
- Coverage and Hybrid improve test-to-labeled coverage versus their own Round-0 state, but Random is at least as good on seed42; geometric coverage is not training utility.
- At the final selected batch (round 7), seed42 Coverage/Hybrid are more test-relevant in fixed latent space than Ensemble and have higher unique-compound coverage. Morgan similarity and condition distance vary by seed/strategy and do not support one mechanism.
- Seed42 shows mixed compound-wise changes rather than a broad all-compound collapse under the registered minority rule. This is compatible with composition/extreme-compound sensitivity, not proof of it; no universal OOD-shift claim is supported.
- Coarse centroid shifts show no consistent active-only movement away from test across rounds/seeds (`active_selection_moves_away_from_test_region=False`). Round-0 difficulty versus AULC gain is descriptive (`n=3`) only.
- Selected-to-test relevance is reported in `selected_test_relevance.csv` using fixed latent distance, Morgan radius-2/2048-bit maximum Tanimoto, and condition distance.
- `per_compound_error_trajectory.csv` and the four `seed42_*.png` heatmaps separate a few-compound failure from broad degradation. Overall error changes are descriptive and n=3; no significance or generalization claim is made.
- `strategy_distribution_shift.csv` reports concentration and coarse centroid shifts from fixed reference space. It is not a formal divergence.
- Historical `label_efficiency.csv` remains unchanged; its Random-final threshold interpretation is degenerate when Round-0 is already better than Random final and must not be cited as compound label-saving evidence.

## Audit boundary

`implementation_bug_found=False`, `data_leakage_found=False`, `evaluation_bug_found=False`. Risk ranking, training utility, and held-out compound OOD utility are separate quantities. All test relevance calculations are post-hoc only; test data never enters acquisition or E2 reruns.

## Decision

`safe_to_proceed_to_e4_preregistration=True`. The current failure hypothesis is descriptive only and does not justify changing the predictor or E2 strategy. Formal E4 training, E2 Quantile curve, causal controls, representation ablations, and advanced AL remain deferred.
