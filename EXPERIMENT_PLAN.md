# Experiment Plan — Research Reset

## Current question

Two independent branches remain open: whether 4g in-domain AL can become robust beyond row splits (Track A), and which mechanism best adapts 4g knowledge to scarce 8g labels (Track B). Active transfer (Track C) depends on Track B and is deferred.

## Frozen historical conclusions

- E2 row: Hybrid/Coverage beat Random in 3/3 seeds; strong pilot evidence, not a final method.
- E2 compound: Hybrid/Coverage beat Random in 2/3 seeds; suggestive only.
- E4 and A2a: tested generic acquisitions under `current_last2_head` did not stably beat Random.
- D45/D46: post-hoc only. D46's seeds did not create independent stochastic fits; 3/18 unique-candidate paired test-row intervals excluded zero.
- Historical scientific results and predictor behavior remain unchanged by repository refactoring.

## Current allowed stage

Repository/document reset, dependency-based artifact pruning, behavior-preserving extraction into `src/qgeognn_al`, literature mapping, and proposal design only. No new training or acquisition experiment may run.

## Next preregistered candidates

**A1a — Hybrid causal mechanism control:** Random, Ensemble, uncertainty-top25%-Random, Hybrid(top25%-farthest-first), Coverage. It isolates whether filtering or batch de-redundancy drives Hybrid. A1b may select only 1–2 advanced methods after A1a supports a diversity mechanism.

**T1 — Transfer Adaptation Benchmark:** `current_last2_head`, `target_readout_only`, `source_prediction_residual`, `frozen_source_feature_target_regressor`; Random target labels only.

## Stop and gate conditions

- Manual approval selects A1 or T1; neither starts automatically.
- Track A cannot freeze a strategy without stable row plus compound/scaffold evidence and more than the current three outer seeds.
- Track C cannot reopen until T1 establishes a stable low-label transfer formulation.
- A1b stops if A1a does not support the diversity mechanism.
- Test truth cannot select methods, tune weights, or retroactively change E2/E4 conclusions.
