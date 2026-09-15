# Protocol — POST-PRIMARY matched V2 Hybrid extension

## Scope and frozen relationship to the primary study

This independent extension compares Random, Gradient-LCMD, and V2-Hybrid only on the primary study's confirmation cohort: `157, 887, 2357, 6101, 12203`.  It is explicitly **POST-PRIMARY MATCHED EXTENSION**.  It does not edit, replace, or reanalyse the frozen primary decision `BEST_CURRENT_ROW_ACQUISITION` in `qgeognn_v2_row_small_batch_benchmark`.

Before any new acquisition labels are revealed, the runner verifies the frozen primary result manifest; exact confirmation seeds; deterministic row-split hashes; L0/U0/validation/test ID hashes; QGeoGNN-V2 model provenance; B=32; K=3; L0 target scales; validation checkpoint criterion; training configuration; evaluation initialization hashes; old B=32 pre-test freezes; and content-plus-contract hashes for every reused graph and gradient cache.  Any mismatch writes `reuse_audit.json` with `BLOCKED` and stops.  Old Random and LCMD are never retrained automatically.

## V2-Hybrid definition

For each seed, the frozen current QGeoGNN-V2 L0 member 0 checkpoint and frozen independently initialized current-V2 members 1 and 2 predict the whole canonical U0.  Only V1_q50 and V2_q50 are used.  With L0-derived scales `s_V1` and `s_V2`, the label-free score is:

`sqrt((std(V1_q50)/s_V1)^2 + (std(V2_q50)/s_V2)^2)`.

The score is ranked descending.  Exact score ties retain canonical U0 order.  The Top-25% shortlist follows historical Hybrid's `ceil(0.25 × |U0|)` rule: with 2,997 U0 rows its fixed size is 750.  The current authoritative 128D QGeoGNN-V2 representation is obtained with `model.extract_representation(...)` through the current V2 extraction adapter.  L0 representations are centers; deterministic greedy Euclidean farthest-first runs only within the shortlist and selects exactly B=32 distinct U0 rows.

No legacy predictor, legacy representation, latent normalization, uncertainty weighting, test truth, U0 labels, shortlist sweep, or acquisition modification is used.

## Evaluation and global barrier

Each Hybrid evaluation fit uses the exact old L0, selected B=32 labels, validation IDs, target scales, architecture, deterministic initialization, shuffle seed, early stopping, and validation combined-normalized-RMSE checkpoint criterion.  It is trained from the same initialization as the corresponding Random and LCMD fits; it is never warm-started.

All five selections, Hybrid checkpoints, and test predictions are content-frozen before any test target is revealed.  Only after the global barrier are the five test sets revealed for metric computation.

## Interpretation rules

- `HYBRID_BEATS_RANDOM`: Hybrid mean NRMSE is lower than Random mean and Hybrid beats the within-seed Random median in at least 4/5 seeds.
- `HYBRID_BEST_IN_EXTENSION`: the preceding rule holds, Hybrid mean NRMSE is lower than LCMD, and Hybrid beats LCMD in at least 3/5 seeds.
- `LCMD_REMAINS_BEST`: LCMD mean NRMSE is lower than Hybrid and LCMD beats Hybrid in at least 3/5 seeds.
- `LCMD_HYBRID_COMPARABLE`: neither method wins at least 3/5 matched seeds and their mean NRMSEs differ by at most 0.01.  This is a predeclared descriptive margin, not a p-value threshold.
- Otherwise Hybrid is reported as `HYBRID_DOES_NOT_BEAT_RANDOM` or `BLOCKED`, as applicable.

The endpoint guard reports an explicit warning when the lower-combined method's V1 or V2 mean RMSE is more than 2% worse than its direct comparator.  Selection and mechanism diagnostics are explanatory only and cannot alter acquisition.
