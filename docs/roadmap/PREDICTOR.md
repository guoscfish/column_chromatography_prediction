# Predictor roadmap

Current status: `POINT_PREDICTOR_CANDIDATE_BASELINE`.

Legacy historical → Condition Completion V2 → R2-pruned candidate baseline.

[R2-pruned requalification](../../studies/predictor/r2_pruned_requalification/R2_PRUNED_COMPARISON.md) removed 318,856 dead registered parameters, leaving 458,952 trainable and gradient-bearing parameters. Random-init and trained-checkpoint mapping have zero prediction differences. The full 262-epoch retrain reproduces R2 exactly; best epoch 162, test V1/V2 R² 0.889217973 / 0.941562533.

The next separate study is `R2_PRUNED_QUANTILE_HEAD_QUALIFICATION`: current Linear/ReLU head versus a monotonic quantile head, changing only the output head and preserving the effective R2-pruned architecture and controlled protocol. This comparison has not been run. Candidate status is not formal baseline or UQ-contract qualification.

`qgeognn_clean_fusion_v1` is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`; engineering reachability PASS, performance FAIL. Retain R0/R1/R2/R3 and all historical branches/results for provenance. No Clean MLP/FiLM/LayerNorm/bottleneck repair is on the current roadmap.

UQ, 8g transfer, AL and active transfer remain paused/deferred. This cleanup did not execute any downstream study, head alternative or hyperparameter sweep.
