# Next-stage Decision after Predictor Regression Audit and T1b-1

## Current predictor gate

The frozen E0 point-predictor regression ladder is complete and all artifact audits pass. R0 reproduced historical Legacy test R2 values (V1 `0.866943896`, V2 `0.902874589`) within the `0.03` gate. R1 and R2 remained strong, while R3 current Clean reached only V1 `0.525054` and V2 `0.606885`; the first regression is therefore `R2_CONDITION_COMPLETE_V2 -> R3_CLEAN_CURRENT`.

The confirmed finding is a Clean architecture/forward-path package regression with substantial training-set underfit. The leading supported mechanism is removal of Legacy early molecule-condition interaction and replacement with additive late fusion. The 128D -> 64D bottleneck, LayerNorm, and monotonic softplus head are plausible but not individually isolated. The split, Clean training protocol, condition completion, global q50 compression, and q50 clamp are not supported as primary causes.

Current decision: `FUNCTION_PRESERVING_PARAMETER_CLEANUP_SUCCESS / POINT_PREDICTOR_CANDIDATE_BASELINE`. R2-pruned has 458,952 registered, trainable, gradient-bearing parameters and zero prediction-unreachable parameters. Both initialization and trained-checkpoint conversion have zero six-output difference over all E0 splits. Controlled retraining exactly reproduces all R2 metrics and the full 262-epoch history (best epoch 162). See [the comparison](../studies/predictor/r2_pruned_requalification/R2_PRUNED_COMPARISON.md).

The next separate experiment is `R2_PRUNED_QUANTILE_HEAD_QUALIFICATION`: retain R2-pruned and change only the current output head versus a monotonic quantile head. It was not executed in this round. Formal baseline/UQ claims, 8g transfer and AL remain downstream. `qgeognn_clean_fusion_v1` is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`: engineering reachability PASS, performance FAIL. The prior nonlinear-fusion repair proposal is superseded; preserve Clean only for regression provenance.

T1a and T1b-1 formal execution are complete. T1b-1 found no stable Adapter improvement over `target_head_only`, so this document recommends independent validation before further architecture work. It does not authorize T1b-2 or active transfer.

| Criterion | A1-family: 4g AL Method Extension | T1: Transfer Adaptation Benchmark |
|---|---|---|
| Scientific question | Why does Hybrid help, and can 4g label efficiency generalize beyond row splits? | Which 4g→8g adaptation mechanism is best at low target-label budgets? |
| Current evidence | E2 beat FullPool-Random in 3/3 row seeds but only 2/3 compound seeds; A1a found no stable farthest-first gain over Random within the same Top25% uncertainty shortlist. | G0-4 retained `last2_head` under richer target data; S1 made affine a strong low-cost baseline but did not solve transfer. |
| Novelty potential | Moderate–high if condition-aware batch diversity is isolated. | High if chromatography-specific adaptation mechanisms differ clearly at low label counts. |
| Expected information gain | High from one causal control; directly resolves the Hybrid ambiguity. | High; establishes whether Track C rests on a suitable predictor. |
| Implementation complexity | Moderate; existing E2 infrastructure plus one shortlist-random control. | Moderate–high; four adaptation families require a shared interface and careful capacity matching. |
| Compute cost | Lower. | Higher, though Random labels avoid acquisition overhead. |
| Negative-result risk | Moderate; diversity may not be causal. | Moderate–high; source/target shift may defeat all simple formulations. |
| Test contamination risk | Low if new frozen row/compound/scaffold evaluation is preregistered. | Low–moderate; must avoid reusing E4 test outcomes to select formulations. |
| Publication value | Strengthens the AL-method story and QGeoGNN label-efficiency claim. | Establishes the transfer story and is prerequisite for defensible active transfer. |
| QGeoGNN paper relationship | Direct extension of in-domain label-efficient QGeoGNN. | Direct extension to new column specifications and scarce target labels. |
| Literature relationship | Tests the diversity mechanism suggested by LCMD/MaxDet/3D graph AL. | Tests readout/residual/frozen-feature ideas suggested by small-data and multi-fidelity GNN transfer. |

## Current decision

**A1a is complete and stopped.** Its failed diversity gate does not mean 4g AL failed: it narrows the unsupported mechanism to farthest-first's incremental value. Because E2 Random was FullPool-Random while A1a Random was shortlist-conditioned, uncertainty filtering remains plausible but is not causally isolated by a direct paired comparison. A1b remains unauthorized.

**T1b-1 completed its frozen row-protocol schedule without an intermediate-capacity benefit.** Head mean normalized AULC was 0.6577; r8/r16/r32 were 0.6583/0.6587/0.6578. Their paired deltas versus Head were +0.00065/+0.00100/+0.00011, with only 2/5, 2/5, and 3/5 wins. No Adapter passed the frozen stability gate. The practical recommendation is to retain output-only Head correction as the working low-label formulation and seek independent compound-level, another-column, or new-target validation. A matched-capacity T1b-2 location comparison may be reconsidered only with a new rationale; it is not the default next run. Track C remains deferred because no stable transfer baseline was established.

## Frozen T1a design

Fixed Random nested target-label budgets are 30/50/70/100, each including eight validation labels. All methods share identical train/validation/test partitions per outer seed.

Primary methods: `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. `target_head_only` means training the prediction head `graph_pred_linear`; graph readout remains fixed sum pooling. Full fine-tuning is disabled and optional; `paper_style`, active acquisition, and Protocol B are excluded.
