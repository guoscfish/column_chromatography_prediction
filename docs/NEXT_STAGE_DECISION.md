# Next-stage Decision after A1a: T1 Engineering Gate

T1a formal execution is complete without a stable winner. T1b-1 engineering/preregistration and smoke are complete, but this document does not authorize its formal run, T1b-2, or active transfer.

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

**T1a completed its frozen row-protocol schedule without establishing a stable winner.** `target_head_only` led mean performance but won only 3/5 paired AULC seeds. T1b-1 now preregisters only a graph-level residual-adapter capacity sweep at r=8/16/32. It is explicitly post-T1a developmental work on an already-consumed row protocol. Track C stays deferred, and adaptation-location methods remain an unauthorized T1b-2 placeholder.

## Frozen T1a design

Fixed Random nested target-label budgets are 30/50/70/100, each including eight validation labels. All methods share identical train/validation/test partitions per outer seed.

Primary methods: `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. `target_head_only` means training the prediction head `graph_pred_linear`; graph readout remains fixed sum pooling. Full fine-tuning is disabled and optional; `paper_style`, active acquisition, and Protocol B are excluded.
