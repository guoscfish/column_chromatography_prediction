# Next-stage Decision: A1 or T1

`manual_approval_required = true`. This document does not authorize a run.

| Criterion | A1-family: 4g AL Method Extension | T1: Transfer Adaptation Benchmark |
|---|---|---|
| Scientific question | Why does Hybrid help, and can 4g label efficiency generalize beyond row splits? | Which 4g→8g adaptation mechanism is best at low target-label budgets? |
| Current evidence | Positive row evidence; compound evidence suggestive; mechanism unresolved. | Current `last2_head` is only a historical baseline; active acquisitions were null under it. |
| Novelty potential | Moderate–high if condition-aware batch diversity is isolated. | High if chromatography-specific adaptation mechanisms differ clearly at low label counts. |
| Expected information gain | High from one causal control; directly resolves the Hybrid ambiguity. | High; establishes whether Track C rests on a suitable predictor. |
| Implementation complexity | Moderate; existing E2 infrastructure plus one shortlist-random control. | Moderate–high; four adaptation families require a shared interface and careful capacity matching. |
| Compute cost | Lower. | Higher, though Random labels avoid acquisition overhead. |
| Negative-result risk | Moderate; diversity may not be causal. | Moderate–high; source/target shift may defeat all simple formulations. |
| Test contamination risk | Low if new frozen row/compound/scaffold evaluation is preregistered. | Low–moderate; must avoid reusing E4 test outcomes to select formulations. |
| Publication value | Strengthens the AL-method story and QGeoGNN label-efficiency claim. | Establishes the transfer story and is prerequisite for defensible active transfer. |
| QGeoGNN paper relationship | Direct extension of in-domain label-efficient QGeoGNN. | Direct extension to new column specifications and scarce target labels. |
| Literature relationship | Tests the diversity mechanism suggested by LCMD/MaxDet/3D graph AL. | Tests readout/residual/frozen-feature ideas suggested by small-data and multi-fidelity GNN transfer. |

## Recommendation

Prefer **A1a first** if the immediate goal is the smallest decisive experiment: it is cheaper and causally isolates the strongest unresolved mechanism behind existing positive evidence. Prefer **T1 first** if the program's priority is new-column adaptation and reopening Track C; T1 is the necessary gate.

Portfolio recommendation: approve A1a as the next minimal scientific study, then reassess T1 with a clean adaptation interface. This is a recommendation, not an automatic decision. Manual approval remains required, and neither experiment was run during this reset.

## Frozen proposals

**A1a:** Random, Ensemble, uncertainty-top25%-Random, Hybrid(top25%-farthest-first), Coverage. **A1b:** only after a positive diversity-mechanism gate, select 1–2 advanced methods.

**T1:** `current_last2_head`, `target_readout_only`, `source_prediction_residual`, `frozen_source_feature_target_regressor`; Random target labels only.
