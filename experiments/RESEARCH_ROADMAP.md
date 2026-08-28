# Deferred Research Roadmap

## Track A — 4g In-domain Active Learning

Status: `PROMISING_BUT_UNFINISHED`.

Future questions, all unrun: A1 Hybrid causal control; A2 Quantile Width secondary AL; A3 LCMD/B³AL-LCMD/MaxDet shortlist; A4 geometry-aware molecular AL; A5 epistemic-UQ variants; A6 graph/condition representation ablation; A7 row, compound, and Bemis–Murcko scaffold validation. Keep QGeoGNN fixed while comparing acquisitions.

Stop/freeze gate: stable improvement over Random across row and compound/scaffold protocols, adequate seeds/intervals, and an isolated mechanism. A row-only result cannot establish novel-molecule success.

## Track B — 4g→8g Transfer Adaptation

Status: `OPEN`.

Next candidate T1 uses Random target labels and compares `current_last2_head`, `target_readout_only`, `source_prediction_residual`, and `frozen_source_feature_target_regressor`. Future deferred topics include multi-fidelity formulations, Bayesian transfer/UQ, 25g/40g, and downstream SQ.

Stop gate: reject a family that fails paired low-label efficiency and stability without using test truth for selection.

## Track C — Active Transfer

Status: `DEFERRED`.

Trigger: Track B first identifies a stable low-label adaptation formulation. Only then may active domain adaptation or active transfer be preregistered. E4/A2a negative results remain specific to the tested `current_last2_head` setup.
