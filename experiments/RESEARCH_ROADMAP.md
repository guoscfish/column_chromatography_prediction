# Deferred Research Roadmap

## Predictor prerequisite

The current predictor mainline is `Legacy historical → Condition Completion V2 → R2-pruned candidate baseline`. The function-preserving cleanup and frozen E0 retraining succeeded. The next separate step is quantile-head qualification; it has not been run. Clean is a failed point-performance architecture experiment retained for provenance. See the [predictor roadmap](../docs/roadmap/PREDICTOR.md).

## Track A — 4g In-domain Active Learning

Status: `PAUSED_AFTER_A1A / PROMISING_BUT_UNFINISHED`.

A1a is completed and stopped: its shared-shortlist farthest-first mechanism gate failed. Historical E2 Hybrid/Coverage pilot evidence remains, but A1b and compound follow-up were not launched.

不再因为当前Hybrid结果自动探索 LCMD / MaxDet / B3AL / geometry-aware diversity。Track A waits for a new independent hypothesis and manual review.

Stop/freeze gate: stable improvement over Random across row and compound/scaffold protocols, adequate seeds/intervals, and an isolated mechanism. A row-only result cannot establish novel-molecule success.

## Track B — 4g→8g Transfer Adaptation

Status: `T1B1_FORMAL_COMPLETE_NO_INTERMEDIATE_CAPACITY_BENEFIT`.

S1 is completed and stopped. T1a and T1b-1 formal runs are complete. T1b-1 completed 180/180 Adapter fits and 120/120 contexts, but r=8/16/32 won only 2/5, 2/5, and 3/5 seeds versus Head and all failed the frozen gate. The T1a → T1b-1 capacity question is therefore complete with no tested intermediate sweet spot. Independent compound-level, another-column, or new-target validation is next in priority. T1b-2 remains a proposed, unimplemented, unauthorized matched-capacity location study; active transfer remains deferred.

Stop gate: reject a family that fails paired low-label efficiency and stability without using test truth for selection.

## Track C — Active Transfer

Status: `DEFERRED`.

Trigger: Track B first identifies a stable low-label adaptation formulation. Only then may active domain adaptation or active transfer be preregistered. E4/A2a negative results remain specific to the tested `current_last2_head` setup.
