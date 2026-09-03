# Deferred Research Roadmap

## Track A — 4g In-domain Active Learning

Status: `PAUSED_AFTER_A1A / PROMISING_BUT_UNFINISHED`.

A1a is completed and stopped: its shared-shortlist farthest-first mechanism gate failed. Historical E2 Hybrid/Coverage pilot evidence remains, but A1b and compound follow-up were not launched.

不再因为当前Hybrid结果自动探索 LCMD / MaxDet / B3AL / geometry-aware diversity。Track A waits for a new independent hypothesis and manual review.

Stop/freeze gate: stable improvement over Random across row and compound/scaffold protocols, adequate seeds/intervals, and an isolated mechanism. A row-only result cannot establish novel-molecule success.

## Track B — 4g→8g Transfer Adaptation

Status: `T1B1_PREREGISTERED_READY_FOR_FORMAL_AUTHORIZATION`.

S1 is completed and stopped. T1a completed all 180 neural fits and 120 contexts, but no candidate passed its stability gate. T1b-1 fixes one graph-level residual adapter and varies only r=8/16/32 capacity between head-only and Last1; engineering smoke and the truth-free preflight passed, leaving the frozen 180-fit protocol ready for separate authorization. This is post-T1a developmental work on known row outcomes. T1b-2 is a future matched-capacity adaptation-location study placeholder only; Message-Passing Adapter and Adaptive/Learnable Readout are not implemented. Active transfer remains deferred.

Stop gate: reject a family that fails paired low-label efficiency and stability without using test truth for selection.

## Track C — Active Transfer

Status: `DEFERRED`.

Trigger: Track B first identifies a stable low-label adaptation formulation. Only then may active domain adaptation or active transfer be preregistered. E4/A2a negative results remain specific to the tested `current_last2_head` setup.
