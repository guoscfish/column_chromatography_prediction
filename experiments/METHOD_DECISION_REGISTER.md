# Active Method Decision Register

更新日期：2026-09-01。Closed D01–D46 decisions live only in `DECISION_ARCHIVE.md`.

## Major frozen decisions

| Decision | Status | Boundary |
|---|---|---|
| QGeoGNN predictor | frozen for acquisition comparisons | Do not change predictor and acquisition simultaneously. |
| 4g E2 evidence | PROMISING_BUT_UNFINISHED | Row evidence is strong pilot evidence; compound evidence is suggestive only. Hybrid is not a final frozen strategy. |
| E4/A2a generic active transfer | historical null | Does not establish that active transfer is impossible; it rejects the tested acquisitions under `current_last2_head`. |
| D45/D46 oracle diagnostics | post-hoc only | D46 nominal seeds are not independent; ICC/rank agreement are degenerate under deterministic training. Three of 18 unique candidate intervals excluded zero. |

## Open decisions

| ID | Question | Current decision | Gate |
|---|---|---|---|
| A1a | Does farthest-first beat random selection inside the exact same uncertainty shortlist? | Complete; mechanism gate failed; stopped | Hybrid exceeded the control median in 3/5 seeds and reached ≥8/10 wins in 2/5; A1b is not authorized. |
| A2 | Should Quantile Width enter a full 4g AL comparison? | Offline-qualified secondary; not run | It may be a secondary baseline but cannot alter E2 primary conclusions. |
| A3 | Which advanced batch method is compatible with QGeoGNN? | Literature shortlist only | Review relevance, feasibility, compatibility, and novelty before selecting 1–2 methods. |
| T1 | Which 4g→8g adaptation mechanism is label-efficient? | Proposed, not run | Random target labels only; compare four frozen adaptation families. |
| C1 | When may active transfer reopen? | Deferred | Only after T1 establishes a stable low-label adaptation baseline. |
| S1 | What structure dominates 4g→8g correction? | Exploratory audit complete; affine was the strongest simple group-CV correction, condition Ridge did not beat it | Reserved truth unconsumed; T1 must include simple calibration and remains manual-only. |

No item in this register authorizes an experiment. Manual approval is required.
