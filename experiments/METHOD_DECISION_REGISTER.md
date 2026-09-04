# Active Method Decision Register

更新日期：2026-09-04。Closed D01–D46 decisions live only in `DECISION_ARCHIVE.md`.

## Major frozen decisions

| Decision | Status | Boundary |
|---|---|---|
| QGeoGNN predictor | frozen for acquisition comparisons | Do not change predictor and acquisition simultaneously. |
| 4g E2 evidence | PROMISING_BUT_UNFINISHED | Row evidence is strong pilot evidence; compound evidence is suggestive only. Hybrid is not a final frozen strategy. |
| E4/A2a generic active transfer | historical null | Does not establish that active transfer is impossible; it rejects the tested acquisitions under `current_last2_head`. |
| D45/D46 oracle diagnostics | post-hoc only | D46 nominal seeds are not independent; ICC/rank agreement are degenerate under deterministic training. Three of 18 unique candidate intervals excluded zero. |
| Legacy QGeoGNN semantics | I0 audit complete; behavior preserved | Ten continuous edge columns are constructed but only positions 0–4 are consumed. Historical results remain valid within this legacy contract. |

## Open decisions

| ID | Question | Current decision | Gate |
|---|---|---|---|
| A1a | Does farthest-first beat random selection inside the exact same uncertainty shortlist? | Complete; mechanism gate failed; stopped | Hybrid exceeded the control median in 3/5 seeds and reached ≥8/10 wins in 2/5; A1b is not authorized. |
| A2 | Should Quantile Width enter a full 4g AL comparison? | Offline-qualified secondary; not run | It may be a secondary baseline but cannot alter E2 primary conclusions. |
| A3 | Which advanced batch method is compatible with QGeoGNN? | Literature shortlist only | Review relevance, feasibility, compatibility, and novelty before selecting 1–2 methods. |
| T1 | Does the best adaptation capacity change with scarce target-label budget? | T1a and T1b-1 formal complete; no stable candidate | `target_head_only` retained the best mean AULC; T1b-1 found no benefit for tested 3k–9k graph adapters. Shared optimization was controlled, not method-optimal. |
| T1b-1 | Is there a parameter-efficient capacity sweet spot between Head-only and Last1? | Formal complete; no tested low-capacity graph-adapter benefit | r8/r16/r32 mean AULC 0.6583/0.6587/0.6578 versus Head 0.6577; all failed the frozen gate. The 9k–93k gap was not tested. |
| T1b-2 | At matched capacity, where should adaptation be inserted? | Proposed, not authorized; lower priority than independent validation | Reconsider graph-level, message-passing, and learnable-readout locations only under a new rationale after independent validation. |
| PV2 | Should a condition-complete predictor replace the legacy contract? | Engineering candidate; implementation preflight passed; not selected final predictor | Residual branch passed schema, reachability, normalization, legacy-load, source-identity, and checkpoint checks. A separate 4g source-qualification authorization is still required. |
| C1 | When may active transfer reopen? | Deferred | Only after T1 establishes a stable low-label adaptation baseline. |
| S1 | What structure dominates 4g→8g correction? | Exploratory audit complete; affine was the strongest simple group-CV correction, condition Ridge did not beat it | Reserved truth unconsumed; T1 must include simple calibration and remains manual-only. |

T1a and T1b-1 formal execution and analysis are complete. T1b-1 did not establish a tested 3k–9k graph-adapter benefit or stable low-label winner; it did not test the 9k–93k gap. Predictor V2 engineering preflight is complete, but this register authorizes neither formal V2 source training, transfer qualification, T1b-2, expanded widths, active transfer, nor a Track A restart.
