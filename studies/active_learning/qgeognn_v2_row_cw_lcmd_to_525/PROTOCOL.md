# Protocol

- Method: Center/Width-LCMD only.
- Seeds: 157 and 6101.
- Source: exact frozen CW-LCMD state at 429 active labels.
- New budgets: 461, 493, 525; batch size 32.
- Center/Width representation: existing L0-scaled full-network CountSketch-512 gradients.
- Selector: existing LCMD-TP with all current L_t centers.
- Training: exact matched QGeoGNN-V2 scratch protocol; no warm start.
- Comparators: exact historical result reuse only.
- Primary metric: AULC 429-525.
- Hard stop: 525; no 557 or 653 continuation.
- Evidence class: developmental continuation, not independent confirmation.
