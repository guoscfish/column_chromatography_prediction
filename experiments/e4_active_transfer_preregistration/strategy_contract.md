# Strategy contract

- `random`: seeded random pool query.
- `coverage`: feature-wise standardized `[h_graph; conditions]` farthest-first, using only frozen predictor inputs.
- `ensemble`: top uncertainty/error-risk score from the three independent source members.
- `hybrid`: ensemble top-25% followed by the same farthest-first selection; 25% is frozen.
- `quantile_width`: legacy low-cost score `0.5*((q90-q10)/S_V1 + (q90-q10)/S_V2)`; no conformal alpha is used for ranking.

Test relevance, Morgan fingerprints, and condition distances are post-hoc diagnostics and are prohibited from acquisition.
