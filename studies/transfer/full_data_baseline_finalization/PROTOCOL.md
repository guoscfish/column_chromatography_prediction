# Protocol

Compound split is primary and row split secondary. Existing reference predictions are hash-verified and reused. M3 is fit per focal column. Shared fits pool only the current seed/protocol gradient-train rows from 8g, 25g and 40g; every validation and test row remains excluded from fitting and context normalization. One penalty is selected from the frozen grid using the equal-column mean validation score. Candidate predictions are globally frozen before test truth is read. Legacy descriptors retain their code names and no units or causal meaning are asserted.
