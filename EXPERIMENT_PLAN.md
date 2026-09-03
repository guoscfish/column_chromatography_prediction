# Experiment Plan — Research Reset

## Current question

Two independent branches remain open: whether 4g in-domain AL can become robust beyond row splits (Track A), and which mechanism best adapts 4g knowledge to scarce 8g labels (Track B). Active transfer (Track C) depends on Track B and is deferred.

## Frozen historical conclusions

- E2 row: Hybrid/Coverage beat Random in 3/3 seeds; strong pilot evidence, not a final method.
- E2 compound: Hybrid/Coverage beat Random in 2/3 seeds; suggestive only.
- E4 and A2a: tested generic acquisitions under `current_last2_head` did not stably beat Random.
- D45/D46: post-hoc only. D46's seeds did not create independent stochastic fits; 3/18 unique-candidate paired test-row intervals excluded zero.
- Historical scientific results and predictor behavior remain unchanged by repository refactoring.

## Current stage

S1 and A1a are completed and stopped. T1a formal execution is complete: 180/180 neural fits and 120/120 evaluation contexts passed the completion gate, but no candidate passed the preregistered stable-improvement gate.

## Next preregistered candidates

**A1a — completed and stopped:** the one-step shared-shortlist mechanism gate failed. Hybrid exceeded the same-shortlist random median in 3/5 seeds, the mean Hybrid-minus-control-mean was negative, and only 2/5 seeds reached an 8/10 beat count. A1b is not authorized.

**S1 — completed and stopped:** affine calibration substantially improved analysis-only compound GroupKFold; condition-aware Ridge did not add stable improvement. Reserved truth remains unconsumed.

**T1a — formal run complete, no stable winner:** row protocol with fixed Random nested target labels and budgets 30/50/70/100 (8 validation labels included). `target_head_only` achieved the best mean normalized AULC (0.6577) and best mean combined NRMSE at every budget, but its favorable mean/median paired delta versus `current_last2_head` came with only 3/5 seed wins, below the required 4/5. `last1_head` also won only 3/5. Historical `target_readout_only` means this fixed-sum, prediction-head-only candidate; no learnable graph readout was tested.

## Stop and gate conditions

- Any T1b genuine learnable-readout comparison requires a new preregistration and separate manual authorization; it is not implemented here.
- Track A cannot freeze a strategy without stable row plus compound/scaffold evidence and more than the current three outer seeds.
- Track C cannot reopen until T1 establishes a stable low-label transfer formulation.
- A1b is stopped because A1a did not support the diversity mechanism.
- Test truth cannot select methods, tune weights, or retroactively change E2/E4 conclusions.
