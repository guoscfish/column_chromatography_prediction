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

S1 and A1a are completed and stopped. T1a formal execution is complete without a stable winner. T1b-1 engineering, preregistration, capacity audit, and nine-fit smoke are complete; its 180-fit formal capacity sweep is not authorized.

## Next preregistered candidates

**A1a — completed and stopped:** the one-step shared-shortlist mechanism gate failed. Hybrid exceeded the same-shortlist random median in 3/5 seeds, the mean Hybrid-minus-control-mean was negative, and only 2/5 seeds reached an 8/10 beat count. A1b is not authorized.

**S1 — completed and stopped:** affine calibration substantially improved analysis-only compound GroupKFold; condition-aware Ridge did not add stable improvement. Reserved truth remains unconsumed.

**T1a — formal run complete, no stable winner:** row protocol with fixed Random nested target labels and budgets 30/50/70/100 (8 validation labels included). `target_head_only` achieved the best mean normalized AULC (0.6577) and best mean combined NRMSE at every budget, but its favorable mean/median paired delta versus `current_last2_head` came with only 3/5 seed wins, below the required 4/5. `last1_head` also won only 3/5. Historical `target_readout_only` means this fixed-sum, prediction-head-only candidate; no learnable graph readout was tested.

**T1b-1 — engineering/preregistered, formal not authorized:** one fixed graph-level residual adapter is inserted after fixed sum pooling and before the existing head. Widths 8/16/32 provide 2,958/5,014/9,126 trainable adapter-plus-head parameters. The primary comparison is adapter minus `target_head_only`; the frozen gate requires negative mean and median paired normalized AULC plus at least 4/5 wins. Because T1a row test outcomes are already known, this is developmental hypothesis testing rather than independent pristine confirmation.

## Stop and gate conditions

- T1b-1 formal execution requires separate manual authorization; smoke cannot select a width.
- T1b-2 is only a future matched-capacity adaptation-location placeholder for graph adapter versus message-passing adapter versus learnable readout; it is not implemented or authorized.
- Track A cannot freeze a strategy without stable row plus compound/scaffold evidence and more than the current three outer seeds.
- Track C cannot reopen until T1 establishes a stable low-label transfer formulation.
- A1b is stopped because A1a did not support the diversity mechanism.
- Test truth cannot select methods, tune weights, or retroactively change E2/E4 conclusions.
