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

S1 and A1a are completed and stopped. T1a formal execution is complete without a stable winner. T1b-1 is also formal complete: all 180 Adapter fits and 120 six-method evaluation contexts completed, with no failed or missing fits and no test-truth read before predictions were frozen.

## Next preregistered candidates

**A1a — completed and stopped:** the one-step shared-shortlist mechanism gate failed. Hybrid exceeded the same-shortlist random median in 3/5 seeds, the mean Hybrid-minus-control-mean was negative, and only 2/5 seeds reached an 8/10 beat count. A1b is not authorized.

**S1 — completed and stopped:** affine calibration substantially improved analysis-only compound GroupKFold; condition-aware Ridge did not add stable improvement. Reserved truth remains unconsumed.

**T1a — formal run complete, no stable winner:** row protocol with fixed Random nested target labels and budgets 30/50/70/100 (8 validation labels included). `target_head_only` achieved the best mean normalized AULC (0.6577) and best mean combined NRMSE at every budget, but its favorable mean/median paired delta versus `current_last2_head` came with only 3/5 seed wins, below the required 4/5. `last1_head` also won only 3/5. Historical `target_readout_only` means this fixed-sum, prediction-head-only candidate; no learnable graph readout was tested.

**T1b-1 — formal complete, no intermediate-capacity benefit:** T1a raised the narrower question of whether capacity between 774-parameter Head and 93,454-parameter Last1 could improve low-label transfer. One fixed graph-level residual adapter, `h'_G = h_G + W_up(ReLU(W_down(h_G)))`, was inserted after fixed sum pooling and before the monotonic head. Widths 8/16/32 supplied 2,958/5,014/9,126 trainable adapter-plus-head parameters. Mean normalized AULC was 0.6583/0.6587/0.6578 versus 0.6577 for Head. Paired wins were 2/5, 2/5, and 3/5; no Adapter met the frozen negative-mean, negative-median, at-least-4/5 gate. This rejects an intermediate sweet spot in the tested range, not latent adaptation in general or larger/smaller capacity universally. Because T1b-1 reuses the already-consumed T1a row protocol, it remains developmental hypothesis testing rather than independent pristine confirmation.

## Stop and gate conditions

- T1b-1 is complete and does not authorize an expanded width sweep.
- T1b-2 remains only a proposed matched-capacity adaptation-location study. Given the null capacity result, independent compound-level, another-column, or new-target validation has higher priority; T1b-2 is not implemented or authorized.
- Track A cannot freeze a strategy without stable row plus compound/scaffold evidence and more than the current three outer seeds.
- Track C cannot reopen until T1 establishes a stable low-label transfer formulation.
- A1b is stopped because A1a did not support the diversity mechanism.
- Test truth cannot select methods, tune weights, or retroactively change E2/E4 conclusions.
