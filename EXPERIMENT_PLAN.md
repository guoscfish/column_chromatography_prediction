> Historical record: its original stage decisions are superseded by the final standalone predictor qualification and final-source transfer study. T1/T1b/G0/S1 are HISTORICAL_LEGACY_PREDICTOR_EVIDENCE. Current authority: `docs/NEXT_STAGE_DECISION.md`; active API: `src/qgeognn_al/models/qgeognn_v2.py`.

# Experiment Plan — Research Reset

## Current question

Two independent branches remain open: whether 4g in-domain AL can become robust beyond row splits (Track A), and which mechanism best adapts 4g knowledge to scarce 8g labels (Track B). Active transfer (Track C) depends on Track B and is deferred.

## Frozen historical conclusions

- E2 row: Hybrid/Coverage beat Random in 3/3 seeds; strong pilot evidence, not a final method.
- E2 compound: Hybrid/Coverage beat Random in 2/3 seeds; suggestive only.
- E4 and A2a: tested generic acquisitions under `current_last2_head` did not stably beat Random.
- D45/D46: post-hoc only. D46's seeds did not create independent stochastic fits; 3/18 unique-candidate paired test-row intervals excluded zero.
- Historical scientific results and predictor behavior remain unchanged by I0. They are retained as Legacy QGeoGNN evidence or clean reproduction evidence derived from the legacy implementation, not as a line-by-line reproduction claim.

## Current stage

S1 and A1a are completed and stopped. T1a formal execution is complete without a stable winner. T1b-1 is also formal complete: all 180 Adapter fits and 120 six-method evaluation contexts completed, with no failed or missing fits and no test-truth read before predictions were frozen.

## Next preregistered candidates

**A1a — completed and stopped:** the one-step shared-shortlist mechanism gate failed. Hybrid exceeded the same-shortlist random median in 3/5 seeds, the mean Hybrid-minus-control-mean was negative, and only 2/5 seeds reached an 8/10 beat count. A1b is not authorized.

**S1 — completed and stopped:** affine calibration substantially improved analysis-only compound GroupKFold; condition-aware Ridge did not add stable improvement. Reserved truth remains unconsumed.

**T1a — formal run complete, no stable winner:** row protocol with fixed Random nested target labels and budgets 30/50/70/100 (8 validation labels included). `target_head_only` achieved the best mean normalized AULC (0.6577) and best mean combined NRMSE at every budget, but its favorable mean/median paired delta versus `current_last2_head` came with only 3/5 seed wins, below the required 4/5. `last1_head` also won only 3/5. Historical `target_readout_only` means this fixed-sum, prediction-head-only candidate; no learnable graph readout was tested.

**T1b-1 — formal complete, no tested low-capacity graph-adapter benefit:** T1a raised the narrower question of whether added low-capacity adaptation could improve transfer. One fixed graph-level residual adapter, `h'_G = h_G + W_up(ReLU(W_down(h_G)))`, was inserted after fixed sum pooling and before the monotonic head. Widths 8/16/32 supplied 2,958/5,014/9,126 trainable adapter-plus-head parameters. Mean normalized AULC was 0.6583/0.6587/0.6578 versus 0.6577 for Head. Paired wins were 2/5, 2/5, and 3/5; no Adapter met the frozen gate. This is a null result for the tested 3k–9k Adapter range. The large 9k–93k gap, including possible 17k/34k/67k capacities, was not tested, and no expanded sweep is authorized. Because T1b-1 reuses the already-consumed T1a row protocol, it remains developmental hypothesis testing rather than independent pristine confirmation.

**I0 — semantic implementation audit complete:** the frozen clean legacy predictor constructs ten continuous edge features but reads only positions 0–4; the acquisition representation uses all nine condition dimensions. I0 also separates 775,476 nominal requires-grad parameters from 456,620 gradient-bearing parameters.

**Predictor V2 — implementation preflight complete:** the versioned residual condition-completion candidate adds only the five previously unreachable conditions after fixed sum pooling. Three source checkpoints retained exact initialized predictions across ten fixtures, all intended features passed reachability, and source-train-only normalization passed leakage checks. This is not performance qualification. Formal 4g training, 8g transfer, UQ qualification, and active transfer remain unauthorized.

## Stop and gate conditions

- T1b-1 is complete and does not authorize an expanded width sweep.
- T1b-2 remains only a proposed matched-capacity adaptation-location study. Given the null capacity result, independent compound-level, another-column, or new-target validation has higher priority; T1b-2 is not implemented or authorized.
- Track A cannot freeze a strategy without stable row plus compound/scaffold evidence and more than the current three outer seeds.
- Track C cannot reopen until T1 establishes a stable low-label transfer formulation.
- A1b is stopped because A1a did not support the diversity mechanism.
- Test truth cannot select methods, tune weights, or retroactively change E2/E4 conclusions.
- The next Predictor V2 step would be a separately authorized 4g source qualification, not an automatic formal run. No authorization is granted by this plan.
