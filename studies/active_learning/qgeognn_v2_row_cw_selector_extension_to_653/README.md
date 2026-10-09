# CW selector factorial: exploratory continuation to653

Authorized after inspecting the completed333–525 development results. Three new arms × two seeds × four additional batches =24 new fits maximum. Own exact L525 histories continue; all retraining remains scratch from the original shared initialization. Forty historical control budget points were audited and reused. No LLM calls or new test labels.

Primary descriptive extension metric: normalized validation AULC525–653. Secondary: AULC333–653, NRMSE653, V1/V2 RMSE/MAE/R². Original AULC333–525 conclusions and failed new-selector confirmation gates remain unchanged. This extension is post-development exploration, not confirmation. Hard stop at653 regardless of effect.

Protocol/code/input hashes: protocol.json. Running/completed state: RUN_STATE.json and fit_ledger.json. Runtime remains local and ignored. A partial fit retains its quota reservation rather than silently retrying.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py run
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py report
```

## User stop amendment (current scope)

Whitening-LCMD stopped at525 for both seeds before any extension fit; preserve its historical result. Continue CW-MaxDet and CW-IVR only. Effective additional fit limits:16 through653 and44 thereafter,60 total. Seven arms × two seeds reach1005; Whitening's two curves end525. No late-window Whitening comparison is available. The original code/protocol is retained; the explicit stop amendment overrides execution membership and quota only.

Use `python scripts/studies/continue_cw_without_whitening.py` in the fish environment with `KMP_DUPLICATE_LIB_OK=TRUE`. Do not resume the old unamended supervisor.
