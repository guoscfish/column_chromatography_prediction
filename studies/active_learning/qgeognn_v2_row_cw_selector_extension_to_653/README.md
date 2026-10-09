# CW selector factorial: exploratory continuation to653

Authorized after inspecting the completed333–525 development results. Three new arms × two seeds × four additional batches =24 new fits maximum. Own exact L525 histories continue; all retraining remains scratch from the original shared initialization. Forty historical control budget points were audited and reused. No LLM calls or new test labels.

Primary descriptive extension metric: normalized validation AULC525–653. Secondary: AULC333–653, NRMSE653, V1/V2 RMSE/MAE/R². Original AULC333–525 conclusions and failed new-selector confirmation gates remain unchanged. This extension is post-development exploration, not confirmation. Hard stop at653 regardless of effect.

Protocol/code/input hashes: protocol.json. Running/completed state: RUN_STATE.json and fit_ledger.json. Runtime remains local and ignored. A partial fit retains its quota reservation rather than silently retrying.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py run
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py report
```
