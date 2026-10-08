# CW selector factorial and offline feedback ablations

See PROTOCOL.md, RUN_STATE.json, MECHANISM_AUDIT.md and RESULTS.md. Primary development window is validation AULC333–525. Runtime stays local and ignored.

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py run
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_geometry_feedback.py report
```

Completed fit hashes are verified before reuse. An interrupted fit without fit_audit.json stops explicitly: its reservation is retained and another attempt must be counted within the 36-fit cap. No automatic LLM transport is provided or enabled.
