# Full-pool feedback LLM active learning

Run with the existing `fish` environment and `KMP_DUPLICATE_LIB_OK=TRUE` on
this host, where Torch and RDKit otherwise load conflicting OpenMP libraries.

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_full_pool.py validate
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_full_pool.py execute --seed 157 --method cw16_random16_full_pool
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_full_pool.py execute --seed 6101 --method cw16_random16_full_pool
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_full_pool.py execute --seed 157 --method cw16_llm16_full_pool
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_full_pool.py execute --seed 6101 --method cw16_llm16_full_pool
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_full_pool.py report
```

Set `OPENAI_API_KEY` in the process environment to run C. Do not put a key
in the repository or command log. The selector uses the frozen model and
settings in `protocol.json`; no model choice is made interactively. If an API
attempt is interrupted, inspect its transcript and pass `--retry-selector`
to record a fresh attempt against the same frozen packet. Re-running a
completed round verifies and reuses its frozen batch, feedback, and fit.

`selections/seed_*/method/round_*/` holds each allowed input, the complete
currently measured record index, numeric references, selector transcript,
selection freeze, and post-freeze feedback. `runtime/` holds regenerable
checkpoints and test predictions and is ignored by Git. `report` refuses to
reveal test truth until all four new trajectories finish.
