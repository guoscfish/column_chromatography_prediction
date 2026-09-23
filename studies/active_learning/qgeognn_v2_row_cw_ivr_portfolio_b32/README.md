# Balanced CW / IVR B32 portfolio

Two-seed development study, L333 through L653. See `PROTOCOL.md` for the frozen
scientific design and `IMPLEMENTATION_AUDIT.md` for repository/method provenance.
No historical study is overwritten or retrained. Runtime artifacts are local
and intentionally git-ignored; aggregate scientific outputs and hashes are committed.

Use the repository's `fish` Python environment with `KMP_DUPLICATE_LIB_OK=TRUE`:

```sh
python scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py --prepare
python scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py --selection-smoke
python scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py --execute-seed 157
python scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py --execute-seed 6101
python scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py --finalize-pre-test
python scripts/studies/run_qgeognn_v2_row_cw_ivr_portfolio_b32.py --reveal-test-and-report
```

Run preparation only after the passing `preflight_tests.xml` has been generated.
`--validate` verifies the protocol, environment and protected historical sources.
The second seed requires a complete, hash-verified first trajectory. Completed
fits and acquisitions resume deterministically. Report generation verifies both
complete trajectories before opening the test label store.
