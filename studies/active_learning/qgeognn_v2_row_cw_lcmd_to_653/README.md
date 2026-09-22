# Center/Width-LCMD continuation: 525 to 653

This study asks only whether the frozen pure-CW mid-stage advantage persists from L525 through L653. It starts from the exact L525 state for seeds 157 and 6101 and does not run another method.

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_653.py --prepare
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_653.py --execute-seed 157
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_653.py --execute-seed 6101
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_653.py --finalize-pre-test
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_653.py --reveal-test-and-report
```

Do not execute any later budget from this study.
