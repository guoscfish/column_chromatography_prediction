# Center/Width-LCMD continuation 429 -> 525

This independent extension reads the frozen CW-LCMD round-3 state from the short sequential study and runs only three new acquisitions for seeds 157 and 6101: 429 -> 461 -> 493 -> 525. It does not retrain or rerun any comparator, and it hard-stops at 525.

```bash
conda run -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_525.py --prepare
conda run -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_525.py --execute-seed 157
conda run -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_525.py --execute-seed 6101
conda run -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_525.py --finalize-pre-test
conda run -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_525.py --reveal-test-and-report
```
