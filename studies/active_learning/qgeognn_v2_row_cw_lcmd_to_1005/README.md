# Center/Width-LCMD continuation: L653 to L1005

This study continues only the two frozen pure-CW trajectories. It runs no comparator or switch method and ends at L1005.

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_1005.py --prepare
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_1005.py --execute-seed 157
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_1005.py --execute-seed 6101
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_1005.py --finalize-pre-test
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_1005.py --reveal-test-and-report
```
