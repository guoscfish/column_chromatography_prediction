# QGeoGNN-V2 row short sequential B=32 screen

This independent study directory contains a two-seed, three-acquisition-round developmental screen for `center_width_lcmd` and `direction_lcmd` only. It reuses exact historical round-zero artifacts and matched historical comparator trajectories, then stops at 429 active labels.

`ensemble_uncertainty` is intentionally excluded from execution in this revision.

Run through the dedicated entry point:

```bash
conda run -n fish python scripts/studies/run_qgeognn_v2_row_short_sequential_b32.py --prepare
conda run -n fish python scripts/studies/run_qgeognn_v2_row_short_sequential_b32.py --execute-seed 157
conda run -n fish python scripts/studies/run_qgeognn_v2_row_short_sequential_b32.py --execute-seed 6101
conda run -n fish python scripts/studies/run_qgeognn_v2_row_short_sequential_b32.py --finalize-pre-test
conda run -n fish python scripts/studies/run_qgeognn_v2_row_short_sequential_b32.py --reveal-test-and-report
```
