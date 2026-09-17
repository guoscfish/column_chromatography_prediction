# QGeoGNN-V2 row active-learning innovation screen

Status: `SELECTION-ONLY`.

This study compares nine fixed acquisition geometries at B=32 on development
outer seeds 73, 311, 1297, 4093 and 8191. It reuses frozen current-V2 L0
checkpoints and raw gradient caches, reveals only L0 labels, and does not train
after acquisition or evaluate any test target.

Run from the innovation worktree with the repository containing the read-only
development runtime artifacts supplied explicitly:

```bash
conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_4g_row_innovation_screen.py \
  --artifact-repository /Users/fish/Documents/GitHub/column_chromatography_prediction
```

The command has no confirmation-seed, retraining, selected-label, test-metric,
Target-IVR, MaxDet, BAIT, or sequential execution mode.
