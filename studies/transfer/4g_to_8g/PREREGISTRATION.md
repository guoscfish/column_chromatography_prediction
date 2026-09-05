# Final QGeoGNN-V2 4g-to-8g transfer baseline

Use the preregistered row-seed-42 standalone 4g checkpoint and the frozen T1 target partitions, budgets, and source-only preprocessing. Compare only zero-shot, affine, target-head-only, last-two effective message layers, and full fine-tuning. Validation selects neural checkpoints; target test labels are evaluated only after fits freeze. No target threshold, adapter sweep, uncertainty acquisition, or post-test tuning is allowed.
