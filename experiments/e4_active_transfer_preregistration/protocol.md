# Frozen protocol

1. Load the D28 partition files; do not regenerate splits. For each seed and protocol, use the eight validation rows inside L0 and count all 50 as target-label budget.
2. Fit no target scaler. Reuse the frozen 4g source-train scaler. Keep test entirely isolated.
3. For each strategy and round, start from the designated source checkpoint(s), fine-tune only the last two GNN layers and prediction head on revealed target training labels, select by validation score, and reveal exactly 10 new labels. Persist selected IDs and RNG state before proceeding.
4. Ensemble scores must use three independent source checkpoints (different source training seeds), not copies of one checkpoint. Every member is independently fine-tuned from its own source checkpoint.
5. Report target NRMSE, RMSE by V1/V2, AULC, recovery, labels-to-90/95, and Protocol-B tail (`V1>60 OR V2>120`) versus common slice. Tail is a diagnostic only.

Formal guard: `L0=50`, `B=10`, `rounds=15`, `K=3`; changing any value requires a new decision record.
