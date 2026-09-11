# 8g transfer context audit

The early G0-4 study used 574 no-threshold 8g rows and three seeds under paired
row and compound splits. It compared `last2_head`, full fine-tuning, and a
paper-style variant. The paper-style implementation appended repository column
diameter/length/packing-density values to every Graph-G edge, loaded compatible
4g weights, zero-initialized new column adapters, initialized a new monotonic
head from source quantiles, and trained the new adapters, last two GNN layers,
and head (242,216 trainable of 831,558 parameters). The shared setup used the
source scaler, equal V1/V2 loss, learning rate 1e-4, and validation-best
checkpointing.

Mean full-slice normalized test RMSE was 0.4482 for `last2_head`, 0.4692 for
full fine-tuning, and 0.4450 for paper-style. The validation-only rule retained
`last2_head`; paper-style worsened mean validation score by 15.3% and won 1/6
paired contexts. The legacy summary does not provide directly comparable
per-endpoint RMSE/R2 aggregates; its prediction and slice artifacts retain the
underlying results.

These results cannot enter the current 25g/40g ranking: they use a different
574-row population, only three seeds, older checkpoints, no-threshold labels,
different training architectures and parameter counts, and a different
normalization/evaluation contract. This study therefore records 8g context
without retraining it. A future architecture study must first establish one
matched protocol across all three columns.
