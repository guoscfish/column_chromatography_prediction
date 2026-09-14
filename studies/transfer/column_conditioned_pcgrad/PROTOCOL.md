# Preregistered A2-PCGrad mechanism protocol

Frozen before any A2-PCGrad fit. `A2_ADAM_FROZEN` is the completed historical
reference; only `A2_PCGRAD` is trained. Architecture, source, preprocessing,
trainable scope, equal-task batches, raw quantile loss, Adam parameter groups,
learning rates, weight decay, epoch budget, patience, and joint checkpoint
selection are identical to completed A2.

## Sole intervention

At every optimization step, compute separate 4g, 25g, and 40g gradients for
the exact tuple returned by completed A2's `shared_gradient_parameters()`:
node blocks 3/4 and the index-3 edge/angle update feeding block 4. For each task
gradient, visit the other original task gradients in a deterministic shuffled
order. If their dot product is negative, subtract
`dot(g_i,g_j)/(||g_j||^2+1e-12) * g_j`. Average the three projected gradients
and write them only to this shared late-backbone tuple.

Every other trainable parameter—including all heads, task embedding, FiLM
generators, and condition-completion parameters—receives the ordinary gradient
of the arithmetic mean of the three task losses. There is no gradient
renormalization. The order seed is a stable unsigned-64-bit mix of outer seed,
inner fold (zero for fixed refits), epoch, and optimization step. Diagnostics
read cloned task gradients and do not alter optimizer gradients.

At epoch 1 and every 10 epochs, record raw and projected norms/cosines, directed
projection trigger counts, projected-minus-original mean-update norm, and the
projected/original mean-update norm ratio. Mechanism screening uses all fixed
diagnostic observations. "Materially reduced" is preregistered as both mean
pairwise cosine increasing by at least 0.05 and mean negative-cosine burden
`mean(max(0,-cos))` falling by at least 25%. At least 5% of directed projection
attempts must trigger. These thresholds are diagnostic gates, not training
controls.

## Data and matched reference

Use the exact completed FULL-data split SHA
`33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3`,
the same five seeds, the same 4g source supervision, and the same global
canonical-SMILES grouping across joined 25g+40g gradient-train rows. Before
each fit, compare all inner identities and train-only endpoint scales with the
frozen A2 record. The qualified source SHA is
`fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`.
Tail thresholds remain each endpoint's outer-gradient-train 80th percentile.

## Sequential gates

COMPOUND trains 25 new fits. Each column must have at least 2% mean paired-fold
NRMSE gain, 3/5 seed wins, 13/25 fold wins, no mean endpoint RMSE or MAE
deterioration above 2%, and no seed-pooled tail endpoint RMSE deterioration
above 2%. The mechanism thresholds above must also pass. Any failure stops the
study before ROW and outer prediction.

Only after COMPOUND passes, train 25 ROW fits without changes. For each column,
mean NRMSE and every endpoint RMSE/MAE may deteriorate by at most 2% versus the
matched frozen A2 ROW inner record. Failure stops before outer prediction.

Only after both gates pass, refit the candidate on each complete outer
gradient-train population for the round-half-up median of its five joint inner
best epochs. Generate every COMPOUND and ROW validation/test prediction without
outer truth, then freeze all prediction/completion SHA256 values in one global
manifest. A separate score action may then read outer test truth. All results
are `DEVELOPMENTAL CONFIRMATION`; inherited COMPOUND outer partitions remain
per-column rather than globally molecule-isolated.

`MECHANISM_SUCCESS` requires positive COMPOUND A2-PCGrad gain versus A2-Adam in
both columns, at least 3/5 wins in each, all endpoint RMSE/MAE and tail guards,
and no ROW NRMSE deterioration above 2%. `ROBUST_TRANSFER_GAIN` additionally
requires at least 3% COMPOUND gain in both columns, removal of ROW instability,
and superiority to the strongest compatible frozen COMPOUND and ROW references.

No CAGrad, GradNorm, NashMTL, weighting, MMoE, adapters, architecture changes,
embedding/FiLM/head/readout changes, hyperparameter or loss searches, 8g,
calibration, physical descriptors/scaling, Active Learning, or post-result
candidate is authorized.
