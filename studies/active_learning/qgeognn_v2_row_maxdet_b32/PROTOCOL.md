# Frozen scientific protocol

This run is a two-seed developmental subset (`157`, `6101`) of the historical
five-seed row cohort. The historical row test cohort was previously exposed, so
the results are not independent confirmation.

Primary priority is **row-split performance and label efficiency**. Compound/OOD
behavior is outside the optimization target of this study.

Both methods use B=32, the two registered execution seeds (`157`, `6101`), identical row partitions,
L0 labels, validation/test rows, scratch initialization, predictor, loss,
checkpoint criterion, preprocessing, target scales, 512D CountSketch and sketch
seed. No ridge, gate, dimension, ensemble-size, transform, or output-weight search
is permitted.

For current features `phi`, define

`psi(x) = phi(x) / sqrt(mean_{i in L_t} ||phi(i)||^2)` and
`A_t = I + sum_{i in L_t} psi(i) psi(i)^T`.

Each batch greedily maximizes

`log(1 + psi(x)^T A^{-1} psi(x))`,

updating `A` after every selection. The numerical path is float64 and exact ties
follow canonical current-pool order.

The uncertainty-gated method first retains exactly
`ceil(0.50 * |U_t|)` rows under the existing K=3 normalized q50 ensemble
disagreement, with stable canonical-order ties. MaxDet then runs unchanged within
that fixed membership set.

Every method runs all 21 acquisition rounds (333 through 1005 labels). Test labels
are inaccessible to acquisition and checkpointing. All selected IDs, checkpoints,
and predictions must be globally frozen before test reveal.
