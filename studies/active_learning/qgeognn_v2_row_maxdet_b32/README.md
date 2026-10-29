# 研究记录：qgeognn_v2_row_maxdet_b32

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

# Row Gradient-MaxDet B32 study

This development study compares two new acquisition policies over the complete
333-to-1005 row-split learning curve:

- `gradient_maxdet`: greedy conditional D-optimal design on the authoritative
  512-dimensional full-network q50 gradient CountSketch;
- `u50_gradient_maxdet`: the identical selector restricted to the fixed top 50%
  of current-pool K=3 ensemble disagreement.

The study reuses exactly matching historical round-zero artifacts and all
published Random, Gradient-LCMD, Hybrid, Kernel-IVR, and full-data reference
results. Later Hybrid ensemble artifacts are forbidden because the U50-MaxDet
trajectory has a different labeled state.

The executed cohort is intentionally reduced to two seeds (`157`, `6101`) to
control runtime. This is a developmental subset of the historical five-seed
cohort, not an independent confirmation. All executed trajectories are frozen
behind one global pre-test barrier.

<a id="记录-2"></a>

## 记录 2：`PROTOCOL.md`

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
