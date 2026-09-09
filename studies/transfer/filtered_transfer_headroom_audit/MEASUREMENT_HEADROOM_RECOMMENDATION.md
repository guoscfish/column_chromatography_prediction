# Measurement headroom recommendation

The train-only decision is **NO_LIGHTWEIGHT_CALIBRATION_HEADROOM_IDENTIFIED**. These observations are prioritization evidence, not measurements of unobserved variables.

| rank | measurement | expected information gain | experimental cost | relevance to current bottleneck | identifiability |
| --- | --- | --- | --- | --- | --- |
| 1 | exact replicate experiments | high | low-medium | high: estimates the unknown noise floor | high |
| 2 | TLC Rf anchors | high | low | high: compound-specific signal for compound holdout | medium-high |
| 3 | actual void/hold-up volume V_M | medium | medium | medium: tests physical normalization | high once measured |
| 4 | crossed mass x flow | medium | high | medium: resolves current column/flow confounding | high for causal effects |

## Ranking

1. **Exact replicate experiments**: highest value for estimating the experimental noise floor, which is currently not identifiable from sparse duplicate rows.

2. **TLC Rf anchors**: next most relevant to the compound-split bottleneck because they are compound-specific and inexpensive; do not extrapolate literature effect sizes to this repository.

3. **Actual void/hold-up volume V_M**: physically motivated for column-volume normalization, but current residual evidence cannot establish its value without measured V_M.

4. **Crossed mass x flow**: important for causal identifiability, but lower immediate information gain because flow is nearly fixed within each current column.

## Train-only residual signal

| variable | target | partial_source_rank_corr | compound_controlled_rank_corr | support |
| --- | --- | --- | --- | --- |
| source_width | V2 | -0.2947 | 0.0048 | 364 |
| source_center | V1 | 0.2942 | 0.1582 | 367 |
| source_center | V1 | 0.2855 | 0.0993 | 359 |
| source_center | V1 | 0.2752 | 0.1396 | 354 |
| source_width | V2 | -0.2711 | 0.0332 | 354 |


No V_M or Rf values were imputed. The next computational stage after a negative gate is a filtered random learning curve at B=30/50/100/150/200/FULL, followed by active learning only if a label-scarcity regime is demonstrated.
