# Final report: full-data architecture headroom

## Decision

**HIERARCHICAL_CENTER_WIDTH_PROMOTED**. Selected baseline: **HIER_CW_25G_40G**.

## Compound-primary

| column | method | V1_rmse_mean | V2_rmse_mean | combined_normalized_rmse_mean | V1_mae_mean | V2_mae_mean | Center_rmse_mean | Width_rmse_mean | Var_eC_mean | Var_eW_mean | Cov_eC_eW_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | HIER_CW_25G_40G | 6.556 | 8.770 | 0.704 | 4.463 | 6.667 | 6.934 | 6.643 | 46.195 | 46.307 | 15.521 |
| 25g | HIER_CW_8G_25G_40G | 6.563 | 8.795 | 0.705 | 4.496 | 6.835 | 6.933 | 6.748 | 45.922 | 46.963 | 15.250 |
| 25g | M3_CENTER_WIDTH_FULL | 6.624 | 9.199 | 0.719 | 4.516 | 6.893 | 7.204 | 6.774 | 50.500 | 47.633 | 19.064 |
| 25g | M3_LATENT_RESIDUAL_RIDGE | 6.788 | 9.268 | 0.733 | 4.758 | 6.924 | 7.349 | 6.689 | 52.478 | 46.779 | 18.682 |
| 25g | M3_LATENT_TINY_ADAPTER | 6.629 | 9.224 | 0.720 | 4.525 | 6.909 | 7.222 | 6.772 | 50.478 | 47.430 | 19.042 |
| 40g | HIER_CW_25G_40G | 15.426 | 20.113 | 1.644 | 9.944 | 13.351 | 17.201 | 10.137 | 298.577 | 101.017 | 82.744 |
| 40g | HIER_CW_8G_25G_40G | 15.437 | 20.249 | 1.648 | 9.946 | 13.484 | 17.238 | 10.451 | 299.882 | 105.640 | 84.781 |
| 40g | M3_CENTER_WIDTH_FULL | 15.994 | 21.411 | 1.718 | 10.400 | 14.749 | 17.686 | 13.327 | 309.775 | 136.959 | 85.135 |
| 40g | M3_LATENT_RESIDUAL_RIDGE | 15.406 | 19.638 | 1.631 | 9.935 | 12.952 | 16.923 | 10.042 | 288.729 | 97.810 | 74.202 |
| 40g | M3_LATENT_TINY_ADAPTER | 15.397 | 19.417 | 1.626 | 10.170 | 12.933 | 16.706 | 10.599 | 281.775 | 106.730 | 68.430 |

## Row-secondary

| column | method | V1_rmse_mean | V2_rmse_mean | combined_normalized_rmse_mean |
| --- | --- | --- | --- | --- |
| 25g | HIER_CW_25G_40G | 8.132 | 11.231 | 0.882 |
| 25g | HIER_CW_8G_25G_40G | 8.216 | 11.520 | 0.895 |
| 25g | M3_CENTER_WIDTH_FULL | 8.081 | 11.646 | 0.889 |
| 25g | M3_LATENT_RESIDUAL_RIDGE | 7.749 | 11.075 | 0.850 |
| 25g | M3_LATENT_TINY_ADAPTER | 7.992 | 11.348 | 0.875 |
| 40g | HIER_CW_25G_40G | 14.270 | 17.441 | 1.494 |
| 40g | HIER_CW_8G_25G_40G | 14.277 | 17.611 | 1.499 |
| 40g | M3_CENTER_WIDTH_FULL | 14.296 | 17.406 | 1.496 |
| 40g | M3_LATENT_RESIDUAL_RIDGE | 13.763 | 16.797 | 1.441 |
| 40g | M3_LATENT_TINY_ADAPTER | 13.972 | 16.925 | 1.460 |

## Ranking and promotion gate

| rank | method | 25g_compound_nrmse | 40g_compound_nrmse | mean_compound_nrmse | relative_gain_vs_m3 | wins_vs_m3 | max_column_deterioration | max_endpoint_rmse_deterioration | aggregate_mae_change | promotion_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | M3_LATENT_TINY_ADAPTER | 0.720 | 1.626 | 1.173 | 0.037 | 6 | 0.001 | 0.003 | -0.055 | False |
| 2 | HIER_CW_25G_40G | 0.704 | 1.644 | 1.174 | 0.037 | 8 | -0.022 | -0.010 | -0.058 | True |
| 3 | HIER_CW_8G_25G_40G | 0.705 | 1.648 | 1.176 | 0.034 | 7 | -0.020 | -0.009 | -0.049 | False |
| 4 | M3_LATENT_RESIDUAL_RIDGE | 0.733 | 1.631 | 1.182 | 0.030 | 7 | 0.019 | 0.025 | -0.054 | False |
| 5 | M3_CENTER_WIDTH_FULL | 0.719 | 1.718 | 1.218 | 0.000 | 10 | 0.000 | 0.000 | 0.000 | False |

## Compound R2

| column | method | V1_r2_mean | V2_r2_mean |
| --- | --- | --- | --- |
| 25g | HIER_CW_25G_40G | 0.574 | 0.771 |
| 25g | HIER_CW_8G_25G_40G | 0.571 | 0.767 |
| 25g | M3_CENTER_WIDTH_FULL | 0.568 | 0.750 |
| 25g | M3_LATENT_RESIDUAL_RIDGE | 0.542 | 0.745 |
| 25g | M3_LATENT_TINY_ADAPTER | 0.568 | 0.749 |
| 40g | HIER_CW_25G_40G | 0.685 | 0.710 |
| 40g | HIER_CW_8G_25G_40G | 0.685 | 0.706 |
| 40g | M3_CENTER_WIDTH_FULL | 0.662 | 0.671 |
| 40g | M3_LATENT_RESIDUAL_RIDGE | 0.686 | 0.723 |
| 40g | M3_LATENT_TINY_ADAPTER | 0.686 | 0.729 |

## Exact inherited label accounting

The study copies the parent's complete `split_manifest.csv` and verifies every `(column, protocol, outer_seed, sample_id, role)` identity. Across compound seeds, gradient-train counts are 8g 429-440, 25g 306-334, and 40g 355-367; outer validation is 6-7 rows and test is 68-95 rows. The exact 20-row accounting table is `FULL_DATA_LABEL_ACCOUNTING.csv`. Donor outer-test rows used: zero.

## Models and selection

For target column k, `r_k = packing_mass_k / 4`, hence `(r_8g,r_25g,r_40g)=(2,6.25,10)`. A1/A2 fit `theta_C,k = theta_C,global + delta_C,k` and `theta_W,k = theta_W,global + delta_W,k` to `C_t/r_k` and `W_t/r_k`, where each theta has base slope, intercept, and EA slope effect; predictions multiply by `r_k` before `V1=C-W/2`, `V2=C+W/2`. The global M3 prior penalty is fixed at 0.1 and only `lambda_delta in {0.1,1,10,100}` is selected.

Ridge fits `rC=C_true-C_M3` and `rW=W_true-W_M3` from standardized frozen h. The adapter is exactly `Linear(128,16) -> GELU -> Dropout(0.1) -> Linear(16,2)`, has 2,098 trainable parameters, and has a zero-initialized output layer, so epoch 0 equals M3 exactly. QGeoGNN has zero trainable parameters. Ridge alpha and the four fixed adapter optimizer configurations are selected by at most five-fold GroupKFold on canonical SMILES wholly within each outer gradient-train. Every normalization is fold-train-only; outer validation is sanity-only and test is never a selector.

## Scientific questions

1. Mass anchor plus partial pooling beats independent M3: mean compound NRMSE improves 3.66% with 8/10 seed wins.
2. 25g/40g partial sharing lowers NRMSE on both columns (0.719 to 0.704; 1.718 to 1.644).
3. Adding 8g is mild negative transfer: mean NRMSE increases by 0.0026 (0.22%), and A2 beats A1 in only 4/10 direct pairs.
4. Frozen latent Ridge is column-dependent rather than stable incremental signal: it improves 40g by 5.04% but worsens 25g by 1.93% and misses the gate.
5. The useful 40g Ridge change is mainly Width: Width RMSE falls 24.65%, versus 4.32% for Center.
6. Tiny adapter has the lowest raw mean NRMSE (1.173) but only 6/10 wins and a slight 25g deterioration, so it does not robustly exceed Ridge or pass the gate.
7. Because tiny nonlinear mapping is slightly better in aggregate but unstable across columns/seeds, this study does not justify claiming residual structure is purely linear or reliably nonlinear.
8. HIER_CW_25G_40G is most balanced: all four endpoint mean RMSEs improve and aggregate MAE improves 5.83%.
9. HIER_CW_25G_40G puts 40g V1/V2 at 15.426/20.113, below 15.994/21.411, while also improving both 25g endpoints.
10. The promotion gate is met, establishing meaningful low-capacity architecture-level headroom over M3 on these frozen developmental splits.

The selected model reduces both Center and Width RMSE, and all four endpoint RMSEs improve, so promotion is not driven by `Cov(eC,eW)` cancellation.

## Contract

The 128D latent is `h = global_add_pool(backbone(atom, angle), atom.batch) + condition_branch(atom)[0]`, immediately after sum pooling and condition fusion and before `head: Linear(128,6) -> ReLU`. Predictions were hash-frozen before test truth was read. No Active Learning, reduced-label run, source-model update, feature sweep, or extra candidate was performed. Partial pooling is predictive evidence, not evidence that mass causally determines transfer. Latent results do not establish that QGeoGNN learned a physical transfer mechanism. Historical test exposure makes all results developmental rather than external validation.
