# Converged P0/P1 neural-transfer baseline: final report

## Design

This is a ROW-only developmental confirmation because the frozen outer tests had historical exposure. P0 and P1 use only Adam/current BN/raw quantile loss/historical-shallow scope. Each outer gradient-train population is split by canonical-smiles GroupKFold; no outer validation or test label reached fitting or epoch selection. Final refits use the median inner-fold selected epoch(s) and a fixed-epoch API with no validation input.

## Selected epochs

| column | outer_seed | method | stage | inner_fold_count | best_epochs | median_best_epoch | selected_epoch | rounding_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | 769539383 | P0 | B | 5 | [185, 160, 157, 355, 94] | 160.0000 | 160 | round-half-up |
| 25g | 769539383 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 25g | 769539383 | P1 | B | 5 | [55, 52, 35, 48, 44] | 48.0000 | 48 | round-half-up |
| 25g | 1425370602 | P0 | B | 5 | [79, 91, 180, 467, 259] | 180.0000 | 180 | round-half-up |
| 25g | 1425370602 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 25g | 1425370602 | P1 | B | 5 | [33, 29, 105, 122, 52] | 52.0000 | 52 | round-half-up |
| 25g | 536279090 | P0 | B | 5 | [187, 143, 223, 162, 184] | 184.0000 | 184 | round-half-up |
| 25g | 536279090 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 25g | 536279090 | P1 | B | 5 | [101, 69, 151, 83, 52] | 83.0000 | 83 | round-half-up |
| 25g | 2767143051 | P0 | B | 5 | [91, 89, 266, 230, 242] | 230.0000 | 230 | round-half-up |
| 25g | 2767143051 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 25g | 2767143051 | P1 | B | 5 | [48, 33, 153, 129, 96] | 96.0000 | 96 | round-half-up |
| 25g | 1362771960 | P0 | B | 5 | [83, 103, 73, 264, 152] | 103.0000 | 103 | round-half-up |
| 25g | 1362771960 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 25g | 1362771960 | P1 | B | 5 | [496, 37, 27, 173, 77] | 77.0000 | 77 | round-half-up |
| 40g | 769539383 | P0 | B | 5 | [471, 181, 198, 285, 244] | 244.0000 | 244 | round-half-up |
| 40g | 769539383 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 40g | 769539383 | P1 | B | 5 | [490, 94, 127, 114, 191] | 127.0000 | 127 | round-half-up |
| 40g | 1425370602 | P0 | B | 5 | [277, 200, 214, 199, 159] | 200.0000 | 200 | round-half-up |
| 40g | 1425370602 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 40g | 1425370602 | P1 | B | 5 | [206, 145, 96, 105, 44] | 105.0000 | 105 | round-half-up |
| 40g | 536279090 | P0 | B | 5 | [225, 255, 228, 168, 211] | 225.0000 | 225 | round-half-up |
| 40g | 536279090 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 40g | 536279090 | P1 | B | 5 | [109, 206, 164, 209, 131] | 164.0000 | 164 | round-half-up |
| 40g | 2767143051 | P0 | B | 5 | [193, 198, 178, 215, 190] | 193.0000 | 193 | round-half-up |
| 40g | 2767143051 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 40g | 2767143051 | P1 | B | 5 | [75, 152, 188, 94, 138] | 138.0000 | 138 | round-half-up |
| 40g | 1362771960 | P0 | B | 5 | [319, 242, 500, 185, 206] | 242.0000 | 242 | round-half-up |
| 40g | 1362771960 | P1 | A | 5 | [500, 500, 500, 500, 500] | 500.0000 | 500 | round-half-up |
| 40g | 1362771960 | P1 | B | 5 | [233, 195, 412, 83, 127] | 195.0000 | 195 | round-half-up |



## Developmental test summary and convergence

| column | method | n_seeds | V1_rmse_mean | V1_rmse_std | V1_rmse_median | V1_rmse_min | V1_rmse_max | V1_mae_mean | V1_mae_std | V1_mae_median | V1_mae_min | V1_mae_max | V1_r2_mean | V1_r2_std | V1_r2_median | V1_r2_min | V1_r2_max | V2_rmse_mean | V2_rmse_std | V2_rmse_median | V2_rmse_min | V2_rmse_max | V2_mae_mean | V2_mae_std | V2_mae_median | V2_mae_min | V2_mae_max | V2_r2_mean | V2_r2_std | V2_r2_median | V2_r2_min | V2_r2_max | combined_normalized_rmse_mean | combined_normalized_rmse_std | combined_normalized_rmse_median | combined_normalized_rmse_min | combined_normalized_rmse_max | combined_source_normalized_rmse_mean | combined_source_normalized_rmse_std | combined_source_normalized_rmse_median | combined_source_normalized_rmse_min | combined_source_normalized_rmse_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P0 | 5 | 7.2714 | 1.3006 | 7.1420 | 5.6200 | 9.0578 | 4.3738 | 0.3352 | 4.5496 | 3.9287 | 4.6851 | 0.3779 | 0.3334 | 0.5261 | -0.1328 | 0.6427 | 10.9866 | 2.2166 | 10.0043 | 8.9594 | 13.6191 | 6.9250 | 0.5885 | 7.0784 | 6.0525 | 7.6262 | 0.5381 | 0.1727 | 0.6273 | 0.3026 | 0.6866 | 0.6683 | 0.1205 | 0.6221 | 0.5205 | 0.8214 | 0.8126 | 0.1478 | 0.7586 | 0.6400 | 1.0097 |
| 25g | P1 | 5 | 7.4314 | 1.5171 | 7.4289 | 5.3814 | 9.2383 | 4.4388 | 0.4502 | 4.6765 | 3.7572 | 4.8583 | 0.3437 | 0.3688 | 0.4873 | -0.1784 | 0.6724 | 10.9465 | 2.2626 | 9.9118 | 8.5965 | 13.6520 | 6.9545 | 0.6803 | 6.9834 | 5.8687 | 7.6836 | 0.5407 | 0.1761 | 0.6341 | 0.2992 | 0.7115 | 0.6757 | 0.1344 | 0.6450 | 0.4989 | 0.8317 | 0.8230 | 0.1657 | 0.7869 | 0.6133 | 1.0236 |
| 40g | P0 | 5 | 13.7451 | 2.2680 | 14.8751 | 10.5490 | 16.0087 | 8.7661 | 0.6119 | 9.0216 | 7.8924 | 9.3999 | 0.6945 | 0.0893 | 0.7063 | 0.5836 | 0.8109 | 16.4034 | 2.5318 | 16.3556 | 14.0169 | 20.2636 | 11.1422 | 1.2399 | 10.8765 | 9.8388 | 13.0178 | 0.7844 | 0.0397 | 0.8042 | 0.7358 | 0.8209 | 0.4778 | 0.0801 | 0.4914 | 0.3867 | 0.5804 | 1.4298 | 0.2258 | 1.5302 | 1.1359 | 1.6906 |
| 40g | P1 | 5 | 13.6559 | 2.4500 | 14.5119 | 10.3766 | 16.1988 | 8.5932 | 0.6814 | 8.6757 | 7.6114 | 9.5174 | 0.6968 | 0.1006 | 0.7205 | 0.5676 | 0.8170 | 16.3374 | 2.5679 | 16.7197 | 13.7514 | 20.1579 | 11.1212 | 1.3325 | 11.3512 | 9.8190 | 13.0385 | 0.7853 | 0.0464 | 0.8116 | 0.7311 | 0.8232 | 0.4751 | 0.0838 | 0.5032 | 0.3822 | 0.5827 | 1.4213 | 0.2420 | 1.4956 | 1.1207 | 1.7027 |



| column | method | stage | inner_folds | best_at_protocol_ceiling | run_reached_protocol_ceiling | negative_last_20_slope | mean_best_epoch | mean_epochs_run | mean_last_20_slope | still_budget_censored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P0 | B | 25 | 0 | 1 | 1 | 180.7600 | 258.8800 | 0.0002 | False |
| 25g | P1 | A | 25 | 25 | 25 | 25 | 500.0000 | 500.0000 | -0.0013 | True |
| 25g | P1 | B | 25 | 0 | 1 | 2 | 92.0000 | 168.9600 | 0.0004 | False |
| 40g | P0 | B | 25 | 1 | 2 | 0 | 237.6400 | 312.4000 | 0.0001 | False |
| 40g | P1 | A | 25 | 25 | 25 | 25 | 500.0000 | 500.0000 | -0.0009 | True |
| 40g | P1 | B | 25 | 0 | 1 | 0 | 165.1200 | 242.3200 | 0.0001 | False |



The convergence adequacy rule marks a stage `STILL_BUDGET_CENSORED` only where at least 40% of folds both selected epoch 500 and retained a negative last-20 validation slope. It does not silently extend the budget.

## P0 vs P1 developmental test comparison

| column | candidate | reference | metric | mean_paired_delta | std_paired_delta | median_paired_delta | seed_wins | n | relative_improvement_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P1 | P0 | V1_rmse | 0.1600 | 0.3013 | 0.1805 | 1 | 5 | -2.2007 |
| 25g | P1 | P0 | V1_mae | 0.0651 | 0.1601 | 0.1079 | 1 | 5 | -1.4876 |
| 25g | P1 | P0 | V1_r2 | -0.0342 | 0.0548 | -0.0388 | 4 | 5 |  |
| 25g | P1 | P0 | V2_rmse | -0.0401 | 0.2310 | -0.0558 | 3 | 5 | 0.3650 |
| 25g | P1 | P0 | V2_mae | 0.0295 | 0.1666 | 0.0574 | 2 | 5 | -0.4259 |
| 25g | P1 | P0 | V2_r2 | 0.0026 | 0.0164 | 0.0050 | 2 | 5 |  |
| 25g | P1 | P0 | combined_normalized_rmse | 0.0074 | 0.0198 | 0.0103 | 2 | 5 | -1.1011 |
| 25g | P1 | P0 | combined_source_normalized_rmse | 0.0104 | 0.0257 | 0.0139 | 2 | 5 | -1.2792 |
| 40g | P1 | P0 | V1_rmse | -0.0892 | 0.3116 | -0.1724 | 3 | 5 | 0.6487 |
| 40g | P1 | P0 | V1_mae | -0.1729 | 0.3776 | -0.2810 | 3 | 5 | 1.9722 |
| 40g | P1 | P0 | V1_r2 | 0.0023 | 0.0142 | 0.0061 | 2 | 5 |  |
| 40g | P1 | P0 | V2_rmse | -0.0660 | 0.3511 | -0.1057 | 4 | 5 | 0.4022 |
| 40g | P1 | P0 | V2_mae | -0.0210 | 0.3612 | 0.0207 | 2 | 5 | 0.1887 |
| 40g | P1 | P0 | V2_r2 | 0.0009 | 0.0100 | 0.0028 | 1 | 5 |  |
| 40g | P1 | P0 | combined_normalized_rmse | -0.0027 | 0.0100 | -0.0045 | 3 | 5 | 0.5678 |
| 40g | P1 | P0 | combined_source_normalized_rmse | -0.0085 | 0.0307 | -0.0151 | 3 | 5 | 0.5913 |



## Historical paper-style reference

| column | V1_rmse | V1_mae | V1_r2 | V2_rmse | V2_mae | V2_r2 | historical_source_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | 7.4088 | 4.4615 | 0.3490 | 11.1114 | 6.9671 | 0.5249 | 0.8257 |
| 40g | 13.1091 | 7.9872 | 0.7209 | 15.8968 | 10.6430 | 0.7972 | 1.3699 |



## P0/P1 raw endpoint gap versus paper-style

| column | method | V1_rmse_mean_delta_vs_paper_style | V1_mae_mean_delta_vs_paper_style | V1_r2_mean_delta_vs_paper_style | V2_rmse_mean_delta_vs_paper_style | V2_mae_mean_delta_vs_paper_style | V2_r2_mean_delta_vs_paper_style |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | P0 | -0.1374 | -0.0878 | 0.0290 | -0.1248 | -0.0422 | 0.0132 |
| 25g | P1 | 0.0226 | -0.0227 | -0.0053 | -0.1649 | -0.0127 | 0.0158 |
| 40g | P0 | 0.6360 | 0.7788 | -0.0264 | 0.5066 | 0.4993 | -0.0128 |
| 40g | P1 | 0.5468 | 0.6060 | -0.0241 | 0.4406 | 0.4783 | -0.0119 |



## Promotion decision

`NO_UNIVERSAL_STAGED_TRANSFER_GAIN`. The frozen gate requires both columns to improve mean train-normalized NRMSE, at least 3/5 paired seed wins per column, no mean endpoint RMSE/MAE deterioration >2%, no systematic opposite R2 direction, and no serious inner-CV budget censoring. Per-column checks: `[{"column": "25g", "nrmse_improved": false, "wins_at_least_3": false, "endpoint_ok": false, "still_budget_censored": true}, {"column": "40g", "nrmse_improved": true, "wins_at_least_3": true, "endpoint_ok": true, "still_budget_censored": true}]`.

## Next scientific action

If convergence is adequate, the next isolated neural ablation is endpoint-normalized quantile loss; q50-only/weak-quantile loss follows to test point-loss mismatch. Normalized L2-SP must first correct its parameter-count scaling and be selected by inner CV. Scope/discriminative-LR work follows those loss/regularization controls. Structured HIER and measured-4g-anchor are separate structured/domain branches, not additions to this neural comparison. COMPOUND and Active Learning remain blocked by this ROW developmental result alone.
