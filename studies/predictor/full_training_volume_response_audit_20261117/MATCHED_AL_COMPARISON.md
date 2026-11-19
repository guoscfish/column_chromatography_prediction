# 同一实验对上的AL与充分训练模型比较

**同pair主结果：** 在正式模型见过两端的体积对上，Fully相对AL L1005的差分MAE，V1仅改善0.02201mL，V2反而恶化0.00289mL。三级等权汇总的Fully预测幅度V1/V2仅为真实绝对变化均值的8.11%/5.83%，均不足10%。因此，当前充分训练模型没有在这些已见pair上普遍解决体积幅度压缩；还不能唯一归因于网络、loss、覆盖、优化或噪声。

本报告只读取已通过source角色与3ID隔离门控的派生评分表；没有新训练、推理、API或新的原始目标store访问，真值差分全部来自根分析已授权的派生表。比较对象为AL五seed的L525/L1005及预先纳入的L1485扩展，与正式Row seed42/525/1101三模型。**正式seed525不是AL预算L525。** 每个比较行使用同一对ID与同一个合法delta_true，四种方法的实际变化值逐项一致。

## 分母与曝光必须先说明

原始评分表33296行，包含4方法×2端点；去掉方法与端点复制后为4162个“正式模型×AL campaign×pair”比较记录。跨模型/AL历史去重后仅1427个物理pair ID，仍可共享分子与系列。不能将cross 3×5的15个组合当成15个配对随机seed或据此做强显著性结论。

| axis | exposure | comparison_pair_rows | unique_physical_pair_ids | unique_molecules | unique_series | formal_models | formal_AL_campaign_cells |
|---|---|---|---|---|---|---|---|
| PE/EA | all_exposures | 3062 | 1060 | 183 | 349 | 3 | 15 |
| V/ul | all_exposures | 1100 | 367 | 136 | 265 | 3 | 15 |
| PE/EA | both_source_train | 2399 | 1020 | 183 | 343 | 3 | 15 |
| PE/EA | both_source_validation | 41 | 35 | 28 | 30 | 3 | 13 |
| PE/EA | mixed_source_train_validation | 622 | 426 | 130 | 191 | 3 | 15 |
| V/ul | both_source_train | 905 | 354 | 132 | 254 | 3 | 15 |
| V/ul | both_source_validation | 6 | 6 | 6 | 6 | 3 | 5 |
| V/ul | mixed_source_train_validation | 189 | 133 | 76 | 113 | 3 | 15 |

| formal_seed | exposure | comparison_pair_rows | unique_physical_pair_ids | unique_molecules | formal_AL_campaign_cells |
|---|---|---|---|---|---|
| 42 | both_source_train | 313 | 243 | 113 | 5 |
| 42 | both_source_validation | 3 | 3 | 3 | 2 |
| 42 | mixed_source_train_validation | 57 | 51 | 35 | 5 |
| 525 | both_source_train | 297 | 239 | 113 | 5 |
| 525 | both_source_validation | 2 | 2 | 2 | 2 |
| 525 | mixed_source_train_validation | 73 | 59 | 43 | 5 |
| 1101 | both_source_train | 295 | 236 | 107 | 5 |
| 1101 | both_source_validation | 1 | 1 | 1 | 1 |
| 1101 | mixed_source_train_validation | 59 | 51 | 36 | 5 |

两端source-validation的体积对仅6条跨模型/AL比较记录（正式seed42/525/1101分别3/2/1条），样本不足，不能单凭该小子集确认泛化恢复或失败。原source-validation也参与历史checkpoint选择，本来就不是全新的确认集。source-train两端表示充分训练模型见过该pair；mixed只见过其中一端；曝光类别随正式模型变化，同一物理pair可能在不同模型下落入不同类别，类别数不能简单相加充当独立实验数。涉及当前正式模型source-test或隔离ID的记录全部不进入本表。

## 聚合顺序与主结果

**幅度变化：** 训练内同pair真实V1/V2绝对变化均值为3.2854/3.2665mL；AL L525预测0.0049/0.0066，L1005预测0.2145/0.1027，L1485预测0.4338/0.2261，Fully预测0.2664/0.1906mL。相对很早期有所恢复，不等于随训练规模普遍达到充分响应。

先在正式模型×AL campaign×曝光×端点的完全相同pair集合上计算指标，再在每个正式模型内对可评价AL campaigns等权，最后对正式模型等权。主阈值为真实绝对变化>0.5mL，flat为预测绝对变化<0.1mL；0.1/1.0mL敏感性完整保存。方向/flat没有有意义真实变化时保持NA，计数列标注有效campaign/模型数，不用0填充。

以下为体积主表，物理单位mL；差分MAE越低越好，响应幅度大不等于预测正确。

| exposure | method | target | n_formal_models | comparison_pair_rows | unique_pair_ids | observed_abs_mean | predicted_abs_mean | difference_mae | direction_accuracy | flat_fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| both_source_train | AL_L1005 | V1 | 3 | 905 | 354 | 3.2854 | 0.21446 | 3.1648 | 0.91054 | 0.6 |
| both_source_train | AL_L1005 | V2 | 3 | 905 | 354 | 3.2665 | 0.10266 | 3.2545 | 0.53467 | 0.61161 |
| both_source_train | AL_L1485 | V1 | 3 | 905 | 354 | 3.2854 | 0.43377 | 3.0825 | 0.90929 | 0.49992 |
| both_source_train | AL_L1485 | V2 | 3 | 905 | 354 | 3.2665 | 0.22607 | 3.2528 | 0.61201 | 0.5586 |
| both_source_train | AL_L525 | V1 | 3 | 905 | 354 | 3.2854 | 0.0049065 | 3.2822 | 0.89096 | 1 |
| both_source_train | AL_L525 | V2 | 3 | 905 | 354 | 3.2665 | 0.006625 | 3.2657 | 0.52487 | 1 |
| both_source_train | Fully_trained | V1 | 3 | 905 | 354 | 3.2854 | 0.26635 | 3.1428 | 0.90929 | 0.55523 |
| both_source_train | Fully_trained | V2 | 3 | 905 | 354 | 3.2665 | 0.19058 | 3.2574 | 0.6227 | 0.56296 |
| both_source_validation | AL_L1005 | V1 | 3 | 6 | 6 | 1.5997 | 0.24798 | 1.3519 | 1 | 0.66667 |
| both_source_validation | AL_L1005 | V2 | 3 | 6 | 6 | 4.2347 | 0.14507 | 4.1099 | 0.66667 | 0.66667 |
| both_source_validation | AL_L1485 | V1 | 3 | 6 | 6 | 1.5997 | 0.55335 | 1.2803 | 0.83333 | 0.16667 |
| both_source_validation | AL_L1485 | V2 | 3 | 6 | 6 | 4.2347 | 0.31457 | 3.9775 | 0.66667 | 0.33333 |
| both_source_validation | AL_L525 | V1 | 3 | 6 | 6 | 1.5997 | 0.0060564 | 1.5938 | 1 | 1 |
| both_source_validation | AL_L525 | V2 | 3 | 6 | 6 | 4.2347 | 0.011516 | 4.2308 | 0.66667 | 1 |
| both_source_validation | Fully_trained | V1 | 3 | 6 | 6 | 1.5997 | 0.37355 | 1.5457 | 0.83333 | 0.33333 |
| both_source_validation | Fully_trained | V2 | 3 | 6 | 6 | 4.2347 | 0.25344 | 4.0225 | 0.83333 | 0.5 |
| mixed_source_train_validation | AL_L1005 | V1 | 3 | 189 | 133 | 3.7036 | 0.20363 | 3.6092 | 0.84048 | 0.6 |
| mixed_source_train_validation | AL_L1005 | V2 | 3 | 189 | 133 | 3.0157 | 0.10135 | 2.9917 | 0.51735 | 0.61333 |
| mixed_source_train_validation | AL_L1485 | V1 | 3 | 189 | 133 | 3.7036 | 0.42489 | 3.557 | 0.82937 | 0.48683 |
| mixed_source_train_validation | AL_L1485 | V2 | 3 | 189 | 133 | 3.0157 | 0.22128 | 2.9644 | 0.64412 | 0.57667 |
| mixed_source_train_validation | AL_L525 | V1 | 3 | 189 | 133 | 3.7036 | 0.0053731 | 3.7004 | 0.82937 | 1 |
| mixed_source_train_validation | AL_L525 | V2 | 3 | 189 | 133 | 3.0157 | 0.0067634 | 3.0147 | 0.5345 | 1 |
| mixed_source_train_validation | Fully_trained | V1 | 3 | 189 | 133 | 3.7036 | 0.25692 | 3.5493 | 0.82937 | 0.55032 |
| mixed_source_train_validation | Fully_trained | V2 | 3 | 189 | 133 | 3.0157 | 0.18469 | 3.0044 | 0.6623 | 0.56021 |

方向准确率包含预测幅度极小但符号相同的情况，可以与高flat比例同时出现；不能由较高方向准确率认定幅度已学会。`correct_nonflat_fraction`及有效分母在CSV保留，避免把近零预测的正确符号当成明显条件响应。

正mae_improvement表示Fully相对对应AL的差分MAE更低；正flat_reduction表示平坦比例降低。比值使用上述分层均值之比，另保留逐campaign相对变化，不用近零真实变化作分母。

| exposure | comparison | target | mae_improvement | mae_relative_improvement | predicted_amplitude_change | direction_change | flat_reduction | n_improved_formal_models | n_formal_models |
|---|---|---|---|---|---|---|---|---|---|
| both_source_train | Full_vs_AL1005 | V1 | 0.022013 | 0.0069556 | 0.051895 | -0.0012579 | 0.044769 | 1 | 3 |
| both_source_train | Full_vs_AL1485 | V1 | -0.060271 | -0.019552 | -0.16742 | 0 | -0.055314 | 1 | 3 |
| both_source_train | Full_vs_AL525 | V1 | 0.13942 | 0.042476 | 0.26145 | 0.018322 | 0.44477 | 3 | 3 |
| both_source_train | Full_vs_AL1005 | V2 | -0.0028932 | -0.00088898 | 0.087921 | 0.088039 | 0.048654 | 1 | 3 |
| both_source_train | Full_vs_AL1485 | V2 | -0.0045609 | -0.0014021 | -0.035491 | 0.010696 | -0.0043535 | 2 | 3 |
| both_source_train | Full_vs_AL525 | V2 | 0.0083558 | 0.0025586 | 0.18396 | 0.097835 | 0.43704 | 3 | 3 |
| both_source_validation | Full_vs_AL1005 | V1 | -0.19379 | -0.14334 | 0.12558 | -0.16667 | 0.33333 | 2 | 3 |
| both_source_validation | Full_vs_AL1485 | V1 | -0.26543 | -0.20732 | -0.1798 | 0 | -0.16667 | 1 | 3 |
| both_source_validation | Full_vs_AL525 | V1 | 0.048103 | 0.030181 | 0.3675 | -0.16667 | 0.66667 | 3 | 3 |
| both_source_validation | Full_vs_AL1005 | V2 | 0.087435 | 0.021274 | 0.10837 | 0.16667 | 0.16667 | 1 | 3 |
| both_source_validation | Full_vs_AL1485 | V2 | -0.045058 | -0.011328 | -0.06113 | 0.16667 | -0.16667 | 2 | 3 |
| both_source_validation | Full_vs_AL525 | V2 | 0.20827 | 0.049226 | 0.24193 | 0.16667 | 0.5 | 2 | 3 |
| mixed_source_train_validation | Full_vs_AL1005 | V1 | 0.059954 | 0.016611 | 0.053286 | -0.011111 | 0.049683 | 1 | 3 |
| mixed_source_train_validation | Full_vs_AL1485 | V1 | 0.0077742 | 0.0021856 | -0.16797 | 0 | -0.063492 | 1 | 3 |
| mixed_source_train_validation | Full_vs_AL525 | V1 | 0.15119 | 0.040857 | 0.25155 | -6.9389e-18 | 0.44968 | 3 | 3 |
| mixed_source_train_validation | Full_vs_AL1005 | V2 | -0.012736 | -0.0042571 | 0.083345 | 0.14495 | 0.053127 | 1 | 3 |
| mixed_source_train_validation | Full_vs_AL1485 | V2 | -0.040037 | -0.013506 | -0.036584 | 0.018182 | 0.01646 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL525 | V2 | 0.010237 | 0.0033957 | 0.17793 | 0.12781 | 0.43979 | 2 | 3 |

## 正式模型逐一查看

| formal_seed | exposure | method | target | n_al_campaigns | difference_mae | predicted_abs_mean | direction_accuracy | flat_fraction |
|---|---|---|---|---|---|---|---|---|
| 42 | both_source_train | AL_L1005 | V1 | 5 | 3.1855 | 0.21314 | 0.9246 | 0.6 |
| 42 | both_source_train | AL_L1005 | V2 | 5 | 3.0751 | 0.10266 | 0.51614 | 0.61481 |
| 42 | both_source_train | AL_L1485 | V1 | 5 | 3.0983 | 0.43346 | 0.9246 | 0.49615 |
| 42 | both_source_train | AL_L1485 | V2 | 5 | 3.0749 | 0.22614 | 0.60634 | 0.55238 |
| 42 | both_source_train | AL_L525 | V1 | 5 | 3.302 | 0.004868 | 0.91691 | 1 |
| 42 | both_source_train | AL_L525 | V2 | 5 | 3.0873 | 0.0065482 | 0.5039 | 1 |
| 42 | both_source_train | Fully_trained | V1 | 5 | 3.2474 | 0.078798 | 0.9246 | 0.66569 |
| 42 | both_source_train | Fully_trained | V2 | 5 | 3.0711 | 0.09116 | 0.61418 | 0.68887 |
| 42 | both_source_validation | AL_L1005 | V1 | 2 | 0.66753 | 0.019721 | 1 | 1 |
| 42 | both_source_validation | AL_L1005 | V2 | 2 | 1.0762 | 0.015723 | 0.5 | 1 |
| 42 | both_source_validation | AL_L1485 | V1 | 2 | 0.5915 | 0.096843 | 1 | 0 |
| 42 | both_source_validation | AL_L1485 | V2 | 2 | 1.1266 | 0.072262 | 0.5 | 0.5 |
| 42 | both_source_validation | AL_L525 | V1 | 2 | 0.68463 | 0.0025513 | 1 | 1 |
| 42 | both_source_validation | AL_L525 | V2 | 2 | 1.0674 | 0.0059536 | 0.5 | 1 |
| 42 | both_source_validation | Fully_trained | V1 | 2 | 0.629 | 0.089842 | 1 | 0 |
| 42 | both_source_validation | Fully_trained | V2 | 2 | 1.0891 | 0.097108 | 0.5 | 0.5 |
| 42 | mixed_source_train_validation | AL_L1005 | V1 | 5 | 3.9077 | 0.20779 | 0.74476 | 0.6 |
| 42 | mixed_source_train_validation | AL_L1005 | V2 | 5 | 3.3058 | 0.1101 | 0.61301 | 0.6 |
| 42 | mixed_source_train_validation | AL_L1485 | V1 | 5 | 3.8754 | 0.44965 | 0.71143 | 0.45714 |
| 42 | mixed_source_train_validation | AL_L1485 | V2 | 5 | 3.2728 | 0.24137 | 0.75392 | 0.58 |
| 42 | mixed_source_train_validation | AL_L525 | V1 | 5 | 3.9869 | 0.0057479 | 0.74476 | 1 |
| 42 | mixed_source_train_validation | AL_L525 | V2 | 5 | 3.3251 | 0.0073893 | 0.58444 | 1 |
| 42 | mixed_source_train_validation | Fully_trained | V1 | 5 | 3.9571 | 0.080974 | 0.71143 | 0.65095 |
| 42 | mixed_source_train_validation | Fully_trained | V2 | 5 | 3.2919 | 0.094294 | 0.80846 | 0.68062 |
| 525 | both_source_train | AL_L1005 | V1 | 5 | 3.1287 | 0.21355 | 0.89476 | 0.6 |
| 525 | both_source_train | AL_L1005 | V2 | 5 | 3.7625 | 0.099405 | 0.54778 | 0.61132 |
| 525 | both_source_train | AL_L1485 | V1 | 5 | 3.0555 | 0.42501 | 0.89476 | 0.49943 |
| 525 | both_source_train | AL_L1485 | V2 | 5 | 3.7632 | 0.21888 | 0.61377 | 0.552 |
| 525 | both_source_train | AL_L525 | V1 | 5 | 3.2321 | 0.0048971 | 0.86276 | 1 |
| 525 | both_source_train | AL_L525 | V2 | 5 | 3.7699 | 0.006555 | 0.53449 | 1 |
| 525 | both_source_train | Fully_trained | V1 | 5 | 2.8734 | 0.70942 | 0.89476 | 0 |
| 525 | both_source_train | Fully_trained | V2 | 5 | 3.763 | 0.46655 | 0.62601 | 0 |
| 525 | both_source_validation | AL_L1005 | V1 | 2 | 2.9717 | 0.015816 | 1 | 1 |
| 525 | both_source_validation | AL_L1005 | V2 | 2 | 3.498 | 0.033365 | 0.5 | 1 |
| 525 | both_source_validation | AL_L1485 | V1 | 2 | 2.9034 | 0.092234 | 0.5 | 0.5 |
| 525 | both_source_validation | AL_L1485 | V2 | 2 | 3.4538 | 0.081703 | 0.5 | 0.5 |
| 525 | both_source_validation | AL_L525 | V1 | 2 | 2.9806 | 0.0069003 | 1 | 1 |
| 525 | both_source_validation | AL_L525 | V2 | 2 | 3.4935 | 0.018412 | 0.5 | 1 |
| 525 | both_source_validation | Fully_trained | V1 | 2 | 2.9008 | 1.0133 | 0.5 | 0 |
| 525 | both_source_validation | Fully_trained | V2 | 2 | 2.8565 | 0.64349 | 1 | 0 |
| 525 | mixed_source_train_validation | AL_L1005 | V1 | 5 | 2.9058 | 0.21018 | 0.98333 | 0.6 |
| 525 | mixed_source_train_validation | AL_L1005 | V2 | 5 | 1.8626 | 0.10596 | 0.49572 | 0.62 |
| 525 | mixed_source_train_validation | AL_L1485 | V1 | 5 | 2.7889 | 0.44525 | 0.98333 | 0.50333 |
| 525 | mixed_source_train_validation | AL_L1485 | V2 | 5 | 1.8481 | 0.23161 | 0.56845 | 0.6 |
| 525 | mixed_source_train_validation | AL_L525 | V1 | 5 | 3.0785 | 0.0050461 | 0.98333 | 1 |
| 525 | mixed_source_train_validation | AL_L525 | V2 | 5 | 1.8912 | 0.0062738 | 0.49572 | 1 |
| 525 | mixed_source_train_validation | Fully_trained | V1 | 5 | 2.6583 | 0.67914 | 0.98333 | 0 |
| 525 | mixed_source_train_validation | Fully_trained | V2 | 5 | 1.8955 | 0.44632 | 0.56845 | 0 |
| 1101 | both_source_train | AL_L1005 | V1 | 5 | 3.1802 | 0.21668 | 0.91228 | 0.6 |
| 1101 | both_source_train | AL_L1005 | V2 | 5 | 2.9258 | 0.10592 | 0.54007 | 0.6087 |
| 1101 | both_source_train | AL_L1485 | V1 | 5 | 3.0938 | 0.44285 | 0.9085 | 0.50417 |
| 1101 | both_source_train | AL_L1485 | V2 | 5 | 2.9203 | 0.2332 | 0.61591 | 0.57143 |
| 1101 | both_source_train | AL_L525 | V1 | 5 | 3.3125 | 0.0049545 | 0.89323 | 1 |
| 1101 | both_source_train | AL_L525 | V2 | 5 | 2.94 | 0.0067717 | 0.53622 | 1 |
| 1101 | both_source_train | Fully_trained | V1 | 5 | 3.3076 | 0.010835 | 0.9085 | 1 |
| 1101 | both_source_train | Fully_trained | V2 | 5 | 2.938 | 0.014034 | 0.62791 | 1 |
| 1101 | both_source_validation | AL_L1005 | V1 | 1 | 0.4166 | 0.7084 | 1 | 0 |
| 1101 | both_source_validation | AL_L1005 | V2 | 1 | 7.7555 | 0.38613 | 1 | 0 |
| 1101 | both_source_validation | AL_L1485 | V1 | 1 | 0.34598 | 1.471 | 1 | 0 |
| 1101 | both_source_validation | AL_L1485 | V2 | 1 | 7.3519 | 0.78976 | 1 | 0 |
| 1101 | both_source_validation | AL_L525 | V1 | 1 | 1.1163 | 0.0087175 | 1 | 1 |
| 1101 | both_source_validation | AL_L525 | V2 | 1 | 8.1315 | 0.010183 | 1 | 1 |
| 1101 | both_source_validation | Fully_trained | V1 | 1 | 1.1074 | 0.017559 | 1 | 1 |
| 1101 | both_source_validation | Fully_trained | V2 | 1 | 8.1219 | 0.019738 | 1 | 1 |
| 1101 | mixed_source_train_validation | AL_L1005 | V1 | 5 | 4.0142 | 0.19293 | 0.79333 | 0.6 |
| 1101 | mixed_source_train_validation | AL_L1005 | V2 | 5 | 3.8067 | 0.08799 | 0.44333 | 0.62 |
| 1101 | mixed_source_train_validation | AL_L1485 | V1 | 5 | 4.0068 | 0.37978 | 0.79333 | 0.5 |
| 1101 | mixed_source_train_validation | AL_L1485 | V2 | 5 | 3.7722 | 0.19086 | 0.61 | 0.55 |
| 1101 | mixed_source_train_validation | AL_L525 | V1 | 5 | 4.0359 | 0.0053253 | 0.76 | 1 |
| 1101 | mixed_source_train_validation | AL_L525 | V2 | 5 | 3.8277 | 0.0066272 | 0.52333 | 1 |
| 1101 | mixed_source_train_validation | Fully_trained | V1 | 5 | 4.0324 | 0.010649 | 0.79333 | 1 |
| 1101 | mixed_source_train_validation | Fully_trained | V2 | 5 | 3.8258 | 0.013465 | 0.61 | 1 |

不同正式模型有各自scaler、训练/validation划分和随机训练轨迹。它们与AL五seed不是相同随机划分的重复对照；本结果能描述预测差异，不能把差异唯一归因于训练行数。AL三个budget的共同pair均在各自已见H中，正式模型的训练曝光则必须按上表分开解释。L1485仅是已有checkpoint的同pair扩展，不是新AL运行，也不与L525/L1005使用不同样本凑比较。

## PE/EA作为因素参照

| exposure | method | target | observed_abs_mean | predicted_abs_mean | difference_mae | direction_accuracy | flat_fraction |
|---|---|---|---|---|---|---|---|
| both_source_train | AL_L1005 | V1 | 12.478 | 12.195 | 1.9449 | 0.96024 | 0.0071568 |
| both_source_train | AL_L1005 | V2 | 27.551 | 26.366 | 3.0175 | 0.96795 | 0.0040307 |
| both_source_train | AL_L1485 | V1 | 12.478 | 12.196 | 1.896 | 0.95539 | 0.012438 |
| both_source_train | AL_L1485 | V2 | 27.551 | 26.676 | 2.8982 | 0.96686 | 0.0044324 |
| both_source_train | AL_L525 | V1 | 12.478 | 11.566 | 2.4822 | 0.95753 | 0.007516 |
| both_source_train | AL_L525 | V2 | 27.551 | 24.684 | 4.1284 | 0.95852 | 0.0052293 |
| both_source_train | Fully_trained | V1 | 12.478 | 12.414 | 2.2087 | 0.9452 | 0.014584 |
| both_source_train | Fully_trained | V2 | 27.551 | 27.22 | 3.7033 | 0.94878 | 0.004543 |
| both_source_validation | AL_L1005 | V1 | 7.18 | 6.1826 | 2.1276 | 1 | 0 |
| both_source_validation | AL_L1005 | V2 | 19.155 | 18.612 | 6.429 | 0.83194 | 0 |
| both_source_validation | AL_L1485 | V1 | 7.18 | 6.1593 | 2.1319 | 0.90833 | 0.05 |
| both_source_validation | AL_L1485 | V2 | 19.155 | 18.591 | 6.4419 | 0.83194 | 0.041667 |
| both_source_validation | AL_L525 | V1 | 7.18 | 6.3782 | 2.311 | 1 | 0 |
| both_source_validation | AL_L525 | V2 | 19.155 | 15.43 | 5.0685 | 0.87222 | 0 |
| both_source_validation | Fully_trained | V1 | 7.18 | 7.2416 | 3.0822 | 0.98333 | 0 |
| both_source_validation | Fully_trained | V2 | 19.155 | 20.633 | 8.6955 | 0.87361 | 0 |
| mixed_source_train_validation | AL_L1005 | V1 | 12.645 | 11.711 | 2.39 | 0.96572 | 0.010038 |
| mixed_source_train_validation | AL_L1005 | V2 | 27.692 | 26.524 | 3.7354 | 0.94692 | 0.0012346 |
| mixed_source_train_validation | AL_L1485 | V1 | 12.645 | 11.778 | 2.1805 | 0.96852 | 0.014413 |
| mixed_source_train_validation | AL_L1485 | V2 | 27.692 | 26.893 | 3.3954 | 0.94332 | 0.0012346 |
| mixed_source_train_validation | AL_L525 | V1 | 12.645 | 11.226 | 2.7393 | 0.9669 | 0.0098118 |
| mixed_source_train_validation | AL_L525 | V2 | 27.692 | 24.824 | 4.6514 | 0.948 | 0.0028837 |
| mixed_source_train_validation | Fully_trained | V1 | 12.645 | 12.008 | 3.0364 | 0.94797 | 0.014224 |
| mixed_source_train_validation | Fully_trained | V2 | 27.692 | 27.786 | 4.7753 | 0.92296 | 0.0058153 |

| exposure | comparison | target | mae_improvement | mae_relative_improvement | predicted_amplitude_change | direction_change | flat_reduction | n_improved_formal_models | n_formal_models |
|---|---|---|---|---|---|---|---|---|---|
| both_source_train | Full_vs_AL1005 | V1 | -0.26376 | -0.13561 | 0.2184 | -0.015048 | -0.0074273 | 1 | 3 |
| both_source_train | Full_vs_AL1485 | V1 | -0.31263 | -0.16489 | 0.21786 | -0.010189 | -0.0021461 | 1 | 3 |
| both_source_train | Full_vs_AL525 | V1 | 0.27359 | 0.11022 | 0.84768 | -0.012334 | -0.0070681 | 2 | 3 |
| both_source_train | Full_vs_AL1005 | V2 | -0.68581 | -0.22728 | 0.85406 | -0.01917 | -0.00051227 | 1 | 3 |
| both_source_train | Full_vs_AL1485 | V2 | -0.80512 | -0.27781 | 0.54431 | -0.018086 | -0.00011058 | 1 | 3 |
| both_source_train | Full_vs_AL525 | V2 | 0.42515 | 0.10298 | 2.5363 | -0.0097386 | 0.0006863 | 2 | 3 |
| both_source_validation | Full_vs_AL1005 | V1 | -0.95456 | -0.44865 | 1.059 | -0.016667 | 0 | 1 | 3 |
| both_source_validation | Full_vs_AL1485 | V1 | -0.95025 | -0.44572 | 1.0824 | 0.075 | 0.05 | 1 | 3 |
| both_source_validation | Full_vs_AL525 | V1 | -0.77122 | -0.33372 | 0.86344 | -0.016667 | 0 | 1 | 3 |
| both_source_validation | Full_vs_AL1005 | V2 | -2.2665 | -0.35254 | 2.0204 | 0.041667 | 0 | 0 | 3 |
| both_source_validation | Full_vs_AL1485 | V2 | -2.2535 | -0.34982 | 2.0414 | 0.041667 | 0.041667 | 0 | 3 |
| both_source_validation | Full_vs_AL525 | V2 | -3.627 | -0.71561 | 5.203 | 0.0013889 | 0 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL1005 | V1 | -0.64641 | -0.27047 | 0.29677 | -0.017747 | -0.0041861 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL1485 | V1 | -0.85592 | -0.39254 | 0.2304 | -0.020546 | 0.00018886 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL525 | V1 | -0.29704 | -0.10844 | 0.78241 | -0.018926 | -0.0044119 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL1005 | V2 | -1.0399 | -0.27839 | 1.2622 | -0.023959 | -0.0045807 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL1485 | V2 | -1.3799 | -0.4064 | 0.89307 | -0.020355 | -0.0045807 | 0 | 3 |
| mixed_source_train_validation | Full_vs_AL525 | V2 | -0.12384 | -0.026623 | 2.9613 | -0.025031 | -0.0029315 | 1 | 3 |

PE/EA与体积的真实物理变化范围不同，不以两者幅度高低评价因素的重要性；它只作为同一模型是否存在选择性响应不足的参照。

## 分子异质性与限制

以下只用于描述Full相对AL1005的异质性，要求每个分子/campaign cell至少2个pair，避免仅一个pair撑起分布。所有分子仍完整保存在CSV。表中cell可共享分子、pair与模型，正负计数不是独立科研重复。

| formal_seed | exposure | target | molecule_campaign_cells | distinct_molecules | improved_cells | worse_cells | median_mae_improvement |
|---|---|---|---|---|---|---|---|
| 42 | both_source_train | V1 | 55 | 36 | 32 | 22 | 0.028618 |
| 42 | both_source_train | V2 | 55 | 36 | 27 | 26 | 0 |
| 42 | mixed_source_train_validation | V1 | 9 | 8 | 3 | 6 | -0.00011737 |
| 42 | mixed_source_train_validation | V2 | 9 | 8 | 4 | 5 | -2.9073e-05 |
| 525 | both_source_train | V1 | 49 | 32 | 37 | 12 | 0.13142 |
| 525 | both_source_train | V2 | 49 | 32 | 28 | 20 | 0.022347 |
| 525 | mixed_source_train_validation | V1 | 12 | 8 | 12 | 0 | 0.20997 |
| 525 | mixed_source_train_validation | V2 | 12 | 8 | 5 | 7 | -0.1234 |
| 1101 | both_source_train | V1 | 59 | 39 | 17 | 42 | -0.024922 |
| 1101 | both_source_train | V2 | 59 | 39 | 32 | 27 | 0.00029124 |
| 1101 | mixed_source_train_validation | V1 | 6 | 4 | 2 | 4 | -0.007943 |
| 1101 | mixed_source_train_validation | V2 | 6 | 4 | 4 | 2 | 0.0096692 |

不能只选择改善最大的分子或正式seed宣布问题解决。所有逐pair、逐campaign和分子cell的正反结果均保留；恢复或平坦曲线的展示由根分析的预定metadata曲线池处理，本脚本不事后更换样本。训练内响应改善不等于source-validation泛化改善，整体Row NRMSE也不能替代这些局部差分指标。

## 文件与复现

`matched_exposure_equal_models.csv`包含两级分层均值与有效分母；`matched_paired_improvements.csv`同时保存campaign、正式模型及正式模型等权三层配对变化；`matched_denominators.csv`区分比较记录和物理pair；`matched_molecule_heterogeneity.csv`保留所有分子cell。输入SHA及权限检查见`matched_summary_inputs.json`、`matched_summary_verification.json`。

入口：`python -B studies/predictor/full_training_volume_response_audit_20261117/matched_summary.py`。这里的“配对”仅指同一模型比较中的同pair，不是随机化干预或随机seed因果配对。没有打开原始AL真值列、正式目标文件或test预测/真值。
