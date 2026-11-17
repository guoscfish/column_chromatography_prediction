# 三模型固定标签对照结果

15次预定拟合全部完成：五个固定seed各训练A原模型、B上样量直接残差、C独立样品体积直接残差。没有新增seed、调参、LLM请求或AL重跑。所有模型从头初始化，原L1005 checkpoint只用于独立的零训练审计，不作为A的结果。

## 数据与可比性

每seed固定H1005，剔除先前metadata-only方案预定的10个完整体积系列。训练行数分别982/981/984/985/984，检查行数23/24/21/20/21；原Row validation均416。H内同系列及完全等价条件的训练重叠均0。A参数458952，B/C各458954；每个端点仅新增一个零初始化斜率，三个分位共享预激活偏移，随后执行原ReLU/eval clamp。B取原typed normalization的amount；C独立raw V经stable sample ID连接，五seed固定30–150µL缩放。

三个模型的共同初始参数及初始化后torch RNG逐seed相同，零skip初始输出逐值一致。loss、Adam、batch2048、lr0.001、weight decay0、最大1000 epoch、patience100、每epoch样本顺序、固定L333目标尺度及validation选择标准均保持原训练规则。独立模型variant/schema拒绝旧checkpoint。

检查标签没有进入训练或early stopping，但原validation存在同系列关联：五seed分别2/3/2/0/2个检查系列与validation相连，2357和12203各有一条完全同条件validation记录。因此这里不是完全独立的确认集。另报删除所有validation关联系列后的固定预测敏感性结果；不改checkpoint，不补训练。

## 逐seed结果先行

主局部指标为同系列所有体积pair差分MAE，先对V1/V2等权，再五seed等权；pair不是独立重复。Row指标用原validation和L333尺度。下表均越低越好；逐endpoint方向、幅度、平坦率与0.1/0.5/1mL阈值见CSV。`paired_endpoint_differences.csv`另外逐seed列出B−A/C−A/C−B的V1/V2 Row RMSE、局部差分MAE、幅度、方向和平坦率配对差。

| seed | kind | difference_error_mean | combined_normalized_rmse | V1_rmse | V2_rmse | best_epoch | epochs_run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | A_original | 2.88369 | 0.40053 | 3.04001 | 4.20871 | 399 | 499 |
| 157 | B_amount_skip | 2.97822 | 0.42111 | 3.23433 | 4.31016 | 257 | 357 |
| 157 | C_volume_skip | 3.00988 | 0.41204 | 3.17137 | 4.19665 | 158 | 258 |
| 887 | A_original | 4.96221 | 0.36183 | 3.34021 | 5.40707 | 220 | 320 |
| 887 | B_amount_skip | 4.95612 | 0.34572 | 3.20743 | 5.12351 | 177 | 277 |
| 887 | C_volume_skip | 4.95451 | 0.34820 | 3.16422 | 5.33402 | 135 | 235 |
| 2357 | A_original | 2.64889 | 0.35626 | 2.38696 | 6.07893 | 149 | 249 |
| 2357 | B_amount_skip | 2.66232 | 0.36232 | 2.48509 | 6.09146 | 149 | 249 |
| 2357 | C_volume_skip | 2.64202 | 0.36006 | 2.47144 | 6.05066 | 171 | 271 |
| 6101 | A_original | 3.55077 | 0.44253 | 2.66331 | 4.81260 | 124 | 224 |
| 6101 | B_amount_skip | 3.54753 | 0.42553 | 2.51393 | 4.75619 | 148 | 248 |
| 6101 | C_volume_skip | 3.53898 | 0.43931 | 2.61586 | 4.85494 | 195 | 295 |
| 12203 | A_original | 2.56607 | 0.41950 | 3.68148 | 6.07561 | 165 | 265 |
| 12203 | B_amount_skip | 2.52863 | 0.41799 | 3.70071 | 5.96894 | 203 | 303 |
| 12203 | C_volume_skip | 2.61529 | 0.41684 | 3.67843 | 5.98439 | 104 | 204 |

## 五seed等权汇总与配对差

| kind | difference_error_mean | combined_normalized_rmse | V1_rmse | V2_rmse | observed_abs_mean | predicted_abs_mean | direction_accuracy_meaningful | flat_fraction_meaningful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_original | 3.32233 | 0.39613 | 3.02240 | 5.31658 | 3.37260 | 0.09318 | 0.70334 | 0.70333 |
| B_amount_skip | 3.33457 | 0.39453 | 3.02830 | 5.25005 | 3.37260 | 0.07283 | 0.70334 | 0.77841 |
| C_volume_skip | 3.35214 | 0.39529 | 3.02026 | 5.28413 | 3.37260 | 0.03977 | 0.70334 | 0.95889 |

正local improvement表示改善；正Row change表示变差。B−A检验强化已有量信号；C−A检验独立V直接通路；C−B比较相同两参数通路中的数值表示。相同表示可能因缩放及优化轨迹不同而表现不同，不能从C−B推断物理因果。

| comparison | local_improvement_ratio_equal_seed_means | local_positive_seeds | row_relative_change_equal_seed_means | row_nonworse_seeds | row_worse_over_one_percent_seeds | mechanism_reference_pass | row_relevance_reference_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-A | -0.00368 | 3 | -0.00402 | 3 | 2 | False | False |
| C-A | -0.00897 | 3 | -0.00211 | 3 | 2 | False | False |
| C-B | -0.00527 | 3 | 0.00192 | 3 | 1 | False | False |
本轮B/C局部MAE相对A分别变差0.37%和0.90%，各只有3/5 seed改善，均未通过20%/4seed机制门槛。Row平均仅改善0.40%和0.21%，各3/5 seed不变差，也未通过预定Row relevance门槛。两种skip都没有得到所需局部收益，停止本组简单残差路线，不追加调参或模型。

检查集真实绝对体积变化均值3.373mL，A/B/C预测幅度仅0.093/0.073/0.040mL；meaningful pair近似平坦率分别70.33%/77.84%/95.89%。三模型方向准确率相同，但大多数预测变化很小；微小同号变化不能被解释为学会了响应幅度。正确非平坦/错误非平坦比例另外保留在`local_response_metrics.csv`，不由方向准确率单独作决策。


20%局部改善、至少4/5 seed局部同向、平均Row恶化不超过预定1%是工程参考门槛。Row relevance另要求平均改善至少1%且至少4/5 seed不变差；逐seed恶化也明确列出。这些不是统计显著性定理，没有以pair数量扩大科研重复数。

计划固定了20%/4seed/1%的数值门槛，但未完全细化相对改善采用均值之比还是逐seed相对值均值，也未细化1%是均值还是逐seed限制。主表明确采用五seed等权均值之比，另列逐seed相对值均值及超过1%恶化的seed数；两种相对改善口径和更严格的逐seedRow限制均不改变本轮未通过的判断，不能将这些具体聚合公式追称为完全预注册。

## 稳健性、输出约束与其他因素

按series等权及剔除validation关联series的描述性敏感性见`control_robustness.csv`；声明时间见`ANALYSIS_SUPPLEMENT.json`，未用于选择模型或改变主门槛。三种局部MAE五seed等权汇总如下。

| kind | difference_error_mean_primary | difference_error_mean_series_equal | difference_error_mean_validation_unrelated |
| --- | --- | --- | --- |
| A_original | 3.32233 | 3.01087 | 3.62558 |
| B_amount_skip | 3.33457 | 3.01912 | 3.63739 |
| C_volume_skip | 3.35214 | 3.03882 | 3.65383 |

输出约束下表是每seed先算频率再等权；负值率按全部6输出统计，分位交叉指任一相邻分位倒置，V2<V1按q50比较。ReLU保证非负并不保证分位顺序或V2≥V1，未在评价时投影或修正。

| kind | scope | negative_output_fraction | V1_quantile_crossing_fraction | V2_quantile_crossing_fraction | V2_below_V1_fraction |
| --- | --- | --- | --- | --- | --- |
| A_original | check | 0.00000 | 0.03810 | 0.02000 | 0.02000 |
| A_original | validation | 0.00000 | 0.07019 | 0.02308 | 0.00048 |
| B_amount_skip | check | 0.00000 | 0.05714 | 0.00000 | 0.00000 |
| B_amount_skip | validation | 0.00000 | 0.04712 | 0.02596 | 0.00000 |
| C_volume_skip | check | 0.00000 | 0.01905 | 0.02000 | 0.00000 |
| C_volume_skip | validation | 0.00000 | 0.04087 | 0.01779 | 0.00096 |

原validation中其余单因素pair（含PE/EA）全部保存在`local_response_metrics.csv`和`control_response_pairs.csv`。它们是validation内的描述性响应检查，并非新增独立holdout。不存在某因素pair时不编造指标。`other_axis_equal_seed.csv`列明每因素/端点预测幅度与差分MAE，供检查意外变化。

本实验只能判断固定标签条件下简单残差路径是否有效，不能证明新增测量、更换采集策略或LLM科学推理更优。最终行动按全部五seed结果和预定门槛决定，失败后不追加网络搜索。

| kind | axis | target | n | difference_error_mean | observed_abs_mean | predicted_abs_mean | direction_accuracy_meaningful | flat_fraction_meaningful | n_seeds | total_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_original | PE/EA | V1 | 125.20000 | 1.95109 | 4.57284 | 4.49163 | 0.86977 | 0.03509 | 5 | 626 |
| A_original | PE/EA | V2 | 125.20000 | 3.52704 | 10.44542 | 9.47309 | 0.85094 | 0.01152 | 5 | 626 |
| A_original | Volume of loading solvent/ul | V1 | 1.50000 | 1.18850 | 1.18333 | 0.02740 | 0.00000 | 1.00000 | 2 | 3 |
| A_original | Volume of loading solvent/ul | V2 | 1.50000 | 2.29603 | 2.30625 | 0.02360 | 0.50000 | 1.00000 | 2 | 3 |
| A_original | loading solvent | V1 | 1.00000 | 0.10589 | 0.34167 | 0.23578 | nan | nan | 1 | 1 |
| A_original | loading solvent | V2 | 1.00000 | 3.91017 | 2.85833 | 1.05184 | 0.00000 | 0.00000 | 1 | 1 |
| B_amount_skip | PE/EA | V1 | 125.20000 | 1.95664 | 4.57284 | 4.37266 | 0.86828 | 0.02378 | 5 | 626 |
| B_amount_skip | PE/EA | V2 | 125.20000 | 3.37054 | 10.44542 | 9.50400 | 0.85296 | 0.01694 | 5 | 626 |
| B_amount_skip | Volume of loading solvent/ul | V1 | 1.50000 | 1.17891 | 1.18333 | 0.01593 | 0.00000 | 1.00000 | 2 | 3 |
| B_amount_skip | Volume of loading solvent/ul | V2 | 1.50000 | 2.29741 | 2.30625 | 0.01533 | 0.50000 | 1.00000 | 2 | 3 |
| B_amount_skip | loading solvent | V1 | 1.00000 | 0.29471 | 0.34167 | 0.04696 | nan | nan | 1 | 1 |
| B_amount_skip | loading solvent | V2 | 1.00000 | 3.23541 | 2.85833 | 0.37707 | 0.00000 | 0.00000 | 1 | 1 |
| C_volume_skip | PE/EA | V1 | 125.20000 | 1.91863 | 4.57284 | 4.42772 | 0.85841 | 0.03192 | 5 | 626 |
| C_volume_skip | PE/EA | V2 | 125.20000 | 3.36665 | 10.44542 | 9.38262 | 0.86525 | 0.00842 | 5 | 626 |
| C_volume_skip | Volume of loading solvent/ul | V1 | 1.50000 | 1.18440 | 1.18333 | 0.00335 | 0.00000 | 1.00000 | 2 | 3 |
| C_volume_skip | Volume of loading solvent/ul | V2 | 1.50000 | 2.30277 | 2.30625 | 0.00774 | 0.50000 | 1.00000 | 2 | 3 |
| C_volume_skip | loading solvent | V1 | 1.00000 | 0.34461 | 0.34167 | 0.00294 | nan | nan | 1 | 1 |
| C_volume_skip | loading solvent | V2 | 1.00000 | 2.79457 | 2.85833 | 0.06377 | 1.00000 | 1.00000 | 1 | 1 |

PE/EA覆盖全部五seed；loading solvent仅seed157，loading solvent volume仅157和12203，Density没有可比较pair。后两因素的描述不能推广为五seed稳定性结论。

直接残差的逐seed系数见`learned_skip_coefficients.csv`，分别以每归一化单位及每mg/µL给出。它只表示该条附加通路，完整模型还会重新学习其余权重，不能把skip斜率单独解释为总预测的物理导数。
