# 15次固定拟合结果的独立QA

结论：**PASS**。直接读取15个拟合各自的check与原validation预测CSV，共30个文件；只使用本轮授权投影中的`check_y`与`validation_y`。每文件校验ID唯一性、ID集合完整性，再按identity的ID索引对齐，未用预测行号猜测标签。30份文件的原始顺序也全部一致。没有训练、模型推理、API请求、checkpoint读取或test/未揭示池标签读取。

## 独立重算口径与分母

本QA没有导入原评估、汇总、pair分类或point_metrics函数，也没有导入torch。独立解析SMILES、PE/EA分数、密度、上样溶剂与其体积，以同分子且全部非样品体积条件相同建立完整系列。每系列中样品体积不同的所有ID pair纳入；重复条件记录仍保留为独立ID，体积相同的两条记录不作为体积变化pair。H1005训练与检查系列重叠仍为0。

主指标：每seed内先对全部体积pair计算`abs((pred_high−pred_low)−(truth_high−truth_low))`，每端点求均值，再V1/V2等权，最后五seed等权。0.1/0.5/1mL阈值仅改变方向/平坦率的有意义真实变化集合，不过滤主MAE。平坦预测为绝对差分严格小于0.1mL；有意义真实变化为严格大于阈值。不得把pair数量作为科研重复数。

Row指标使用原416行validation的q50，每端点RMSE除以固定L333目标尺度，再求两端点归一化RMSE的平方均值开方；五seed等权，未以检查集Row成绩替换。

| seed | check_rows | validation_rows | volume_series | volume_pairs_per_endpoint | training_related_series | validation_related_series | validation_exact_condition_check_rows | validation_unrelated_series | validation_unrelated_pairs_per_endpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157 | 23 | 416 | 10 | 16 | 0 | 2 | 0 | 8 | 14 |
| 887 | 24 | 416 | 10 | 18 | 0 | 3 | 0 | 7 | 15 |
| 2357 | 21 | 416 | 10 | 12 | 0 | 2 | 1 | 8 | 10 |
| 6101 | 20 | 416 | 10 | 10 | 0 | 0 | 0 | 10 | 10 |
| 12203 | 21 | 416 | 10 | 12 | 0 | 2 | 1 | 8 | 10 |

## 逐seed与五seed等权

| seed | kind | difference_error_mean | combined_normalized_rmse |
| --- | --- | --- | --- |
| 157 | A_original | 2.883691597 | 0.400525371 |
| 157 | B_amount_skip | 2.978223838 | 0.421108122 |
| 157 | C_volume_skip | 3.009884869 | 0.412037569 |
| 887 | A_original | 4.962211508 | 0.361834984 |
| 887 | B_amount_skip | 4.956124144 | 0.345721944 |
| 887 | C_volume_skip | 4.954508550 | 0.348200238 |
| 2357 | A_original | 2.648888300 | 0.356258077 |
| 2357 | B_amount_skip | 2.662322221 | 0.362316714 |
| 2357 | C_volume_skip | 2.642021508 | 0.360062769 |
| 6101 | A_original | 3.550774663 | 0.442529419 |
| 6101 | B_amount_skip | 3.547528708 | 0.425532449 |
| 6101 | C_volume_skip | 3.538983268 | 0.439311209 |
| 12203 | A_original | 2.566070079 | 0.419495328 |
| 12203 | B_amount_skip | 2.528632691 | 0.417994676 |
| 12203 | C_volume_skip | 2.615285329 | 0.416842715 |

| kind | difference_error_mean | combined_normalized_rmse |
| --- | --- | --- |
| A_original | 3.322327229 | 0.396128636 |
| B_amount_skip | 3.334566320 | 0.394534781 |
| C_volume_skip | 3.352136705 | 0.395290900 |

逐endpoint及逐pair对应的核对覆盖全部408条体积pair×端点×模型事件。逐seed主MAE、Row RMSE/MAE/NRMSE、各阈值方向/平坦率、幅度、pair身份与validation关联标记均与发布表一致；每项最大绝对误差与输入SHA见JSON。这是预测评分事件数量，不能当作独立实验重复数。

## 配对与参考门槛

| comparison | local_improvement_ratio_equal_seed_means | local_positive_seeds | row_relative_change_equal_seed_means | row_nonworse_seeds | row_worse_over_one_percent_seeds | mechanism_reference_pass | row_relevance_reference_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-A | -0.003683891 | 3 | -0.004023579 | 3 | 2 | False | False |
| C-A | -0.008972468 | 3 | -0.002114808 | 3 | 2 | False | False |
| C-B | -0.005269166 | 3 | 0.001916482 | 3 | 1 | False | False |

15条逐seed模型配对差与全部参考门槛逐项一致。主要相对改善使用“五seed等权均值之比”；另存的“逐seed相对改善再平均”不是门槛计算依据。20%局部改善、至少4/5seed局部改善、平均Row恶化不超过1%三个条件联合判定机制参考门槛；Row相关性另需平均改善至少1%且至少4/5seed不变差。原计划没有完全明确相对改善聚合公式和Row1%应对均值还是每seed应用，不能宣称这些具体公式全部预注册；当前报告分列两种相对口径及逐seed恶化，结论不随这些解释改变。三组对比均未通过这两个参考门槛；没有用稳健性结果替换主口径。

## 两项描述性敏感性

series等权先在每个系列内平均pair误差，再在seed内系列等权、端点等权，最后五seed等权。validation关联剔除则独立从metadata定位任何有同系列原validation记录的检查系列，将其整体剔除，再保留原pair权重及同一冻结预测。剔除系列数为2/3/2/0/2，剩余8/7/8/10/8；2357及12203各有一个检查ID与validation的完整条件相同，已整体排除其系列。这些关联不含检查ID直接出现在训练中。

| kind | difference_error_mean_primary | difference_error_mean_series_equal | difference_error_mean_validation_unrelated |
| --- | --- | --- | --- |
| A_original | 3.322327229 | 3.010867642 | 3.625579435 |
| B_amount_skip | 3.334566320 | 3.019122585 | 3.637392377 |
| C_volume_skip | 3.352136705 | 3.038820399 | 3.653832483 |

两项敏感性分别匹配`control_robustness.csv`的15条seed记录；声明发生在完成1次拟合时，明确标注描述性，不改变主门槛或checkpoint，也不补训练。原validation关联意味着检查集不是完全独立确认集，此限制不能因训练系列重叠为0而省略。

## 精度与复现

授权投影标签为float32，发布评估的真实pair差先以float32相减；QA逐项重现该精度，同时独立用float64标签相减检查舍入影响。JSON列出最大差异及主MAE影响，不改变任何结果或门槛。Q50 Row误差用CSV浮点预测直接计算。

运行`results_independent_qa.py`即可只用已冻结数值重建本报告与JSON。全部读取文件的SHA均记录，运行结束再次核对无变化；训练冻结代码SHA仍与DATA_FEASIBILITY一致。没有改动任何原始预测、标签、评估代码、训练文件或既有报告。
