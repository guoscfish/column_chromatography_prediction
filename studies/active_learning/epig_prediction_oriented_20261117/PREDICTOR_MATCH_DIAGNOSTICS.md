# 预测器与 EPIG / IVR 条件信息的匹配诊断

这是科学预注册封存后新增的独立、无标签附录；只描述输入结构、原L333预测器响应和已完成同状态批次。它不改变模型族、资格门槛、目标分布或正式两臂，不能据此调参或挑选策略。

## 输入与严格边界

五seed 157/887/2357/6101/12203，各使用其原L333 QGeoGNN-V2 checkpoint及该seed outer-train 3330输入。模型字节SHA同时匹配本研究science freeze和原Free协议；scrubbed graph cache同时核对原context contract、cache字节SHA、原Free保存SHA，并逐图验证y全零与canonical index。只载入feature-only字段，没有调用RestrictedLabelStore、读取候选/validation/test真值或历史feedback。原float32模型以CPU/1线程、inference batch256计算q50，不新增任何神经网络或辅助模型拟合。

严格pair要求canonical molecule相同、Density以及除指定因子之外的全部记录条件相同，仅V/ul、加载溶剂体积或PE/EA的EA分数不同。相同比例的PE/EA字符串先按精确分数归为同一级。全部合法Row组合保留；重复row/重叠series使pair不独立，不能把pair数充当统计样本数。

响应为|Δq50|，同时保留mL和各seed固定L333端点标准差单位。combined幅度是sqrt(mean((Δq50/s)²))；它是冻结预测幅度，**不是真实响应、学习收益、误差、模型不确定性或oracle信息增益**。

## 原L333预测响应

| seed | 因子 | pairs | V1平均预测幅度 / L333std | V2平均预测幅度 / L333std | 两端均<0.1mL比例 |
|---|---|---:|---:|---:|---:|
| 157 | eluent_EA_fraction | 8044 | 0.597733 | 0.634810 | 0.1% |
| 157 | loading_solvent_volume | 17 | 0.000060 | 0.000468 | 100.0% |
| 157 | sample_volume | 2750 | 0.000856 | 0.000757 | 100.0% |
| 887 | eluent_EA_fraction | 8136 | 0.527147 | 0.559793 | 0.3% |
| 887 | loading_solvent_volume | 26 | 0.000108 | 0.000112 | 100.0% |
| 887 | sample_volume | 2752 | 0.000324 | 0.000056 | 100.0% |
| 2357 | eluent_EA_fraction | 8076 | 0.506988 | 0.605084 | 0.2% |
| 2357 | loading_solvent_volume | 20 | 0.000635 | 0.000129 | 100.0% |
| 2357 | sample_volume | 2746 | 0.006745 | 0.000710 | 87.0% |
| 6101 | eluent_EA_fraction | 8018 | 0.760621 | 0.700437 | 0.1% |
| 6101 | loading_solvent_volume | 25 | 0.000131 | 0.000042 | 100.0% |
| 6101 | sample_volume | 2751 | 0.000780 | 0.000586 | 100.0% |
| 12203 | eluent_EA_fraction | 8048 | 0.471715 | 0.565185 | 0.2% |
| 12203 | loading_solvent_volume | 22 | 0.000047 | 0.000068 | 100.0% |
| 12203 | sample_volume | 2756 | 0.000327 | 0.000137 | 100.0% |

同一seed/因子内按pair、series和molecule分别等权（下表幅度均为combined / L333std），再跨seed解释；不把54k相关pair合并为独立证据：

| seed | 因子 | series | molecules | pair等权 | series等权 | molecule等权 |
|---|---|---:|---:|---:|---:|---:|
| 157 | eluent_EA_fraction | 615 | 217 | 0.633878 | 0.685597 | 0.688603 |
| 157 | loading_solvent_volume | 17 | 2 | 0.000334 | 0.000334 | 0.000356 |
| 157 | sample_volume | 1094 | 195 | 0.000829 | 0.000865 | 0.000869 |
| 887 | eluent_EA_fraction | 612 | 217 | 0.553307 | 0.629331 | 0.615771 |
| 887 | loading_solvent_volume | 26 | 3 | 0.000113 | 0.000113 | 0.000133 |
| 887 | sample_volume | 1105 | 195 | 0.000236 | 0.000212 | 0.000221 |
| 2357 | eluent_EA_fraction | 611 | 217 | 0.578837 | 0.652857 | 0.673576 |
| 2357 | loading_solvent_volume | 20 | 3 | 0.000460 | 0.000460 | 0.000596 |
| 2357 | sample_volume | 1125 | 195 | 0.004798 | 0.005070 | 0.005105 |
| 6101 | eluent_EA_fraction | 612 | 217 | 0.761165 | 0.786324 | 0.789315 |
| 6101 | loading_solvent_volume | 25 | 2 | 0.000098 | 0.000098 | 0.000110 |
| 6101 | sample_volume | 1105 | 195 | 0.000699 | 0.000722 | 0.000713 |
| 12203 | eluent_EA_fraction | 612 | 217 | 0.534781 | 0.615461 | 0.608763 |
| 12203 | loading_solvent_volume | 22 | 3 | 0.000061 | 0.000061 | 0.000067 |
| 12203 | sample_volume | 1107 | 195 | 0.000261 | 0.000265 | 0.000262 |

## 同状态 P1 / M1024 选点触及哪些条件对照

已完成并校验 8/8 个GP状态（按冻结选择规则：本seed所选GP，或无所选GP时两种预注册核）；缺少状态保持缺少。概率资格失败的模型只作明确标记的离线方法诊断，不能由可算score宣称可靠EPIG或正式学习收益。

| seed | GP | 方法 | 因子 | 批内pair | 至L333 pair | 批内pair平均combined预测幅度 / L333std |
|---|---|---|---|---:|---:|---:|
| 157 | gp_matern32 | epig_conditional_greedy | eluent_EA_fraction | 0 | 14 | — |
| 157 | gp_matern32 | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 157 | gp_matern32 | epig_conditional_greedy | sample_volume | 0 | 2 | — |
| 157 | gp_matern32 | ivr_conditional_greedy | eluent_EA_fraction | 0 | 9 | — |
| 157 | gp_matern32 | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 157 | gp_matern32 | ivr_conditional_greedy | sample_volume | 0 | 0 | — |
| 887 | gp_matern32 | epig_conditional_greedy | eluent_EA_fraction | 0 | 5 | — |
| 887 | gp_matern32 | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 887 | gp_matern32 | epig_conditional_greedy | sample_volume | 0 | 3 | — |
| 887 | gp_matern32 | ivr_conditional_greedy | eluent_EA_fraction | 2 | 7 | 0.271407 |
| 887 | gp_matern32 | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 887 | gp_matern32 | ivr_conditional_greedy | sample_volume | 0 | 3 | — |
| 887 | gp_rbf | epig_conditional_greedy | eluent_EA_fraction | 1 | 6 | 0.186884 |
| 887 | gp_rbf | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 887 | gp_rbf | epig_conditional_greedy | sample_volume | 0 | 2 | — |
| 887 | gp_rbf | ivr_conditional_greedy | eluent_EA_fraction | 3 | 7 | 0.189782 |
| 887 | gp_rbf | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 887 | gp_rbf | ivr_conditional_greedy | sample_volume | 0 | 1 | — |
| 2357 | gp_matern32 | epig_conditional_greedy | eluent_EA_fraction | 0 | 11 | — |
| 2357 | gp_matern32 | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 2357 | gp_matern32 | epig_conditional_greedy | sample_volume | 0 | 3 | — |
| 2357 | gp_matern32 | ivr_conditional_greedy | eluent_EA_fraction | 0 | 11 | — |
| 2357 | gp_matern32 | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 2357 | gp_matern32 | ivr_conditional_greedy | sample_volume | 0 | 2 | — |
| 2357 | gp_rbf | epig_conditional_greedy | eluent_EA_fraction | 0 | 10 | — |
| 2357 | gp_rbf | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 2357 | gp_rbf | epig_conditional_greedy | sample_volume | 0 | 1 | — |
| 2357 | gp_rbf | ivr_conditional_greedy | eluent_EA_fraction | 1 | 10 | 0.060530 |
| 2357 | gp_rbf | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 2357 | gp_rbf | ivr_conditional_greedy | sample_volume | 0 | 0 | — |
| 6101 | gp_matern32 | epig_conditional_greedy | eluent_EA_fraction | 1 | 4 | 1.162250 |
| 6101 | gp_matern32 | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 6101 | gp_matern32 | epig_conditional_greedy | sample_volume | 0 | 0 | — |
| 6101 | gp_matern32 | ivr_conditional_greedy | eluent_EA_fraction | 1 | 5 | 1.162250 |
| 6101 | gp_matern32 | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 6101 | gp_matern32 | ivr_conditional_greedy | sample_volume | 0 | 0 | — |
| 6101 | gp_rbf | epig_conditional_greedy | eluent_EA_fraction | 3 | 7 | 0.746693 |
| 6101 | gp_rbf | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 6101 | gp_rbf | epig_conditional_greedy | sample_volume | 0 | 2 | — |
| 6101 | gp_rbf | ivr_conditional_greedy | eluent_EA_fraction | 2 | 7 | 1.497145 |
| 6101 | gp_rbf | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 6101 | gp_rbf | ivr_conditional_greedy | sample_volume | 1 | 1 | 0.001016 |
| 12203 | gp_matern32 | epig_conditional_greedy | eluent_EA_fraction | 1 | 6 | 1.292248 |
| 12203 | gp_matern32 | epig_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 12203 | gp_matern32 | epig_conditional_greedy | sample_volume | 0 | 2 | — |
| 12203 | gp_matern32 | ivr_conditional_greedy | eluent_EA_fraction | 1 | 9 | 0.205071 |
| 12203 | gp_matern32 | ivr_conditional_greedy | loading_solvent_volume | 0 | 0 | — |
| 12203 | gp_matern32 | ivr_conditional_greedy | sample_volume | 0 | 3 | — |

每一选中行“触及pair”分别记录三种口径：批内两端均选中、选中行对原L333已有行、任一端选中且另端可为未选U。第三种只表示存在输入匹配邻居，不能称已设计或购买完整对照。CSV同时保留各选中scope的series/molecule等权响应，以及具有任意匹配邻居的候选基数，避免把原数据网格密度误称为策略独特性。

## 能与不能解释什么

若EPIG选择的体积对照在原模型下预测幅度很小，只能说明选点涉及当前冻结预测器响应弱的输入方向；不能确定真实物理变化存在、不能确定模型随后难以利用，也不能认定EPIG失败。GP辅助模型使用描述符/条件，并不等于原GNN表示，二者响应不匹配是待检验线索。反过来，预测幅度较大也不证明批次有真实Row AULC收益。最终仍需原QGeoGNN scratch重训的完整五seed配对轨迹。

既有上样响应审计曾在已揭示历史数据及充分训练模型中报告响应压缩，同时披露同X响应差、未记录条件、seed异质性及标签边界事故。它支持把预测器可学习性作为独立机制问题，不能替代本附录不存在的真实pair响应。不得把本次无标签小幅度与历史有标签幅度拼接为本次oracle效用。若做预测器结构干预，必须让EPIG与IVR/LCMD同时更换同一模型，作为另一个研究因素。

## 文件与复现

`predictor_match/input_matched_pairs.csv`：五seed全部输入pair及原q50差分；`input_pair_summary.csv`：候选网格与L/U分层、pair/series/molecule等权；`input_series_summary.csv`与`input_molecule_summary.csv`：组内均值明细；`selected_pair_summary.csv` / `selected_pair_details.csv`：已完成GP批次匹配；`predictions_seed_*.npz/json`：全outer冻结q50与严格身份receipt；`appendix_receipt.json`：独立脚本SHA、science freeze SHA、输入输出SHA和零训练/零目标访问声明。

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_20261117/predictor_response_diagnostics.py
# acquisition全部完成后只刷新匹配表；不会再运行原模型预测：
/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_20261117/predictor_response_diagnostics.py --summarize-only
```
