# Hybrid中LLM16替代了哪些LCMD候选？

**新增发现：LLM替代行为最突出的额外内容是现有梯度算法不选的上样量对照，同时减少了仍有明显边际收益的CW几何覆盖；这一取舍不是普遍的“条件对照 vs 不做对照”。** 56/71批两套后16无共同ID。样品体积单因素对照在LCMD后16中全部为零，LLM相对H+前16则在67/71批更高、其余持平。PE/EA对照在两组都很常见。冻结状态候选选择的分工得到直接识别，但实际训练收益仍未知。

## 新计算与精确复原资格

本模块尝试71个真实Hybrid冻结状态，71个状态通过有序前16一致性核验。全部原始bank为3330×512的CW CountSketch缓存，存储float32；直接复用缓存，不重算CW梯度、不加载模型。LCMD纯NumPy实现来自`src/qgeognn_al/active_learning_v2/lcmd.py`，计算转float64，以平方欧氏距离指定最近中心，选择最近中心簇内距离平方和最大的簇，再选择该簇最远点。`argmin/argmax`按原输入顺序处理并列；新中心只有距离严格更小时更新归属。每个状态保留训练/候选的原始有序canonical索引；32步逐步trace落盘。没有用farthest-first代替。

只有重建32步的前16与历史pending16逐个有序相同，才保留第17–32为合格候选集合反事实。排除项见`counterfactual_state_coverage.csv`。checkpoint、bank SHA、catalog与freeze哈希、H/U有序身份逐项核验。旧封存检查先运行；本次所有文件写入新目录，历史实验不变。候选预测仅取当轮catalog，未查询反事实候选真值，也不计算反事实Row Gain。

## LLM增加与牺牲的内容

同一H、U、checkpoint和数值前16下，实际LLM16与反事实LCMD后16可做严格候选集合比较；两组不是训练收益反事实。以下先逐批计算，再按seed汇总（157为36批、6101为35批；可另查共同L1453窗口）。

| seed | llm_cf_candidate_overlap | llm16_cw_distance_pool_percentile | lcmd_back16_cw_distance_pool_percentile | llm16_cw_gain_after_n_relative_h | lcmd_back16_cw_gain_after_n_relative_h | llm16_single_factor_history_fraction | lcmd_back16_single_factor_history_fraction | llm16_new_molecules_hn | lcmd_back16_new_molecules_hn |
|---|---|---|---|---|---|---|---|---|---|
| 157 | 0.27778 | 0.58351 | 0.91593 | 0.015234 | 0.050086 | 0.85417 | 0.71181 | 0.83333 | 0.52778 |
| 6101 | 0.25714 | 0.52904 | 0.9172 | 0.011731 | 0.053331 | 0.85893 | 0.7125 | 1.1143 | 0.25714 |

CW覆盖是固定128个U0参考ID到历史集合平均最近距离的下降。比较起点均为H+实际数值16，表中相对贡献除以同轮选前H参考距离；从未把不同checkpoint的原始距离合并成静态坐标。新分子是H+数值16尚未出现的SMILES数量。单因素对照以同分子、五个记录因素中恰一个不同定义；有科学对照含义，不能自动叫作冗余或低效。

**观察→证据→解释→边界：** LCMD后16并不是前16之后没有几何收益的补位：seed157/6101的给定前16CW边际覆盖分别约为起始参考距离的5.009%/5.333%，实际LLM为1.523%/1.173%。对应数值前16自身为7.079%/6.283%；后16边际虽往往递减，却远非接近零。LLM减去后16的CW贡献在52批为负、18批为正、其余平；两个seed的每个预定阶段均值都为负。因此平均梯度几何覆盖确有牺牲，但“每批LLM都更差”被反例否定，且几何损失不是Row学习损失。

## 被替换的是哪一类条件实验？

| seed | llm16_single_sample_volume_history_fraction | lcmd_back16_single_sample_volume_history_fraction | llm16_single_sample_volume_hn_fraction | lcmd_back16_single_sample_volume_hn_fraction | llm16_full32_single_sample_volume_pairs | lcmd_back16_full32_single_sample_volume_pairs | llm16_single_pe_history_fraction | lcmd_back16_single_pe_history_fraction |
|---|---|---|---|---|---|---|---|---|
| 157 | 0.43924 | 0 | 0.45486 | 0 | 1.1944 | 0 | 0.75347 | 0.71181 |
| 6101 | 0.50357 | 0 | 0.52321 | 0 | 1.0857 | 0 | 0.75714 | 0.7125 |

这里最清楚的差异是**样品体积对照关系**。LCMD后16相对H、相对H+数值前16的样品体积单因素名额都为零，完整数值32内样品体积only pair也全部为零。实际LLM在H+数值前16下约45.49%/52.32%的名额有此锚点；而完整Hybrid32内部仅平均1.194/1.086个这种实验对。两种分母不同：多数LLM条件对照以已知历史为锚，不是重复购买批内相同实验。

PE/EA历史对照比例则LLM约75.35%/75.71%、后16约71.18%/71.25%；不能笼统声称数值方法不选条件对照。平均样品体积本身也很接近或跨seed异向：157约97.83对97.55μL，6101约97.77对103.91μL。差异在“固定同分子其他因素进行对照”的选择结构，而非简单偏好较大上样量。加载体积和加载溶剂only对照稀少，完整表保留而不夸大。

上述结论还由`verify_counterfactual.py`独立以Fraction/Decimal逐因素枚举142个比较子批复核，没有调用被测condition函数；与主表最大误差为零。LCMD实现另用20套小整数例子，独立标量重算每一步簇分配、平方距离和、最远点及并列顺序，逐步trace完全一致。

## 阶段一致性与反例

以下均为LLM减去LCMD后16的配对差，CW列为相对于H起始距离的边际覆盖下降，joint/molecular列为固定结构/条件几何的绝对覆盖增量差。完整方向计数、逐批极值与条件水平分布分别见`counterfactual_paired_directions.csv`和`counterfactual_condition_levels.csv`。

| seed | phase | delta_llm_minus_lcmd_cw_gain_after_n_relative_h | delta_llm_minus_lcmd_joint_gain_after_n | delta_llm_minus_lcmd_molecular_gain_after_n | delta_llm_minus_lcmd_single_sample_volume_history_fraction | delta_llm_minus_lcmd_single_pe_history_fraction | delta_llm_minus_lcmd_width_mean |
|---|---|---|---|---|---|---|---|
| 157 | 1005-1453 | -0.040332 | 8.84e-05 | 0 | 0.61607 | -0.013393 | 3.0267 |
| 157 | 1453-1485 | -0.14796 | -0.00039453 | 0 | 1 | 0.125 | 12.389 |
| 157 | 333-525 | -0.020038 | 0.00074806 | 0.0026164 | 0.17708 | -0.03125 | 1.5259 |
| 157 | 525-813 | -0.019325 | -0.00024042 | 0.00075978 | 0.29861 | 0.15972 | 3.3313 |
| 157 | 813-1005 | -0.041316 | -0.00036277 | 0 | 0.40625 | 0.052083 | 2.408 |
| 6101 | 1005-1453 | -0.050983 | -0.00012407 | 0 | 0.72321 | 0.044643 | 2.0879 |
| 6101 | 333-525 | -0.014414 | 0.0030922 | 0.008514 | 0.03125 | 0.03125 | 0.71879 |
| 6101 | 525-813 | -0.058553 | 0.00087166 | 0.0021908 | 0.50694 | 0.020833 | 2.5725 |
| 6101 | 813-1005 | -0.021463 | -0.00079787 | 0 | 0.45833 | 0.09375 | 1.9121 |

静态分子空间为Morgan radius2/2048 Tanimoto距离；条件空间采用旧审计固定、无标签的四个连续范围与一个溶剂类别，联合距离各占一半。它们不等同于CW梯度空间。不能因某一覆盖变大而断定学习更好。

LLM相对后16的新增分子优势主要出现在L333–813，后续状态已没有未覆盖分子空间，不能从后期零分子增量推断探索意愿下降。联合静态覆盖差的逐批方向混合，157为14正/19负/3平，6101为19正/15负/1平；“LLM普遍增加一切覆盖”不成立。样品体积对照差却随预算明显扩大，L1005–1453两seed的LLM额外对照名额分别约61.61/72.32个百分点；这是与模型条件响应审计值得连接的机制信号，而不是通用切换阈值。

## 响应尺度、宽度与重复结构

| seed | llm16_pred_V1_q50_mean | lcmd_back16_pred_V1_q50_mean | llm16_pred_V2_q50_mean | lcmd_back16_pred_V2_q50_mean | llm16_width_mean | lcmd_back16_width_mean | llm16_same_molecule_pairs_fraction | lcmd_back16_same_molecule_pairs_fraction |
|---|---|---|---|---|---|---|---|---|
| 157 | 12.273 | 8.2456 | 26.314 | 17.852 | 9.3509 | 6.3413 | 0.042824 | 0.0055556 |
| 6101 | 14.124 | 8.5945 | 30.784 | 18.309 | 5.3916 | 3.444 | 0.025238 | 0.0038095 |

q50为预测保留体积，不是候选实验价值；q90−q10为现有单模型区间宽度，未校准为epistemic uncertainty。两方案均取同一状态预测，不使用未来模型。重复SMILES的候选对比例衡量条件探索集中程度，不能单凭此指标认定浪费。

LLM内部61个样品体积only实验对的预测V1/V2 q50绝对变化均值为0.3066/0.2154mL；同组122个PE/EAonly对为5.394/11.598mL。分seed样品体积对分别34/27个，预测变化0.4047/0.3400及0.1831/0.05846mL。LCMD后16内部只有14个PE/EAonly对，没有样品体积only对。此处只描述候选当前预测，**没有读取这些对的真值**；不同因素的真实变化尺度不同，不能仅凭预测差大小比较学习价值。它提供“LLM补充的部分条件实验在当前表示/预测中变化很小”的具体依据，需与已揭示反馈的独立模型响应审计共同解释。

LLM可见理由的透明规则统计保存在`counterfactual_reason_factors.csv`，候选级原文见`counterfactual_candidates.csv`。规则只识别PE/EA、上样量、加载溶剂/体积及结构转移措辞；LCMD反事实没有原LLM理由，保持缺失，不生成解释。理由中出现某条件不证明该条件是唯一选择原因。

## 结论边界与下一步

这次新增证据回答“在相同状态下替换了谁”，而非“替换后训练收益是多少”。实际两组ID重叠、CW与固定几何边际、条件对照、分子覆盖及预测尺度均可在同状态比较；真实响应及行级损失贡献仍不可识别。任何下一轮学习收益验证都需要另行授权新训练或未揭示训练池标签；本轮未执行。两seed只能支持机制描述，阶段结果不产生通用预算切换阈值。

## 复现与核验

先运行旧审计`run.py --check`，再运行本模块`counterfactual.py`。默认自动检查旧封存；`--limit`仅用于调试，不能用部分状态冒充全71批。`counterfactual_input_manifest.json`保存读取路径及SHA；`counterfactual_verification.json`记录有序前缀、覆盖率、算法SHA和零训练/零LLM/零真值查询。关键单元测试见`tests/test_counterfactual.py`。
