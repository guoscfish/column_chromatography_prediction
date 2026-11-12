# 必须保留的GP与IVR负结果

这些是已有研究的有限结论，不因“负结果”退掉其科学身份。完整数据、原始门槛、失败链和修订在统一archive，当前不自动启动新训练或扩大预算。

## Kernel-IVR连续探索：没有明确AULC改善

`NO_CLEAR_IVR_IMPROVEMENT`。五个既有outer seeds157/887/2357/6101/12203；row outer-train3330、validation416、test417；L333–1005，每批32，共22点/21采集。当前scratch predictor的512D raw q50 gradient、固定原L0+U0 reference、global RMS scaling与unit prior/noise是冻结的单一surrogate假设，未自动调参。

| 方法 | 原报告归一化AULC | NRMSE@1005 |
| --- | ---: | ---: |
| Random | 0.624260 | 0.543301 |
| LCMD | 0.479539 | 0.408614 |
| Kernel-IVR | 0.500070 | 0.395913 |

**本表是旧全轨迹冻结后的test评价**，不与Free/LCMD的validation表混排。IVR在AULC上只赢LCMD 1/5 seed；终点较低不等于全程标签效率更好。历史test已启发假设，虽先冻结110个新方法点才揭示其test，整个cohort也不再是独立盲确认。原实验有105次新scratch拟合；本轮没有复跑。

[报告](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/FINAL_REPORT.md) · [协议与预算](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/PROTOCOL.md) · [test消费审计](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_row_kernel_ivr_b32/results/test_label_access_audit.csv)。

## IVR机制audit：停止在Gate C之前

`STOP_BEFORE_GATE_C`。对上述五seed已有scalar-IVR轨迹的round1/5/10/15/20共25个冻结态，使用原L/U与3330固定reference、stored L0 scales和label-scrubbed图。它是label-free选样机制/数值audit：没有新拟合、标签揭示或test-error门槛。

scalar cross-map平均batch overlap约0.5767，512-vs-2048约0.6871，block约0.6438，未过原稳定门槛；forward/backward只在7/25态正增益，中位相对gain=0。multi-output在自己的surrogate下25/25优于same-state LCMD，不等于真实预测效果更好。没有候选获得新训练资格。raw kernel低秩本身不等于正则化求解数值失稳。

[完整机制报告](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md) · [原门槛/范围](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/qgeognn_v2_ivr_mechanism_audit/PROTOCOL.md)。

## 表示+GP开发基准：不进入正式LLM-GP主动学习

`NO_GO_FOR_FORMAL_LLM_GP_ACTIVE_LEARNING`。五seed×L333/525/813/1005共20个既有LCMD状态、720次端点GP拟合；只使用已授权L_t与原416行validation开发评价，没有新训QGeoGNN或新AL闭环。validation已参与QGeoGNN checkpoint selection，预算嵌套，20态不是20个独立重复。

本地语言对照是冻结 **Qwen1.5-0.5B-Chat causal hidden state**，不是T5、通用embedding或完整GOLLuM。其主误差是两端点 `RMSE / L0 sigma` 的算术均值，**不同于Combined NRMSE**。原开发汇总：Frozen Qwen conditions=0.8776、Structured=0.7663、GNN hidden=0.5053；Qwen Projection改善到0.6443，但90%覆盖从95.4%降至22.8%、NLPD恶化到86.1771。GNN fixed覆盖也仅68.6%；360/720有限fit未在冻结预算内收敛，均未因validation差而补优化/重启。

这是当前明确方案下的概率可靠性/文本稳定性缺口，不等于否定所有语言模型。mean prediction改善、surrogate信息量或offline选点差异不能当AL AULC收益。不得事后用validation缩放方差或挑单个初始化改写结论。

[裁决](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/representation_gp_benchmark_20261104/NEXT_STAGE_DECISION.md) · [配对比较与指标定义](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/representation_gp_benchmark_20261104/COMPARISON_REPORT.md) · [概率校准](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/representation_gp_benchmark_20261104/GP_CALIBRATION_REPORT.md) · [协议](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/active_learning/representation_gp_benchmark_20261104/PROTOCOL.md)。

更多失败/部分证据（Clean、loading-cap、Agent合法性、文本模板稳定性）仍在完整archive，而不是被删除出科研历史。数据已曝光的事实继续见 [数据消费登记](../configs/data_consumption_register.json)。
