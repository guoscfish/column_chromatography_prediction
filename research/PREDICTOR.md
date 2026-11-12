# 独立QGeoGNN-V2点预测资格

状态：PRIMARY_FROZEN。原决策 `4G_POINT_PREDICTOR_QUALIFIED_FOR_TRANSFER_STUDIES`，只授予普通点预测研究资格，不授予最终UQ模型资格，也不授权新迁移实验。科学实现是458,952有效参数的独立V2，旧Clean是失败的另一结构；见 [当前代码合同](../docs/model/CODE_CONTRACT.md)。

## 冻结身份与预算

- 原域：4,163行、217个compound，V1≤60、V2≤120。六次run为seeds42/525/1101 × row/compound，完整source训练，不是AL的L333预算。
- row每seed：train3330 / validation416 / test417。compound按整分子互斥：seed42为3401/389/373，seed525为3321/428/414，seed1101为3275/430/458；不能与row插值混作同一个泛化任务。
- 原固定协议：Adam lr0.001、weight_decay0、batch2048、max1000、patience100、deterministic epoch shuffle、V1/V2等权、预处理仅训练集。validation选checkpoint，reload后才测test；没有因分数差重跑seed。
- 固定row seed42源checkpoint SHA256为 `fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0`，在资格结果之前预选，不按最佳test挑选。它是原本地ignored runtime，不能因有Git报告就假定fresh clone拥有。

## 已公布结果

| split任务 | validation Combined NRMSE均值±seed样本SD | test均值±seed样本SD |
| --- | ---: | ---: |
| row | 0.314827987 ± 0.063232719 | 0.359253926 ± 0.013896380 |
| compound | 0.620328856 ± 0.201281233 | 0.699251411 ± 0.112096901 |

row有点预测信号，unseen-compound泛化差距明显。测试真值已经用于上述冻结评价，不能再叫盲确认集。quantile审计显示交叉和欠覆盖，q50用MSE而非pinball中位数；保留点预测头不等于区间已校准。

本地小表 [predictor_qualification_metrics.csv](predictor_qualification_metrics.csv) 是原 `results/all_metrics.csv` 的原字节副本，SHA256 `ae9b20b46dde2f46551467743ff0cc47a1d1735c7d10ae92f9ac476a5c7b5162`。这是聚合指标，不是原始test行标签。

## 原始协议与数据入口

[最终报告](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/predictor/final_4g_qualification/FINAL_4G_QUALIFICATION_REPORT.md) · [冻结protocol](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/predictor/final_4g_qualification/protocol.json) · [split清单](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/predictor/final_4g_qualification/splits/split_manifest.json) · [工程等价](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/predictor/final_v2_engineering/README.md)。

输入语义原始权威源为 [I0审计](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/i0_predictor_semantic_audit/README.md)；论文/代码详细差异在 [原合同审计](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/docs/model/PAPER_CODE_CONTRACT_AUDIT.md)。原数据/图/scaler/checkpoint由 [保护登记](../configs/protected_artifacts.json) 和协议hash定位。这里只摘要既有结果，本轮未训练或读取新test目标。
