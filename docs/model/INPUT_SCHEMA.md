# QGeoGNN-V2 输入语义

状态：`CURRENT`。更新：2026-11-11。本页说明候选 [predictor 包](../../src/qgeognn_al/predictor/) 现有 V2 输入，不修改特征或模型。本页以 [清理前V2语义](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/docs/model/INPUT_SCHEMA.md) 为来源；旧Clean是失败的独立结构试验，不能套到当前V2。其原始沿革指针在该固定历史版本中检索。

## 图和条件

原子/键类别描述拓扑；键长进入 atom-bond 图连续特征，键角与分子描述符进入 bond-angle 图。现有图构造见 [data.py](../../src/qgeognn_al/predictor/data.py)，原编码器见 [application/QGeoGNN.py](../../application/QGeoGNN.py)。

V2 保留旧 backbone 实际消费的条件通路，在 graph/sample 层补齐未消费条件。构造 edge_attr 列不等于旧 encoder 确实消费了该语义，见 [当前代码合同](CODE_CONTRACT.md) 与 [归档I0语义审计](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/studies/i0_predictor_semantic_audit/README.md)。

| 条件 | 类型与物理语义 | 当前实现 |
| --- | --- | --- |
| PE/EA | 比例构造六维混合物描述符：ExactMolWt、TPSA、rotatable bonds、H donors、H acceptors、logP | 训练分区 min/max；backbone 消费前四维，V2 分支补齐 H acceptors/logP |
| `loading solvent` | PE / EA / DCM 类别，代码不表示序数大小 | 词表与4D embedding |
| `Density g/ml` × `V/ul` | 上样量 mg；1 g/ml × 1 ul = 1 mg | float32，source-train min/max |
| `Volume of loading solvent/ul` | 上样溶剂体积 ul | float32，source-train min/max |

[conditions.py](../../src/qgeognn_al/predictor/conditions.py) 的 `condition_schema()` 是机器合同：四维补齐连续量加四维 solvent embedding，组成8D分支输入，经 hidden16 输出128D residual，加在固定 sum-pooled backbone 表示后、六输出头前。`data.py` 的9D `condition_matrix` 是诊断表示，不等于这个8D模型输入。现用 V2 没有加入裸 `column=4/8/25/40` 特征。

## 预处理与标签

全 source 预测器的 scaler/条件统计只用 source-train，不能用 validation、test 或目标柱数据。AL 的 [preprocessing.py](../../src/qgeognn_al/predictor/preprocessing.py) 保持冻结协议：covariate normalization 来自 outer-train X，目标 sigma 来自授权 L0 标签；未揭示候选真值不能用于 target scaling。两种拟合人群和角色分别记录。

预处理/checkpoint 保存 fit ID hash、scaler hash、model variant、input schema/hash、初始化/训练来源和输出顺序。表示图移除未来标签，训练图的 y 只能来自授权标签；传入 CSV 路径不构成标签访问授权。

输出固定为 `V1_q10, V1_q50, V1_q90, V2_q10, V2_q50, V2_q90`。q50 用 MSE 学习，交叉和 coverage 在排序前统计；点预测资格不授予校准 UQ 资格，见 [科学状态](../NEXT_STAGE_DECISION.md)。
