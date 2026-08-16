# E0-3 4g→8g 单seed迁移基线

状态：10个配置的row-level seed=42矩阵已完成；候选尚未最终冻结。

## 数据与协议

- 当前仓库`dataset/dataset_8g.csv`：574行；原代码60/120 mL阈值后552行。
- 88个唯一canonical SMILES全部构图成功；87个复用4g冻结图，1个8g新生成。
- 重复实验保留；2行`t1>t2`保留并标记。
- row-level 80/10/10：train/valid/test=441/55/56。
- compound-group比较split：441/57/54，同一化合物不跨集合，尚未训练。
- 训练损失保持原代码`L_V1+0.5L_V2`。
- checkpoint只根据validation选择，使用尺度归一化分数：`MSE_V1/Var_train(V1)+MSE_V2/Var_train(V2)`。
- 预训练模式复用4g source scaler；scratch仅在8g train拟合target scaler。

## 单seed比较

按normalized validation score排序的前三名：

| 配置 | best epoch | Valid R² V1/V2 | Test R² V1/V2 | Test coverage V1/V2 | Test crossing |
|---|---:|---:|---:|---:|---:|
| last2+head, 1e-4 | 50 | 0.8715 / 0.8503 | 0.0093 / 0.3765 | 0.8571 / 0.5893 | 0.0000 |
| last2+head, 1e-5 | 385 | 0.8656 / 0.8510 | -0.0068 / 0.3858 | 0.8393 / 0.4643 | 0.0893 |
| full, 1e-4 | 42 | 0.8711 / 0.8356 | -0.1044 / 0.2820 | 0.7679 / 0.4464 | 0.1071 |

单seed候选为`last2_head_lr1e-4`，但不能作为最终迁移配置：validation只有55行、test只有56行，且源行224一个样本贡献该候选test中V1和V2各约80%的SSE。该记录必须回查原始实验；随后需要paired多seed和compound-group复现。

## 关键限制

- 当前图缓存忠实于原代码实际行为：优化10个构象后使用默认conformer 0；论文方法文字要求选择最低能构象。两种图口径需要成对重跑。
- row-level split有69个化合物跨集合，7组重复条件中3组跨集合，只用于论文可比基线。
- 单seed结果未达到论文8g迁移表现，当前证据显示数据版本、构象选择、极端记录和划分方差均可能造成差异。

## 论文叙述与仓库实现的区别

- 论文 Figure 4A 把新任务的输入层和输出层标为新层，正文写明迁移隐藏层参数、适配新柱规格输入、更新输出层，并以 `1e-4` 微调。
- 原仓库 `QGeoGNN_transfer_8g` 实际建立与4g完全相同的 `GINGraphPooling(num_tasks=6)`，直接加载整个4g `state_dict`，再把全部参数交给 Adam；没有新建输入/输出层，也没有冻结隐藏层。原函数学习率是 `1e-5`，全局 `Use_column_info=False`，所以没有把柱规格送入模型。
- 本阶段没有把论文示意图猜成代码，而是显式比较 scratch、zero-shot、head-only、last1/last2+head、full，以及 `1e-5/1e-4`。当前候选 `last2+head, 1e-4` 只代表单seed validation最优。

## 产物说明

- `canonical_8g.csv` / `sample_decisions_8g.csv`：保留后的数据与全部原始行决策。
- `graph_audit_8g.csv`：88个结构的构图来源和状态。
- `graph_cache_8g_only.pt`：只保存4g缓存中不存在的1个8g结构；其余87个运行时引用4g缓存。
- `split_seed_42.csv` / `compound_group_split_seed_42.csv`：论文可比的行级划分与严格化合物划分。
- `scalers/target_train.json`：scratch仅由8g train拟合的缩放器；预训练模式直接引用`../e0_4g_baseline/scaler.json`，不再保存重复副本。
- `histories/`：9个训练配置的validation轨迹；zero-shot没有训练历史。
- `comparison.csv`：10个配置的validation/test汇总，是完整结果主表。
- `test_predictions.csv`：每个配置逐测试样本预测，用于异常点和配对误差复核。
- `selected_transfer.json` / `checkpoints/last2_head_lr1e-4.pt`：当前单seed候选索引及唯一保留权重。
- `config.json` / `data_metadata.json` / `environment.json` / `artifact_manifest.json`：协议、数据、环境和校验和。

完整指标见`comparison.csv`，逐样本预测见`test_predictions.csv`，争议项见`../METHOD_DECISION_REGISTER.md`。
