# 正式 Row 模型的严格条件对覆盖与标签边界

三个模型各有 **3330 条 source-train、416 条 source-validation、417 条 source-test**，完整特征域为4163条、217个分子。Row seed 525 是模型随机种子，与 AL 标签预算 L525 无关。

## 无标签覆盖计数

| row_seed | axis | exposure | n_pairs | n_molecules | n_series | n_endpoint_rows | n_global_quarantine_pairs | n_scoring_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | PE/EA | train_train | 8020 | 217 | 614 | 3322 | 27 | 7993 |
| 42 | PE/EA | train_validation | 2082 | 182 | 325 | 2067 | 0 | 2082 |
| 42 | PE/EA | validation_validation | 96 | 68 | 80 | 168 | 0 | 96 |
| 42 | V/ul | train_train | 2771 | 195 | 1092 | 3008 | 0 | 2771 |
| 42 | V/ul | train_validation | 654 | 164 | 333 | 981 | 0 | 654 |
| 42 | V/ul | validation_validation | 48 | 39 | 42 | 87 | 0 | 48 |
| 525 | PE/EA | train_train | 8079 | 217 | 613 | 3320 | 20 | 8059 |
| 525 | PE/EA | train_validation | 2010 | 185 | 305 | 1930 | 0 | 2010 |
| 525 | PE/EA | validation_validation | 132 | 73 | 88 | 198 | 0 | 132 |
| 525 | V/ul | train_train | 2767 | 195 | 1120 | 3047 | 0 | 2767 |
| 525 | V/ul | train_validation | 684 | 167 | 337 | 1013 | 0 | 684 |
| 525 | V/ul | validation_validation | 44 | 35 | 41 | 84 | 0 | 44 |
| 1101 | PE/EA | train_train | 8072 | 217 | 615 | 3325 | 18 | 8054 |
| 1101 | PE/EA | train_validation | 2005 | 178 | 294 | 1871 | 2 | 2003 |
| 1101 | PE/EA | validation_validation | 140 | 77 | 99 | 218 | 0 | 140 |
| 1101 | V/ul | train_train | 2730 | 194 | 1122 | 3039 | 0 | 2730 |
| 1101 | V/ul | train_validation | 713 | 167 | 348 | 1049 | 0 | 713 |
| 1101 | V/ul | validation_validation | 48 | 35 | 39 | 83 | 0 | 48 |

train_train 衡量已见端点上的拟合；validation_validation 才是两端均未用于梯度拟合的对照，仍属于原本用于 checkpoint 选择的 validation，不能称为全新独立测试。train_validation 单独标记为部分已见。所有涉及 source-test 的 pair 只保留 X 身份与隔离计数，不请求目标，也不参与评分。

“系列”固定 canonical SMILES 及另外四项条件，至少有两个变化因素水平；同一系列的多个 pair 与同分子的多个系列均不是独立科研重复。`n_series_all_domain_levels` 表示相应曝光子集包含该系列在4163条特征域出现过的全部水平；全部重复行也覆盖的更严格计数另列在CSV。它不表示潜在物理实验域已穷尽。

## 精确规则与稳定身份

仅改变样品体积 `V/ul` 或仅改变 `PE/EA`，其余四项条件与 canonical SMILES 必须精确相等，不设近似容差。样品体积和 `Volume of loading solvent/ul` 分别匹配。数值仅消除格式差异（例如30与30.0），不四舍五入；PE/EA固定条件使用源字符串精确匹配，排序按EA体积分数增加。体积以uL、PE/EA变化坐标以EA体积分数表示。没有把名义范围30–150内不存在的60/90水平当作已观察点。

`source_metadata.csv` 是完整4163条无目标特征；`formal_split_roles.csv` 给出三个原始split的12489条ID角色。`formal_pairs.csv` 的 pair_id=SHA256(axis,无序端点IDs)，跨模型保持相同，series_id由分子与固定条件决定。全部重复测量的跨水平ID组合都保留；同水平重复不冒充因素变化。`formal_pair_series.csv`、`formal_pair_level_counts.csv` 与 `formal_pair_delta_distribution.csv` 提供系列、水平、跨度和原始多重性。所有覆盖统计保留完整metadata域，但三条预览ID全局隔离，任何触碰它们的pair均`global_quarantine=True,scoring_allowed=False`；原split角色和3330训练数不改。`n_scoring_pairs`为真正允许下游评分的子集；`formal_role_counts.csv`另列每角色剔除后的可评分行数。

## 同条件重复与噪声边界

| row_seed | source_role | n_rows | n_unique_condition_groups | n_duplicate_groups | n_rows_in_duplicate_groups | n_excess_duplicate_rows | targets_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | train | 3330 | 3115 | 199 | 414 | 215 | False |
| 42 | validation | 416 | 413 | 3 | 6 | 3 | False |
| 42 | all_metadata | 4163 | 3829 | 309 | 643 | 334 | False |
| 525 | train | 3330 | 3107 | 205 | 428 | 223 | False |
| 525 | validation | 416 | 413 | 3 | 6 | 3 | False |
| 525 | all_metadata | 4163 | 3829 | 309 | 643 | 334 | False |
| 1101 | train | 3330 | 3115 | 203 | 418 | 215 | False |
| 1101 | validation | 416 | 413 | 3 | 6 | 3 | False |
| 1101 | all_metadata | 4163 | 3829 | 309 | 643 | 334 | False |

重复指所有已记录输入条件相同、sample_id不同。此表只依据协变量，不能判定是重复测量、独立实验批次、误录或未记录因素，也不凭无标签表宣称噪声大小。授权train/validation若比较重复目标离散程度，必须按各模型角色分别投影；含test的重复组仍隔离。

## 可复用的角色受限接口

历史正式资格验证runner直接读取全canonical表，未提供专用角色store；本轮不执行那个runner。`FormalLabelBoundary(seed)`先核验canonical字节SHA与冻结split字节SHA及逐ID/SMILES/行序，调用官方 `transfer.protocol.validate_label_roles`。独立适配器只把该模型原始train映射到官方 `RestrictedLabelStore` 的l0，validation/test保持原角色，不修改冻结split，也不重新划分。`reveal(ids, role='train'|'validation')` 先全局阻断三条预览ID、再校验请求完全属于声明角色，调用 `assert_fit_ids_authorized` 与官方 `RestrictedLabelStore.reveal(..., purpose='initial_fit')`；test、未知、重复和跨角色请求在打开目标store之前被拒绝。本接口不暴露freeze或final_test_evaluation解锁。`labels_for_pairs`在任何数值投影之前拒绝整个含test/未知/全局隔离端点的输入。

本脚本CLI只生成无标签覆盖表，**目标投影0行、模型计算0次、训练0次、LLM/API0次**。官方streaming store读取CSV字符串行后，只对请求授权ID的V1_ml/V2_ml转换float32；文件hash也经过所有字节。因此不声称目标文件字节从未被操作系统读取。下游调用会产生独立官方请求receipt；本覆盖生成器没有目标receipt。

## 文本查看边界例外

初次核查源manifest格式时，审阅命令误用了 `head -4 qualification_dataset_manifest.csv`，显示了头部3行目标文本。其中第一行是Row1101的test、第三行是Row525的test。此文本查看发生在正式loader建立之前；这些值未写入本轮表、未数值投影、未用于任何指标/模型/选择。不能声称本轮全过程test目标文本零可见。`data_boundary_verification.json`与`LABEL_BOUNDARY_INCIDENT.json`透明保存例外的身份及角色、零分析使用范围；不复制数值。**三个ID在所有seed全局隔离**，即使原角色为train也不再投影或评分，且接口在官方store访问前硬阻断。另一个agent初期sed命令把同一目标manifest列为第二文件，累计行号使其没有输出任何目标内容；记录为打开顺读、0额外预览ID，不等同新增数值评分。后续所有标签请求必须经过上述封闭接口。

## 复现与范围

```bash
/Users/fish/miniforge3/envs/chromatography/bin/python studies/predictor/full_training_volume_response_audit_20261117/data_boundary.py
/Users/fish/miniforge3/envs/chromatography/bin/python -m pytest -q studies/predictor/full_training_volume_response_audit_20261117/tests/test_data_boundary.py
```

覆盖表只回答哪些身份可比，不替代预测响应评价。AL共同对由独立模块按相同pair身份及每个正式seed的角色再分层；不同Row seed共享数据，不能当作独立随机种子配对实验。

## AL共同对补充：严格身份交集与角色门控

New CW五个AL seed的L525∩L1005交集共有1741个“AL campaign×pair”身份记录，其中样品体积464、PE/EA 1277；它们的两端也都已包含在同一AL seed的H1485中。跨AL seed可能重复同一个物理pair，1741不能当作全域互不相关实验对数。L525/L1005/L1485的15份已有AL checkpoint均已通过官方严格CPU eval加载，未新增训练；正式Row seed525与AL预算L525是不同概念。

逐正式模型按source角色并排除三条全局隔离ID后，体积共同对可评分比较记录在正式seed42/525/1101分别为373/372/355。两端都属于正式source-validation的AL共同体积对子集，仅有6个比较记录（各3/2/1）；这不是正式validation自己全部体积对的总数，不能用该小共同子集确认泛化。

`al_common_pair_exposure.csv`逐行给出正式seed、两端角色、曝光、test冲突、全局隔离与评分许可；任何当前模型source-test或隔离ID都阻断新评分。正式三seed×AL共同pair共5223个身份比较行，其中1057行涉及source-test、6行触及隔离ID（重叠2行），允许4162行；不能把这些复制比较行当作独立科研重复。许可与本模块`formal_pairs.csv`已逐行独立交叉核验。

同pair数值结果与分层方法见[MATCHED_AL_COMPARISON.md](MATCHED_AL_COMPARISON.md)、`matched_denominators.csv`和`matched_exposure_equal_models.csv`；本补充只说明身份/权限，不改变冻结split或原始数据。
