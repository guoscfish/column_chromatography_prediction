# 主动学习checkpoint与同pair身份/权限审计

## checkpoint资格

New CW的五个AL seed是157/887/2357/6101/12203，标签预算L525/L1005/L1485对应round06/21/36。已找到15/15份已有checkpoint，官方`load_predictor_checkpoint`严格CPU eval加载通过15/15。无R2转换、无替代模型、无新增拟合，未执行forward；本模块不读取任何目标值。

**正式模型的Row seed525是随机划分种子，不是AL标签预算L525。** 文件中用`formal_row_seed`与`al_seed/label_budget`分开命名。完整checkpoint路径、SHA、schema、参数数、H身份、scaler与归一化hash见`al_checkpoint_inventory.csv`和`al_checkpoint_identities.json`。

H525/H1005由对应选前contract取得，H1485由最后一个实际H1453和已冻结接受32个ID构成，再核对最终fit/prediction receipt的train ID hash及行数。L1485五个AL历史checkpoint均存在，可作为同pair扩展，无需训练。每份保存自身preprocessing、target_scales、normalization与六输出顺序；不会向正式3330训练模型强套AL scaler，也不将不同模型状态的梯度或预测当作同一固定空间。

## 同pair交集与身份

只从旧`frozen_response_pairs.csv`投影seed/budget/因素/ID/条件水平/checkpoint SHA列，**没有将actual_delta_ml或任何真值列作数值解析或载入分析**；源文件全字节只用于SHA绑定。checkpoint本身没有用于新标签查询。以label-free AL metadata重新核对同分子、恰一个因素变化、物理单位与低高水平，且两ID均在对应H。按AL seed保留L525与L1005的相同pair ID交集，并确认端点也在H1485；可选1485只预测这些相同ID，不扩大标签范围。

| AL seed | 因素 | L525∩L1005共同对 | 分子数 |
|---|---|---|---|
| 157 | PE/EA | 277 | 82 |
| 157 | V/ul | 111 | 56 |
| 887 | PE/EA | 282 | 85 |
| 887 | V/ul | 101 | 47 |
| 2357 | PE/EA | 254 | 84 |
| 2357 | V/ul | 108 | 54 |
| 6101 | PE/EA | 251 | 100 |
| 6101 | V/ul | 53 | 37 |
| 12203 | PE/EA | 213 | 87 |
| 12203 | V/ul | 91 | 51 |

同一全量Row模型与不同AL seed的比较是跨模型预测比较，不是配对随机种子试验。多个实验对可共享分子/条件系列，不能把所有pair当独立科研重复。

## 正式曝光与隔离

共5223个AL seed×正式seed×同pair比较记录；允许4162，其余明确隔离。详见al_common_pair_exposure.csv。

每个正式Row seed分别标注两端source-train、mixed train-validation、两端validation或涉及source-test。涉及该模型source-test的pair全部阻断新评分；AL中曾揭示并不覆盖正式source-test禁令。事故隔离名单中的任一ID在所有模型下都拒绝评分。另存任一正式seed涉及test的标记和三正式seed均非test的敏感性子集，不能用其它seed的train角色放宽当前模型的test边界。

完整正式X域为4163行，而旧AL label-free metadata为4162行；缺少的正式独有行不补造成AL历史，也不强行扩展共同pair。正式角色映射及元数据由独立模块提供，目标读取与评分由根分析严格按这些许可执行；本模块只产身份与权限，不读取正式train/validation/test truth。

## 接口与复现

`model, identity = load_al(al_seed, label_budget)`返回官方严格加载模型与该checkpoint自身scaler/目标尺度；不加载标签。`python -B .../al_checkpoint_audit.py --checkpoints-only`只运行15份身份与共同pair检查；默认进一步要求formal_split_roles.csv、source_metadata.csv和带quarantined_ids数组的LABEL_BOUNDARY_INCIDENT.json全部已固定后才产生许可。
