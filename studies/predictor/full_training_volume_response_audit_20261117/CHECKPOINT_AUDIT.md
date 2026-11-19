# 正式充分训练Row checkpoint身份核查

结论：三份正式权重均存在，身份与原资格验证记录一致，官方严格加载通过。每个模型真正训练3330行，原validation416行、test417行；完整4163行数据域不能称为全部训练数据。模型seed525与主动学习预算L525没有对应关系。

| Row模型seed | 训练/validation/test行数 | best epoch / 已运行epochs | 参数 | checkpoint SHA256 |
| --- | --- | --- | --- | --- |
| 42 | 3330 / 416 / 417 | 95 / 195 | 458952 | fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0 |
| 525 | 3330 / 416 / 417 | 318 / 418 | 458952 | d86d314c55a4d6068a58d965755b2b4e5a664f5e18d877016f2da5af85acae07 |
| 1101 | 3330 / 416 / 417 | 67 / 167 | 458952 | df37cdff353536e825ae269ca03418024913adc39d6061f143c8c5d63aca5901 |

## 权重、split与预处理身份

三份权重均位于`studies/predictor/final_4g_qualification/runtime/row/seed_{seed}/best.pt`，不是仓库根目录runtime，也不是早期E0模型。以对应run_summary记录定位，再同时验证固定SHA和原资格验证run_summary的SHA。split从final资格验证的splits目录读取，字节与clean资格验证原始split相同；逐ID唯一、seed/mode与3330/416/417行数均核验。source-train ID排序哈希同时匹配checkpoint的preprocessing与condition normalization。

| seed | split SHA256 | V1目标尺度mL | V2目标尺度mL |
| --- | --- | --- | --- |
| 42 | a8a4a322cd9c75253b1a4ed8c5c91ededed74a933df2db2685015c8b3e008008 | 7.879659034632 | 16.076509553563 |
| 525 | 58876dc5a9033a0584da964e1e52bd0fa5d1aeba8131410060cce947fe11d0a1 | 7.779784662002 | 15.796294775218 |
| 1101 | fb51a225dd2b802b4dbc2982fa45166a7c11962fef9add2015e6637a1a2a6bf1 | 7.872146175179 | 16.194650683003 |

条件量及分子/eluent scaler均来自source-train：fit_rows=3330，validation/test/target_rows_used均0。checkpoint的scaler、runtime/scaler.json与results/normalization.json逐值相同；scaler原文件SHA同时匹配condition normalization。三个模型的分子/eluent scaler恰好相同，不据此强行替换其各自身份。目标尺度来自冻结记录，未重新读取或计算标签。q50输出已为mL，目标尺度仅用于归一化评价，不能重复反归一化。

## 官方加载及R2→V2等价边界

实际二进制均为原生`schema_version=1`、`model_variant=qgeognn_v2`，状态键是backbone/head/condition_branch；这些正式模型直接构建有效V2网络，不是旧R2归档，因此本轮无转换。只读`load_formal(seed)`调用官方`load_predictor_checkpoint`，保留schema、input schema hash、归一化、6输出顺序、458952参数与strict state_dict校验，再逐tensor确认加载前后bitwise相同，并确认eval模式。未知seed、身份不符或旧格式直接阻断，不临时改键绕过契约。

历史官方`final_v2_engineering/equivalence_audit.json`已证明R2_PRUNED经`historical.conversion.convert_pruned_to_standalone`迁移后，在固定4163行工程fixture上全部六输出最大差0、指标最大差0、保留参数bitwise相同、无missing/unexpected keys。final资格验证protocol用SHA绑定此gate。本轮验证gate状态与SHA，不重跑它，因为其历史执行会访问test标签。该证据属于模型工程语义等价，不表示本轮三个Row模型都由同一个R2 checkpoint转换而来。

## 推理接口与权限

在本目录Python路径下导入`from checkpoint_audit import load_formal`，调用`model, identity = load_formal(42)`。identity包含preprocessing/scaler、normalization、目标尺度、output_order、split路径与SHA、训练配置及权重SHA；使用每个模型自身输入契约进行图构建。接口只构建和加载CPU eval模型，不执行forward，不读取图缓存、target CSV、标签或数值预测缓存。训练3330行角色来自冻结split元数据。源canonical与分子图缓存SHA仅转录原protocol，未在此子审计直接读取其内容。

## 执行边界与下游隔离

本loader程序及14项契约测试不读取任何target CSV、标签或正式predictions.gz。初始人工式shell检查曾误把qualification_dataset_manifest.csv列为sed的第二个文件参数；sed行号跨文件累计，工具实际只返回第一个row split的4行元数据，没有返回目标值或该manifest中的ID，但进程确打开并顺读了目标承载文件。该范围失误已向主审计披露，不能把整个工具历史描述为从未打开过目标文件；它未进入loader、标签或评分输入。

主审计另有事故预览ID的全局quarantine；后续所有新评分须先排除该集合，独立结果QA也以此为硬门。排除评分行不改变原模型3330行训练事实、任何原始source split或权重。本报告仅核验原模型身份，不把排除后的评分子集重新称为其训练清单。

复现：`PYTHONDONTWRITEBYTECODE=1 /Users/fish/miniforge3/envs/chromatography/bin/python -B studies/predictor/full_training_volume_response_audit_20261117/checkpoint_audit.py`。新manifest记录loader实际读取的配置、split、权重及源码SHA，并在结束时再次核对未变化。自动审计与测试均0新训练、0模型forward、0LLM/API、0标签数组访问；不更改旧checkpoint或报告。
