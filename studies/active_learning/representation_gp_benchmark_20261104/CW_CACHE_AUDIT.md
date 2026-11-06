# L333 跨目录 CW 缓存补充审计与更正

先前“LCMD 没有可复用 CW 缓存，必须全部重算”的判断过强。它把 Raw LCMD 目录内未存 CW，错误推广为相同状态下不存在 CW。补审确认：五个 seed 的共同 L333 初始 checkpoint、L/U 和预处理完全相同，历史 Free 目录已有可复用的 CW bank；seed157、6101 另有 short-sequential 的显式 CW bank。目录归属不应替代 checkpoint、输入和特征合同核验。

本研究在发现此遗漏前，已重复计算各 seed 的 333 个 L0 和 256 个 U 行，共 2945 个 CW 行，占完整 20 状态实际 CW 提取 26820 行的 10.98%。每个 L333 状态另有 416 个 validation 行不在旧 bank 中，仍需补提。这里不估算节省时间，也不把早期遗漏包装成缓存无法使用。

## 来源桥接

- 五个初始 Free bank 的 byte SHA-256、sidecar SHA-256 均与既有 `nocw_free_selection_diagnostics_20261101/geometry_input_manifest.json.xz` 相符。sidecar 的 checkpoint、完整 ordered canonical indices、protocol hash，以及 manifest 的 ordered L+U IDs，与本研究 LCMD L333 一致。对应历史 context 的 preprocessing 和 normalization 与 BASE 完全一致。
- seed157、6101 的 `qgeognn_v2_row_short_sequential_b32/runtime/seed_{seed}/center_width_lcmd/round_00/acquisition_artifacts/current_transformed_gradient_features.npz` 有完整显式合同：checkpoint、L/U hash、333 个 center、3330 个有序 indices、512维、`seed+4000037` 的 sketch seed。`gradient_audit.json` 的 CountSketch mapping hash 和 L0-only center/width transform 与本研究完全一致。两个 seed 的旧 Free CW 与旧 short CW 的完整 3330×512 矩阵逐位相同。
- seed887、2357、12203 的旧 Free sidecar 没有重复存储 transform/scales 字段，因此不能假称侧车直接证明这些字段。其已冻结 producer `reports/free_llm_five_seeds_to1005_20261023/run.py::activate` 明确把初始 round 送入 `llm_screen.banks` 的既有重算路径：相同 `context.cw_transform`、512维和 `sketch_seed(seed)`；runner byte hash 与历史 protocol 匹配；其 `llm_screen.py`、`gradient_features.py`、`gradient_transforms.py`、`benchmark_protocol.py`、`scientist_study.py` 五个 producer 依赖的当前文件 SHA-256 也逐一等于历史 protocol 冻结值。context 使用相同初始 L0 和 preprocessing，CW 坐标由 `center_width_transform(L0)` 构建。下述589行数值比较进一步核验此来源桥接，不读取标签来重建 transform。

完整路径、SHA-256、来源字段与比较统计见 `CW_CACHE_AUDIT.json`。

## 旧缓存与本研究重算的数值比较

仅读取 NPZ 的 `cw`/`features` 和 `indices`/`canonical_indices` 字段；没有加载本研究 NPZ 内的 `train_y`、`validation_y`，也没有读取响应 CSV、feedback、预测或残差。每个 seed 比较相同 589 行，即333个 L0 和固定256个 U；validation 不参与比较。

| Seed | 比较行数 | 逐位相等 | 最大绝对误差 | 相对 Frobenius 误差 |
|---|---:|---|---:|---:|
| 157 | 589 | 否 | 3.8147×10⁻⁶ | 1.3200×10⁻⁸ |
| 887 | 589 | 否 | 7.6294×10⁻⁶ | 1.3684×10⁻⁸ |
| 2357 | 589 | 否 | 7.6294×10⁻⁶ | 1.2151×10⁻⁸ |
| 6101 | 589 | 否 | 7.6294×10⁻⁶ | 1.3406×10⁻⁸ |
| 12203 | 589 | 否 | 7.6294×10⁻⁶ | 1.2214×10⁻⁸ |

全部比较通过仓库既有历史梯度回归容差 `rtol=2e-4, atol=2e-5`。它们属于同一状态和特征定义，数值上可复用，但不能声称本研究重算与旧缓存逐位一致。最大非零旧元素相对误差为0.5988%，出现在约−9.95×10⁻⁶的近零特征坐标；该坐标绝对差仅5.96×10⁻⁸。整体相对 Frobenius 误差均约10⁻⁸；所有元素有限。

这些差异与 float32 梯度/CountSketch 运行的微小数值差异相容；本审计未单独定位运算顺序或运行库的具体贡献，不把可能原因写成已证实机制。

## 对冻结结果的处理

本研究已有 `state.audit.historical_cw_available=False`、`representation_source_audit.json` 及各缓存 receipt 中的相同字段，是早期仅搜索 LCMD 所属目录后形成的发现结果。该字段对 L333 的广义缓存可用性判断由本补充审计更正；不能继续解读为不存在跨目录同状态 CW。为保留已冻结运行的真实记录，不改这些字段、cache byte hash、核心代码或已算 GP 分数，不用旧缓存替换结果来择优。

本更正只适用于共同初始 L333。L525、L813、L1005 的后期 Free、NoCW、CW-Free 与 Raw LCMD 已分叉，不能借相同预算之名混用 checkpoint 或 L/U。那些状态仍按本研究冻结的 Raw LCMD checkpoint 计算 CW。

本补审没有启动梯度计算、模型训练或采集；只做 metadata 和特征数组比较。
