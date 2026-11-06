# 表示实现与历史状态审计

本审计只读取源代码、无标签身份分割、轮次合同、冻结 checkpoint 字节、Raw 特征缓存及标签清零的图缓存。未打开任何响应总表；未读取 test 真值或未来 U 响应。可执行输入审计在 `representation_source_audit.json`，包含全部 20 状态的完整 SHA-256。

## 可直接复用的实现

| 功能 | 既有实现 | 本研究使用方式 |
|---|---|---|
| GNN 隐藏表示 | `src/qgeognn_al/models/qgeognn_v2.py::QGeoGNNV2.extract_representation` | 分子图第五层节点输出求和池化，加类型化条件分支残差，处于 `Linear(128,6)` 之前；输出 N×128，冻结模型 |
| 批量隐藏提取 | `src/qgeognn_al/active_learning_v2/coverage.py::extract_representations` | 保留其无梯度、eval、顺序核验和 finite 核验 |
| 冻结点预测 | `src/qgeognn_al/active_learning_v2/runner.py::predict_outputs` | 只返回预测与位置；不调用会收集 `a.y` 的训练模块 `predict` |
| checkpoint 加载 | `src/qgeognn_al/models/qgeognn_v2.py::load_predictor_checkpoint` | 先核验历史文件 SHA-256，再核验 schema 和加载后 state_dict hash |
| Raw 梯度 | `gradient_features.py::extract_q50_gradient_sketches` | 历史 L/U 精确缓存优先；仅未缓存的合法 validation 行需计算 |
| CW 梯度 | `gradient_features.py::extract_linear_output_gradient_sketches` | 同一 checkpoint、同一 sketch seed、同一原始参数顺序，重算许可行 |
| CW 坐标 | `gradient_transforms.py::center_width_transform` | 仅用对应 seed 的历史 L0 333 个已观测响应计算 population std |
| 条件编码 | `src/qgeognn_al/data.py::condition_matrix` | 复用六个 PE/EA 混合描述符、密度乘样品体积、加载溶剂体积；序数溶剂代码替换为三类 one-hot |
| 条件标准化 | `src/qgeognn_al/data.py::fit_standardizer` / `apply_standardizer` | 只在当前 L_t 拟合均值和尺度，验证与候选只变换 |
| 分子指纹 | 既有 `full_pool_selector.py::_fingerprints` 与 `llm_screen.py` 使用 RDKit Morgan radius=2, bits=2048 | 同样调用 RDKit 成熟生成器并复用唯一分子缓存；保留二进制位 |

GNN、Raw、CW 的原始推理/梯度图遵守既有 float32 模型运行合同；输出统一转换为 float64。转换不会凭空增加原始数值精度；后续 GP 和标准化可在 float64 中进行。不同状态的 hidden、Raw、CW 均不可直接当作同一坐标空间混合。

## 统一历史轨迹与四个状态

五个 seed 为 157、887、2357、6101、12203；预算与历史 round 对应为 L333→0、L525→6、L813→15、L1005→21。全部 20 个状态均已执行文件及合同审计，通过 checkpoint 字节、模型状态 hash、训练 ID hash、validation ID hash、L/U disjointness、canonical indices 与 Raw bank 同态绑定检查。

冻结训练状态统一位于：

`studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/lcmd/round_{round}/`

该目录内读取 `labeled_ids.npy`、`unlabeled_ids.npy`、对应 indices、`input_contract.json`、`contract.json`、`model/fit_audit.json`、`model/best.pt`。`model/predictions.csv.gz` 是 test 输入上的预测表，不需要访问；其真值不在本模块接口内。`model/history.csv` 记录既有训练历史，反映 validation 曾参与 checkpoint 选择。

L333、L525、L813 的 Raw 特征位于同状态的 `acquisition_artifacts/current_gradient_features.npz`。L1005 的 fit 仍由上述 BASE 目录所有，但该状态下一批 acquisition 由扩展目录 `studies/active_learning/qgeognn_v2_row_lcmd_to1485_20261026/` 所有，因此其 Raw acquisition cache 位于扩展目录相同 seed/round 路径。

每个 Raw cache 有 `features` 与 `canonical_indices` 两个字段，形状为 3330×512，顺序严格等于 L_t 后接 U_t。侧车 `.npz.contract.json` 绑定 checkpoint 文件与模型状态 hash、seed、round、L/U hash、ordered indices、L0 endpoint scales、512 维和 `seed+4000037` 的 sketch seed。全部数据均 finite。

早期审计只确认 Raw LCMD 所属目录没有持久化 CW，曾错误推广为完全不存在可复用缓存。补审确认：五个 seed 的共同 L333 在历史 Free 目录已有相同 checkpoint、L/U、ordered indices 和特征定义的 CW；157/6101 还在 short-sequential 目录有显式 CW 合同。2945个 L333 的 L+256U 行因此被冗余重算，占全部26820个 CW 行的10.98%；416个 validation 行/seed仍需补提。新旧589行/seed比较全部通过既有梯度回归容差，非逐位相等，最大绝对差7.63×10⁻⁶、最大相对 Frobenius 差1.37×10⁻⁸。完整证据见 `CW_CACHE_AUDIT.md` / JSON。冻结缓存中的 `historical_cw_available=False` 保留早期记录，由该更正覆盖；不修改核心或原结果。后期 L525/L813/L1005 已分轨，仍不得从 Free/CW-Free 回填。

## 图缓存及数据权限

每个 seed 的既有 `runtime/seed_{seed}/scrubbed_graphs.pt` 已封存全部输入图。加载时核验对应合同与 SHA-256，并验证每个 `y` 均为零 sentinel、canonical_position 与位置一致。模型本身不使用 y。提取接口仅允许本状态 L_t、U_t 和既有 validation 的 indices；test 或未知 indices 会直接被拒绝。

身份分割文件为 `studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv`，只含身份、canonical SMILES、canonical_index、角色及 seed；416 个 validation、417 个 test、333 个 L0、2997 个 U0。

既有 LCMD 没有单独持久化 L_t 和 validation 真值表。本模块不打开响应总表，也不提供标签读取接口。研究主驱动必须显式验证当前 L_t 与历史 acquisition lineage 后，只请求已揭示 L_t 和既有 validation 响应。`RestrictedLabelStore` 是既有逐 ID 授权实现；不能调用其 `final_test_evaluation` 路径。

仅用于结构化特征时，成熟 `benchmark_protocol.load_features` 通过明确的七列 `usecols` 读取合法输入。本研究 encoder 同样只接受这七列，任何真值、残差或预测列均拒绝。Structured 输出为 Morgan2048 + 标准化连续条件8 + solvent one-hot3 = 2059 维。

## 接口与已执行验证

`representations.py` 提供 `audit_lcmd_state`、`load_raw_cache`、`load_scrubbed_graphs`、`load_model`、`extract_hidden`、`extract_gradient`、`structured_features`。所有路径通过传入项目根目录确定；返回的审计路径均为项目相对路径。

已执行 4 项合成单元测试，全部通过：拒绝 test/未知及重复 indices；标准化只用当前训练行；拒绝响应列；同分子条件变化可改变结构化表示。另对真实 seed157/L333 的 2 个 L 行与 1 个 validation 行执行冻结 hidden 预检，输出 3×128 与 q50 的 3×2，全 finite；3 行提取耗时约 0.023 秒，不含模型及图缓存加载。未运行大规模梯度计算，未训练 GNN。

现有 chromatography 环境具备依赖。独立进程遇到重复 libomp 初始化时，采用仓库 README 已有的 `KMP_DUPLICATE_LIB_OK=TRUE` 调用设置，不安装或修改依赖环境。
