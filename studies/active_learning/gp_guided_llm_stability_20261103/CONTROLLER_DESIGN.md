# Numerical Decision Controller：离线设计与原型

## 1. 本次结论与适用范围

当前原型可以计算可复算的批次联合条件信息增益、扫描完整 Batch=32 的 496 个候选对、核实严格单因素实验结构，并演示“原提案—数值反馈—完整修正提案”的审计记录。它没有获得正式部署资格；`configs/controller_development.json` 的全部决策阈值均为 `null`，`calibration_status=not_calibrated`，所有结果均携带 `deployable=false`。

本次只运行合成数据 mock 与单元测试。合成提案修正后信息增益上升仅证明代码路径可工作，不能支持真实选点或 QGeoGNN 泛化改善。控制器没有标签文件读取器、测试真值入口、GP 拟合器、自动候选替换器或付费 LLM 传输实现。

优先研究可以由矩阵和实验元数据直接验证的两类约束：**完整批次的高度相关冗余检查，加上严格单因素对照的结构豁免；相同状态下、预先冻结参考批次的联合条件信息增益相对下界。** 风险盲区覆盖需要额外的可信风险校准证据，暂不能成为默认硬约束。

## 2. 与现有工程控制的边界

当前 CW-Free 对照应使用冻结 `run_free_cw_sol61_five_20261101.py::validate_plan` 作为 opaque shared hook；其 `activate()` 覆盖 `nocw_study.validate_plan`，调用 `validate_direct` 及原有正向字段白名单，不能直接换成未运行 paired 版 `nocw_inputs.validate_plan`。仓库其他可复用工程机制包括：

- `src/qgeognn_al/active_learning_v3_2/selector.py:validate_selection`：冻结 packet 绑定、确切选择数、已查看合法候选、唯一 ID、合法科学角色与假设引用、到期假设复核、响应长度。
- `src/qgeognn_al/active_learning_v3_2/response_contract.py`：严格响应 schema 和非生成 JSON 恢复。
- `src/qgeognn_al/active_learning_v3_2/catalog.py:batch_diagnostic`：分子、科学角色、CW percentile、结构相似度等描述性统计；角色计数不施加配额。
- `src/qgeognn_al/active_learning_v3/catalog.py` 的 matched contrast 查询已经能提供条件对照证据。

原型通过 `shared_validate` 注入同一个既有校验入口，不能据此声称 CW-Free 实际使用 v3_2 selector。A/B/C 都必须使用相同冻结版本、修复预算与证据权限。mock 的注入函数只是队列演示，不是正式校验器的替代实现。索引范围、重复索引和矩阵形状检查只是数值函数的输入前置条件，不作为新研究贡献。

Controller 新增的研究对象是数值决策质量反馈及其引发的提案变化。它不根据合法性直接宣称数值质量改善，也不从 LLM 自述中生成单因素对照证据。

## 3. 数学输入和联合条件信息增益

GP 适配层必须先给出相同 checkpoint、相同表示和同一已标注集合 `L_t` 条件下的候选潜变量后验协方差 `Σ_U|L`。控制器本身不重新条件化，不读取未来标签。

对完整批次 B，记 `Σ_B|L = Σ_U|L[B,B]`，未来观测噪声为 `R_B=diag(σ²_i)`，则

\[
I(f_B;y_B\mid D_{L_t}) = \tfrac12\log\det\left(I+R_B^{-1/2}\Sigma_{B\mid L_t}R_B^{-1/2}\right).
\]

这一式子要求给定 GP 潜函数后观测噪声相互独立，并且噪声方差与潜变量协方差在同一输出尺度。初版支持单输出、同方差或逐候选异方差。V1、V2 应分别计算；未经独立性或多输出联合协方差建模，不能把两个输出的量相加后称为联合信息增益。

`conditional_information_gain` 用 `float64` Cholesky，返回 `sum(log(diag(cholesky)))`，并记录 jitter、选中协方差的最小特征值、完整选中子矩阵与噪声以供复算。非有限数、明显非对称、选中子矩阵非 PSD、非法噪声均失败；这里只检查 batch/reference 子块 PSD，不对完整候选池执行昂贵的全矩阵特征分解，完整后验来源认证依赖上游 GP。微小数值 jitter 如实际使用，则决策状态标为不可判定。后验协方差已条件于全部 `L_t`，因此也显式包含与历史已观测数据的相关性。

单点信息量之和

\[
\sum_{i\in B}\tfrac12\log(1+\Sigma_{ii\mid L}/\sigma_i^2)
\]

只作为“忽略相关性的边际上界”诊断字段，不作为批次 CIG。两个完全相关、方差 1、噪声 1 的候选，真实联合信息量是 `0.5 log 3`，边际求和是 `log 2`；测试显式区分两者。

固定梯度核 IVR 的历史单位 ridge/noise 参数不能自动作为已校准 `σ²`。若 GP 噪声只有代数意义，则信息量也只能解释为该代理模型的模型内量，不能标作已校准真实实验不确定性。

## 4. 无加权目标的可校准约束

### 4.1 相对冻结参考的 CIG

在看正式结果前冻结同一候选池、相同批次大小的参考批次 `B_ref`，定义

\[
r_{IG}(B)=I(B\mid L_t)/I(B_{ref}\mid L_t),\qquad r_{IG}(B)\ge\tau_{IG}.
\]

原型检查参考大小完全相同。实际触发还要求研究者预先冻结的 `reference_scope` 明确给出相同实验域、相同实验结构指纹以及冻结 manifest；缺失、不匹配、参考信息量为零或缺失则返回 `indeterminate`，不向 LLM 施加低 IG 修正请求。scope 必须来自研究设计适配层，不能采用 LLM 临时自报字段。候选池、输出、核超参数、噪声和状态都必须一致。参考可由预先声明的、与未来标签无关的开发规则构造；参考并不等价于最优批次，也不能据此声称 QGeoGNN 误差改善。

必须预先声明参考生成算法、随机种子、参考批次 ID、参考属于哪类实验设计；不能临时更换参考以制造触发。此约束如果对局部对照存在系统偏差，应采用预注册 matched local reference，或返回不可判定并进入人工协议审查，不能循环逼迫换成全局覆盖点。当前没有对“一个声称有假设的候选”实行整批 CIG 豁免，结构豁免仅用于具体冗余对。

本次实际历史 Free r0 的八个 seed/表示/输出组合，其模型内 IG 都高于各自 64 个随机批次的最大值（详见 OFFLINE_REPORT）。这不能证明 Free 已最优，但直接否定了“先假设 A 的 IG 低再设下界”的开发方式。这些历史结果和随机 IG 对照均未用于校准正式门槛。

### 4.2 全批高度相关冗余

计算

\[
\rho_{ij}=\Sigma_{ij\mid L}/\sqrt{\Sigma_{ii\mid L}\Sigma_{jj\mid L}}.
\]

扫描 B 中全部候选对，用 `|ρ_ij| ≥ τ_ρ` 标记高度相关对，再统计未获得结构豁免的对数 `N_unexempt`，可预先校准 `N_unexempt ≤ m` 的审查规则。负相关同样可能表示相似的信息方向，因此采用绝对相关系数。零潜变量方差的相关性定义为 0，并同时由 CIG 呈现其低增量信息。

相关性只是在指定 GP 内的冗余，不是真实信息效用的定理；错误或退化的核可能把所有候选判断为高度相关。这类情况要先否决 GP 可靠性或报告不可判定，不能强迫 LLM 给出实际上不存在的“合格”批次。

### 4.3 严格单因素结构豁免

豁免仅限特定高相关候选对，需要冻结的可信候选目录同时满足：

1. 两个不同合法候选属于同一分子。
2. 完整的实验条件字段列表已冻结，两行元数据都具备全部字段。
3. 只有一个字段不同，而且它就是声明的被操纵因素。
4. `hypothesis_id` 来自三个 Arm 共用工程校验器确认过、并在本次数值评估前冻结的假设清单。来源可以是科学记忆，也可以是当前完整提案中的合法新假设；r00没有历史记忆不应导致所有新局部对照被排除。B/C共享初案时清单也相同，后续新假设仍先经过共同校验和冻结，不能由C临时自报字段获得豁免。

元数据省略字段、跨分子、多因素同时变化、自由文本“这很科学”或虚构假设都不能获得豁免。匹配使用精确相等；正式目录中的浮点表示必须先由已有数据规范统一，不能由控制器临时放宽容差。

报告只称 `verified_structural_only`：它证明实验结构，不证明假设正确、实验一定有价值或风险预测可信。豁免只解除该对的冗余触发，所有原始相关系数仍记录。它不施加 CW16+LLM16、GP16+LLM16 或任何固定分子/条件比例。

### 4.4 风险盲区与历史相关性

只有外部 GP 离线资格报告明确给出 `qualified_on_frozen_development`、校准 manifest、相同 `checkpoint_sha256`、有限非负风险分数及冻结布尔盲区定义，原型才计算盲区选择数。否则固定返回 `indeterminate`；已配置盲区门槛也不得制造低覆盖“失败”。池中盲区本来少于门槛时同样标为不可满足、不可判定。

外部资格核验必须依赖合法 prequential / OOF 残差；训练内残差和混杂 checkpoint 的历史残差不具备这一资格。控制器只核对证据接口，不替代 GP 校准审查。

可传入同状态 Raw/CW 候选与历史已观测特征，记录最近历史距离及批内几何距离。它们仅作描述；不设置全局几何覆盖下界，不假定局部条件实验低距离就没有价值。所有实际距离都需与表示版本、状态 hash 一起解释。

## 5. 阈值冻结与不可判定

本次没有“看起来合理”的正式阈值，也没有以 validation 改善为目标调权重。默认 JSON 全部阈值为空。mock 中 `|ρ|≥0.99、未豁免对数≤0` 只是人为制造可验证路径的合成单元测试夹具，不是任何真实实验建议。

正式阈值冻结前需要：

1. 单独冻结开发状态清单，标注此前已参与开发的 seed；不得称外部确认。
2. 先确认 GP 的方差、噪声与风险目标具备所声明的可靠性。
3. 只以开发状态中的退化批次、历史批次、严格对照结构和预先指定无标签参考校准触发规则；禁止看到正式 AULC 后回调门槛。
4. 预先确定局部对照豁免、无可行批次处理、修正次数与成本上限，保存配置与输入 hash。
5. 将校准报告、时间、代码版本、参考 ID 和门槛冻结为 manifest；正式重复调用不再修改。

`decision_status=indeterminate` 包括尚未校准、无激活阈值、参考缺失或为零、风险证据不合格、数值失败。一次 mock 通过不会改变 `deployable=false`。数值失败记录具体错误和原始提案，不产生替换批次，也不假装“Controller 通过”。

## 6. A/B/C 冻结输入适配器

`freeze_shared_inputs` 保存完整基线组件并逐项 hash：checkpoint、`L_t` ID 顺序、`U_t` ID 顺序、科学记忆、候选输入、候选元数据、原始 packet、selector 协议、LLM 配置、工具协议。组件 hash 之外，还比较三个 Arm 完整 shared payload 的 hash，因此即使有人重算单臂 manifest 也会被发现。

- A 保留相同基线，新增 GP 信息为 `null`。
- B/C 初次模型输入完全相同，新增 GP 信息 hash 相同。
- 只有 C 的追加 `controller_feedback` 可以不同。
- A/B 的任何 controller 反馈、单臂记忆变化、模型参数变化或未注册隐藏字段都会失败。

hash 证明提供给适配器的 JSON 输入相同，不能证明调用者没有遗漏真实外部输入，也不能自行证明标签权限合法。Phase 1 还必须把实际工具返回、动态查询记录、服务模型信息、系统提示与传输配置纳入归档；不能只比较摘要 hash 后宣称整个动态工具对话完全相同。若三臂自由查询产生不同工具返回，应明确这是相同只读目录/权限下的决策过程差异，不能称 token 级相同候选可见信息；严格同输入子实验应冻结返回转录。

## 7. 审计记录与失败处理

`run_mock_controller` 只调用注入的内存 mock 修正函数，不导入 LLM 客户端。每一步记录完整原始提案、数值诊断、触发约束、是否请求修正、反馈、最终 32 个候选、添加/移除/保留候选、逐候选原理由与新理由、约束反馈理由、LLM 自报修正理由和前后 hash。数值源记录 covariance/noise hash，CIG 保存可复算子矩阵。

控制器不自行排序替换，不要求固定比例，不以更多修正次数获取无限采样优势。正式 C 的模型、temperature、reasoning、查询预算应与 A/B 冻结；额外修正 token 和调用成本必须单列，并额外报告固定总 token/call 预算的敏感性分析。修正后仍不满足条件时记录 `unresolved_constraint`，不得偷偷使用 GP Top-32 回填。

无效响应继续进入三臂共用的既有工程修复路径；数值质量修正另计。超预算、GP 不合格或无法判定时的正式处理须在 Phase 1 前声明，本次原型不会自动进入在线执行。

## 8. API、复现命令和 mock 证据

核心接口：

```python
diagnose_batch(posterior_covariance, noise_variance, batch_indices,
    config=config, reference_batch_indices=reference_indices,
    reference_scope=frozen_same_domain_and_structure_manifest,
    metadata=candidate_metadata, matched_control_claims=claims,
    approved_hypothesis_ids=shared_validated_ids,
    risk_evidence=qualified_risk_or_none, geometry=raw_or_cw_features_or_none)
```

从仓库根目录运行（当前系统 `python3` 没有 numpy；现有 Miniforge 的 `python` 可运行，无需安装）：

```sh
/Users/fish/miniforge3/envs/chromatography/bin/python -B -m unittest discover -s studies/active_learning/gp_guided_llm_stability_20261103 -p test_controller.py -v
/Users/fish/miniforge3/envs/chromatography/bin/python -B studies/active_learning/gp_guided_llm_stability_20261103/mock_controller_demo.py
```

当前 23 项测试覆盖联合 logdet 与边际求和的区别、异方差噪声、float64、无效矩阵/噪声、完整 496 对、同大小参考、参考域缺失/局部全局冲突不触发、严格单因素与假设引用、checkpoint 风险门禁、无全局距离门槛、逐组件输入漂移、B/C GP 漂移、内存队列无网络回退、完整 32 个提案修正、失败与未解决记录。

可复算合成记录位于 `results/mock_controller/mock_trace.json` 与 `results/mock_controller/frozen_input_check.json`。合成例中一次修正将一个高度相关且无对照结构的候选换成独立候选，CIG 从 `10.946846908449158` 到 `11.090354888959126` nats；最终状态仍是 `indeterminate`，因为门槛未校准。真实 GP 概率可靠性和 QGeoGNN 泛化收益必须另行验证。
