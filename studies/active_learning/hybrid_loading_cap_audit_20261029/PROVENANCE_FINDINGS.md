# Scientist V2 来源与 validation 曲线复核

2026-10-29。仅使用已有冻结实验，主区间为 seed 157、6101 的 333→813：每个方法 15 批 acquisition、16 个拟合预算点。已完成的检查为 PASS。本次没有调用 LLM、训练模型、提取梯度、揭示新标签或执行 test 评价。

## 来源及原执行 owner

主对象为 `cw16_llm16_scientist_v2`，不使用顶层 query 试验轨迹替代正式全候选 Scientist V2。

| 方法 | 选点 round | 原执行目录 |
|---|---|---|
| Hybrid，两个 seed | 0–14 | Scientist V2 `revisions/astra_high_full_pool_20261001` |
| Free，seed 157 | 0–5 / 6–14 | 同上 / `revisions/free_sol_xhigh_continuation_20261012` |
| Free，seed 6101 | 0–9 / 10–14 | 同上 / `revisions/free_sol_xhigh_continuation_20261012` |
| CW-LCMD32 | 0–2 / 3–5 / 6–9 / 10–14 | `qgeognn_v2_row_short_sequential_b32` / `qgeognn_v2_row_cw_lcmd_to_525` / `qgeognn_v2_row_cw_lcmd_to_653` / `qgeognn_v2_row_cw_lcmd_to_1005` |
| raw-gradient LCMD32 | 0–14 | `qgeognn_v2_row_sequential_b32` |

拟合 r 属于产生选点 r−1 的 revision，边界不能按 acquisition r 直接套用。Scientist V2 后续 `sol_xhigh_hybrid_free_to1005_20261013` 的 60 批继承副本与上述原 owner 的冻结文件逐一按字节一致；几何和行为分析可使用这些继承副本而不改变来源。逐轮 owner、选中 ID、pending CW ID 和 SHA 见 [source_audit.json](source_audit.json)，逐文件 SHA256 见 [source_hashes.json](source_hashes.json)。

`reports/llm_hybrid_complete_20261010/validation_curves.json` 当时的 Hybrid 已到813，而 Free 仅分别到525、653。因此完整 Free 曲线必须从其真实续接 owner 读取，不能外推旧报告。重新读取的共同点与该报告以及后来的五 seed 冻结 validation 快照完全一致。813之后的历史延长段未进入本次任何共同区间平均；其他 seed 缺少 Hybrid 时没有补造该策略数据。

## 完整性与标签边界

共复核128个“方法×预算”拟合点、122个唯一 `fit_audit.json` 和120批选点。L333每个 seed 四法共用一个 checkpoint，所以128不等于128次独立训练。120批分为 Scientist60批、CW30批、raw30批，总计3840条逐轨迹选点记录。

对 Scientist 的每批检查覆盖 `selection`、`feedback`、`batch_freeze`、`catalog`、`input`、`contract`、已冻结 turn 和 label-access receipt：

- 32个 ID 合法且不重复；Hybrid 前16个 pending CW与LLM16分开，Free无pending。
- 当前已观测集合恰等于自身 L_t，候选+pending恰等于自身 U_t；跨轮 L_t/U_t 增减与前一批冻结ID一致。
- 输入数据包与原catalog的每条ID、SMILES、五条件、预测数值及已有观测值逐字段一致，所有候选均完整送入；压缩表没有漏行。
- 候选及pending不含真实V1/V2；科学假设的实测证据ID只能来自自身已观测集合。
- `batch_freeze` 明确标记新批标签尚未揭示；receipt绑定其哈希，且要求 `after_acquisition_fit`、`acquisitions_frozen=True`、仅u0及恰好32个请求ID。反馈顺序、事前预测和揭示后的误差等式一致。
- 后一预算模型的训练ID哈希必须恰为 L_t 加上本批；本次没有从全量目标表或预测CSV补充真值。

CW和raw的选点表、acquisition contract、label-access ledger也按32个实际ID及冻结后u0请求复核；raw没有可用于历史误差邻近分析的已审计反馈响应载荷，不能据此补造残差。

各预算checkpoint和预测文件只按字节校验，未反序列化checkpoint或解析预测CSV。拟合ID、初始化、validation划分、固定目标尺度和测试标签禁用字段均一致；`history.csv` 的实际epoch数量、最低validation分数及best epoch与fit audit一致。历史元数据归档的SHA也与manifest一致，没有解压覆盖历史文件。

**边界声明：历史test曾被评价及曝光。** 原协议自身将证据列为“two-seed development; historically exposed test; no significance claims”。现有冻结receipt只证明记录中的选点/训练阶段访问边界，不能证明过去所有人工分析从未看过test，也不是独立时间戳服务。本次性能计算只使用已有validation fit audit，不按test选K，不执行新的test评价。validation同时承担历史checkpoint选择和开发比较，结果应按开发证据解读。

## 初始化、尺度及实现谱系

两法在同一seed下共享L333、split和初始化。seed157初始目标population std为V1=6.502957820892334、V2=13.158523559570312；seed6101为V1=5.462151050567627、V2=12.266972541809082。它们整条轨迹固定，未重新按L813或全量数据定标。

NRMSE保持原定义：`sqrt(((RMSE_V1/s_V1)^2 + (RMSE_V2/s_V2)^2)/2)`。AULC为对标签数的梯形积分除以该阶段标签跨度，未除以初始误差。CSV端点均来自实际冻结预算点，不插值和外推。

原训练规则为相同初始化从头训练，Adam、lr0.001、weight_decay0、batch2048、最多1000 epoch、patience100，按validation combined NRMSE选择checkpoint；实际各预算的best epoch可以不同，训练动态并未相同。

当前Scientist study/selector/full-delivery/transport、llm_screen、gradient、CW transform、LCMD及runner均与Astra协议登记源码哈希相符。与git历史逐字节和AST比较结果保存在来源审计：`93609bd9`对应完整全候选Scientist实现；`1ec395f8`较早query版本的stage/transport等与后来的全候选实现不同。原始gradient源`8081e865`的五个共同符号AST与当前一致；后来的linear-output梯度功能是扩展，不能混称原始q50表示。LCMD实现与`8081e865`逐字节一致。

raw LCMD使用当前模型q50全网络512维CountSketch梯度；CW使用L333-only的中心/宽度变换：`[(V1+V2)/(2*s_C), (V2−V1)/s_W]`，再做对应线性输出梯度。二者使用同一个corrected LCMD-TP选择器：全部当前L_t先作为中心，按最近中心平方距离之和最大的簇，选择该簇最远点，稳定输入顺序打破并列。CW32和raw32是两个独立方法及轨迹，不能互换。

## 模型、Prompt与路径混杂

| seed/方法 | 精确对应已接受selection的served model |
|---|---|
| 157 / Hybrid | 15批gpt-6-astra |
| 6101 / Hybrid | 15批gpt-6-astra |
| 157 / Free | 5批gpt-6-astra；r5为历史授权ID抄写修正，精确served model记unknown；9批gpt-6.1-sol |
| 6101 / Free | 10批gpt-6-astra；5批gpt-6.1-sol |

Astra配置high reasoning；Sol续接配置xhigh。served model优先取实际接受的turn，而非用当前配置倒填；授权抄写修正和额外纠错保留原始记录。

三目录的系统Prompt文件完全相同，文件SHA为 `f304b667ec2e1d97f0726e24bced2535a9e386e5ec206ea979cab0b5426ff441`；协议中的 `prompt_sha256` 为string稳定JSON哈希 `3fea9e22b2b325117f68e22b06ab0030767de6edddf4017df27128a4308840c4`，两种哈希不可混用。因此主轨迹之间不能无证据声称“系统Prompt版本不同”。实际输入仍因选16/32、pending CW16、各自L_t、预测器、反馈和科学记忆而不同，且Free中途切LLM版本/推理强度。首轮以后是**匹配seed与预算的观察比较**，并非同模型同状态的因果实验。

旧 `qgeognn_v2_row_llm_dialog_feedback_v1` 是独立辅助证据：gpt-6-sol/high、query式交互、最多240个候选可见，与全候选Scientist V2不同。已读取其协议、最终报告和12轮边界审计（384个冻结事前预测误差记录）；未把其任何ID轨迹或性能曲线拼入主结果。

## 逐seed曲线

| seed | 方法 | NRMSE525 | NRMSE653 | NRMSE813 |
|---:|---|---:|---:|---:|
| 157 | Hybrid | 0.451110 | 0.384773 | 0.322792 |
| 157 | Free | 0.645484 | 0.619968 | 0.431253 |
| 157 | CW-LCMD32 | 0.504888 | 0.409144 | 0.391350 |
| 157 | raw-gradient LCMD32 | 0.538682 | 0.403946 | 0.389443 |
| 6101 | Hybrid | 0.630022 | 0.428822 | 0.380292 |
| 6101 | Free | 0.824845 | 0.520703 | 0.422698 |
| 6101 | CW-LCMD32 | 0.550836 | 0.461569 | 0.422949 |
| 6101 | raw-gradient LCMD32 | 0.727990 | 0.462123 | 0.405376 |

| seed | 方法 | AULC333–525 | AULC525–653 | AULC653–813 | AULC333–813 |
|---:|---|---:|---:|---:|---:|
| 157 | Hybrid | 0.659929 | 0.417850 | 0.352198 | 0.492798 |
| 157 | Free | 0.753584 | 0.651583 | 0.526178 | 0.650581 |
| 157 | CW-LCMD32 | 0.616361 | 0.458227 | 0.392942 | 0.499719 |
| 157 | raw-gradient LCMD32 | 0.671241 | 0.461078 | 0.397824 | 0.524059 |
| 6101 | Hybrid | 0.764431 | 0.527136 | 0.404847 | 0.581291 |
| 6101 | Free | 0.908936 | 0.647074 | 0.487370 | 0.698584 |
| 6101 | CW-LCMD32 | 0.778342 | 0.499207 | 0.442531 | 0.591969 |
| 6101 | raw-gradient LCMD32 | 0.827271 | 0.575534 | 0.424329 | 0.625827 |

Hybrid的整个333–813 AULC和813端点均在两个seed优于CW32，但**不能称其每个早期阶段都更优**：157的333–525 AULC劣于CW32，6101的525–653 AULC也劣于CW32。曲线不能单独说明loading补全有害或有益；需要与具体行为及几何证据一起讨论，并保留上述状态混杂。

完整128点和32个阶段指标见 [validation_rounds.csv](validation_rounds.csv)、[validation_stages.csv](validation_stages.csv)。

## 正式1005双臂的计算量规则

333→1005需要21次32点采集，最终1005拟合后不再选下一批。每seed两个Arm合计42次新QGeoGNN拟合、42次正常LLM selection；同一L333模型可共用，若不能复用冻结L333则额外训练1次/seed。两seed为84次新拟合及84次正常LLM selection；五seed为210次新拟合及210次正常LLM selection，另可需2/5次共享L333拟合。

“LLM selection”和网络请求不是同一计数：若每批最多3个完整计划响应（初始1+纠错2），五seed最坏630个完整计划响应；网络重试还会增加HTTP尝试，应另设上限、另行记录。历史纠错次数不应用作未来突破上限的授权。若只做同一冻结L333的单轮配对选择，每seed仅2个正常selection，选择分析本身无需训练。

以上为未来条件性成本核算，不构成启动正式实验的授权。复现命令：

```bash
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/provenance_analysis.py
/Users/fish/miniforge3/envs/chromatography/bin/python -m unittest discover -s studies/active_learning/hybrid_loading_cap_audit_20261029 -p test_provenance_analysis.py -v
```

12个测试覆盖阶段积分不重定标、缺点拒绝、owner边界、候选真值注入、非已观测证据ID、未冻结/提前揭示、数据包漏行、重复ID及实际输出完整性；全部通过。
