# 代码、实验来源与资源审计

日期：2026-11-04，Asia/Shanghai。工作区为 main / `32fdc85cb8411f9e93c2e3db56f6648c14d620b2`，本地 origin/main 与其一致；远端核验另见本研究根执行记录。保留用户原有两项已修改运行状态和大量未跟踪记录。没有checkout/reset/clean、覆盖历史文件、停止进程、执行正式AL或安装依赖。本次新写只在独立研究目录。

审计使用源码、冻结协议、历史报告、contract/fit metadata、模型/feature文件身份及无标签metadata；实际响应按本研究白名单另行获取。历史报告里已有测试曝光只影响证据等级，不读取测试逐行真值，不用旧测试评分选表示或调参数。

## 当前运行与禁止写入路径

CW-Free 权威目录为 `studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/`；README指向 `immediate_retry_20261102/controller_overlay.py`，串接persistent、HTTP/2、HTTP403与zero-backoff修订。早期PROTOCOL的有限重试/串行调度不能代表现状。根执行者在11-04约15:36 +08进行只读进程核验：controller PID10050及887/12203 worker存活；其余三seed的持久化状态为FAILED_CLOSED/upstream_error，不能仅由该字段推断进程。该时刻已完成budget157/887/2357/6101/12203分别为813/1101/1037/973/909；这是进度观察，非原子终局快照。

NoCW延长目录 `qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1389_20261101/`有明确11-01用户停止回执 `progress/user_stop_20261101T112435Z.json`。最后完成L分别1069/1005/1133/1165/1229；停止回执优先于旧SELECTING状态。本研究不恢复/停止/改写两类运行。禁止写它们的selections、runtime、progress、frozen_source、controller、锁和回执；已有GPPhase0目录也只读复用。

## 可复用成熟实现

| 功能 | 已有入口 | 本研究用法与限制 |
| --- | --- | --- |
| 冻结预测器 | `src/qgeognn_al/models/qgeognn_v2.py` | `load_predictor_checkpoint`校验variant、schema、normalization、参数数与输出顺序；eval CPU，不训练 |
| GNN隐藏层 | 同文件`QGeoGNNV2.extract_representation`；`active_learning_v2/coverage.py::extract_representations` | pre-head图节点add-pool加typed condition residual，N×128；不是6个预测值，模型float32、导出float64 |
| Raw梯度 | `active_learning_v2/gradient_features.py` | q50两端点以L0 scale标准化后全458,952参数CountSketch512；缓存精确绑定checkpoint、ordered L/U与sketch map |
| CW梯度 | `gradient_transforms.py::center_width_transform`及既有linear-output梯度接口 | L0-only center/width尺度；与Raw分开。共同L333五seed有跨目录同状态CW缓存；后期轨迹分叉不得Free回填，见CW_CACHE_AUDIT |
| 多map bank | `gradient_bank.py` | 有共享参数endpoint blocks及nested sketch工具，本任务不新做dimension/noise网格搜索 |
| 固定核IVR | `active_learning_v2/ivr.py::conditional_batch_ivr` | 成熟无标签线性条件协方差/批更新；unit prior/noise未校准，无响应mean拟合，不等同本次监督GP |
| 图与条件预处理 | `data.py`、`schemas/conditions.py`、`training/predictor.py` | 复用graph缓存、checkpoint preprocessing、typed solvent和density×V；不得重新拟合/改变原QGeoGNN归一化 |
| 分子指纹与描述符 | `active_learning_v2/full_pool_selector.py`、`data.py` | 已有RDKit Morgan radius2/2048与化学descriptor；显示用rounded descriptor不能冒称无损实验条件 |
| GPPhase0 | `gp_guided_llm_stability_20261103/gp.py` | 复用LinearGP、stable_cholesky、probability_metrics、batch_information；不是重写同一成熟功能 |
| selector真实协议 | `scientist_selector.py`、`scientist_full_delivery.py`、`nocw_inputs.py`及CW-Free冻结runner | 仅审计历史输入语义；新研究不调用selector、transport或记忆模块 |

源码确认V2确有density×V、loading solvent、loading-solvent volume输入，不能把条件方向弱误写成输入缺失。数据实有五字段：PE/EA、Density g/ml、V/ul、loading solvent、Volume of loading solvent/ul。无flow/pH/额外concentration，不新增虚构条件。PE/EA=0/1合法，纯EA必须保留；样品体积μL与加载溶剂体积μL为不同字段。

## 二十个合法LCMD状态

表示审计确认五seed×L333/L525/L813/L1005全部存在，对应round00/06/15/21。fit统一来自 `qgeognn_v2_row_sequential_b32/runtime/seed_<seed>/lcmd/round_<round>/`；round21 Raw acquisition延长来源为 `qgeognn_v2_row_lcmd_to1485_20261026/`，fit仍在原来源。完整身份与SHA见 [representation_source_audit.json](representation_source_audit.json) 和 [REPRESENTATION_AUDIT.md](REPRESENTATION_AUDIT.md)。

Raw缓存每状态3330×512且ordered L_t+U_t严格匹配；validation不在bank，需从同冻结checkpoint补提416行。早期仅发现LCMD所属目录未存CW，漏查共同初始模型的跨目录缓存；补审确认L333五seed均有同checkpoint/L/U/特征定义的旧Free CW，157/6101还可桥接short-sequential CW。2945/26820个CW行被冗余重算（10.98%），新旧589行/seed均通过既有容差但非逐位一致，详见CW_CACHE_AUDIT。L525/L813/L1005后期不得混轨。本研究实际CW提取限定当前L_t+固定256U+416validation，不新算全U。每seed已有scrubbed_graphs.pt且全graph y=0，隐藏层/梯度只访问X；test索引从接口白名单拒绝。GNN隐藏3行冻结模型预检约0.023s（不含load）；不能将其外推成CW梯度或语言编码成本。

标签由根执行者的RestrictedLabelStore按状态显式允许L_t与validation，U_t/test拒绝；新模块不直接读取全响应表。若流式经过原始CSV未授权行，禁止将其target字符串数值化、返回或用于任何计算。完整L/U/validation身份、checkpoint/normalization/hash必须先于标签读取与GP评分冻结。

同seed后期Free/NoCW/CW-Free的L/U、checkpoint已分叉，同预算不恢复同状态。本研究不混轨迹，不借外轨迹后续响应。既有validation多次参与开发及checkpoint selection，只能称开发评价；20状态相关，不能称20次独立重复。

## 既有GP与几何结论

11-03 Phase0实现只有固定Raw/CW的线性Gaussian GP，训练幅度和同方差noise，float64、Cholesky，无Matérn、非线性projection或LLM encoder。它在seed157/6101固定r00/L333状态、其后已经观测的六历史批次上核验；响应90%覆盖61.5%–74.5%，并未通过概率可靠性门槛。固定旧模型风险GP有排序线索，但监督在前96点揭示之后才可用，不能回送历史r00，也不能当后期当前checkpoint风险。新研究需要的Matérn/Projection是缺失功能；概率度量与稳定求解直接复用。

`qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md`审计25冻结状态，结论STOP_BEFORE_GATE_C。512维跨seed或512对2048总体rank相关虽高，Top32重叠稳定性未过；没有候选获准进入独立训练实验。故本研究必须报告排名、重叠和冗余，不能以高Spearman代替选择稳定性，更不能在自己surrogate中IVR较优就宣称AL更强。

`free_lcmd_mechanism_diagnostics_20261029/matched_condition_geometry_probe.json`覆盖157/887、四历史状态、Free Raw/CW及LCMD Raw三空间、三条件轴。它只分析几何、未取真实响应；PE/EA对比距离远大于上样量方向，显示压缩线索。变量幅度/单位不同，比例不是物理导数。原报告也明确小距离并不证明标签无用或表示失败。本研究新增严格validation观测对的真实响应/差分误差，才可检验这个问题。

本次无标签validation匹配支持见 [pair_support.json](pair_support.json)：样品体积逐seed39/39/39/39/42对，PE/EA为137/113/123/117/136对；loading-solvent-volume仅seed157有2对、12203有1对，其余0，总共只涉及2个分子。加载溶剂体积轴严重缺乏独立支持，不能作稳定结论。每seed四预算使用同validation，因此这些pair数不可乘4作为独立观测。

CW-Free/NoCW都保留QGeoGNN预测、分位宽度、结构、五条件、新颖性及本轨迹既测反馈；NoCW仅去显式CW字段。分位宽度不是已校准epistemic uncertainty，历史residual属于获取时模型。原Free部分seed含Astra/high前缀；执行时期/重试也有差异，不能从这些历史行为直接归因语言或CW信息。本研究不使用LLM理由来替代实测效果。

## 本地语言模型、依赖与实际预检

标准HF cache、配置环境变量指向、项目及本地frozen runtime的有限范围扫描未发现T5/BERT/sentence encoder权重；不能由此证明整台电脑绝对不存在模型。标准hub只见完整 `Qwen/Qwen1.5-0.5B-Chat`，revision `4d14e384a4b037942bb3f3016665157c8bcb70ea`，约1.2GB。它是chat decoder预训练权重，不是T5；根执行者明确将它单列探索性D/E/F。未下载、未付费调用、未generate。

默认miniforge Python没有torch/transformers/RDKit；既有chromatography环境有torch2.13.0、PyG2.7.0、transformers5.4.0、tokenizers0.22.2、safetensors0.7.0、RDKit2025.9.3、NumPy2.4.6、SciPy1.17.1；缺sentencepiece/gpytorch/botorch。Qwen fast tokenizer无需sentencepiece，GP用现有SciPy/torch，不安装新库。CUDA/MPS当前均不可用。

未设既有OpenMP运行约定时，torch import触发duplicate libomp abort；使用已有研究惯例 `KMP_DUPLICATE_LIB_OK=TRUE` 并设OMP/OPENBLAS/MKL线程为1后正常。本次不改环境包；保留这项运行可移植性风险。

实际离线载入Qwen AutoModel：463,987,712个base参数、输出1024维、fp32CPU，load约3.14s。strict load审计missing/mismatched/error均空；仅checkpoint多余 `lm_head.weight`被明确忽略，不随机补参数。4条合成feature-only模板（非效果数据）共291tokens，编码0.948s，重复1.537s，bitwise相同/maxdiff=0。混合短/全模板粗外推7680条约30min，实际需按文本长度并跨状态去重，不能把此估计冒称全矩阵执行耗时。详见 [environment_audit.json](environment_audit.json)。

## 外部方法及许可证边界

[GOLLuM正式论文](https://www.nature.com/articles/s42256-026-01283-z)发表于2026-08-28；其PLLM为冻结语言向量后的投影，φ变体进一步训练语言模型。完整主变体PLLMφ不能与本次Frozen+Projection混称。官方代码的[PLLM配置](https://github.com/schwallergroup/gollum/blob/main/configs/pllm.yaml)提供T5/平均pooling/64维Matérn思路；其EI优化目标不替代本任务预测学习目标。

[T5-base模型卡](https://huggingface.co/google-t5/t5-base)列220M参数、Apache2.0；[权重文件目录](https://huggingface.co/google-t5/t5-base/tree/main)单份safetensors约892MB，含多框架仓库总量4.46GB。若将来获准下载，tokenizer另约数MB，FP32完整权重内存约0.9GB外加activation/runtime；encoder-only载入可减少执行模块，但仍需明确下载范围。当前未有下载授权，也未下载。[Qwen模型卡](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)列tongyi-qianwen-research许可，与T5的Apache2.0不同；本研究不重分发权重。来源与核验结论见 [literature.json](literature.json)。

## 缺失与风险清单

- T5对照blocked，本地Qwen结果仅限这一真实模型/模板/pooling，不能泛化到所有语言表示。
- 后期Raw LCMD缺可复用同状态CW；L333已有五seed跨目录CW，早期遗漏造成2945行冗余计算，详见CW_CACHE_AUDIT。Raw/CW validation仍需同checkpoint补提；实际成本按执行回执报告。
- validation已用于checkpoint selection、五seed同源、预算嵌套；不会得到独立确认性校准保证。
- 同方差noise、plug-in超参数、40步优化、投影多初值和scale不可识别可能产生欠覆盖/过拟合；如实输出失败/未收敛，不因训练NLL低就放行。
- Fusion预处理、PCA及输入来源也变化；即使参数匹配仍需分清容量与信息贡献。Qwen1024→64容量比GNN128→64大，专门报告，不隐藏。
- 256U离线诊断是资源明确的固定子池，不是全候选、实际实验效率或AULC。
- 真实labels、cache、graph与权重有部分本地runtime依赖；新克隆不能仅凭git文本完全重放。所有路径从项目root/配置解析，报告不嵌个人机器绝对路径。

全部阶段最终结论需以本研究真实CSV/JSON/图表为准；本审计不预判LLM或GP胜出，亦不授权启动后续AL。
