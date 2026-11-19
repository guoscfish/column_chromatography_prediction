# 代码与历史证据审计

审计日期：2026-11-17（Asia/Shanghai）。本文件只审计保存的代码、协议、验证汇总、fit audit 和 epoch history；没有打开候选 U 真值、测试真值或测试逐行预测。历史研究已经反复暴露，本研究只能形成 **development evidence**。历史报告中的停止建议不能代替本次用户明确要求的新 EPIG 研究，也不构成本次自动停止许可。

## 1. 仓库现场与保护边界

审计开始时 HEAD 和 origin/main 均为 `b1f54b5132f671ab1c6068127fe8cc313df5b3ac`。父任务已重新 fetch origin main。存在两个历史监控 JSON 的未提交修改及未跟踪的 `studies/predictor/full_training_volume_response_audit_20261117/`，均不是本子任务创建。本次只在 `studies/active_learning/epig_prediction_oriented_20261117/` 写新文件，不修改、移动、重生成任何历史研究。旧 lock 文件存在不等于进程仍持锁；进程与实时锁资格由父任务资源预检单独记录。

用户列出的六个源码与五个报告路径均存在，未发生目录替代。最新 Free 的真实正式源是 `free_cw_sol61_xhigh_five_to1485_20261101/protocol.json` 和其 runtime；已发表结果位于 `reports/free_cw_final_20261112/`。最新充分训练体积响应审计不在 active_learning 下，而在未跟踪的 `studies/predictor/full_training_volume_response_audit_20261117/`；保留其现状，仅只读引用。

## 2. 六个源码的职责与复用判断

| 文件 | 实际内容 | 本次可复用及边界 |
|---|---|---|
| `src/qgeognn_al/active_learning_v2/ivr.py` | scalar row-kernel 条件化 greedy IVR，float64、Cholesky 初解、Sherman–Morrison 更新、每步 risk 差复核 | 可作历史 Kernel-IVR 精确实现或解析对照；不是已校准 GP，也不是多输出 Bayesian posterior |
| `block_ivr.py` | 同参数两输出 Jacobian block、共同 Frobenius RMS、Woodbury 加点/删点、BAIT式 forward/backward | 可复用稳定更新与数值核验思路；其单位 prior/noise 假设不得继承为唯一新 GP |
| `ivr_study.py` | seal、标签屏障、历史L333复用、105次新scratch fit、checkpoint/selection freeze、最终测试评价 | 可借鉴不可变 receipt、断点与历史初态核对；**不能直接调用其 report 当作本次 validation 主结果** |
| `sequential_runner.py` | 合法 X-only 数据载入，L0/validation 通过 store 分角色揭示，scrubbed graphs，当前模型特征、选择冻结、重训、逐轮审计 | 适合复用原训练与状态转换；默认 test prediction graph 不含标签，但新 acquisition 必须只获得训练侧 X 与 L truth，不能获得 `validation_truth` |
| `benchmark_protocol.py` | 原始训练配置、两个RNG公式、Row输入字段、版本、split和hash校验 | 原训练常数/初始化/尺度必须匹配；旧 `assert_formal_authorized` 绑定旧研究 Commit A，不应伪造或绕过旧研究 seal；新研究另行封存 |
| `cw_ivr_portfolio.py` | LCMD和scalar IVR共享pending context，轮流16+16；两者观察所有前选点 | 可作共享条件化状态和确定性tie的参考；其 scalar IVR 是历史单位噪声代理，不应称 GP-EPIG |

## 3. 历史 IVR 的确切概率假设

历史 Kernel-IVR 每轮从当前 QGeoGNN-V2 提取全网 V1/V2 q50 梯度，分别除以固定 L333 端点尺度，拼接后 CountSketch 到512维。固定参考集合是该 seed 最初的 L0∪U0 全部3330行，成员不随轮次变化；不含 validation/test X。全参考均匀加权。

令原特征为 φ，`s²=mean_R ||φ||²`，`X=φ/s`，先验权重 `w~N(0,I)`，代理观测 `z(x)=X(x)w+ε`、`ε~N(0,1)`。后验权重协方差 `C=(I+X_LᵀX_L)⁻¹`，参考二阶矩 `M=X_RᵀX_R/3330`。每点得分为 `xᵀ C M C x/(1+xᵀ C x)`，对每个 pending 点更新 C 后再选择下一点。精确平局取原 U0 剩余顺序的第一行。

这是**显式 dimensionless、单位 prior precision、单位同方差 noise 的线性核代理**：未用残差拟合噪声，未学习核长度尺度，无边际似然/OOF 概率资格；仅 L 行身份影响条件化，标签数值不进入该协方差。scalar 拼接把一个实验压到一个 scalar latent observation，尽管实际揭示两个端点；因此不是严格双端点联合 GP。q10/q50/q90 也没有决定该代理的跨样本协方差。

block-IVR 保留每实验2×d Jacobian，两个端点共享参数空间；用全参考每实验 Frobenius 能量统一缩放，观测噪声与端点risk权重均为 I。它有跨输出结构这一不同假设，但仍是对 q50 的近似线性化，不是六输出分位loss完整后验，也没有概率校准证明。标量与block绝对risk不能直接比较。

历史机制审计25个冻结状态（5 seeds×rounds 1/5/10/15/20），不新增标签或fit：scalar cross-map512平均batch重合0.5767，512vs2048为0.6871，block跨map为0.6438，均未满足旧研究0.75工程门槛；分数相关仍约0.99。forward/backward仅7/25有正增益，median为0；block在自己的risk上25/25优于同状态LCMD。结论只说明旧表示/代理/门槛，**不能推出 EPIG 的互信息目标无效**，也不能把 raw Gram 低秩当正则求解病态。

## 4. 必须保持一致的正式协议

- 4g canonical数据4163行；Row split每seed outer-train3330、L0=333、U0=2997、validation416、test417；formal outer seeds 为157/887/2357/6101/12203。样本ID与原split逐行对齐。
- QGeoGNN-V2 458952参数，Adam lr=.001、weight_decay=0、训练batch2048、max1000epoch、patience100、V1/V2原始loss等权、每epoch确定性shuffle、validation combined NRMSE选checkpoint。
- 每轮scratch：模型初始化 `outer_seed+2000003`；训练RNG `outer_seed+3000017`；原CPU两线程。不warm start，不缩短epoch，不改网络原dtype。
- X预处理只用该seed完整outer-train X；target scales仅用L333标签std(ddof=0)，之后固定。端点误差单位mL；combined NRMSE=`sqrt(((RMSE_V1/s1)^2+(RMSE_V2/s2)^2)/2)`，不是端点NRMSE的算术均值。
- AULC=`trapz(raw_NRMSE,active_budget)/(end−333)`，保留全部非单调点；主窗口333..1005为22点21批；1485为37点36批。L不含共用416个validation标签。
- 内部原始时间到体积转换应沿用数据导入定义 `t*Flow/1200`，不能在新GP中按常见秒/60自行重新解释。辅助模型如标准化端点，其协方差也须按 `D Σ D` 还原；联合多输出EPIG须声明所处单位。
- validation仅允许原训练checkpoint选择，不能进入 GP监督拟合、噪声/核调参或acquisition；训练侧OOF折只能用L。预测器encoder若已用全L训练，其表示再做residual OOF仅为条件于该encoder的诊断，不能声称encoder也严格OOF。

## 5. 标签访问屏障与新的适配风险

`protocol.py::RestrictedLabelStore` 在 `initial_fit` 只允许 l0/validation，在 `after_acquisition_fit` 只允许已freeze的u0，在 `final_test_evaluation` 要求预测已freeze且角色仅test。其源CSV流式扫描所有文本，但只对wanted IDs转换目标数值；不能用普通pandas读整张含目标表替代该受限接口。

`SequentialSeedContext` 持有 `validation_truth` 是为原训练所需；若把整个 context 传给新selector，类型/对象边界并不能保证不泄漏，应将selector设计为只接收合法训练X、当前L标签与冻结模型预测/表示。选择后先保存唯一、U成员正确的32个ID及hash，再freeze后reveals。不能把某row在另一个seed已揭示的值跨seed接入当前U。

历史 `ivr_study.report()` 在全局freeze后确实揭示test真值，`results/learning_curve_metrics.csv` 与 `FINAL_REPORT.md` 的 AULC .500070、终点 .395913 是**test口径**；该行为属于它原已完成协议，不是本次新报告的validation主依据。本审计没有读取它的test truth、prediction rows或results误差表，而从其保存fit_audit/history恢复validation。

## 6. 已核验、可复用的真实历史 validation 基线

新建 `audit_baselines.py` 从既有综合审计的validation汇总逐点回查fit_audit SHA与checkpoint选择分数，另外从Kernel-IVR每轮freeze定位真实model目录，在history精确取best_epoch。所有记录再用最新CW protocol的固定L333端点尺度独立重算combined NRMSE。结果1117个validation点、130个完整seed-window指标；没有新fit和标签揭示。

| 方法 | seeds | L333–1005 AULC均值±样本SD | L1005 NRMSE均值±样本SD |
|---|---:|---:|---:|
| Random32 | 5 | .614742±.102912 | .513940±.108788 |
| raw Gradient-LCMD32 | 5 | .462549±.067389 | .366328±.038413 |
| historical Kernel-IVR32（本次恢复validation） | 5 | .472446±.079777 | .373631±.072092 |
| Numerical Hybrid32 | 5 | .438420±.067272 | .351918±.039880 |
| 最新 CW-Free32 | 5 | .532876±.072537 | .378752±.042341 |
| 历史 Free32（混合LLM） | 5 | .508802±.082160 | .349855±.030140 |
| NoCW32 | 5 | .497741±.065443 | .405025±.046221 |
| CW-gradient LCMD32 | **2** | .500029±.059870 | .368524±.047338 |

Random、raw LCMD、Numerical Hybrid、Kernel-IVR均兼容同一L333数据/初始化/validation设定；新方法仍必须逐项核对冻结SHA与实际训练配置后复用。raw LCMD、历史Free、新CW-Free五seed完整到1485。CW-gradient LCMD在该统一历史表只有157/6101两seed，不能替代五seedraw LCMD。两种LCMD的梯度定义不同。Kernel-IVR到1005，其105个后续fit已保存，不需重复训练；不能直接借它的后期checkpoint给分叉的EPIG。

产物：`baseline_validation_curves.csv` 保留每seed每预算原始点、端点RMSE/MAE、训练耗时、epoch、初始化/validation/train/checkpoint hash与审计路径；`baseline_seed_metrics.csv` 给各完整525/813/1005/1485窗口；`baseline_summary.csv` 汇总；`baseline_inventory.json` 记录输入SHA及实际checkpoint文件可用性。尚未执行的新GP-EPIG/GP-IVR不在这些历史表中，不以baseline结果填补新实验。

## 7. 最新 Free LLM 的协议和真实结果

`free_cw_sol61_xhigh_five_to1485_20261101` 已完成五seed各36轮/1485，180次被接受且真正重训的batch，185个含L333的模型状态。完整记录保留430份完整响应、173511物理传输尝试；物理尝试不等于付费成功请求，不能推测货币成本。本次不调用任何LLM。

其新运行自由选32、完整lossless候选catalog、CW显式可见，QGeoGNN与L333 seed-matched模型复用规则同上；served lock为Sol6.1/xhigh。历史Free的157/6101前缀混Astra/high后续Sol，另三seedSol，必须单列混合版本；新CW与NoCW仅历史观察比较，轨迹、日期、prompt及纠错行为同时不同，不能当CW可见性的因果消融。

最新CW五seed L333..1485 AULC .454195±.048511，终点 .322703±.031694；raw LCMD为 .417049±.049452 与 .347582±.030467。最新CW长期终点五seed均优，完整AULC五seed均差，二者应同时保留。L525..813的损失对全程差异贡献大，不能事后只报后期窗口；同样不能因早期差就掩去后期优势。

## 8. 联合概率与预测器能力的既有警示

综合审计 `NUMERICAL_SIGNAL_AUDIT.md`：新CW全部5760个已选事件raw80覆盖V1/V2为63.21%/57.29%，末期仍62.08%/55.54%，跨seed异质性明显；q50本身对绝对困难度的排序常强于width，OOF增量亦未证实。边际区间既不保证校准，也不决定跨样本joint covariance。因此不得从独立q10/q50/q90采样估计EPIG，不得把width、GP variance或代理risk直接叫科学信息增益。

`decision_mechanism_followup_20261117/NEXT_EXPERIMENT_DECISION.md` 与最新fully-trained审计均支持上样体积响应压缩这一具体线索，不支持已证实网络容量单因。fully-trained三个Row模型在3330训练行下V1训练幅度比5.58%/51.58%/.71%，V2为5.13%/27.31%/.79%；validation亦压缩。相同记录X的响应差和缺实验批次等元信息是额外混杂。充分训练相对AL1005的同pair差分MAE只微小改善甚至恶化，削弱“只是标签不够”的单一解释。

因此新EPIG如倾向上样体积对照但Row收益差，最多提出预测器瓶颈候选解释；必须与校准、batch冗余、同posterior IVR差异分开。不能读未揭示U真值算oracle gain。若改预测器，须成为独立研究因素，并同时给EPIG/IVR/LCMD匹配新预测器，而非仅为一个方法换模型。

这些最新体积报告自身披露历史标签边界事故与隔离，不能将其称为完全盲证据。本审计只引用已公布聚合结果，未打开该事故源目标文件，也不将其已暴露的目标带入任何新selector。

## 9. 新研究应锁定的区别

历史Kernel-IVR、同一新GP posterior下GP-IVR、同GP的单点Top32 EPIG、同GP的条件化greedy32 EPIG、以及EPIG-inspired proxy须使用独立方法名和配置。GP更强产生的收益不能记到EPIG objective；同posterior、同参考输入、同noise与尺度的IVR对照才识别目标函数差异。独立两端点score求和只对应独立任务假设，不是带跨输出协方差的joint GP。

本报告不作EPIG Go/No-Go。可计算性、资格校准、同状态排名、真实五seed Row学习性能是分层问题；旧IVR停止结果不能提前裁决新研究。五outer seeds划分重叠，配对SD/区间应明确为这一开发队列的描述性不确定性，不冒充五独立化学数据集。

## 10. 复现本审计

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_20261117/audit_baselines.py
```

此脚本仅打开白名单路径的已有validation汇总、协议、fit audit、epoch history和freeze元数据，不打开canonical目标CSV或`predictions.csv.gz`，不写任何历史目录。原始输入hash见 `baseline_inventory.json`。重新运行会重建本研究的baseline派生表，但不修改原始证据。

## 11. 新实现的独立边界复核与修正

对 `study_support.py`、`config.json`、`EXPERIMENT_PROTOCOL.md` 的独立复核发现并在资格评分前修正：

1. 原计划只封存新EPIG代码，不能保护原GNN训练依赖或split；现已将原 `src/qgeognn_al/**/*.py`、`application/*.py`、五seed原split/context/初始input contract/checkpoint/fit audit/history/prediction字节hash纳入封存，并在恢复时核对科学包版本。
2. 跨seed pooled OOF统一选择核会把某seed合法L333标签通过模型选择间接带入另一个seed的当前U；现改为每seed仅凭自己的L333 OOF合格与选family，pooled仅描述。五seed全部有合格模型才允许完整主对照；不删掉不利seed。
3. 若某seed无合格GP但有预注册BLR，应按本seedOOF选择合格BLR备选，而不是无视已合格备选。正式两臂统一名为 `posterior_epig_conditional_greedy32` / `posterior_ivr_conditional_greedy32`，每seed保存确切family；报告必须分清GP与BLR，不能把混合family队列称作统一GP研究。

新 `epig_formal_runner.py` 的selector接口只暴露训练侧X、L目标、固定尺度与身份；保留全局行位置以对齐原canonical ID，但validation/test行及X均掩为None/NaN。新fit通过原 `SequentialSeedContext.fit` 调用原 `fit_from_same_initialization`，只保存当前U的预测，不请求测试输入或标签；原validation checkpoint selection保留。每次选点JSON和独立字节hash必须落盘并核对，之后才允许store揭示。完整fit由state/模型文件hash恢复而不重复拟合，partial attempt迁入incomplete_attempts。唯一OS-held锁限制单worker；Torch2线程，累计210次fit尝试与首次formal启动起40h绝对墙钟（离线时间亦保守计入），中断尝试也占预算。失败可能导致不足210成功fits，必须报告未完成，不私自扩预算。

16项工程边界/恢复测试通过（5.02s，未训练神经网络），包括非法/重复ID拦截、freeze早于reveal、无validation能力暴露、partial保留、预算跨resume、资格门控、BLR fallback命名、已完成零重训、权重篡改与相同ID不同truth拒绝。`initial_state_preflight.json` 另以保存元数据与hash检查五个真实L333模型、配置、split、角色、尺度和history best_epoch，全部通过；0新fit、0新source标签读取。此工程预检不是EPIG真实学习收益证据。
