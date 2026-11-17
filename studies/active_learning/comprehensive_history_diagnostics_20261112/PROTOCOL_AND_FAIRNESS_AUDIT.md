# 协议、时间顺序与公平比较审计

本研究是历史开发数据的离线诊断。没有新增选点、付费模型请求、QGeoGNN/GP拟合、策略修改或实验信息注入。各计算脚本只处理既有验证汇总、训练元数据、无标签候选特征及已选样本反馈。整个会话的一次搜索越界另见文末，不能以脚本的零测试读取回执冒充全会话零接触。

## 1. 代码与运行状态

2026-11-12（Asia/Shanghai）本地分支 `main`，HEAD `44be2c0070219d71b2560604f8234a390cc14304`。`git ls-remote origin refs/heads/main` 只读确认远端相同；首次沙箱内DNS失败，随后网络只读检查成功。无需fetch或合并，没有切换分支。

开始时已有959个修改/未跟踪文件（逐文件记录在 `verification/repository_start.json`），包括两个持续更新的监视状态文件。新工作只写本目录。进程快照显示五个历史1549观察/恢复/报告监控脚本；未发现当时的CW/NoCW选择或训练worker，但进程瞬时快照不证明未来不会恢复。CW新运行180批已经冻结完成；NoCW归档实际端点为1069/1005/1133/1165/1229，不能将1389目标当成完成。没有停止、恢复或通知任何实验。

## 2. 共同基础与方法身份

五seed是157、887、2357、6101、12203；4g Row split，4163行，outer train3330、初始L333、U2997、validation416、test417。训练标签预算L不包含共用的416个验证标签。Compound只作为行为/误差分组，不改变row目标。跨seed验证行有可能重叠，五seed不是五个独立化学数据集。

主表1088个保存的验证点逐点核对fit_audit SHA、训练行数、best_epoch历史行及分目标指标。每seed八方法可得记录的初始化hash、验证ID hash一致；**只证明初始化/评价集对齐，不能证明后续训练集、checkpoint或LLM输入对齐**。初始L/U与checkpoint相同的证据还包括旧协议审计与新CW的initial_state/contract。每次选点32行，已选ID不重新购买；同分子/同记录条件的不同row仍可重复。

| 名称 | 实际机制 | 不能混同的对象 |
|---|---|---|
| Random32 | 历史嵌套随机轨迹 | 两seed延伸不能冒充五seed延伸 |
| Gradient-LCMD32 | 全网q50梯度512维CountSketch、原始raw几何、corrected LCMD-TP | CW-gradient LCMD |
| Numerical Hybrid32 | 三初始化q50 disagreement归一化，当前U最高25%，member0 latent farthest-first | LLM16＋CW16 |
| Historical Free32 | 全候选交付、LLM自由32、显式CW；混合LLM历史 | 新CW独立重跑 |
| CW-visible Free32 | 全候选交付、自由32，实际served lock为gpt-6.1-sol/xhigh | 不能由名称推断严格NoCW因果对照 |
| NoCW32 | 去掉显式CW字段/说明；保留预测、分位宽度、结构条件、历史反馈 | “没有数值信息” |
| Free16＋CW-LCMD16 | CW先给16个pending，LLM选16；历史混合模型 | 数值Hybrid32、纯LLM32 |
| CW-gradient LCMD32 | center/width梯度几何的LCMD | raw-gradient LCMD32 |

历史Free seed157从L525、6101从L653续用Sol；此前是Astra/high，157 r5最终接受的人工ID抄写修正served identity未知。旧接受响应统计164 Sol、15 Astra、1 unknown只作为已发表审计复用。其余887/2357/12203从L333为Sol/xhigh。Free16＋CW16前缀Astra/high至L813，之后Sol/xhigh。新CW的430份完整响应均由既有审计锁为Sol；模型字符串不能证明供应商内部权重/路由跨日期恒定。

## 3. 训练、尺度与指标

固定QGeoGNN-V2，458952参数，Adam lr=.001，weight_decay=0，训练batch2048，最多1000epoch、patience100，V1/V2原始尺度loss等权，validation combined NRMSE选checkpoint。每轮重新建模，初始化seed=outer_seed+2000003；训练RNG=outer_seed+3000017，CPU2线程。不是warm-start。L≤1549时每epoch只有一个训练batch，因此不能笼统把曲线锯齿称为minibatch随机性。

归一化采用每seed初始L333固定尺度s1/s2：

`NRMSE = sqrt(0.5 × [(RMSE_V1/s1)^2 + (RMSE_V2/s2)^2])`。

`tables/normalization_scales.csv`保存具体分母。1088点全部由V1/V2 RMSE独立重算吻合。不能误用仓库其他旧指标的算术平均公式。所有V1/V2 RMSE/MAE为mL，R²和NRMSE无单位。原始时间的体积换算按项目导入代码 `t × Flow / 1200`，不按普通秒/60自行重解释。

`AULC[a,b] = Σ (L[i+1]−L[i]) × (e[i+1]+e[i])/2 / (b−a)`。

保留每一个原始非单调点；不平滑、不累计最优、不插值或外推缺失轮次。阶段端点同时用于相邻梯形，但面积没有重复计数。gain_t=e_t−e_(t+1)，是“加样本后重训的验证变化”，不是batch纯因果效应。预先固定的.6/.5/.4/.35/.3阈值仅用于本次描述性诊断，不追称原实验预注册；首达与连续三个检查点达标分开，未达保持右删失，不把终点填成首达。

## 4. 同预算、同seed与同状态

六个五seed方法共同窗口L333–1005；CW/历史Free/raw LCMD三者共同L1485。额外两seed CW-gradient、Random历史已到1549（本次补入，不沿用11月3日图的1005截止）；Free16＋CW16终点157=1485、6101=1453，共同1453。任何表中单seed1485的16＋16行只作可用性描述，不能与五seed均值直接排名。配对表自动使用比较双方共同seed和真实预算。

扩展数值方法另表保留MaxDet、U50-MaxDet、gradient/latent fusion、a080 screen、CW/IVR portfolio。它们预算/seed不同，见 `additional_run_registry.csv`；不与八方法主表混成一个排行榜。

相同seed不控制选择历史、模型拟合状态、prompt、纠错历史、运行日期和LLM随机输出；相同预算更不等于同状态。新CW−NoCW与新CW−历史Free均为观察性比较。NoCW在L1005 AULC更好，已经足以否定“增加显式CW必然有益”的无条件主张，却不足以证明CW本身有害。

## 5. 信息角色与输出边界

| 表/变量 | 时间角色 | 允许的用途 |
|---|---|---|
| 候选结构、条件、当前预测/宽度、raw/CW/latent bank | 选前 | 几何/排序诊断，不含候选真值 |
| 已观测L、history memory、假设和引用 | 选前 | 检查当时可引用性和行为 |
| 最终接受IDs/条件对照 | 提案/冻结后、揭示前 | 真实执行的选择行为 |
| selected residual/observed response | 选后揭示 | 事后机制分析，不能当选前信号 |
| validation gain、控制项before NRMSE | 训练后/离线评价 | 解释历史，不供在线selector/critic使用 |
| test prediction freezes | 仅元数据、hash | 不打开预测行或测试标签；不能冒充validation逐行预测 |

公开交付保留聚合统计、可审计ID索引与哈希，不复制完整packet、catalog、请求正文、全量逐行真实响应表或模型权重。科学审计仅保留27条来自已授权且已发表归档selected feedback的定点病例摘录，记录选后揭示角色与来源。读取已选反馈不意味着某row在其他seed中未被选时即可作为选前真值；所有残差都绑定其原seed/轮次/冻结checkpoint。

## 6. 搜索越界记录与限制

核查单位时，一个子任务的宽范围文本搜索意外命中 `studies/predictor/final_4g_qualification/splits/qualification_dataset_manifest.csv` 中3条带响应值的行。没有继续打开该文件或判定行的角色；不能保证这些行不属于某个测试划分。它们未进入任何脚本、表、图、推荐或推断，响应值不写入交付。本次分析数据路径与该文件无依赖。详见 `verification/label_access_incident.json`。

这是对用户严格标签访问要求的执行偏差，不能用后续正确分析抹去。各脚本“零test文件读取”描述的是计算数据流，不是全会话未经授权信息零暴露。修复方式是将后续源码搜索限制为具体 `.py`/报告 `.md` 和已授权目录，独立核查manifest与代码没有引用该CSV。

主要源码/历史依据：`src/qgeognn_al/active_learning_v2/benchmark_protocol.py`、`sequential_runner.py`、`src/qgeognn_al/evaluation/point.py`、新CW `protocol.json`、旧NoCW诊断 `PROTOCOL_AUDIT.md`。精确SHA见总manifest。
