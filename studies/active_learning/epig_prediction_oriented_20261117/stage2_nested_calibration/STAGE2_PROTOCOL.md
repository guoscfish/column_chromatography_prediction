# 第二阶段：有限nested covariance calibration资格验证

2026-11-17，Asia/Shanghai。本阶段在stage1全部60次合法OOF拟合完成、且seed 887/6101未满足原80%区间上界后提出，属于**明确的结果后第二阶段开发研究**。它不修改stage1的freeze、阈值、选模决定、历史结果或已生成的acquisition；所有stage1结果必须继续保留。第二阶段在任何新真实校准结果揭示前另立config、实现/测试hash与seal，只有本阶段seal完成后才可执行。本阶段受用户原研究任务授权；此文不是只列提案，父任务将按新seal实际执行有限资格验证。

## 目的与可识别性

检验训练侧有限样本估计的covariance幅度校准，能否让预定义Gaussian代理在**未参与校准的outer-held-out L333数据**上达到原资格门槛；不预设它必能通过。核心问题是资格与校准，不是从中央coverage改善直接推出QGeoGNN学习收益。

对同一已拟合family，若用正对角D一致变换所有latent、observation-noise和cross covariance blocks为DΣD，保持均值不变，则Gaussian EPIG在该模型下对可逆输出线性变换不变：

\[
\log\frac{\det(D\Sigma_{tt}D)}
{\det\{D(\Sigma_{tt}-\Sigma_{tx}\Sigma_{xx}^{-1}\Sigma_{xt})D\}}
=\log\frac{\det\Sigma_{tt}}{\det\Sigma_{t|x}}.
\]

同一组训练数据、同一family、同一候选/参考集下，逐点及条件化greedy EPIG应在数值容差内相同。它不能因温度调整而产生新的EPIG排序信息。校准可能改变是否达到资格、以及资格后所选family；后者可间接改变EPIG决策，须与纯温度效应分开报告。

IVR的trace下降则变为tr(DΔΣD)，等价于改变端点风险权重。正式QGeoGNN误差和所有代理输入仍以原固定L333尺度为基准，不得再改目标sigma。应报告校准前后的每端点温度与IVR权重变化，不能把IVR目标变化掩盖成仅调整后验可信度。

## 固定模型、折与校准候选

四个family保持stage1定义：RBF GP、Matérn-3/2 GP、独立BLR、coupled matrix-normal BLR。特征只用预定义分子描述符与实验条件，PCA/标准化/响应中心均在各拟合训练子集学习；核/优化上限仍由stage2 config冻结，不扩展模型族、重启数或试验seed。五个outer seeds固定为157、887、2357、6101、12203。

每个seed只可使用自己的L333标签。对每个family：

1. 将L333分成预定义三个outer folds。outer-held-out标签既不参与base模型拟合，也不参与该fold温度估计。
2. 在该outer训练子集内作三个inner folds，各inner模型仅用inner-train拟合，再产生inner-held-out mean和noisy marginal covariance；拼成诚实inner-OOF标准化误差z_e=(y_e−μ_e)/σ_e。
3. 从该outer训练子集的inner-OOF误差一次性估计三个固定候选：identity方差温度T_e=1；每端点边际Gaussian NLL方差温度T_e=mean(z_e²)；central80方差温度T_e=[Q_.80(|z_e|)/Φ⁻¹(.9)]²。代码温度字段记录的是标准差/区间宽度倍率D_e=sqrt(T_e)，不混用variance-temperature与width-temperature。**不加任意bounds，不jitter或替换退化尺度**；只接受finite且严格正的D，退化时明确失败。分位数插值规则及精确seed/折常数以冻结config/实现为准，必须在结果前确定并报告；不得用outer-held-out覆盖反推D。
4. 在完整outer训练子集只再拟合一次base模型；该模型的三个候选共享mean与base covariance，仅用各候选D=diag(sqrt(T_e))变换完整联合covariance。对outer-held-out计算完整OOF概率评分、coverage和跨row联合NLPD。
5. 三个outer folds汇总成每seed、每family、每calibrator的333行OOF。行/fold/seed的原始数值与温度全部保存。

这不是用outer OOF自身直接估计温度后再评分。inner-train行数小于outer-train行数，二者模型样本量导致校准迁移误差，须作为局限披露。identity也在相同nested base模型/outer split下报告，避免把更换fold或模型拟合偶然差异当校准收益。

## 原门槛与本地选择保持不变

仍要求每seed分别、每端点80% coverage∈[.70,.90]、95% coverage∈[.88,.99]、标准化残差RMS∈[.70,1.50]；预测covariance正定；held-out fold full-joint Gaussian NLPD相对其对角版本的加权平均退化每scalar≤.10 nat。不能因stage1只差几行而放宽门槛，也不能把pooled五seed良好替代某个seed失败。

每seed在自己的合格GP family×calibrator中按最低mean marginal NLPD选择；无合格GP才在其合格BLR family×calibrator中同样选择。精确tie-break由config冻结。所有候选按同一规则全部报告，不选最有利中央coverage或AL窗口。跨seed pooled仅作描述，不参与其他seed的选择。

只有五seed各自都有合格本地后验，才允许完整五seed、同后验EPIG/IVR正式配对开始；任何seed无合格模型不得删除，不将三seed结果伪装五seed。正式方法名称及逐seed记录必须说明真实family和calibrator，BLR不冒称GP。未完成或未通过只能说明本有限代理/校准族尚无资格，不能给EPIG普遍NO-GO。

## 预算、恢复与正式训练

资格base拟合上限为4 families×5 seeds×3 outer folds×(3 inner+1 outer)=**240次**。三个calibrator共享base模型，不为每种温度重复拟合。其中outer模型重新拟合的明确目的是构造stage1没有保存的full outer predictive covariance，以正确复核各温度的跨row联合NLPD；这是记账的新阶段计算，不是把stage1完成的资格状态当中断而重复resume。若后续五seed资格齐全，每seed最终校准只对其选定family作三个inner-OOF base fits加一次全L333 fit，五seed合计最多**20次**；本阶段新增完成auxiliary fit上限为**260次**；包括中断重试的累计attempt硬上限为**263次**（至多3次中断额外尝试），墙钟**1800秒**，原有限CPU资源与逐fold attempt账本必须在启动前就位。

每fold完成后保存模型参数/预测/hash/温度/评分；缓存绑定X、合法L333 ID hash、折索引、model/config/source SHA与上游stage1 freeze。重跑先验hash，完成fold不重复拟合。中断attempt计入实际成本，不能把263次attempt硬上限变成无限重试；预算实现和明确中断处理以stage2 config为准。部分结果不得插补或覆盖旧stage1文件。没有新的付费LLM调用。

若五seed资格通过，最终真实性能仍按原458952参数QGeoGNN-V2、scratch retrain、原初始化/1000epoch上限/patience100/validation checkpoint selection、原固定L333标准差、B32、L333→L1005、每臂105次后续拟合执行，同posterior同refs只改变objective；正式fit仍单独受原210次及墙钟预算约束。该成本不能以辅助拟合冒充，也不得用少epoch替代。正式资格失败或未完成时，正式性能仍未知。

## 概率解释、机制与停止边界

这个结果应称**nested校准Gaussian联合代理**，不能声称还是原GP先验/似然下未经修改的精确Bayesian posterior。DΣD保持PSD与相关系数，但温度不修复错误跨row依赖、不修复均值失配、不修复异方差，更不把Gaussian变成尖峰重尾分布。中央80%覆盖改善可能伴随95%尾部、NLL或joint score变差，所有指标须同时报告；保留conditional coverage曲线和outer-fold异质性。

数学消融需验证固定family下EPIG温度不变性和IVR端点权重变化；不能把这一性质称新信息来源。若最终有AL收益，应区分被资格选择的family与objective差异；最终无收益也要区别校准/均值、batch冗余、预测器可学习性与真实效能类别，不从某一个失败否定其他机制。

全部候选一次性执行完或遇冻结预算/依赖/标签屏障故障后，按真实结果作结。不因早期有利或不利选择性停，不新增温度族直到通过，不根据测试或未揭示U标签调整。全部阶段均属于重复暴露的development数据；独立确认仍需新的冻结cohort/实验批次。

全L333最终校准inner seed固定为outer seed+914043。正式阶段D由初始L333训练侧crossfit估计后沿全轨迹冻结；每轮只重新拟合base代理。L333资格不等于L525/L813/L1005校准迁移已验证，这一状态漂移是正式结果必须保留的限制。
