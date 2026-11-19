# 跨样本预测依赖的实际验证

本文件只评价有限Gaussian代理，不将概率评分、信息积分和真实QGeoGNN Row AULC混同。新OOF数据范围是每个seed合法的原L333；全部held预测来自其余222行训练，不使用当前U、validation或test标签。五个outer划分重叠，fold和行不增加独立重复数。原Stage1/2资格结论及门槛保持原样。

## 评分定义与实现核查

`src/qgeognn_al/active_learning_v2/epig_models.py:470–490`正确区分三个评分：full联合Gaussian NLPD、保留每行V1/V2相关但删除跨行协方差的block diagonal NLPD、完全逐标量diagonal NLPD。独立输出GP的后两者应相等；coupled BLR的差值必须再拆成行内和跨行部分，不能把输出相关带来的改善说成跨样本证据。

新60fit保存每fold完整222×222 noisy predictive covariance及held均值，见`surrogate_evidence/seed_*/F*/fold_*/oof.npz`。这是111行×2输出的OOF验证矩阵，**不是U×U矩阵**；acquisition fullP1仍逐块计算。每fold完整矩阵对称/正定由实际Cholesky和特征值检查，未用未披露jitter修复；原GP预测的tiny负latent roundoff处理仍按原代码。逐fold特征值、非对称程度、联合评分和训练成本保存于`joint_conditional_logscore.csv`。

增加row-conditional logscore：对held的第i行的两个端点联合预测，条件是同一fold其他held行的已知L333标签。若Σ为完整held covariance、P=Σ^-1、e=y−μ，则条件协方差为`P_ii^-1`，条件残差为`P_ii^-1(Pe)_i`。每行评分除以2，单位nat/标量。它检验合法已知L333中“知道其他行后，相关结构是否改善当前行的logscore”，不把这些其他held标签用于OOF基模型fit、特征选择或未来U acquisition。它不是未来单步真实训练oracle，也不是正式效能指标。

新`surrogate_audit.py:116–130`使用Cholesky求解实现；`surrogate_evidence/test_surrogate_v2.py`以独立dense Schur complement逐行复核，并验证跨行独立block covariance时条件增益恰为零。测试4/4通过。原full/block/diag及新conditional分母均为标量数，未把per-row和per-scalar混用。

## 新60fit的实际结果

表中是相同fold/seed权重下的描述均值。`conditional−unconditional`为上述row-conditional分数减去逐行block分数，负值表示相关条件信息改善logscore。

| 配置 | full NLPD | block NLPD | diagonal NLPD | full−block | conditional−unconditional |
|---|---:|---:|---:|---:|---:|
| F0_PCA16_opt | 3.605184 | 3.595327 | 3.595327 | +.009857 | +.019463 |
| F1_PCA16_opt | 3.606784 | 3.594749 | 3.594749 | +.012035 | +.023610 |
| F2_PCA16_opt | 3.618050 | 3.613998 | 3.613998 | +.004051 | +.007931 |
| F2_full_opt | 3.616119 | 3.611757 | 3.611757 | +.004362 | +.008482 |

这些模型在本描述平均上未提供改善的跨样本conditional logscore。该结果只限制这四个训练规则、当前L333三折与此诊断，不足证明所有Gaussian/其他联合模型依赖无用，更不足证明EPIG真实AL无效。单foldfull−block仍有正有负：F0为−.032671到+.154496，F1为−.029384到+.188199；平均掩盖明显fold异质性。没有用15个fold虚构显著性。

各状态noisy covariance数值有效，配置平均最小特征值分别24.2309、26.2816、29.8802、29.4851mL²；逐fold实际值保留在CSV，平均最小值不能冒充整条AL轨迹的数值证明。新模型均独立端点，block=diagonal是预期恒等控制，未声称已经验证真实V1/V2条件相关。

旧Stage1也经只读重算：GP full−block平均接近零且折间异质，coupled BLR的主要NLPD改善来自行内端点相关。证据表为`old_joint_dependence.csv`，不是新fits。Stage2源码`nested_calibration.py:256–316`把温度限在outer训练侧inner OOF；全240fit对应360优化记录，7个inner GP端点noise下限敏感性另见`stage2_optimizer_audit.csv`。这不改变其原qualification失败结论，也不把噪声边界当作联合依赖失败的因果解释。

## 残差形状、边界敏感性与条件校准

`residual_shape.csv`逐seed/endpoint保存NLL相关残差RMS、max|z|、超额峰度、最大4行平方贡献、80%覆盖数、Wilson区间和距旧上限的行数。333行下，旧覆盖上限.90允许最多299行；300/333已越界。新表的`distance_from_old_upper_count=k−299`明确这一离散距离，未把轻微越界改判为合格。

四族的残差超额峰度描述均值12.83–12.93，最大的4行贡献约35.5%–35.7%平方和；同方差Gaussian的中央质量/尾部形状仍失配。95%覆盖均值约.944–.946不能单独证明可靠性。反过来，80%上限轻微超标也不能证明整个联合covariance无效。Wilson95区间仅描述固定OOF预测下有限行数敏感性；数据复用、分子聚类、训练折重叠使其不能当独立二项严格置信保证。

按predictive sigma四分位：

| 配置 | 80%条件覆盖范围 | 标准化残差RMS范围 | sigma与绝对误差Spearman跨状态均值 |
|---|---:|---:|---:|
| F0_PCA16_opt | .785714–.975904 | .387546–1.562407 | −.001881 |
| F1_PCA16_opt | .821429–.975904 | .391893–1.499173 | −.017915 |
| F2_PCA16_opt | .843373–.951807 | .519160–1.424229 | −.062784 |
| F2_full_opt | .843373–.951807 | .516225–1.426037 | −.060025 |

还完整输出Density、V、加载溶剂体积、EA fraction的去重分位分组、溶剂类别和n≥4的分子组；较小分子组不被当独立资格检验。分位切点只依据输入/预测标准差，未用未来真实标签分层。部分低变异因素的quartile因重复切点自动减少，表中保留实际分组数/行数，不人为打散相同输入以制造四组。

这里有条件不均匀性与弱不确定性排序的证据；没有独立验证其原因究竟是常数均值、噪声、未记录条件、核距离还是超参数点估计。下一步若拟合条件噪声，必须训练侧cross-fitting或明确的层次likelihood，并重新报告合法联合covariance；若换非Gaussian likelihood，Gaussian闭式MI不自动适用。

## 输出温度与相关结构

一致可逆`DΣD`同时作用于latent、noise和cross时，同一Gaussian posterior的EPIG及条件greedy决策不变；IVR的trace可能改变端点权重。该不变量已在原stage2测试和新数学审计中检验。温度不能改变相关系数、修复错误均值或Gaussian尾部形状，本研究没有追加全局温度搜索以期待改进EPIG排名。若校准改变了所选family，必须把family变化和objective变化分开解释。

## 与GNN匹配的证据边界

原L333冻结GNN在完整合法outer输入的预测响应，按严格同分子/其余条件相同的pairs，可直接复核原`PREDICTOR_MATCH_DIAGNOSTICS.md`及其NPZ/hash：加载体积/样品体积方向通常远弱于EA比例方向。但这只表示当前已训练函数的响应，不能替代真实物理响应或后续新标签学习收益。

新增冻结encoder128维BLR路线构造了真实跨输入Gaussian covariance：5个seed，64行抽样noisy covariance与noise均正定，EPIG排序相对原GP大幅改变（Spearman描述均值.1472、Top32重合.10）。这为“表征选择会改变信息方向”提供数值可行性证据。由于encoder看过全部L333，该路线不具备honest L333 OOF资格；没有把这些训练残差与GP OOF误差比较来证明泛化误差匹配。后续honest encoder实验需15个训练侧GNN cross-fit，不能假装由现有checkpoint免费获得。

本任务的主要效能证据仍须完整五seed、两臂、21轮的原QGeoGNN scratch轨迹。代理batch MI更大、概率NLPD更小、排名改变与真实Row AULC更好是四种不同命题。新main runner采用预先固定F0 Matérn规则，因此本代理消融结果不会反向挑选主模型。

## 可复现与保存

执行命令及全部源码封存见`FEATURE_AND_SURROGATE_AUDIT.md`；完成后使用只读`surrogate_verify.py`，不要重跑有已披露PCA汇总遗漏风险的原初次执行入口。`surrogate_protocol.json`与`surrogate_followup_protocol.json`分别绑定60fit/5fit规则、代码hash、输入hash与预算。`complete.json`保存CSV哈希，fold `complete.json`绑定模型audit、完整OOF covariance和contract；label_access回执逐seed确认0 future-U/validation/test访问。`unit_tests.xml`为新增4项数值检查回执。所有结论限定于实际已完成范围，原研究目录未被改写。
