# 特征与概率代理：实际代码和有限消融

本文件属于 **post-stage1/stage2 exploratory development study**。完成了预先冻结的60个真实L333三折OOF辅助模型fit及单独冻结的5个encoder条件诊断fit；没有新增GNN训练，没有读取任何未来U、validation或test标签。新结果不更改原Stage1/2资格，不用于事后更换主AL代理。证据等级：**E**为已执行并保存数组/哈希的数值结果，**S**为直接源码审查，**U**为仍待验证的机制。

## 逐项审计

| 判断、源码位置 | 实际设置 / 预期 | 影响、验证和证据 |
|---|---|---|
| 原特征，原`study_support.py:94–124` | 16分子描述符、6洗脱剂描述符、3溶剂one-hot、Density×V、加载溶剂体积及二者log1p，共29维；没有独立Density/V | 这是有限的表征选择，可能限制条件归纳偏置和外推。F1明确增加Density及V，保留乘积和其余坐标。不能仅由“用了乘积”断言本数据已经发生信息碰撞；见下述实际检查。S+E |
| 单位，`data.py:163–172`、`protocol.py:173–179` | 输入Density列标为g/mL、V及加载溶剂体积为uL，标签`V1_ml/V2_ml`为mL；Density×uL的数值在这些单位真实成立时等于mg | 代理输入逐列标准化，常数单位换算不改变标准化坐标；目标尺度、noise、cross须整体还原到同一输出单位。列名不能验证原始实验记录的物理含义，未擅自重新解释或修改原数据。S |
| 比例字符串，`data.py:140–154` | float解析`PE/EA`，除以两项和，形成六维溶剂描述符；旧函数本身未检查负值/零和 | 当前输入10种合法字符串，正数比例均有限；新审计显式拒绝负值和零和，并按EA/(PE+EA)归一。现数据没有坏值，因此这是通用输入校验风险而非本次已发生错误。E |
| 变换，`epig_models.py:55–81` | 标准化仅用fit的训练侧X；SVD取方差最大的前16方向，不是低方差方向；零方差列scale=1 | PCA保留总方差不能保证条件方向或监督信息保留。新输出逐坐标投影能量及完整维对照；不能把投影能量当监督相关性。S+E |
| 均值、输出尺度，`epig_models.py:35–46,92–104` | 默认常数经验均值，训练折ddof1输出尺度；均值/noise/latent covariance按相同尺度回到输入y单位 | 现实现没有OOF输出尺度泄漏，但经验均值及超参数的不确定性没有积分。常数均值并非唯一可用路线；旧BLR已经提供非平凡线性均值。新GP消融未独立改变均值，不能声称已经验证线性/非线性均值修复。S+U |
| 核、优化、噪声，`epig_models.py:140–258` | 独立V1/V2、isotropic Matérn/RBF、每端点同方差Gaussian噪声、60iter/1start；精确条件化在empirical-Bayes点参数上进行 | 独立输出不等于跨输入独立；核产生合法跨样本latent covariance。但同方差/固定超参数可以错配条件误差和重尾。新预算200iter/3starts；条件异方差、ARD、稳健似然没有被虚报为已运行。S+E+U |
| BLR输出耦合，`epig_models.py:263–355` | 线性后验；coupled模型的权重先验和噪声共享同一个B，B是训练中心化y的收缩协方差 | B提供真实输出相关性；完整双输出Gaussian MI在可逆输出白化后可消去这个公共B，不能把其边际/行内NLL改善归因于跨样本EPIG决策创新。S |
| OOF，`epig_models.py:394–511` | 分折、标准化、PCA、center、scale、核及噪声全部在训练侧；每fold held只评分 | 该函数不能修复上游监督encoder泄漏。新60fit沿用原三折与seed偏移，使OOF差值可对照；fold/endpoint不是新的独立outer重复。S+E |
| nested calibration，原`stage2_nested_calibration/nested_calibration.py:256–316` | outer222中做inner三折，温度只由inner-OOF产生，再用outer222拟合模型，outer111仅评分；三个scheme共用同一outer posterior | 审查未发现outer-held标签用于温度。DΣD一致变换不改变同一posterior的EPIG；更多全局温度搜索不能修复排名。S；原测试中的held-y扰动检查亦保留 |

## 输入结构：确定风险与实际数据分开

`surrogate_evidence/input_structure.json`与`aliasing_audit.json`实际使用feature-only输入：4163行、217分子、3829个不同完整条件，643行处于至少重复两次的完整条件组。未查看这些重复条件的标签，不能据此断言噪声、错误记录或不可约误差。

Density范围0.77–2.182（151值），V为30–150uL（8值），加载溶剂体积200–2000uL（11值）；EA归一比例10级。217分子中仅2个有多个Density值；3829个不同F0向量中，**相同完整F0但Density或V不同的组为0**。因此，在这份实际有限池内，“乘积编码已经合并了不同Density/V条件”没有实证支持；F1更合理地检验表达方式与外推能力，而不是宣称恢复已证实丢失的标签信息。

16维PCA在原F0 folds保留98.87%–99.22%标准化总方差。新F0中，amount轴投影能量均值0.4963、加载溶剂体积轴0.5633；新F1的独立V轴0.4504、Density轴0.8624。高度相关的条件轴仍可在共同子空间中编码，以上数值不能解释为相应比例的预测信息损失。完整维对照才是本次实际可检查的干预。

## 冻结的小模型族与预算

`surrogate_protocol.json`在任何新fit/标签读取之前绑定脚本、源数据、原freeze和以下四组。每组全5seed×3fold，共60模型fit；每fit双端点×3starts，共360端点优化启动。运行活动12.297s，fit自身总耗时9.265s，macOS `ru_maxrss`为580,124,672字节；不将导入/font-cache等待冒充模型训练时间。有限上限为60fit/1800活动秒，无自动重试。

| 配置 | 表征 / 变换 | 优化 |
|---|---|---|
| F0_PCA16_opt | 原29维、逐列标准化、PCA16 | Matérn32，200iter，3starts |
| F1_PCA16_opt | F0+独立Density/V，共31维，PCA16 | 同上 |
| F2_PCA16_opt | F1标准化后，分子16/洗脱剂6/溶剂3/加载量6各乘1/sqrt(组维数)，再PCA16 | 同上 |
| F2_full_opt | 相同分组权重，保留31维 | 同上 |

权重在标准化**之后**施加，避免后续逐列标准化把权重抵消。`surrogate_v2.py:20–53`实现并经独立合成测试验证；所有训练方向仍只用训练折。

原Stage1的60模型fit中，90个端点/BLR优化记录90/90成功、无noise边界。原Stage2的240fit对应360优化记录360/360成功，**7个inner GP端点命中noise下限exp(-11.51)=1.00293e-5**，outer记录无命中；均发生在较小inner训练折，涉及157/6101/12203。不能把Stage1“无边界”扩展到nested inner，也不能由这7项单独解释所有资格失败。逐项数据在`stage2_optimizer_audit.csv`。旧缓存未保存梯度，不能事后宣称已审查旧梯度；新360starts全部success，最大nit26，梯度max-abs的中位数1.744e-4、最大0.006115（包括边界梯度，非普适最优性证书）。

## 实际OOF概率结果

以下是5seed×2endpoint等权描述均值，NLPD为nat/标量，行内RMSE/MAE另存逐seed CSV，不把V1/V2不同物理幅度混合为一个RMSE。没有把端点或fold当独立重复。

| 配置 | marginal NLPD | 80%覆盖 | 95%覆盖 | 标准化残差超额峰度均值 | 最大4行对残差平方和贡献均值 |
|---|---:|---:|---:|---:|---:|
| F0_PCA16_opt | 3.595327 | .898198 | .945345 | 12.8912 | .3567 |
| F1_PCA16_opt | 3.594749 | .897898 | .945646 | 12.8307 | .3573 |
| F2_PCA16_opt | 3.613998 | .900000 | .944444 | 12.9326 | .3558 |
| F2_full_opt | 3.611757 | .900601 | .944745 | 12.8986 | .3550 |

F0增强优化相对原Matérn的最大端点OOF NLPD差值绝对值2.94e-6、RMSE差值4.52e-6mL，基本复现原解。因此当前证据不支持“60iter/单起点导致这批概率失败”；这也不证明所有未来L状态都能达到好最优解。

增加Density/V对平均NLPD仅−0.000578nat，F2权重的平均NLPD较高，完整维较相同权重PCA略低。重尾与中部过覆盖仍存在。不能由很小均值差宣称概率可靠、挑选新赢家，或把本有限族结果外推为所有特征/噪声模型无效。

## 实际排序敏感性

另行冻结`surrogate_followup_protocol.json`后，恢复已完成fold0参数及训练222行的后验，并严格复现其已保存held均值/协方差（容差2e-10）；恢复过程没有调用fit/minimize、没有新增优化。四族共享相同L222、原U2997、全P1的3330个X。所有pair积分逐块128×128、最大协方差请求256×256，未存U×U矩阵。该诊断**不是主AL的全L333后验**。

| EPIG单点比较 | 5seed Spearman均值 | Top32重合均值 | Top32重合范围 |
|---|---:|---:|---:|
| F0→F1（独立Density/V） | .968777 | .74375 | .50000–.93750 |
| F1→F2（条件分组权重） | .915772 | .65000 | .43750–.84375 |
| F2 PCA16→完整维 | .998271 | .93125 | .90625–.93750 |

这证实特征/距离规则可改变同一训练标签状态下的采集排序，尤其Top32；没有证据显示这些差异改善真实GNN学习。IVR/BALD也在相同参考输入与各自合理目标下完成，逐seed值见`followup/ranking_comparisons.csv`，BALD不依赖目标积分。

## 与QGeoGNN相关的合法联合协方差路线

原`models/qgeognn_v2.py:64–71`公开128维`extract_representation`。新增5个seed各复用已哈希确认的原L333 checkpoint和原scrubbed graph cache；逐图验证y为零、canonical index一致，提取outer3330表示。在该表示上拟合PCA16独立Gaussian BLR，明确生成跨输入协方差，而不是把q10/q50/q90当相关结构。

**这些encoder用全部L333监督训练过，所以这个BLR只属于“条件于已训练encoder”的可行性诊断，未做OOF、未报告资格通过。** honest OOF必须每个held折重新训练对应222行encoder（共15个真实GNN fit），或者采用严格无标签表征；本65fit辅助预算没有暗中实施这些GNN训练。当前结果未将全L333监督特征伪装为独立OOF。

5个BLR的64行抽样noisy covariance最小特征值0.008802–0.035405，noise最小特征值同范围；这是数值有效性证据，不是全数据依赖可靠性或校准证明。与旧F0同seed全L333 GP相比，EPIG单点Spearman均值.147158、Top32重合均值.10（0–.4375）。因此与GNN相关的代理可能大幅改变排序，但其可靠性和真实学习效用完全未证实，也不能将换后验的变化归为EPIG objective贡献。

本方向5fit及30个完整积分score状态共活动47.512s，峰值RSS1,275,625,472字节，预算5fit/300s；其20个fold恢复模型与5个旧GP复用不计作新训练，但明确列在完成回执中。

## 未完成的模型因素与复现

本次没有声称跑过ARD、条件异方差、Student-t/混合似然、训练侧cross-fit encoder、独立非线性均值或这些路线的正式AL。旧BLR与新输入分组提供有限对照，不能消除上述未决。非Gaussian路线若后续实施，必须重新验证MI估计；全局温度不能修复同一Gaussian posterior的EPIG排名。

完成后的推荐复核入口是只读验证，不重新生成科学汇总：

```bash
/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/surrogate_verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python -m pytest -q studies/active_learning/epig_prediction_oriented_v2_20261117/surrogate_evidence/test_surrogate_v2.py
```

实际首次执行命令为相同环境下的`python .../surrogate_audit.py`和`python .../surrogate_followup.py`。如需从零重跑，应在隔离副本中保留这两个已封存源码/协议而使用空的新结果目录；不得覆盖本目录的已完成证据。

审计发现初始`surrogate_audit.py`虽能复用fold模型，但再次运行汇总时，PCA方向表只收集新fit，会漏掉复用fold。**未实际再次执行，没有污染当前完整1365行方向表。** 保留原封存源码及其hash，新增`surrogate_verify.py`作为完成后安全入口；它已实际核对219项文件hash、60fold/360starts/5encoderfit账目，0新fit、0标签读取。这里不把“能复用fold”夸大为整个旧汇总入口已无瑕疵。本问题不影响主AL runner，其缓存逻辑有独立测试。

4项新增测试全部通过，包括新旧Matérn后验等价、分组顺序、precision条件logscore对dense Schur oracle及独立block零条件增益。执行回执见`surrogate_evidence/unit_tests.xml`，科学完成回执分别为`surrogate_evidence/complete.json`和`followup/complete.json`。
