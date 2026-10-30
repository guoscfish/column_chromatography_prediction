# CW16+LLM16 机制复核协议

日期：2026-10-29（Asia/Shanghai）。对象仅为 4g QGeoGNN-V2 Row split；这是对已有冻结轨迹的探索性观察分析。H1–H4 都是待检验假设，不由历史曲线认定机制因果关系。

## 范围与来源

主共同窗口为 seed157、6101，L333→813，15批选择、每批32点，16个拟合预算。Scientist V2 的 CW16、LLM16、合并32分别列行。Free使用相同seed与预算的真实冻结轨迹。CW-LCMD32 与 raw-gradient LCMD32独立命名，不能互换。历史延长段另列，不参与两seed共同区间均值；Dialog Feedback V1只作为独立来源说明。

来源按冻结contract、batch_freeze、input、catalog、selection、feedback、receipt和fit_audit相互绑定确定，不按最新目录猜测。实际读取文件保存SHA256。候选只读ID、结构、条件、预测和事前几何；观测只使用该方法当时已揭示记录。训练/预测文件可按字节校验，但不解析候选或test真值。记录历史是否已曝光test，不把“本次不读取”误写成历史从未曝光。

## 条件与计数

分子键沿用冻结catalog canonical SMILES；五条件是 PE/EA、Density g/ml、V/ul、loading solvent、Volume of loading solvent/ul。严格单轴对照要求同分子、除目标轴外四条件完全相同，且目标轴不同。逐轴列出V、Density、加载溶剂体积、加载溶剂种类与PE/EA。

loading_amount_mg=Density×V是派生量，不加第六个独立轴。主口径为四个原始loading轴合并、仅对选前L_t匹配、排除已见完整tuple的补全，每个被选点最多计一次；这里的轴合并不代表把batch加入已观测集。prior-or-batch关系union单列为敏感性口径。另保留旧五seed报告的V-only、prior-L、排除已见完整tuple的curve_extension口径作逐行精确对齐。原报告34.65%覆盖五seed至1485，不能直接与两seed至813比较。

每点分别标记与选前L匹配、与整批32其他点匹配、仅L、仅batch、两者皆满足，以及union。Hybrid子来源的batch参考仍为完整32，包含跨CW/LLM对照；pending从未作为已测量L。原始matched contrast与真正补全分开：若分子+完整五条件tuple已在L出现，则记重复测量，不记新条件补全。批内完全相同tuple也单列；各口径的排除规则以脚本字段为准。

宽口径B为同分子、存在loading原始变量差异的局部设计，可同时改变PE/EA或多个loading变量；它不等于严格单轴补全A，也不意味着样本有用或浪费。

CW16与LLM16的比例各以16为分母，完整Hybrid/Free/CW32/raw32各以32为分母。汇总先在seed内计算，再等权列两seed与均值；不把轮次或点作为独立实验重复。报告阶段333–525、525–653、653–813分别对应6、4、5批。

## 几何与反馈

每种方法用自己的当轮冻结模型、L和U。距离分位在当轮U内计算；raw与CW空间独立。无匹配bank则NA，不重提梯度，不拿其他方法bank补齐。覆盖收益在固定事前U比较加批前后最近L距离；包含与排除当前批selected自身归零的口径均披露。Hybrid两来源可在同一空间计算各自收益及加在对方之后的边际收益；跨方法不属于同状态因果比较。

本审计没有核算当前模型的高误差邻域。选点自身当前真实误差需要未测量真值；若另有合法冻结的当前L预测，可定义当前L高残差邻近代理，但本次不以历史采集时残差冒充该代理。这里报告该轨迹选择前已揭示的历史采集时残差邻域，初始L333没有此残差。当前批feedback不能用于构造本轮事前特征；LLM文字理由仅为可复现叙述代理，与真实ID条件核对，不等于意图或信息依赖的因果证据。

## 性能与推断边界

NRMSE沿用各seed冻结L333目标population std：sqrt(mean((RMSE_t/s_t)^2))。validation normalized AULC为对真实非单调NRMSE曲线的梯形积分除以标签跨度，越小越好。阶段差保留全部预算点，不择优截断。validation只用于描述历史性能；不得据反复尝试历史validation挑选K。

不同方法自首轮后即具有不同L_t、训练状态和反馈；版本、prompt、seed、历史修复也可能混杂。n=2配对seed不能支持“显著高于”的常规统计结论；只报告方向、大小、逐seed一致性和反例，不给点级伪重复p值。

## 阶段B决策门与停止条件

完成阶段A后，只有行为证据足以支持一个可区分的loading-only干预时才实现新策略。若预期行为差异不成立或证据不支持统一loading约束，遵守任务要求暂停B，只给反证与更合理的后续方案，不为交付代码而忽略决策门。

任何后续方案保持Free每批32、全部候选、原预测/CW/化学信息、自己的反馈记忆、模型与reasoning、训练初始化split与checkpoint规则。唯一拟干预是严格loading数量及必要审计纠错，不加入CW配额、采集分数或自动替代点。若进入B，只做离线测试和mock dry-run，然后停止。

本次不调用真实LLM，不训练或推理新模型，不访问新标签，不读取新test结果，不覆盖历史冻结文件。目录中的历史分析脚本不能触发训练或标签store。
