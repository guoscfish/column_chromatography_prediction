# Loading-only 后续设计与暂停决定

**状态：阶段B暂停；未实现、未注册、未运行新的loading-cap策略。** 这是用户规定的阶段A决策门的结果，不是待批准的正式实验。最终数值与反证见REPORT.md、loading_mechanism_findings.md。

## 为什么不直接实现

Free全批loading比例较高与Hybrid内部LLM更少loading是两个不同命题。后一个命题在两seed共同窗口没有稳定成立；Hybrid全批比例下降可主要由其16个CW点组成解释。后段还出现Hybrid loading更多的反例。强制CW既改变局部条件比例又主动增加覆盖；只限制loading不保证获得同样的覆盖。

因此现有数据不足以把“Free可能过度局部采样”收窄成“统一限制严格loading数量就应改善效率”。这不证明loading有效或有害，也不禁止未来检验；它意味着此时不能以Hybrid优势为由启动闭环实验。

## K的处理

**当前不冻结主实验K。** 若之后单轮同状态诊断支持这一干预，可以比较两个事先说明强度的设计候选：K=8（32点的25%）与K=12（37.5%）。这些候选针对prior-or-batch关系union，区别于主分析prior-L口径；尚不注册。这些只是行为敏感性候选，不是已经选定的主Arm，也不是由历史validation拟合得到的最优值。

K=8约落在这两条Free历史轨迹严格union批次数量的中位附近；K=12更接近高批次数量的上四分位，只限制局部采样最集中的尾部。正式采用前必须选定一个K并冻结，而不能把多个K的闭环validation中最好者事后当主结果。历史次数分布与每个候选会触及哪些批次由汇总脚本输出到cap_behavior_sensitivity.csv；它只读取冻结行为，不读取性能来优化K。

## 更合理的下一步

先准备同一seed、同一冻结L_t、同一checkpoint、同一候选池、同一LLM版本与reasoning的单轮配对选择。状态覆盖L333、L525、L653并分别处理两seed；状态在比较前固定，不能按某个K的validation获益选状态。对比原Free32与一个预注册loading cap，先看约束是否只改变局部条件，是否获得raw/CW覆盖、是否引入更多完整tuple重复、以及LLM纠错代价。单轮不揭示新标签、不训练；结果不能代替长程AULC。

当前任务不执行这一步的真实LLM调用。也不引入化学、CW、覆盖权重或多样性配额来“帮助”cap工作。若单轮显示只减少loading而未改变覆盖，loading-only作为解释Hybrid优势的依据会进一步减弱。

## 若将来通过决策门：唯一干预的精确定义

Arm A为Free LLM32 Unconstrained；Arm B为相同Free32加固定K。两臂均保留完整QGeoGNN预测、quantile widths、CW分位、分子描述符、原始条件、全部候选、各自反馈和科学记忆。冻结相同模型服务版本/reasoning/transport、相同初始化split/归一化/训练与checkpoint协议。L333可复用同一个冻结模型。后续分歧是各自轨迹结果，不跨臂共享观测。

沿用同一原Free prompt及校验器，B只追加如下约束说明（K在注册时填入）：

> Of the 32 experiments you choose, at most K may complete a strict matched-loading contrast: the same molecule with exactly one of V/ul, Density g/ml, loading-solvent volume or loading-solvent type changed and all other recorded conditions identical, relative to observed L_t or another row in this selected batch. Count each selected row once, exclude tuples already observed in full, and do not treat Density×V as an independent axis.

全批关系按无序最终32点同时计算，不能按返回顺序规避；批内两个端点可能都计数。报告同时保留prior-L口径以对齐历史五seed。必须提前冻结到底采用prior-only还是上述prior/batch union；上述union更符合防止把loading全部转成批内对照的研究目的，但比旧V-only/prior口径更宽，不能隐瞒此差别。

主干预之外不增加CW配额、acquisition score、LCMD推荐要求、探索利用权重、模型结构或自动替代点。完全tuple重复作为监测指标报告，不暗中加一项新配额。

## 合法性、纠错与fail-closed规格（未实现）

1. 先核验全池输入、L/U隔离、32个合法唯一ID、schema、packet绑定和实测证据ID边界，再由不接触标签的纯元数据函数逐点计算四轴loading union。
2. B超K时给明确错误：K、实际计数、违规selected IDs、匹配axis、prior或batch关系；保留整个原池和模型输入，要求LLM重新返回完整32。不得静默删点、自动用CW替换或生成validation/test驱动候选。
3. 两臂冻结同样的总返回预算，例如首次+最多2次纠正。B只是增加loading规则违反这一错误类型。每个请求、失败与纠正按次记账，重启不重置限额；耗尽则写失败且不freeze合法批、不揭示标签、不训练。
4. 若相对L不属于loading的候选少于32−K，这是全池无法满足cap的一个必要条件失败，可立即停止。反之并不保证可行，因为批内配对会增加计数。只有元数据组合可行性检查给出合法32集合才称存在证书；它仅证明可行，不向LLM推荐/替换选点。无证书或超资源上限则停止，记录“未证实可行”，不能误报数学上无解。
5. 宿主不能自动以已见tuple重复填充“非loading名额”，也不从validation/test误差选择替代候选。若要进一步禁止重复，必须单独披露为新的干预并重新决定实验问题，不能悄悄加入本loading-only主实验。

## 正式闭环评价与资源预算（条件性，尚不启动）

终点固定1005；333→1005为21轮，每臂22个拟合预算。主指标validation AULC333–1005；预先列重点窗口333–653，次指标固定预算NRMSE、loading分布、raw/CW覆盖、批内分子与tuple重复。所有指标逐seed报告；旧Free五seed和Hybrid为历史参考，不能替代同版本Arm A。

| 假设规模 | 新训练（复用各seed共同L333） | 基础LLM选择请求 | 若每轮首次+2次纠正的逻辑请求上限 |
|---|---:|---:|---:|
| 2 arms × 2 seeds × 21轮 | 84 | 84 | 252 |
| 2 arms × 5 seeds × 21轮 | 210 | 210 | 630 |

如每臂独立重新拟合L333，分别为88或220次训练；若每seed只新拟合一次共享L333，则为86或215次。每次训练是原单predictor的从头fit，不含未授权新增ensemble、重试、梯度bank计算或终点延长。传输失败重试须另冻结并单独计费，不把上述逻辑请求上限冒称服务商实际调用上限。未来仅两seed一次状态的配对选择为4个基础请求、0次新训练。上述预算不是本次已发生的调用。
