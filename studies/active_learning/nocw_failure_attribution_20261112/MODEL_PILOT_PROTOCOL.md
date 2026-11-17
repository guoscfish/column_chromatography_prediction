# 固定标签条件交互试验：唯一模型对照与执行门槛

本协议在新模型拟合之前固定。旧报告用于机制动机，不用于扫描结构。本试验只有一个结构改动，最多16次成功拟合；没有付费LLM、选点、AL轨迹或测试真值访问。

## 1. 固定状态与检查组H

五个历史confirmation seeds按数值升序取前两个：**157、887**。两预算状态固定为 **L525、L1005**，均取NoCW自身L/U与数据身份。不依据任何真实响应、gain、原验证误差或可学习性表现挑seed/预算。

H规则先记录于 `verification/pilot_metadata_rule.json`，再读取元数据计算资格：只从当前L_t中L333之后已采集的行选择；与L333或validation有相同canonical molecule及完整五条件的组全部排除。严格单因素对要求同分子且五条件恰好一项不同，配比使用精确Fraction，数值使用Decimal，不四舍五入制造匹配。

**拟合前泄漏审查修订。** Root用X-only字段发现v1的157/L1005与887/L525各有一条H记录与validation同记录条件，可能通过checkpoint选择间接泄漏。v1规则/资格/表原字节保留为`*_v1`；v2在构图前排除validation同条件整组，其他门槛、seed、模型、排序不变。为补齐旧snapshot缺的一条验证metadata，本次资格统一使用canonical CSV的7个身份/条件列（`usecols`严格限制、`dtype=str`），不加载响应列。修订依据只有X，不依据响应或模型结果。此前为确认输入可用性曾解析授权observed值，未用于H或修订；没有训练或试验结果。

将严格单因素边和同记录条件边构图，端点共享的连通分量整体隔离，作为component；不能把共享实验点的pair计成独立组。至少10个合格component，其中至少2个包含样品上样V对、且覆盖至少2分子。选择正好10个component：先含V对者，再按component有序ID的固定SHA256排序；其余可由PE/EA等单因素component补足，不根据响应大小取组。全部同条件重复行随component一起H。component仍可能属于同分子，报告分别保留component、molecule和row数；统计不把pair当独立重复。

四状态均满足预设资格：

| outer seed | 已获L | 可选component | 含V component | H行数 | 实际训练行数 |
|---|---:|---:|---:|---:|---:|
| 157 | 525 | 43 | 8 | 33 | 492 |
| 157 | 1005 | 127 | 77 | 47 | 958 |
| 887 | 525 | 50 | 15 | 35 | 490 |
| 887 | 1005 | 145 | 72 | 45 | 960 |

这只是元数据资格，不是模型结果。H原始ID、组与metadata SHA已固定。L333与其条件重复组都不在H，原目标尺度可以原样沿用。实际拟合预算始终写成 `|L_t\H|`；不能将结果放回原L525/L1005曲线称相同训练量。

## 2. 唯一结构假设

C1指出当前V2将molecular/backbone pooled表示p与条件completion表示r直接相加：`z=p+r`，再经同一Linear128→6＋ReLU预测。上样量/加载溶剂变量只有r通路；在正输出区域，其加性响应缺少显式分子交互。

唯一预定义修改为：

`z = p + r + r * tanh(layer_norm(p, 128, elementwise_affine=False))`。

这是对现有condition residual的有界分子调制，调制系数为 `[0,2]`。没有新参数；baseline与interaction均为 **458952参数**。LayerNorm只在新交互项中作用于网络内部p，不改变原输入数据、原p路径或预处理；loss、head、branch维度和训练预算全部相同。该对照测试“显式分子条件交互是否释放既有标签信息”，不是直接证明上样响应具有某种化学机理。

既有condition output为零初始化，所以r=0，两个模型初始函数完全相同；所有同名参数及buffer按相同seed逐字节配对。没有选多个结构，也没有同时改loss或scale。该交互同时改变现有r参数的梯度路径与优化conditioning；即使有效，也不能唯一归因于表示容量而排除优化机制。若此结构失败，不继续尝试direct skip、加宽网络或重新调系数寻找正结果。

## 3. 完全配对训练

每状态baseline与interaction各用两个预固定训练replicate，rep=0/1。初始化seed=`outer_seed+2000003+rep*1000033`；训练RNG seed=`outer_seed+3000017+rep*1000033`。两臂每rep共享初始化hash、训练数据及顺序、检查组和validation。rep同时改变初始化/训练RNG，不能用两个rep分离两种方差来源。

训练沿用冻结科学设置：CPU2线程、Adam lr=.001、weight_decay=0、batch2048、maximum_epochs1000、patience100；六输出原loss、V1/V2等权；原L333 target scales、原条件/descriptor标准化；按416行row validation combined normalized RMSE选择checkpoint。H完全不参与训练、early stopping、结构选择或超参调整。两臂从头拟合，原本训练过H的历史checkpoint只用于C诊断，绝不作为本试验初始化或baseline结果。

每状态4fit，四状态总上限16成功fit。执行调度最多两个独立worker，分别只拥有outer seed157或887的8个固定slots；每fit仍CPU2线程，总最多4CPU线程。worker写独立状态/输入manifest，只有所有固定slots完成后由只读summarize全局强检≤16并合并，不通过并发增加科学试验数。

执行前需C1–C3 gate通过，并由主代理写入当前研究授权记录。失败即停，不自动增结构、增seed或增成功fit额度；已完成fit必须按冻结contract复用，不覆盖。

## 4. 标签与数据访问

图仅使用既有 `scrubbed_graphs.pt`，加载后确认所有y均为零sentinel。根据冻结partition角色和ID映射，将合法train与validation响应赋到克隆图；未采集行、H、test图保持零，不把H真值放进模型图缓存。

训练与H响应优先从该NoCW状态的已观测input表及其最后一个已接受feedback取回，先验证contract/input SHA与时间角色、精确ID allowlist。若终点没有下一个input，则用前一轮选前observed＋已接受最后32反馈构成L_t。H响应只保存在local-only内存/评估流程，不公开全量逐行标签。

validation必须先冻结416个验证ID白名单，再通过受限CSV流式读取仅这些ID的两个target字段；非白名单target字符串不数值化、不输出、不分析。禁止test IDs、未知IDs、或任何未采集U ID出现在标签请求。原始CSV不进行宽范围文本搜索，源码检索仅限指定 `.py` 文件。

## 5. 评价与停止

主预测评价是固定416行row validation combined NRMSE，附V1/V2 RMSE/MAE/R²及每rep结果。主机制评价是H的样品V单因素有符号差分误差、绝对差分误差、方向、幅度压缩；PE/EA等合格轴作为控制。先按组/分子归并，报告每outer seed、预算、rep，不把多个pair或16fits当16独立研究重复。

本次只做机制/工程筛查，无五seed统计显著性主张。局部响应变强而整体NRMSE恶化必须完整保留；不能选出有利rep或预算写结论。固定标签对照可说明某些信号是否更可学，不能仅凭两模型失败分解不可约实验噪声；噪声/缺失变量还需复测身份或新测量。

所有输出仅写本新目录，weights、逐行预测/标签缓存、训练日志放local-only `runtime/`，公开表只含聚合指标、ID/hash与授权metadata。完成16次以内合格fit后停止。
