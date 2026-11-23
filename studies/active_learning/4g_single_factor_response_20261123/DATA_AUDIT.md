# 数据与单因素设计审计

## 正式数据、单位与列角色

- 本地 HEAD、`origin/main` 与本次 fetch 的 `FETCH_HEAD` 均为 `dff586d0d59e6821cee0a83f4fabae5a79a16d0a`；没有 checkout、rebase、清理或终止正在运行的监控和 Hybrid 拟合。分析只在新目录写入。
- 原始 `dataset_4g.csv` 4,243 行；E0 reader-compatible 4,229 行，canonical 4,163 行、217 canonical SMILES。E0 重复实验策略是保留；旧 reader 拒收缺失或 `t1=-1`，剔除 V1 >60 mL 或 V2 >120 mL、不可解析的 SMILES 或失败构图（正式记录无构图失败）。文件中保留少量负值和窗口倒序警告，不能把“被保留”理解成“物理有效”；使用原判定列而非自行重清洗原始表。原始去除原因、metadata 缺失和变量范围详见 `tables/audit.json`，实际用于物理分析的标签数量另报。未投影 test 真值来补全原始有效标签计数。
- 80 行排除中，14 行同时标记 reader t1 拒收及 SMILES 不可用；剩余 66 行超标（仅 V1 17、仅 V2 6、两者 43）。canonical 有 4,132 个完整记录条件，31 个重复组（62 行），220 个 CAS；3 个 canonical SMILES 各出现多个 CAS，没有 CAS 跨多个 canonical SMILES。CAS 作为配对固定键，未替用户订正记录。正式 qualification row 各 seed 为 3,330/416/417；物理观察覆盖 3,746 个可算双标签，严格质量敏感性保留 3,740 个；在完整 canonical 的已记录质量 flags 中 4,155 行无警告、8 行倒序，其中 1 行又有负标签。
- 数值覆盖：EA 记录份额 0.00990–1.000（10 水平），样品量 30–150 µL（8），加载溶剂体积 200–2,000 µL（11），流速 4–10 mL/min（5），密度 0.770–2.182 g/mL。纯度混含等级值如 `AR`，39 个原始水平，范围表中字符串边界不是定量纯度范围。canonical 与 qualification manifest 的行号、sample ID 和 canonical SMILES 全覆盖一一匹配。保留与排除警告对主要比例的影响小：EA 两端下降为 89.6%/86.2% vs 89.8%/86.3%，上样展宽为 84.7% vs 84.9%；这不能检验被正式协议排除的大体积尾部对外部推论的影响。
- E0 从 `t1/t2`（**不是** `verified t1/verified t2`）计算 `V=t_raw×Flow_mL_min/1200`；等效 `t_raw/1200` 分钟，不能按秒直接除 60。例：847、10 得 7.058333 mL。每个合法投影批的 V1 和 V2 与 canonical 原值做双端 `1e-12 mL` 比对。`C=(V1+V2)/2`，`W=V2−V1`，保留符号。仅使用正相 4g。一行真实实验的 V1 与 V2 同源，不能当两个独立重复。
- `PE/EA=a/b` 使用协议和特征构造均认可的混合份额 `b/(a+b)`；它是记录配比对应的 EA 分数，案卷未提供逐批实配或物性换算，不声称为摩尔分数。样品 `V/ul` 与加载溶剂体积是不同的列；样品体积比较固定密度，并输出 `Δ(密度×体积)`，其量纲为 mg。

## 数据分层和记录级控制

- 主事实表：row seed 42 的 train+validation，不取该参考 split 的 test。该描述性队列独立于某个 AL 时点；**其真值不回流到任何 AL 推理、选点或历史方向分析**。历史 AL 仅取当轮合法的 validation 或 receipt 核验过的反馈；其他 seed 的 role 绝不由 seed 42 推断。
- 条件固定键：canonical SMILES、CAS、柱规格、纯度、密度、EA 分数、样品体积、加载溶剂体积、流速、加载溶剂类型；只移除正在改变的目标列。不把结束时间、名称别名、观察标签当作实验控制。固定记录字段不等于固定未记录的装柱、仪器批次和操作者。
- 在每个合法队列内对相同条件的重复行求均值；保留所有 ID。连续水平只比较相邻，类别按名称确定差分方向，只比较类别组合，不能解释为梯度。斜率使用实际间隔；三水平曲率报告不等距二阶差分及中点与弦的实际 mL 偏差。
- `canonical_non_test` 保留合法队列原警告，`coherent` 排除旧质量 flags、负 V1 或负 W；正式模型不针对敏感性子集重训。历史旧 `4g_error_landscape` PE 曲率/梯度定义的分组少了 CAS、纯度、密度，且其数据口径不是本轮同一白名单，保留为历史定性基准而不直接拼接数字。活动学习历史同批旧筛选只固定分子和五模型条件，新表另固定流速/纯度/CAS、合并重复/比较相邻，因此不要求与旧样本数相等。
- 物理主要比例及 MAE：每个分子内先平均，再等权平均分子；95%区间 1,500 次按 canonical SMILES 有放回重抽。少于 5 个分子不报区间；数个相邻差分不能充当独立样本。`true_median_pair` 与散点描述性相关保留 pair 加权口径，不用于独立性推断；各 seed 的验证 split 可重叠，也不把五个 seed 当五个独立实验室。

## 表的入口

`physical_levels/pairs/curvature` 为逐例；`physical_summary`、`physical_by_molecule`、`physical_tpsa`、`physical_structure`、`physical_fixed_solvent` 为汇总；`repeats` 是可见重复组；`interaction_rectangles` 是同分子、同区间的成对差上差，不是干预因果效应。`interpolation` 的内部水平轮留不同于训练内线性残差；只有至少四水平才纳入，端点不外推。异常列表按已有绝对方向阈值 0.5 mL 及反趋势幅度排序，仅作追查，不反馈给模型。
