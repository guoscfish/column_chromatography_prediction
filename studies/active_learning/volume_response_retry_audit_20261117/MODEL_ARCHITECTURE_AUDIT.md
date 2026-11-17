# Original QGeoGNN-V2条件通路审计

**原模型包含上样量信息，但没有独立Density或原始样品体积V输入。样品体积只能通过Density×V的量进入条件补充分支。** 这不等于当前观测域中的体积必然不可辨识：绝大多数分子的Density固定，且没有发现同分子、其余条件固定而不同体积产生相同/接近量输入的实际别名。

本轮没有修改原模型、runner、checkpoint或Prompt。源码与输入SHA见 [sensitivity_manifest.json](sensitivity_manifest.json)、[sensitivity_summary_manifest.json](sensitivity_summary_manifest.json)。下述是源码事实、无标签元数据统计和零训练检查；三个新模型的训练结果另见主报告。

## 1. 图输入的真实顺序与单位

`data.build_model_data` 构建的edge_attr为3个类别bond特征，之后依次为：bond_length、6个洗脱剂描述符、loading solvent编码、Density×V、加载溶剂体积。样品体积V与加载溶剂体积是两个不同因素，不能按字段末尾的volume字样混用。

Density的g/mL乘V的µL，在此数值关系下等于**mg上样量**。例如1.2g/mL×100µL=120mg。原始V及Density没有分别保留在模型输入中。研究所附`Data.sample_volume_ul`是为独立C模型按稳定sample ID提供的额外字段，Original不读取它；不能从量所在的edge列伪造原始V。

条件量与加载溶剂体积采用每seed冻结的outer-train X min–max（3330个无标签条件行）归一化；不是用新检查集标签拟合。V1/V2损失归一化尺度则来自L333目标，是另一个不同的尺度契约。

## 2. 两条条件路径

Legacy backbone的`BondFloatRBF`按五个名字枚举前五个连续位置，实际取到bond_length和前四个eluent描述符。其历史变量名`prop/e/m/V_e`不能被按名字当作真实Density、amount或V；量所在的第8个连续位置没有进入该RBF路径。

`TypedConditionCompletionBranch`从完整连续字段提取其余两个eluent描述符、amount与加载溶剂体积，再拼接4维loading-solvent embedding，组成8维输入，经过 **Linear(8,16)→ReLU→Linear(16,128)**。128维残差加到五层图网络的global sum pooled表示后，才进入共同预测头。

该分支末层weight与bias在原模型初始化时全为零。最初输出保持旧路径一致，末层尚未更新时，更早隐藏层/embedding的梯度也会被零末层阻断；这是一种初始化性质，不能据此宣称训练后分支仍然为零。A2中所有原冻结模型的末层权重均已非零，hidden也有激活。

量的信息能否产生足够输出响应，还取决于补充分支、128维融合及head学习到的权重；没有一个源码定理保证量必须被充分使用。global sum pooled分子表示与单次图级条件残差的尺度不同是可诊断线索，不是唯一瓶颈的证明。

## 3. 六个输出、截断与loss

共同128维表示经一个Linear(128,6)，输出顺序固定为V1 q10/q50/q90、V2 q10/q50/q90。六个head行有各自参数，但共享上游结构；之后统一ReLU，eval模式再clamp到[0,1e8]。

训练中每endpoint包含q10/q90 pinball、q50 MSE及相邻分位交叉的ReLU惩罚，两endpoint相加；交叉惩罚不等于硬排序约束。原模型也没有强制V2≥V1。B/C共享端点残差必须接入pre-ReLU六维线性量，并保留上述顺序、ReLU与eval clamp；共享残差不新增独立width参数，但经过原非线性后，最终区间宽度仍可能变化。

## 4. 无标签元数据的可辨识性

统计对象为旧审计保存的4162行无响应元数据、217个canonical SMILES；它覆盖本轮全部H，不能冒称重新读取了完整4163行真值表。

- 215/217个分子在该元数据中只有一个Density值；两个有多个Density值。
- 全局有11个float32 amount值对应多个不同Density/V组合，说明乘积在数学上不是全局一一映射。
- 但在**同分子＋固定PE/EA、loading solvent及加载溶剂体积**的实际候选域中，不同V的exact amount别名为0，amount差≤0.1mg的近别名也为0。

因此全局质量碰撞不能被当成模型输入完全相同：分子结构本身仍不同。对大多数分子，Density可由分子身份的经验对应关系间接确定；模型原则上可学习量到V的关系，但这可能比显式V通路复杂。当前证据支持比较“已有量通路是否弱”与“独立V是否更易学习”，不支持预先认定大量实际信息已不可逆丢失。

表格见 [metadata_density_by_molecule.csv](metadata_density_by_molecule.csv)、[metadata_global_amount_collisions.csv](metadata_global_amount_collisions.csv)、[metadata_same_molecule_near_aliases.csv](metadata_same_molecule_near_aliases.csv)。最后一表保留空表schema，表示该阈值下零案例。

## 5. 安全数据通道与构图核验

不实例化可能加载完整数据/标签的RowSeedContext。已有每seed `scrubbed_graphs.pt` 的y全部为零，cache SHA、partition hash和canonical_index与稳定ID映射可核验。原始分子图/descriptor cache不含实验响应；使用旧元数据、冻结scaler和明确y=0即可调用原 `build_model_data` 重建。

本轮A2先按样品ID核对新构图与盲图特征及预测，再做完整条件变化。AD用从物理条件重新组装所有edge字段的可微构图器，并与原构图器及有限差分交叉检查；没有只改一个中间tensor便假定实验条件改变正确。

原validation可直接由`RestrictedLabelStore`对对应416个validation ID以`initial_fit`权限流式投影；不能加载全canonical targets。已揭示H标签来自本seed当轮observed记录，逐ID与H核对。本子任务没有调用该validation读取，也没有新增fit；训练子任务独立管理已授权15fit上限。

静态AST检查在相关仓库模型/data/schema及QGeoGNN/utils的模块顶层未发现直接读取CSV/np.load/torch.load数据的调用；这不等于证明所有第三方库导入内部行为。使用CPU构建函数会设置历史qg.device为CPU，未触发其训练/预测应用入口。
