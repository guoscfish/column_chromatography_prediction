# 固定标签15fit设计的只读独立核查

判定为 **PASS_WITH_FIXED_VALIDATION_ASSOCIATIONS**。未改动已冻结的`prepare_data.py`、`isolated_models.py`、`train_controls.py`，未启动任何fit。逐seed元数据核验见 [training_independent_qa.json](training_independent_qa.json)、[training_independent_qa.csv](training_independent_qa.csv)，复核入口为 [independent_training_qa.py](independent_training_qa.py)。

## 训练与输入的一致性

- 三模型使用同一seed的H1005，扣除原先固定10个完整体积series后的相同训练行、相同check行、原416个Row validation。
- 历史H1005与旧冻结snapshot逐ID及顺序一致；没有纳入当前轮尚未揭示batch。训练/检查/validation的canonical index、稳定ID映射正确。
- 新A直接构建原QGeoGNN-V2，458952参数；B/C仅各加两个零初始化系数。B使用原typed normalized amount，C使用按稳定ID附着的raw V及outer-train X固定min–max；未从amount列伪造V。
- 同endpoint三个quantile共享pre-ReLU修正，保留原head ReLU及eval clamp。没有额外width参数、loss或Prompt变化。
- Adam、lr0.001、weight_decay0、batch2048、max1000epoch、patience100、逐epoch确定性shuffle与原`train_source`一致；调用同一个原`target_loss`。V1/V2训练loss原尺度与原权重不变；checkpoint仍按原validation combined NRMSE。
- 每seed初始化规则为seed＋2000003，shuffle使用seed＋3000017。数据门已验证共同参数初值、post-init Torch RNG和零skip输出相同；新Parameter使用zeros，不消耗额外随机初始化。
- 原冻结preprocessing、normalization及L333目标尺度字典与新identity完全相同，没有用check标签重新估计尺度。
- optimizer/early stopping只接收train与validation；check图的y保持0，check标签仅用于完成后的评价。旧训练过全部1005行的checkpoint没有作为新A训练结果使用。
- `started.json`采用排他创建，异常不自动重跑，计划清单恰为5seed×3模型=15。最终还需核对全部完成回执，不把当前部分完成状态算为15fit完成。

## 完整组隔离结果与固定validation的限制

对每个seed的**整个历史H1005**，独立按同分子＋非V四条件生成response-series key。所有训练行与check均没有同系列或完全相同记录条件交集；L333也不触及这些check系列。每系列H内所有记录（含完全重复条件ID）全部留出，不只是从pair中挑两行。

| seed | train行 | check行 | train同series重叠 | 原validation同series行 | 原validation完全相同条件行 |
|---|---:|---:|---:|---:|---:|
| 157 | 982 | 23 | 0 | 2 | 0 |
| 887 | 981 | 24 | 0 | 3 | 0 |
| 2357 | 984 | 21 | 0 | 2 | **1** |
| 6101 | 985 | 20 | 0 | 0 | 0 |
| 12203 | 984 | 21 | 0 | 2 | **1** |

这没有把check ID或check标签放进训练/early stopping；但原validation中存在相关条件记录，因此不能称check在物理条件层面与validation完全独立。用户同时要求保持原validation不变，故不修改已冻结输入，不追加fit。关联ID与完整series key仅含元数据，见 [check_series_validation_overlap.csv](check_series_validation_overlap.csv)，可在相同已完成预测上另报排除关联series的描述性敏感性；它不能替换预定主结果或用于选择更好模型。

## 评价口径

主局部指标保持：每seed对检查集体积pair计算各endpoint差分MAE，再V1/V2等权平均，最后五seed等权。主门槛不变；series等权仅作附加敏感性。多水平series组合pair的依赖不应被当成更多独立科研重复。

方向主阈值为真实绝对Δ>0.5mL，预测flat为绝对Δ<0.1mL，另报0.1/1.0mL真实变化阈值敏感性。`evaluate_controls.py`按稳定ID核对预测，使用原`point_metrics`及L333尺度；输出负值、分位交叉及V2<V1频率单列，不把原惩罚项误写成硬输出保证。

A2的阈值和单位与此一致，但A2读取的是已经训练过自身H的原checkpoint，报告已明确为in-sample机制审计。它不能与本轮隔离check成绩混合计算B−A/C−A收益，也不能代替15fit的五seed配对结果。
