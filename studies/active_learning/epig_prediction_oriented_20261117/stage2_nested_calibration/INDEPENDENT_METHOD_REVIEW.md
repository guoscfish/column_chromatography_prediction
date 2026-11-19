# 第二阶段独立方法与训练适配复核

本复核在第二阶段真实校准评分前完成；不使用任何新的U/validation/test标签，不改变stage1源、阈值、资格结果或冻结文件。复核对象是有限nested校准的科学边界及原QGeoGNN运行器的隔离适配，**不是EPIG效能确认**。

## 科学边界

- `run_nested_family` 的每个outer-held-out样本，其预测均值/协方差仅来自outer-train拟合，温度仅来自outer-train内部inner-OOF残差。outer-held-out y只进入评分，不进入温度或base fit。所有transform、PCA、响应均值在相应fit的训练子集估计。每seed的模型/温度选择只使用自己的L333，不以跨seed pooled结果决定另一seed的策略。
- identity、NLL和central80三个候选共享每fold的同一base预测；不能把额外拟合次数写成三倍。SD倍率字段D与方差倍率T=D²已在协议中区分。final L333 inner seed的代码和config统一为 `seed+914043`。
- `TemperaturePosterior` 作用于全部latent、noise及cross covariance，均值不变。Gaussian EPIG在可逆DΣD下的det/Schur比不变，条件化greedy同样不变；因此纯温度不新增EPIG排序信息。IVR trace的端点权重会变化，协议已明确这个作用，目标NRMSE分母仍是原L333固定尺度。
- 原stage1的80%/95%覆盖、标准化残差RMS、正定与joint-NLPD门槛未放宽。每seed合格GP优先，否则合格BLR；全部五seed都合格才允许正式对照，不能删除不利seed。此阶段是看过stage1结果后明确注册的development follow-up，不能追称独立确认。
- 有限预算现已统一：资格240次完成fit，资格通过后最多20次final L333 fit，最多260次完成fit；含至多3次额外中断尝试的实际attempt cap为263，累计墙钟1800秒。三个温度不能触发无限重拟合。formal另保留210次辅助尝试与210次原GNN尝试、40h、单worker两线程。

## 隔离适配

直接向原 `run_formal(callback,study=stage2)` 传新路径仍不足以保护第二阶段源，因为原runner每轮只会调用stage1的 `support.verify_freeze`。新增 `stage2_formal_adapter.py` 按以下方式补足而不修改原训练：

1. 核对原runner字节与stage1 seal后，以importlib加载私有实例，不修改原文件或原module globals。
2. 私有support facade每次先验stage1 seal、再验stage2的files/data/packages及parent-freeze绑定；所有runner派生写入只允许落在独立stage2子目录，越界或symlink逃逸拒绝。
3. 原历史BASE仍为只读L333来源；初始化、架构、训练配置、split、角色、尺度、checkpoint和history检查继续由原runner执行。实际神经网络训练仍经原 `SequentialSeedContext.fit` → `fit_from_same_initialization`，没有复制或简化训练实现。
4. stage2门槛保留plain family字段与 `posterior_*` 方法名；callback从本stage决策读取family和冻结D，所有新state、selection、label audit、checkpoint、partial attempt和计算账本落stage2 runtime。
5. stage2 driver保存finalization对qualification-decision与每seed温度文件的SHA绑定；每轮callback核验，防止D在后续预算漂移。执行锁与累计phase deadline已加入，正式辅助fit也在执行前记attempt，不能靠中断重试越过上限。

## 发现与已修复问题

复核过程中识别并由父任务在第二阶段seal前修复：final helper返回dict却按二元组解包、调用不存在的make_model、未将adapter/tests纳入v2封存、final温度缺少hash绑定、final-inner seed配置不一致、260完成fit与263 attempt的预算叙述不一致、缺少操作锁/硬deadline和formal辅助attempt账本，以及seal前只检查完成结果而未检查已启动fit。这些属于执行前修正，没有从真实stage2结果反调方法。

## 验证与适用范围

`tests/active_learning_v2/test_epig_stage2_adapter.py` 六个无训练stand-in测试通过（9.93秒），覆盖双seal核验、原module不变、目录隔离、symlink拒绝、runner字节漂移、stage2资格失败时context/标签/训练前停止，以及原210/40h/单worker门槛保留。driver与adapter语法编译通过。nested模块另有合成数学/泄漏/缓存测试，由总预检统一记录实际结果；本复核不将尚未运行的真实校准或GNN学习曲线写作已完成。

温度迁移、错误joint相关、均值/异方差失配及当前GNN条件学习瓶颈仍可能存在。资格通过也不证明Row AULC收益；资格失败只界定本有限代理/校准族，不能否定一般EPIG。完整正式主评价仍须原GNN的五seed真实非单调曲线。
