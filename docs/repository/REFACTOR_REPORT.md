# 隔离工程重构交付报告（2026-11-09）

已完成附件 Gate 0–2 中可安全实施的审计、最小公共 API、真实删减与离线验证；Gate 3 按要求停止。候选位于 `/Users/fish/Documents/GitHub/column_chromatography_refactor`，分支 `codex/repository-architecture-20261109`。独立孵化仓库位于同级 `column_chromatography_lab`。均为本地交付，没有推送或合并活跃 main。

## 基线与原实验保护

独立 clone 从已提交的 `16eca74769f664bc378a58ccbf380a806723f587` 初始化；无对象 alternates，无硬链接共享抽样命中，拥有自己的 `.git`。没有复制原工作树未提交 runtime、keys 或原始请求。候选采用稀疏展开，只读取代码、文档和允许的元数据；数据/历史工件在独立 Git 对象中保留身份，不解析真实标签。原源码、控制器、锁、运行目录及恢复授权没有由本任务写入、移动或接管。

18:48（Asia/Shanghai）的非原子只读快照显示 CW-Free 尚未完成：

| seed | accepted 批次 / L | completed fit 批次 / L | pending fit |
| --- | --- | --- | --- |
| 157 | 19 / 941 | 18 / 909 | 1 |
| 887 | 29 / 1261 | 29 / 1261 | 0 |
| 2357 | 29 / 1261 | 29 / 1261 | 0 |
| 6101 | 29 / 1261 | 29 / 1261 | 0 |
| 12203 | 28 / 1229 | 27 / 1197 | 1 |

目标仍为每 seed L1485；不得把以上快照当作此刻最新进度。completed fit 以连续完成回执及六项工件哈希相符为准，accepted 仅以 batch_freeze 存在为准，未读取选择值/标签。Supervisor PID 90623 和识别到的 workers/五个旧监控进程均以原 main 为 cwd。采集只读；没有发送信号、获取锁、重启或调用恢复链。263/263 CW 冻结代码、263/263 原始源快照、45/45 恢复授权绑定匹配；最终完整性检查复核原 main 的 565 个源码与 13 个运行目录身份。

完整证据：[运行依赖](RUNTIME_AND_FROZEN_DEPENDENCIES.md)、[结构化快照](runtime_snapshot_20261109/RUNTIME_SNAPSHOT.json)、[最终完整性](validation/integrity.json)。runtime 会正常继续变化，不能声称其全部字节不变。

## 逐文件审计与实际删减

`CODE_INVENTORY.csv` 共875行：820个基线Python、24个新增候选Python、31个仅登记路径/哈希的未提交运行Python；AST解析失败0。审计覆盖相对导入、脚本/子进程、动态字符串、shell、测试、元数据、Git身份、frozen_source/全目录hash约束和当前运行索引。风险为 LIVE_BLOCKER461、FROZEN_HASH_BLOCKER200、IMPORT_BLOCKER4、UNKNOWN188、SAFE22；这是静态证据分类，不是对任意反射/外部调用的完备证明。

137个历史退役候选中，真实退役2个，暂缓135个。删减仅发生于候选分支：

- `scripts/studies/plot_free_cw_snapshot_20261102.py`（42行）。
- `scripts/studies/plot_free_rerun_audit_20261103.py`（83行）。

两份真实绘图实现合并为 `scripts/plot_saved_validation.py`，配套两个配置模板。历史报告保持不变；旧路径的历史文献引用、测试中的 `git show` 恢复引用和完整性验证器白名单已逐条审查为非执行依赖。每次复跑删除门禁都会重核当前依赖、冻结/运行/保护关系、原blob/SHA、替代入口与等价证据。原commit、blob、SHA256、恢复命令及8个合成输出哈希见 [RETIREMENT_MANIFEST.json](RETIREMENT_MANIFEST.json)、[RETIREMENT_PROOF.md](RETIREMENT_PROOF.md)、[PLOT_EQUIVALENCE.json](PLOT_EQUIVALENCE.json)。没有删除唯一科研证据；没有把旧文件复制进 main 的 archive/legacy。

## 实际新架构与迁移

实质提取2个核心组件：LCMD-TP算法、128维latent提取。新增12个公共API Python文件，建立 representations/acquisitions/workflows/agents/infra 边界；评价沿用既有 evaluation，避免重复实现。旧LCMD/coverage因冻结历史依赖保留原字节，新模块拥有实际算法，未套一层转发文件冒充迁移。

表示接口校验有序授权L/U、有限向量、冻结 predictor fingerprint、提取前后模型/图哈希。实际latent提取拒绝图上的y/额外字段、额外候选或重排，要求eval/CPU predictor。Acquisition只执行LCMD或局部RNG random。Workflow独占plan→磁盘freeze→有限授权reveal callback；排他写和reveal-start marker阻止崩溃后自动重复揭示。Agent只输出经候选/预算校验的计划；没有provider调用。合成CLI走通 config→representation→acquisition→selected_ids+trace+freeze envelope。

统一provenance包含 code_commit、dataset_hash、split_hash、model_checkpoint_hash、protocol_hash、representation、acquisition、seed、budget、feature_hash。仓库内合成示例的commit是输入算法基线；交付时 lab 的配置与lock另绑定真实候选实现commit，解决提交SHA自引用问题。后续运行应传入实际执行的固定commit。

模型、损失、特征数学定义、Prompt、随机种子与训练协议未改。原LCMD全零距离下可能选择空cluster并抛异常；这次保留该历史行为并测试，不暗中修算法。详细职责/映射/退出条件见 [ARCHITECTURE.md](../ARCHITECTURE.md)。

## 已真实执行的验证

候选clone完整安全测试：**101 passed，0 failed，118条现有依赖警告，51.71秒**。命令和环境见 [tests.log](validation/tests.log)、[JUnit](validation/tests.xml)、[environment.json](validation/environment.json)。

覆盖实际合成QGeoGNN 458,952参数、5节点层/4边角更新、128维hidden、六输出顺序、固定合成checkpoint严格加载与损坏拒绝、512 CountSketch固定seed/顺序、q50 MSE+软交叉及V1:V2=1:1、train-only归一化；LCMD新旧选择/距离/trace逐元素一致，latent新旧逐元素一致。固定合成数值快照事前容差 `rtol=0, atol=1e-6`。现有MaxDet、IVR、Hybrid在合成输入上通过；未把“NRMSE差不多”当作等价。

两份旧绘图入口对同一合成输入的新旧PNG/SVG/PDF及snapshot REPORT/manifest逐字节相同（固定元数据时间与SVG salt以消除序列化随机性）。只使用虚构曲线和合成checkpoint字节，未解析真实checkpoint。

新增完整性测试验证原源码字节、263冻结路径和28历史兼容路径、全部基线Git对象和12个保护项的可恢复性、独立Git存储、原main源/身份、2条退役授权。静态库存工具也有拒绝不安全删除/载荷扫描的测试。

原有 `audit_repository_hygiene.py` **已运行但exit=1**：research_scope violations=0；稀疏封存导致118个历史 retained_record及12个受保护数据/检查点在文件系统未展开。这些是本次明确的未通过项，不冒充传统hygiene全通过；[原始输出](validation/legacy_hygiene.json)保留。独立对象级完整性检查证明其基线Git身份/对象保留，不为通过旧检查而展开封存真值或修改旧审计器。

模型测试使用当前fish环境；其重复OpenMP库需要子进程级 `KMP_DUPLICATE_LIB_OK=TRUE`。该限制与118条依赖警告记录在验证环境中，没有改原运行环境或安装依赖。

## lab 与晋升门槛

独立lab具有自己的Git、experiments/tests/configs/reports/shared/scripts、registry.csv和PROJECT_BASELINE.lock.json。通过候选提交的独立源码快照（SHA逐文件、444/555、无软/硬链接）使用固定接口；不指向活跃 main 的可写路径。Git clone不能保存只读权限，prepare_baseline先验全部哈希再恢复权限。合成demo在隔离Python进程运行，拒绝错误commit和非synthetic来源。lab验证与具体提交见同目录 `LAB_DELIVERY.json`。

未复制真实数据、预训练检查点或keys；lock中的原资源哈希只是审计来源，不是数据授权。真实科学实验处于未就绪状态。registry只登记合成工程smoke和研究idea。GP/融合/分位约束/CW物理参数化未实现。晋升需独立科学收益或长期复用证据、对照/成本/泄漏/回归/审查、可核验文件SHA及用户明确批准。

## 保留的入口和下一阶段阻碍

28个t1_formal/t1b1/transfer冻结兼容文件仍有效且原字节保留。`benchmark_reporting.py`有benchmark/sequential等调用；`phase1.seal_phase1()`仍封存旧协议，未迁移或退役。`input_schema.py`仍服务conditions；`application/QGeoGNN.py`仍服务图与编码器。V2/V3/V3.2的Prompt/模型/重试协议未合并。

完整fit/validate/feedback工作流迁移、真实预训练checkpoint/缓存/封存历史输入的完整新旧回归、其余表示与选样迁移、传统全展开hygiene和正式恢复演练均尚未完成，不在本报告冒称通过。无法证明安全的文件保持DEFERRED。

**Gate3停止：CW-Free仍运行；须待其全部完成、取得独立授权快照、补足真实回归及恢复演练，并由用户在独立任务明确决定切换后，才能正式合入main或退出旧冻结路径。**
