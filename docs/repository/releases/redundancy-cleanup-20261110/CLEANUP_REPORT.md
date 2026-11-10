# 全仓库历史冗余整理与可用性验收（2026-11-10）

状态：CURRENT。继续使用 `codex/repository-architecture-20261109`，本轮基线 `382fc755c1c22058c18d4b0593f6a686b9a48e31`；main 实测为 `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`。已 fetch 核对两者，无新分支、无自动 merge/rebase/push。

## 实际结果

**实际退出71个文件、11,296,738个跟踪字节（约11.30 MB）**。原路径通过标准Git删除；没有平移进archive/legacy，也没有把原图或日志打包后留在候选。新增的只有本报告与三件紧凑机器交付，详细工作输入留在隔离临时目录。

| 目录 | 跟踪文件数 | 目录数（含该根） | 跟踪字节数 | 实际退出文件 |
| --- | --- | --- | --- | ---: |
| docs | 60 → 38 | 10 → 8 | 8,940,168 → 3,511,876 | 26 |
| experiments | 773 → 769 | 183 → 183 | 56,701,780 → 56,547,282 | 4 |
| reports | 494 → 462 | 48 → 48 | 96,839,166 → 91,567,536 | 32 |
| studies | 20947 → 20938 | 2324 → 2318 | 1,338,880,450 → 1,338,874,936 | 9 |

“清理后”包含本轮新报告/收据与导航更新，所以净减少字节不等于删除字节。统计使用完整Git树和最终候选，不把sparse未展开文件算作丢失；Git历史对象体积不会因此缩小。逐文件身份、原SHA/blob、替代入口、固定来源及比较证据在 [ACTUAL_REMOVALS.csv.gz](ACTUAL_REMOVALS.csv.gz)。

- **docs 26项**：四份旧phase2生成库存/图视图、三份已被final91严格覆盖的component JUnit、上一轮完整16件工程包、两份旧一期环境/源码快照和只剩语义proxy读者的Legacy审计。历史工程包固定在 [382fc755](https://github.com/guoscfish/column_chromatography_prediction/tree/382fc755c1c22058c18d4b0593f6a686b9a48e31/docs/repository/releases/document-cleanup-20261110)。强保留恢复锁/374项源码删除证明/真值消费登记原字节不变。
- **reports 32项**：9 PNG、17 SVG、6 PDF，都是旧进度/滚动导出的衍生图。完整聚合输入、原generator、原始结果与控制/重试链保留。未将不同seed、模型、预算或split的结果合并。
- **studies 9项**：五个仅跳转的proxy README、三份由36-fit账本/交付审计完整覆盖的factorial stdout、一份V3.1由原纯renderer逐字生成的406-byte Markdown。真正的I0/TrackA/原预注册和结果不移动。
- **experiments 4项**：D46 engineering-smoke四张图，原聚合输入、正式D46结果、truth-use声明及smoke身份保留。不是删除科学结果或失效实验。

实际消失的旧目录为：旧 `document-cleanup-20261110/` 工程包、一期 `runtime_snapshot_20261109/` 与 `validation/`，以及 `studies/predictor/condition_completion/`（含design/preflight）、`semantic_input_audit/`、`studies/active_learning/4g/hybrid_batch_control/`及其空父目录4g。报告和smoke目录只删合格输出，因generator/输入/原文或forest仍保留，没有虚报整目录退出。

## 再生成与仍须保留的对象

本轮实跑验证了上轮37张图的再生成门槛；其中29张可退出、8张PNG仍被保留原文读取。新增four-to1005三图也通过并退出。滚动REPORT虽可逐字重建，仍被两个冻结study README读取，连同它引用的两张PNG保留。因此最终保留9个有真实文档读者的候选，未改冻结原文去强行解除引用。

实际退出的37件衍生视图为13 PNG、17 SVG、6 PDF、1 Markdown：PNG原始字节及全部RGBA像素一致；SVG仅规范日期、随机生成ID及等价path空白后完整结构一致，不放宽数值/文字；PDF仅创建日期不同，规范后完整字节及解压内容流一致；Markdown逐字节一致。运行的是原纯绘图/renderer或隔离AST片段，输入为已有公开aggregate，无原始U/test目标、模型拟合或provider。最初因10位小数量化出现的2像素偏差已用原generator指定的高精度aggregate CSV解决，不用近似比较掩盖差异。

上轮其余阻塞也重新审查：32个机器引用候选中，两份旧一期JSON是phase2同名路径/临时fixture误判，现已退出；其余30项作为原始聚合输入、fit/control/provenance或被读的REPORT保留。5项文档引用中，Legacy audit随其唯一proxy读者退出；其余四项仍有模型冻结/原文/PRUNE合同关联。两份protocol-deviation记录继续保留。

[BLOCKED.csv](BLOCKED.csv) 逐项记录384个具体保留对象与原因，包含：GP `verification_run.txt`虽与JSON同blob仍被完整artifact_manifest封存；V3.2曲线、NoCW状态有hash链；D46第五张forest缺独立保存的paired mean CI输入；compound preflight的318/57/3375/413划分和K3回合身份不能被后续试验替代；跨柱兼容文件虽与dadd逐blob相同，仍有保护/ignored资源限制。重要GP/IVR/失败Clean负结果、原始Prompt/response、选择顺序、标签账本、授权、预注册、模型/缓存均保留。

完整逐文件工作表覆盖基线22,274个产物，位于 `/private/tmp/column_redundancy_20261110/FULL_CANDIDATE_REVIEW.csv.gz`；hash与来源记录在 [TEST_RESULTS.json](TEST_RESULTS.json)。这不是对所有原始科学payload逐一解码：原始标签/模型/LLM请求继续受保护，未读取的SHA不伪造，保留Git blob及保护身份。只有经过内容、生成器、消费者与manifest审查的子集执行退出。

## 恢复和最终依赖复核

71个实际退出文件均从独立HTTPS GitHub fetch/checkout目录恢复并验证原bytes/SHA256；40份固定绘图/renderer代码、aggregate输入及权威设计源也独立取回。对象库无alternates、符号链接或硬链接共享。不可把本地ref当远端备份。

同时从远端独立恢复了16eca/4ee锁定的263/375份源码，whole-src/application的165/177文件、166/178旧benchmark hashes及7项已安装包元数据核对通过。**SOURCE_RESTORABLE_ONLY**：没有重建私有模型/原始runtime或安装旧环境，不等于完整历史实验可重新训练。ignored/private资源须另有受控备份；本轮没有凭Git记录删除这些资源。

删前按最终集合做fixed-point读依赖复核；删后重新取Git `--no-renames`的实际D集合，71项与批准集合精确相同。171份安全named manifest/hash配置检查无封存命中；留存Markdown读者=0、无条件机器读者=0，164条保守字符串/目录/writer/旧schema/临时fixture边均有语义说明。217条main跨根观察单独记录，main原物理文件不删。没有用最初80候选集合的闭包代替实际71项结果。

## 本轮实际可用性验收

当前导航相对链接检查通过；原冻结双专家README中一处 `LLM方案沿革.md#记录-1` 失效历史锚点仍单列在TEST_RESULTS，不改原文字节，也不宣称全库所有历史链接都已修复。

| 组 | 本轮passed / failed / skipped | 证据范围 |
| --- | --- | --- |
| `tests/phase2` | 91 / 0 / 0 | 五seed L333真实checkpoint、图缓存、表示及纯算法/工作流/恢复 |
| `tests/document_cleanup` | 26 / 0 / 0 | 原24项与本轮2项compact验收兼容；临时Git负向场景 |
| 合成CLI + resume | PASS | 全新目录预算2→6、两轮、3fit/2freeze/2label/3feedback；resume结果、全部文件集及SHA不变 |
| 历史源码恢复 | PASS | 既有只读树复查及额外独立远端源码checkout；没有新建分支/注册worktree |

合计 **117 passed、0 failed、0 skipped**，均在本轮实际删除后执行。五seed六输出/hidden最大差=0，真实图重建逐字段bitwise一致；33条算法记录的ID/顺序/trace等价，seed157的L333→365、L525→557、L973→1005三个历史LCMD批次32 IDs完全重放。输入来自已有授权只读private manifests，manifest、checkpoint、代码与日志/JUnit的SHA均在TEST_RESULTS；新机器缺fixture应明确BLOCKED，不能回退读live真值。原始详细日志保留在隔离regression目录，本轮未把旧日志复制堆入Git，也未重跑一期105项计数。

唯一CLI仍为 `python -m scripts.run_al --config configs/offline_synthetic.json --output <全新隔离目录>`。当前验证的是synthetic生命周期、纯选样核心、冻结checkpoint推理和SOURCE_RESTORABLE_ONLY；真实生产多轮训练、真实同seed K3 Hybrid、LLM provider/prompt协议端到端仍未验证，没有在本轮声称补齐。

## 最短阅读路线和main边界

1. 当前模型与科学状态：[NEXT_STAGE_DECISION](../../../NEXT_STAGE_DECISION.md) → [输入合同](../../../model/INPUT_SCHEMA.md) / [模型代码合同](../../../model/PAPER_CODE_CONTRACT_AUDIT.md)。
2. 当前AL实现：[ARCHITECTURE](../../../ARCHITECTURE.md) → [method registry](../../../../configs/method_registry.json) → [合成CLI](../../../../scripts/run_al.py)。
3. 最可信五seed与边界：[AL研究索引](../../../../studies/active_learning/README.md) → Free/LCMD L1485；CW仍按固定main快照与其协议解释，不能将旧L973进度当当前最终结果。CW/NoCW/GP/IVR负结果直接由索引进入原报告。
4. 原始身份/数据/恢复：[保护登记](../../../PROTECTED_ARTIFACTS.json)、[真值消费登记](../../../data_consumption_register.json)、[历史源码锁](../../phase2/HISTORICAL_PROTOCOLS.json) 与每项固定SHA。科学复盘仍仅在main的fcb51852来源，不复制到重构分支。

main HEAD/index/staging、623个科学/控制源码及完整路径集合保持本轮基线；src/application/scripts/configs与强机器合同本轮未变。只读PID/cwd观察不等于证明所有worker不变或CW已结束；正常runtime进度可以推进。本任务没有发送信号、执行恢复控制器、更改环境或写运行工件。

main与候选仍从16eca分叉：main旧控制器依赖legacy接口和完整旧源码hash，新稳定布局不是可直接替换的运行树；main新增CW lease/controller/report尚未合入。未来切换仍需CW完成和审计、用户授权、当前main消费者迁移及真实生产AL/LLM/K3验收。**本轮不合并main，不删除main，不自动push。**

本轮只有一份总报告，附一个实际退出清单、一个具体保留表与一份测试/保护汇总；旧工程包从固定Git历史检索，不新增并行archive副本。当前核验命令见 [tools说明](../../../../tools/README.md)。
