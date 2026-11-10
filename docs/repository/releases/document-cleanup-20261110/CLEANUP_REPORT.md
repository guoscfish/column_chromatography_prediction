# 文档与研究产物清理验收（2026-11-10）

状态：`CURRENT` 工程验收。全部工作继续在原仓库的 `codex/repository-architecture-20261109`，基线为 `0e33a86d3dea3945acb0a638ca78a8f5065980a5`。main 单独固定在 `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`，本轮未合并、切换或修改它。

## 完成的阅读入口

- [docs/README](../../../README.md)：唯一文档导航，区分当前、历史冻结、已归档、已取代与 main 运行保护。
- [docs/ARCHITECTURE](../../../ARCHITECTURE.md)：吸收二期架构核心，明确实际 CLI 仅开放合成生命周期。
- [NEXT_STAGE_DECISION](../../../NEXT_STAGE_DECISION.md)：科学状态和新假设，分开描述已确认机制、待验证问题与冻结流程。
- [研究总索引](../../../../studies/README.md)、[44个AL研究](../../../../studies/active_learning/README.md)、[CW扩展关系](../../../../studies/active_learning/CW_RESEARCH_INDEX.md)、[10个预测器研究](../../../../studies/predictor/README.md)。
- [实验索引](../../../../experiments/INDEX.md)：21个候选实验目录与10个仅归档来源；[历史决策摘要](../../../../experiments/HISTORICAL_DECISIONS.md) 收敛三份早期决策/路线图。
- [29个报告的身份索引](../../../../reports/README.md)：current comparison、historical finalized、progress/transport、engineering audit 四类；废止滚动 latest 的权威含义。
- [工程审计版本导航](../../README.md)、[收据release清单](RELEASE_MANIFEST.json) 和 [工具说明](../../../../tools/README.md)：历史收据按固定版本统一检索，避免多套入口互相混淆。

输入文档同步到当前 predictor 路径和 V2 条件语义。原始研究报告、协议、负结果和已用测试真值声明不因导航更新而改写。科学复盘继续只放 main 的 `fcb51852...:studies/diagnostics/quantile_loss_audit_20261109/`，候选只保留固定来源链接。

## 实际退出的文件

本轮实际退出 **39 个文件、6,902,191 个跟踪字节（约6.90 MB）**，逐文件来源见 [DELETED_FILES.csv](DELETED_FILES.csv)。这是候选工作树的真实 Git 删除，不是在仓库里另搬一套 archive 副本；Git 历史体积不会因此缩小。

| 范围 | 文件数 | 处理 |
| --- | ---: | --- |
| docs | 32 | 一期报告/库存/重复测试日志/本地运行快照、被取代的草案与过期设计；机器锁仍保留 |
| experiments | 3 | DECISION_ARCHIVE、METHOD_DECISION_REGISTER、RESEARCH_ROADMAP 的原始副本，统一由历史摘要和固定提交检索 |
| reports | 4 | 10月8日旧进度正文/完成审计与11月6日工程清理收据，固定来源保持可查 |
| studies | 0 | 仅修入口；科学证据、冻结谱系和原始协议保持原路径 |

[39文件子集引用复核](SELECTED_SUBSET_REFERENCE_REVIEW.json) 对**最终实际集合**重新计算：留存候选树机器 caller=0、文档 caller=0；独立 main 有29条目录范围引用，它们仍使用 main 的原始文件，未被本轮删除。没有拿115项候选组的闭包结果冒充39项实际闭包。

仍保留37份未证明可以固定输入再生成的衍生图/PDF、32份有机器引用的候选、5份待进一步文档引用处理的候选及2份受保护候选。它们不是“已删”或“无价值”。旧 Clean/Legacy 科学合同、权威最终指标、数据来源、checkpoint、scaler、分区、采样序列、标签/授权账本和恢复链继续保留。完整分类由机器清单给出，不对报告整目录作删除授权。

## 逐文件审计与恢复门槛

[ARTIFACT_INVENTORY.csv.gz](ARTIFACT_INVENTORY.csv.gz) 覆盖基线 docs/experiments/reports/studies 的 **22,293 个跟踪产物**：路径、字节、hash/blob、owner-study、直接/动态/hash/报告引用、受保护角色、替代入口、恢复提交和决策。`COUNT@SET_ID` 字段在 [DEPENDENCY_REFERENCES.json.gz](DEPENDENCY_REFERENCES.json.gz) 中解析成去重边；目录和歧义 basename 是保守规则，不展开成庞大的笛卡尔依赖表。大件仅保留确定性 gzip，机器审计约1.1 MB，没有同时复制大原文。

这不是遍历并解释所有原始科研 payload。源CSV、checkpoint、标签、原始请求和凭据不解码；这些条目保留 Git blob、字节和保护角色，SHA字段明确标注未读取，不制造 SHA256。可审查文档/源码/普通配置用于静态引用，任意反射和仓库外消费者仍有边界。计划见 [DELETION_CANDIDATES.csv](DELETION_CANDIDATES.csv)，实际结果以 DELETED_FILES 为准。

[REMOTE_RECOVERY.json](REMOTE_RECOVERY.json) 证明从 HTTPS GitHub 在独立临时裸仓库实际取得基线提交和39个精确 blob，并逐一恢复到只读目录、验证大小与 SHA256。对象库无 alternates、符号链接或硬链接共享。远端没有二期两个 archive tag；使用完整可取得的 SHA，而没有把本地 refs 当备份。16eca 与4ee两个历史恢复来源提交也已直接从远端取得。

[HISTORICAL_SOURCE_RECHECK.json](HISTORICAL_SOURCE_RECHECK.json) 单独复查**已存在**的两个只读源码工作树：263/375个源文件、165/177个src/application universe、166/178个旧benchmark hash及7项环境版本匹配。没有为此新建分支或工作树；该复查不等于从远端重建完整训练环境。Git只能恢复已跟踪字节，不能备份 ignored runtime、原始请求、私有checkpoint或凭据。

## 链接、测试与 main 保护

[DOCUMENT_LINKS.json](DOCUMENT_LINKS.json) 扫描整个候选 Markdown，当前导航相对链接、标题anchor和固定自仓库Git目标必须通过。稀疏未展开但在Git中的目标不算丢失；原始历史文档不被改字节，对其中退役路径提供单独 pinned 解析结果并保留fragment，不声称其旧相对链接在候选里已经可执行。报告明确列出所有未解析项，不把历史例外掩盖成全库零断链。 当前全量扫描244份Markdown，日常导航无断链；保留原文中仍有1处旧双专家README的 `LLM方案沿革.md#记录-1`，该锚点未在已登记历史来源中找到，未伪报已修复。相应研究设计可另查 [固定双专家设计源](https://github.com/guoscfish/column_chromatography_prediction/blob/0e33a86d3dea3945acb0a638ca78a8f5065980a5/docs/research/4G_LLM_DUAL_EXPERT_ACTIVE_LEARNING_V1_2026-09-27.md)，不声称它等同于缺失记录的逐字恢复。

[FINAL_TESTS.xml](FINAL_TESTS.xml) / [FINAL_TESTS.log.gz](FINAL_TESTS.log.gz)：**115 passed、0 failed、0 skipped**，含二期91项核心回归与本轮24项文档链接/产物审计/验收门槛测试。真实 checkpoint/cache 和三个历史LCMD批次使用已有授权只读快照，不训练、不调用LLM、不揭示新真值。没有为了本轮计数重跑第一期105项记录。

[FINAL_PROTECTION.json](FINAL_PROTECTION.json) 证明 main HEAD/index/staging、**623个既有科学/控制源码及完整路径集合**保持本轮基线，候选 src/application/scripts/configs 没有更改或新增，强保留机器契约原字节不变，实际删除集合精确等于恢复证明。main 的正常进度文件可自行变化，不被加入本轮提交。

[PROCESS_OBSERVATION.json](PROCESS_OBSERVATION.json) 是本轮只读PID/cwd观察：原二期supervisor PID90623此刻不在列表，另有以main为cwd的Python进程。此事实不构成当前CW完成或中止判断，也不能假定历史PID一直存活；本轮没有发送信号、运行恢复控制器或重启实验。`ACTIVE_RUNTIME` 在导航中表示应保护的运行/恢复边界，不是重新确认某个旧PID仍活着。

## 可复查方式

从候选根目录生成计划到独立临时目录（不会删除文件）：

```sh
python tools/audit_research_artifacts.py --compress \
  --active-root /Users/fish/Documents/GitHub/column_chromatography_prediction \
  --output /private/tmp/column_artifact_audit_recheck
python tools/check_documentation_links.py --output /private/tmp/column_document_links.json
python tools/verify_cleanup_release.py --output /private/tmp/column_cleanup_protection.json
```

gzip可用Python标准库读取：`gzip.open(path, 'rt')`，再使用 `csv.DictReader` 或 `json.load`。历史原文可用 `git show 0e33a86d3dea3945acb0a638ca78a8f5065980a5:<path>` 恢复到独立目录；勿恢复进正在运行的main或新稳定源码。精确路径/hash/来源见删除证明。

本轮未自动 merge/rebase/push。只保留 main、迁移学习归档分支、原重构分支；创建多余分支的临时操作已撤销，改动完整转回原分支。正式科学训练、AL选点、prompt/loss改动和main切换不属于本轮工作。
