# 活跃运行与冻结依赖审计

只读采集时间：2026-11-09T10:48:27.538765+00:00；展示时区 Asia/Shanghai。原仓库 `/Users/fish/Documents/GitHub/column_chromatography_prediction` 保持 `main`，HEAD `16eca74769f664bc378a58ccbf380a806723f587`。Git 完整状态与 worktree 清单保存在 `runtime_snapshot_20261109/` 的初始快照文本中；状态共 487 行，包含持续写入的 runtime 与历史监控文件。本文不把未提交状态当作已提交科研基线。

## CW 五种子实际进度

| seed | accepted batches / budget | completed fit batches / budget | accepted pending fit | state file phase | process capture worker PID |
| --- | --- | --- | --- | --- | --- |
| 157 | 19 / L941 | 18 / L909 | 1 | ADVANCING | 12252 |
| 887 | 29 / L1261 | 29 / L1261 | 0 | SELECTING | 62449 |
| 2357 | 29 / L1261 | 29 / L1261 | 0 | QUEUED | unknown |
| 6101 | 29 / L1261 | 29 / L1261 | 0 | QUEUED | unknown |
| 12203 | 28 / L1229 | 27 / L1197 | 1 | ADVANCING | 21130 |

accepted 表示选择目录存在 batch_freeze.json；未打开选择 ID 或揭示值。completed fit 表示从 round_00 开始连续存在 completed_seed 回执，且回执列出的六个工件逐一 SHA-256 相符，包括下一轮 runtime 的 fit_audit、prediction_freeze 与 history。初始 L333 checkpoint 不计入新增 fit batches。pending 是已冻结选择但尚无有效完成回执的批次。每种子目标为 36 个新增批次、L1485。未把进程退出或旧状态文件当作完成证据。

观测非原子：进程和文件会随实验正常运行而变化。旧 training_seed 与 stream_seed 记录可能落后，必须结合其 PID、更新时间及完成回执判断。完整结构化证据见 `runtime_snapshot_20261109/RUNTIME_SNAPSHOT.json`。当前实验未全部完成，因此 Gate 3 必须停止。

## 进程、目录与锁

只读 ps 证实 manual_resume_20261109/controller_overlay.py supervisor PID 90623 存活；worker PID 与时间戳见 process_snapshot.json。lsof 证实已识别进程 cwd 均为原仓库，记录了 supervisor、worker、operation 与三个 new_free_resource 锁文件的打开描述符。未获取、释放或探测任何锁，打开描述符本身不是内核锁归属的充分证明。另有五个 2026-10-18 旧实验报告/恢复监控进程仍在运行；其入口已纳入 live_runtime_dependents，不能当作完成的过期脚本退役。

## 冻结来源与动态控制链

静态读取确认链路为：manual_resume → three_parallel_lagging → upstream_error_recovery → immediate_retry → http403_recovery，并组合 transport_recovery 的 HTTP/2 shim、persistent_20261102_v2、concurrency_v2、frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py。这些模块通过 importlib 及全局 ENTRY 身份绑定相连；审计没有导入或执行它们。

CW protocol 的冻结 code_hashes 实测匹配 263/263。source_snapshot 的原始源文件现状匹配 263/263；不匹配项保留在 JSON 中，不能自动“修复”。protocol 文档实测匹配 3/3。恢复授权 code_hashes / preserved_fatal_receipts 绑定实测匹配 45/45。手动恢复及三并行授权、launch、controller 的 SHA-256、权限、inode 和时间戳均保留在 identity_files。

manual_resume 保留五个既有 HTTP401 故障回执的原 SHA-256；它们只按字节哈希核验，没有解析响应正文。该恢复授权属于既有实验，不被本审计解释为可启动新实验的授权。source_snapshot 的冻结 resources.ROOT 明确绑定原仓库数据路径，因此仅复制源码并不能使新工作区安全脱离原数据与协议。

早期 `src/qgeognn_al/active_learning_v2/benchmark_protocol.py:63` 的 code_hashes() 扫描整个 src/qgeognn_al/**/*.py、application/*.py 以及一个 benchmark runner；在 main 新增 Python 文件也可能改变身份。RESEARCH_SCOPE 的 frozen_llm_compatibility_paths 包含 28 条历史兼容路径，完整索引见 `runtime_snapshot_20261109/runtime_frozen_index.json`。

## 数据、checkpoint 与权限边界

仅对协议明确列出的初始 sources、五个 seed 的 L333 checkpoint、canonical_4g、source graph cache 与 scaler 计算字节 SHA-256。共 43 条工件记录；带协议期望值的核验匹配 40/40。没有解析标签表、split 行、预测表、graph tensor 或 checkpoint tensor；没有穷举大模型；没有读取 API keys、环境变量值或原始请求内容。

环境版本与 prompt hash 只记录冻结协议声明；未导入 live 科学模块重构 prompt，未声称已验证当前内存环境。源码初始完整性清单 active_source_hashes.json 覆盖 src、application、scripts、CW study 下全部 .py，以及已识别运行中的旧监控入口；最终还需复核此清单。

## 风险分类与退出门槛

- LIVE_BLOCKER：原仓库所有当前控制器、runtime、锁、授权与故障回执，五个旧监控进程及其路径。
- FROZEN_HASH_BLOCKER：CW 263 个冻结 code_hashes、263 个 source_snapshot 原始来源、控制器逐层授权绑定、28 个历史迁移兼容入口；数量以 JSON 实测索引为准。
- IMPORT_BLOCKER：schemas/conditions.py → input_schema、data/model → application/QGeoGNN.py 等须由全仓库 inventory 审计迁移；本运行审计不声称已完成所有调用方审查。
- UNKNOWN：瞬时内存模块内容、内核锁持有细节、未列入协议的外部资源、未重建的 runtime prompt 与当前包环境。
- SAFE：隔离 clone 内的静态/合成测试与新 API 工程工作；该结论不授权向 main 合并、正式删除、启动训练或请求 LLM。

本次只读审计对原仓库写入 0 文件、发信号 0 次、获取锁 0 次、导入控制器 0 次、启动训练/LLM 0 次。实验继续自行推进，runtime 字节自然变化是预期现象；不能声称整个 runtime 字节不变。正式晋升须等待现有实验全完成、恢复演练与数值回归通过，并在独立任务中由用户明确决定。
