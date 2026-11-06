# 仓库清理与分析归档 · 2026-11-06

本次按用户要求保存当前代码、研究产物及失败原因，供后续离线分析。正在运行的实验继续按原授权推进；本次没有调用选点 API、启动训练、修改科学协议或停止 NoCW。

## 清理范围

仅删除 src、scripts、tests、reports 内可再生成的 `__pycache__` 与 Finder 元数据；逐文件大小、SHA256 见 manifest.json。模型、预测、选点输入、原始失败回执、授权、锁、完成标记、NoCW 文件均未删除。付费得到的表征缓存也保留本地。

CW 的高频 attempt/stream 原件继续留在原位置，供控制器累计计数和恢复使用；新增的这类 sidecar 不再逐个加入 Git，而是通过下列无损归档保存。已在 Git 中的历史文件不受忽略规则影响。

## 分析入口

- `../cw_retry_evidence_20261106_1445/manifest.json`：逐 seed / 逐 round 的传输结果、回答校验失败分类、归档 SHA256 与文件数量。
- `../cw_retry_evidence_20261106_1445/seed_*_receipts.jsonl.gz`：每一次传输的 started/terminal、流状态、完整逻辑回答及 rejected 回执；每行包含相对路径、原字节 SHA256、raw_utf8，能够还原原始 UTF-8 文件。含成功和失败，避免把缺少终态的在途请求误算失败。
- `../free_first_vs_second_20261106_1434/`：最新两次 Free32 学习曲线，按已完成训练点比较；历史轨迹保留混合模型标记。
- `../free_rerun_audit_20261103/README.md`：训练协议、两次轨迹分叉、记忆、输出长度与产物绑定审计。
- `../../studies/active_learning/verifiable_scientific_agent_20261030/FAILURE_TAXONOMY.md`：历史失败分类与结构化分析。
- `../../studies/active_learning/representation_gp_benchmark_20261104/README.md`：表征与 GP 基准代码、结果及复核入口。
- `../../studies/active_learning/gp_guided_llm_stability_20261103/`：GP 引导稳定性实验代码与材料。
- `../../studies/active_learning/nocw_free_selection_diagnostics_20261101/`：NoCW 选点诊断代码与材料。

## 当前停滞的已确认事实

14:33 上海时间，五 seed 已完成批次为 15、29、27、27、24，合计 122/180，对应 L813、L1261、L1197、L1197、L1101。最近全局训练产物为 11月6日04:39 左右；157 的最近训练已约83小时以前。

157 的活动 round_15（零起始，即第16批）保存了 turn_00 至 turn_80 共81份完整逻辑回答；对应81份 rejected 均为 `selection packet binding mismatch`。最近 turn_80 的 packet_sha256 少了输入哈希末尾的 `0`（输入以 `...1c0` 结尾，回答以 `...1c` 结尾），该批未被接受。turn_81 是下一次纠错请求编号，不能视为已完成81轮。

当前主要传输阻碍为 HTTP429；此前还出现 HTTP403、HTTP/2 stream error、upstream_error、连接/读取超时及可恢复 worker 退出。不同错误的精确次数以快照 manifest 和原始回执为准。HTTP429 回执没有保存足以判断配额、余额或容量原因的响应体，不能把状态码直接解释成余额不足。传输成功不等于逻辑回答通过，也不等于重训完成。

## 快照边界与复核

实验继续运行，导出依次扫描各 seed；快照不是同一毫秒的全局原子快照。导出开始以后产生的新请求留在本机，由后续快照保存。归档中每条保留自己的 started/ended 时间。

完整请求、候选输入、训练数组与 checkpoint 按仓库既有忽略规则留本地；本次推送覆盖所有新增非忽略代码以及上述失败证据，不把大体积运行目录或凭据当作源码提交。

重新导出：`python scripts/studies/export_retry_evidence_20261106.py --output reports/cw_retry_evidence_NEW_TIMESTAMP`。必须使用新目录，禁止覆盖历史快照。

验证归档：逐行解析 gzip JSONL，将 raw_utf8 编码为 UTF-8 后计算 SHA256，与 sha256 字段对照；另核验 gzip 文件整体哈希与 manifest 一致。只读分析请在独立目录解包，禁止覆盖活动实验文件。

补充文件：失败归档内的 failure_counts.csv 和 round_transport_counts.csv 提供直接可读分类计数；runtime_status_and_logs.jsonl.gz 保存控制器/worker状态、完成标记与诊断日志，覆盖本地worker退出及恢复证据。
