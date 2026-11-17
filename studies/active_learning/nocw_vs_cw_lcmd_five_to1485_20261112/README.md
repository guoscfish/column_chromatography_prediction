# NoCW LLM versus CW-gradient LCMD32: five paired seeds to L1485

User authorized both methods on seeds 157, 887, 2357, 6101, 12203, with 36 batches of 32 after shared L333 starts. Reuse all existing scientific trajectories without changing old files. NoCW needs 57 additional batches; CW-LCMD needs 138. Primary outcome: raw normalized trapezoidal validation NRMSE AULC, L333-L1485; endpoints are secondary. These are established Row splits, not a new untouched confirmation cohort. No test truth is read in this run.

The imported NoCW requests/answers/freezes remain byte-identical. Four interrupted local streams get separate local reconciliation receipts in this revision; provider completion and billing remain unknown. Original records remain intact. Do not substitute partial answers or reconstruct selections. Model, effort, prompt, input encoding, candidate pool, correction rules and training settings are retained.

Two shared compute slots, two CPU threads each; five independent NoCW keys. Each worker is dispatched once and stops on an unexpected error; no automatic abnormal-worker restart. The supervisor writes status.json and validation snapshots. It does not send reports to acquisition.

## Git 归档（2026-11-17）

选点契约、选择 ID、批次冻结、反馈、标签访问回执及最终报告按原字节保存。传输和逻辑回答证据集中到 [`reports/nocw_retry_evidence_20261117/`](../../../reports/nocw_retry_evidence_20261117/)，五份 gzip JSONL 共保留 6,782 个 attempt、stream（含完整回答和帧修复回执）、turn、rejected JSON；每条含原路径、SHA256 与原始 UTF-8 文本。所有条目均已解压并与本地原件逐字节核验，汇总见 `manifest.json`。

传输 JSON 原件和原始 SSE wire 镜像继续留在原位置，由局部 `.gitignore` 排除；完整请求、模型和运行目录沿用仓库既有保留规则。本次归档未修改实验原件，未调用模型 API 或启动训练。
