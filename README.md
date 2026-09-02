# Data-efficient QGeoGNN adaptation for column chromatography

本项目研究：如何利用 4g source knowledge 和有限的新柱规格标签，实现 data-efficient QGeoGNN adaptation，并研究 active selection 是否能进一步减少 target labels。

## Research portfolio

项目不是一条从 E2 到 D46 的线性诊断链，而是三个相关研究分支：

- **Track A — 4g In-domain Active Learning:** `PAUSED_AFTER_A1A / PROMISING_BUT_UNFINISHED`。E2 row-level 保留正向 pilot 证据；A1a 只否定了 shared-shortlist 内 farthest-first 的稳定额外收益，A1b 不获授权。
- **Track B — 4g→8g Transfer Adaptation:** `T1_ENGINEERING_READY_FORMAL_NOT_AUTHORIZED`。`current_last2_head` 是历史 E4 baseline；T1 已完成预注册与工程 smoke，但未运行正式实验。
- **Track C — Active Transfer:** `DEFERRED`。只有 Track B 建立稳定 low-label adaptation baseline 后才可重开。

## Frozen evidence

E2 row normalized AULC：Hybrid 0.542938、Coverage 0.562489、Ensemble 0.626849、Random 0.644749；Hybrid/Coverage 对 FullPool-Random 均 3/3 outer seeds 更好。E2 compound：Hybrid 0.761203、Coverage 0.777107、Random 0.788732、Ensemble 0.802149；Hybrid/Coverage 仅 2/3 seeds 胜 FullPool-Random。A1a 的 Random 则来自同一个 Top25% ensemble-uncertainty shortlist；其 diversity gate 失败仅表示 farthest-first 没有稳定额外收益。联合证据使 uncertainty prefilter 成为 plausible contributor，但尚无 paired Top25%-Random vs FullPool-Random 的同协议因果证明。因此不得把 4g AL 写成失败、completed、solved 或 final strategy found。

E4 Protocol A 与 A2a 表明，在 `current_last2_head` 下 tested generic active acquisitions 没有稳定胜 Random。D45/D46 是 post-hoc diagnostics。D46 的三个 nominal seeds 在同 checkpoint、`shuffle=False`、近似 full-batch、`drop_ratio=0.0` 的 CPU protocol 下没有形成独立 stochastic realizations；ICC=1 与 rank agreement=1 因 zero within variance 而退化。primary bootstrap 数字是 3/18 unique candidates 区间排除零。

## Repository layout

- `src/qgeognn_al/`: reusable data, model, engine, acquisition, metrics, artifact, and diagnostic code.
- `scripts/`: historical/reproduction runners and thin compatibility shims. New studies should use a config-driven family runner.
- `experiments/INDEX.md`: one-row-per-experiment navigation.
- `docs/RESEARCH_DIRECTION.md`: research reset, literature maps, and future method gaps.
- `docs/ARTIFACT_RETENTION_POLICY.md`: tracked-artifact contract.
- `docs/NEXT_STAGE_DECISION.md`: A1 versus T1 decision analysis; manual approval required.

## Environment and tests

The validated project environment is conda `fish` (Python 3.11, PyTorch/PyG/RDKit installed):

```bash
conda run --no-capture-output -n fish pytest -q
```

S1 exploratory shift audit is complete: zero-shot combined NRMSE was about 0.803 and affine about 0.399 ± 0.130, with V1/V2 RMSE about 7.63/11.45 mL; affine won only 3/5 folds. It is a strong simple baseline, not a solved 4g→8g transfer. T1 engineering/preregistration is complete with `formal_authorized=false`; no formal T1, active transfer, Protocol B, or 25g/40g run is authorized.
