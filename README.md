# Data-efficient QGeoGNN adaptation for column chromatography

本项目研究：如何利用 4g source knowledge 和有限的新柱规格标签，实现 data-efficient QGeoGNN adaptation，并研究 active selection 是否能进一步减少 target labels。

## Research portfolio

项目不是一条从 E2 到 D46 的线性诊断链，而是三个相关研究分支：

- **Track A — 4g In-domain Active Learning:** `PROMISING_BUT_UNFINISHED`。E2 row-level 有明确正向 pilot 证据，compound-held-out 仅 suggestive；Hybrid 是当前最强 candidate，不是最终策略。
- **Track B — 4g→8g Transfer Adaptation:** `OPEN`。`current_last2_head` 是历史 E4 baseline，下一候选是只用 Random target labels 的 T1 benchmark。
- **Track C — Active Transfer:** `DEFERRED`。只有 Track B 建立稳定 low-label adaptation baseline 后才可重开。

## Frozen evidence

E2 row normalized AULC：Hybrid 0.542938、Coverage 0.562489、Ensemble 0.626849、Random 0.644749；Hybrid/Coverage 对 Random 均 3/3 outer seeds 更好。E2 compound：Hybrid 0.761203、Coverage 0.777107、Random 0.788732、Ensemble 0.802149；Hybrid/Coverage 仅 2/3 seeds 胜 Random。因此不得把 4g AL 写成 completed、solved 或 final strategy found。

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

Historical runners are reproduction-only and write to a new output directory. Do not overwrite finalized experiment directories. No D47, new AL, Protocol B, or 25g/40g run is authorized by this reset.
