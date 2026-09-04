# Data-efficient QGeoGNN adaptation for column chromatography

本项目研究：如何利用 4g source knowledge 和有限的新柱规格标签，实现 data-efficient QGeoGNN adaptation，并研究 active selection 是否能进一步减少 target labels。

## Research portfolio

项目不是一条从 E2 到 D46 的线性诊断链，而是三个相关研究分支：

- **Track A — 4g In-domain Active Learning:** `PAUSED_AFTER_A1A / PROMISING_BUT_UNFINISHED`。E2 row-level 保留正向 pilot 证据；A1a 只否定了 shared-shortlist 内 farthest-first 的稳定额外收益，A1b 不获授权。
- **Track B — 4g→8g Transfer Adaptation:** `T1B1_FORMAL_COMPLETE_NO_TESTED_LOW_CAPACITY_ADAPTER_BENEFIT`。T1b-1 的 180/180 Adapter fits 与 120/120 capacity contexts 全部完成；r=8/16/32 均未稳定优于 774-parameter `target_head_only`，因此在已测试的约 3k–9k Adapter 区间没有支持的收益。9k–93k 区间未测试。
- **Track C — Active Transfer:** `DEFERRED`。只有 Track B 建立稳定 low-label adaptation baseline 后才可重开。

## Frozen evidence

E2 row normalized AULC：Hybrid 0.542938、Coverage 0.562489、Ensemble 0.626849、Random 0.644749；Hybrid/Coverage 对 FullPool-Random 均 3/3 outer seeds 更好。E2 compound：Hybrid 0.761203、Coverage 0.777107、Random 0.788732、Ensemble 0.802149；Hybrid/Coverage 仅 2/3 seeds 胜 FullPool-Random。A1a 的 Random 则来自同一个 Top25% ensemble-uncertainty shortlist；其 diversity gate 失败仅表示 farthest-first 没有稳定额外收益。联合证据使 uncertainty prefilter 成为 plausible contributor，但尚无 paired Top25%-Random vs FullPool-Random 的同协议因果证明。因此不得把 4g AL 写成失败、completed、solved 或 final strategy found。

E4 Protocol A 与 A2a 表明，在 `current_last2_head` 下 tested generic active acquisitions 没有稳定胜 Random。D45/D46 是 post-hoc diagnostics。D46 的三个 nominal seeds 在同 checkpoint、`shuffle=False`、近似 full-batch、`drop_ratio=0.0` 的 CPU protocol 下没有形成独立 stochastic realizations；ICC=1 与 rank agreement=1 因 zero within variance 而退化。primary bootstrap 数字是 3/18 unique candidates 区间排除零。

## Repository layout

- `src/qgeognn_al/`: reusable data, model, engine, acquisition, metrics, artifact, and diagnostic code.
- `scripts/`: historical/reproduction runners and thin compatibility shims. New studies should use a config-driven family runner.
- `experiments/INDEX.md`: one-row-per-experiment navigation.
- `docs/RESEARCH_DIRECTION.md`: research reset, literature maps, and future method gaps.
- `docs/QGEOGNN_IMPLEMENTATION_VARIANTS.md`: legacy、clean reproduction 与 Predictor V2 的实现合同差异。
- `docs/DATA_CONSUMPTION_REGISTER.md`: dataset/outcome 使用边界与 confirmatory 状态。
- `docs/ARTIFACT_RETENTION_POLICY.md`: tracked-artifact contract.
- `docs/NEXT_STAGE_DECISION.md`: A1 versus T1 decision analysis; manual approval required.

## Environment and tests

The validated project environment is conda `fish` (Python 3.11, PyTorch/PyG/RDKit installed):

```bash
conda run --no-capture-output -n fish pytest -q
```

S1 exploratory shift audit is complete: zero-shot combined NRMSE was about 0.803 and affine about 0.399 ± 0.130, with V1/V2 RMSE about 7.63/11.45 mL; affine won only 3/5 folds. T1a found a low-capacity signal but no stable winner. T1b-1 then isolated graph-level adapter capacity at 2,958/5,014/9,126 trainable parameters while retaining fixed sum pooling. Mean normalized AULC was 0.6577 for Head and 0.6583/0.6587/0.6578 for r8/r16/r32; Adapter wins were 2/5, 2/5, and 3/5, so none passed the frozen gate. This supports retaining output-only correction as the working baseline over the tested low-capacity graph adapters only; widths corresponding roughly to 17k/34k/67k were not tested. T1b-2, active transfer, Protocol B, and new 25g/40g runs remain unauthorized.

I0 confirmed that the legacy clean predictor constructs ten continuous edge features but consumes only the first five, leaving eluent HBA/LogP and all loading fields unreachable. The audited model has 775,476 nominal requires-grad parameters but 456,620 gradient-bearing parameters. These findings refine the implementation contract; they do not erase historical evidence. The explicit `qgeognn_condition_complete_v2` residual branch has now passed implementation preflight with zero legacy prediction drift across all three source members. This is engineering evidence only: formal 4g source qualification is not authorized, 8g transfer has not started, and active transfer remains deferred.
