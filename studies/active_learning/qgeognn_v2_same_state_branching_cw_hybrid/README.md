# QGeoGNN-V2 same-state branching CW/Hybrid pilot

这是一个 development / mechanism pilot，用于检验在 Center/Width-LCMD 已经很强时，state-dependent adaptive acquisition 是否仍有 short-horizon headroom。

冻结设计：

- seeds: `157`, `6101`
- source trajectories: `center_width_lcmd`, `hybrid`
- anchor budgets: `429`, `653`
- branch strategies: `center_width_lcmd`, `hybrid`, `kernel_ivr`
- 每个 exact `(L_t, U_t, checkpoint)` 连续 scratch rollout `+32`, `+64`
- 所有 acquisition、checkpoint、prediction 先 freeze，再统一 reveal test truth

运行：

```bash
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --prepare preflight_tests.xml
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --validate
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --selection-smoke
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --execute-seed 157
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --stage-check 157
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --execute-seed 6101
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --stage-check 6101
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --freeze
python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --report
python scripts/studies/finalize_qgeognn_v2_same_state_branching_cw_hybrid_report.py
```

历史 Hybrid 的 K=3 ensemble artifact 保持只读并写入 lineage audit。由于 branch pilot 不能为 acquisition 再训练 controller 或额外 ensemble，本研究使用已审计的兼容选择器：当前 gradient-norm top-25% shortlist + current-L farthest-first；该兼容层在报告中明确标注，不能与历史 Hybrid 结果直接等同。
