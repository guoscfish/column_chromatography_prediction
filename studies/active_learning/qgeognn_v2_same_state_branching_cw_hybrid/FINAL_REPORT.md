# CW/Hybrid same-state branching pilot（待执行）

这是一个已完成设计、lineage seal 和 selection smoke 的 development/mechanism pilot。8 个 exact same states、3 个 branch strategies、两轮 rollout 和 test firewall 均已固定；48 次新 scratch fit 尚未执行，因此这里不填入任何伪造的 test metric 或 oracle headroom。

当前可确认的结论是：历史 Center/Width-LCMD 与 Hybrid source state 均可在不重训 anchor 的前提下复用，8 个 lineage 的 checkpoint、ordered L/U、source prediction 与 gradient bank 已 hash 封印；selection smoke 为 deterministic，test access 为 0。

待执行后必须直接回答：

1. CW 引入后 adaptive 是否仍有明显 headroom；
2. always-CW 与 local oracle 的平均 short-AULC 差距；
3. 是否值得再补 2 个 seeds；
4. ranking 更受 stage 还是 state/trajectory 影响；
5. 停止 adaptive、补 2 seeds，或继续 controller 的明确推荐。

运行完成后，`finalize_qgeognn_v2_same_state_branching_cw_hybrid_report.py` 会生成 local_oracle_headroom.csv、headroom_summary.csv、winner_matrix.csv、state_diagnostics.csv、三个 figures、decision.json 和最终中文报告。由于历史 Hybrid 的 K=3 ensemble 不能在本 pilot 中额外训练，branch Hybrid 使用协议中明示的 gradient-norm top-25% + current-L farthest-first compatibility selector；其结果必须按 development evidence 解读。
