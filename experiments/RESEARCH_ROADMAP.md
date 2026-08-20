# Research Roadmap（Deferred Only）

`EXPERIMENT_PLAN.md` 和 `METHOD_DECISION_REGISTER.md` 对当前正式实验有更高优先级；本文件只记录 deferred questions、trigger conditions 和长期研究方向，禁止 Codex 因看到 roadmap 就自动执行后续实验。

## A. Near-term deferred controls

## D38R/D40 — Corrected audit and engineering smoke

- **D38R Corrected E2 Compound Failure Audit** remains purely post-hoc. The seed42 README aggregation, gradient-row label, condition standardization, paired-compound comparison, and split shift decisions were corrected. Concentration metrics show large top-compound contributions but broad worsening counts, so localization remains inconclusive rather than asserted. No implementation, leakage, or evaluation bug was found.
- **D39 E4 Preregistered Protocol** freezes 8g no-threshold target data, Protocol A/B D28 partitions, last2+head transfer from three hashed independent source checkpoints, L0=50, B=10, 15 rounds, K=3, five acquisition methods, recovery/AULC metrics, tail diagnostics, and pilot expansion gates. The preregistration is a design artifact; formal training is explicitly deferred.
- **D40 E4 Protocol A Engineering Smoke** checks source loading, partition isolation, five acquisition dry-runs, Random/Hybrid query→reveal→source-reinitialize→retrain, parameter freezing, and resume. It is engineering evidence only and has no scientific conclusion.

Long-term, after the E4 pilot is stable, move reusable code into `src/qgeognn_al/{model,features,engine,acquisition,metrics,partitions,artifacts}.py` so reusable core no longer imports `run_*.py`. This refactor is explicitly deferred until then.

- **Hybrid causal diversity control**：仅当row/compound中Hybrid持续优于Ensemble且fixed common-reference diversity同时更高时，比较Ensemble Top-25%后Random-B与farthest-first-B。
- **Quantile Width post-hoc acquisition**：仅当row/compound risk ranking持续强但Ensemble AL utility弱或不稳定时，作为secondary control，不改写E2预注册主比较。
- **Coverage representation ablation**：仅当Coverage/Hybrid在row与compound稳定优于Random时，比较`h_graph`、conditions、联合、block-balanced联合及MorganFP+conditions。
- **Compound-stratified Random**：仅在需要判断Coverage收益是否只是覆盖更多compounds时使用，不是当前primary baseline。

## B. E4 preregistration checklist

- Protocol B primary固定为canonical-compound-held-out；Bemis–Murcko scaffold OOD延后到E5。
- labels-to-90%-ceiling使用error-gap recovery：`(E_baseline-E_t)/(E_baseline-E_full) >= 0.90`；正式E4前冻结baseline。
- K个Ensemble成员必须具有独立source checkpoints与target fine-tune mapping，并记录checkpoint hashes。
- 8g no-threshold协议每轮记录queried tail fraction、tail/common error与tail uncertainty。
- Pilot候选为3 outer seeds × K=3；Final候选为至少5 seeds × K=5，按E2真实计算成本决定且各策略K一致。

## C. E5 downstream and OOD

- 独立执行Bemis–Murcko scaffold OOD与condition-region OOD，不与compound split混称。
- pair-SQ truth必须由历史真实V1/V2构建；报告SQ MAE/Spearman、NDCG@K、top-K precision/recall与recommendation regret。
- 只有RMSE下降但SQ ranking不改善时，才触发task-aware/SQ-aware acquisition。

## D. Conditional advanced methods

- batch redundancy仍是瓶颈时，再考虑LCMD、B³AL-LCMD或MaxDet。
- Ensemble有效但成本过高时，再考虑PBNN或MC Dropout。
- Compound/scaffold OOD失败时，再考虑geometry-aware molecular AL及Morgan/learned/geometric distance。
- RMSE改善但SQ不改善时，再考虑task-oriented或Pareto acquisition。

## E. Scientific debt

D05、D12、D15–D18、D21–D22、D24–D26与D30记录极端源记录、未使用模块、收敛/BN/shuffle/重复噪声、标准化与异常、纯度/溶剂编码、稳健宏指标及构象非收敛问题。这些是scientific debt / future ablations，不是重开当前冻结Predictor的理由。

## F. 25g/40g

仅在E4 8g active transfer成立后考虑，用于检验跨柱label-efficiency能否泛化；当前不得启动。
