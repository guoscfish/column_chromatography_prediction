# Research Roadmap（Deferred Only）

`EXPERIMENT_PLAN.md` 和 `METHOD_DECISION_REGISTER.md` 对当前正式实验有更高优先级；本文件只记录 deferred questions、trigger conditions 和长期研究方向，禁止 Codex 因看到 roadmap 就自动执行后续实验。

## A. Near-term deferred controls

## D38R/D40 — Corrected audit and engineering smoke

- **D38R Corrected E2 Compound Failure Audit** remains purely post-hoc. The seed42 README aggregation, gradient-row label, condition standardization, paired-compound comparison, and split shift decisions were corrected. Concentration metrics show large top-compound contributions but broad worsening counts, so localization remains inconclusive rather than asserted. No implementation, leakage, or evaluation bug was found.
- **D39 E4 Preregistered Protocol** freezes 8g no-threshold target data, Protocol A/B D28 partitions, last2+head transfer from three hashed independent source checkpoints, L0=50, B=10, 15 rounds, K=3, five acquisition methods, recovery/AULC metrics, tail diagnostics, and pilot expansion gates. The preregistration is a design artifact; formal training is explicitly deferred.
- **D40 E4 Protocol A Engineering Smoke** checks source loading, partition isolation, five acquisition dry-runs, Random/Hybrid query→reveal→source-reinitialize→retrain, parameter freezing, and resume. It is engineering evidence only and has no scientific conclusion.
- **E4-A2a Engineering Smoke** is now complete: exact nested L0=30 partitions, K=3 source/freeze checks, five-strategy dry-run, Random/Hybrid 30→40, strict source reset, and resume identity all passed. Formal A2a remains deferred pending manual approval and was not started.
- **E4-A2a Formal** is complete. Across 3 seeds × 5 frozen strategies × budgets 30–100, no active strategy achieved both a mean AULC improvement over Random and at least 2/3 paired wins; evidence is null. Initial L0=30 recovery was mixed (0.748, 0.895, 1.144), and the headroom hypothesis is not supported by A2a.

Long-term, after the E4 pilot is stable, move reusable code into `src/qgeognn_al/{model,features,engine,acquisition,metrics,partitions,artifacts}.py` so reusable core no longer imports `run_*.py`. This refactor is explicitly deferred until then.

- **Hybrid causal diversity control**：仅当row/compound中Hybrid持续优于Ensemble且fixed common-reference diversity同时更高时，比较Ensemble Top-25%后Random-B与farthest-first-B。
- **Quantile Width post-hoc acquisition**：仅当row/compound risk ranking持续强但Ensemble AL utility弱或不稳定时，作为secondary control，不改写E2预注册主比较。
- **Coverage representation ablation**：仅当Coverage/Hybrid在row与compound稳定优于Random时，比较`h_graph`、conditions、联合、block-balanced联合及MorganFP+conditions。
- **Compound-stratified Random**：仅在需要判断Coverage收益是否只是覆盖更多compounds时使用，不是当前primary baseline。

## B. E4 preregistration checklist

- Protocol B primary固定为canonical-compound-held-out；Bemis–Murcko scaffold OOD延后到E5。
- labels-to-90%-reference使用error-gap recovery：`(E_baseline-E_t)/(E_baseline-E_full) >= 0.90`；`E_full`是full-data reference而非数学ceiling。
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

## G. E4 Protocol A status

三seed正式pilot已完成。Pretrained Random平均normalized AULC最低；Coverage、Ensemble、Hybrid与Quantile Width均只在1/3 seed胜Random，证据分类为null。Pretrained Random仍明显优于scratch Random，支持迁移价值但不支持复杂主动采样优势。按预注册停止，Protocol B、E3、E5及25g/40g均未启动。

## H. D42 — Protocol A Headroom & Acquisition-Shock Audit

D42是纯post-hoc descriptive离线审计，不训练模型、不修改E4方法。Seed42的L0 initial recovery为0.742，明显低于seed525/1101的0.930/0.912；后两者在首次active query前已达到90% reference recovery。高饱和seeds的四种active在50→60均恶化，而Random改善；reveal后truth显示active首批与更高source residual及label extremeness相关。该truth只能作历史机制分析，绝不能进入acquisition。`queried_union_top_decile_*`仅指首轮已查询样本union，不是完整U0 top decile。Batch diversity不能单独解释shock；convergence仅为混合弱线索，normalized validation score也因train-variance denominator随策略/轮次变化而不作跨策略绝对证据。Protocol A primary null保持冻结。

## I. D43 — Transfer-Aware Acquisition Qualification

D43检验新的secondary hypothesis：“在已有强source prior的active transfer中，样本价值不仅由target uncertainty决定，还可能依赖source→target prediction correction magnitude以及该correction region在target pool中的representativeness。”这不修改E4 Protocol A的confirmatory null，也不把D42的post-hoc结果升级为确认性证据。

三seed、L0=50的unlabeled qualification显示T1 prediction shift提供了不同于Ensemble的ranking，T2也不等同于旧Ensemble/Hybrid proxy。T3的deterministic facility-location在3/3 seeds改善top-25% informative shortlist覆盖，并提高所选点的U0 density；但它在3/3 seeds未同时保留T2至少80%的mean transfer shift与target uncertainty，因此整体representativeness objective未通过。Gate为false，条件式seed42 L0=30 smoke未运行，E4-A2 formal未获批准。该结果只验证方法行为，不含performance结论。

以下方向继续deferred，本阶段未实现：Expected Gradient Length / Expected Model Change、BADGE-style regression gradient embeddings、influence-function acquisition、BALD / fully Bayesian transfer、LCMD / B3AL / MaxDet、learned acquisition policy、cost-aware chromatography acquisition，以及显式`y_8g-y_4g` delta surrogate。任何后续实现都需单独人工批准和新预注册，不能用D42 reveal truth或test回调设计。

## J. D44 — AL Suitability & Model-Update Diagnosis

当前项目采用一条方法设计原则：active-learning strategy应在诊断dataset structure、model uncertainty/error relation、source-target shift和model-update behavior后选择，而不只依据常见benchmark方法；这只是本项目的设计原则，不写成普遍定理。

D44复用E4 Protocol A合法历史状态，发现U0同时包含显著compound coverage缺口与大量同compound condition repeats；137D nearest-L0距离多数由graph block主导。High-shift区域在3/3 seeds均为低density。历史Round1 reveal诊断中，shift与uncertainty都和更大per-sample gradient同向；QWidth/Hybrid的参数与function update最大，Random最小，且首轮function update与shock方向描述性一致。但跨全部225个历史batch时，信息score与下一轮test变化没有稳定单调关系，因此不能把“大梯度”直接等同于“高training utility”。

Soft T3R固定只检查λ=0/0.1/0.2/0.3：λ=0.1基本仍是T2，λ=0.2的density改善以seed42/1101 shift retention失败为代价，λ=0.3信息损失过大。无λ通过预注册gate，conditional smoke未运行，transfer-aware performance继续deferred。E4-A2a只预注册低L0 headroom单变量实验，尚未执行。
