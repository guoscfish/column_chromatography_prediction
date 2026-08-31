# Research Direction after the Reset

## Portfolio logic

```text
                    ┌─ Track A: 4g in-domain AL
4g QGeoGNN ─────────┤  promising but unfinished
                    └─ Track B: 4g→8g transfer adaptation
                                      │
                                      └─ Track C: active transfer (deferred)
```

Track A and Track B are independent but related. Track C depends on Track B; D42–D46 do not form an automatic route to another acquisition experiment.

## 4g In-domain AL Status

**Final strategy found? No.** Hybrid is the strongest current candidate and Coverage is a strong, simpler candidate, but neither is frozen.

**Effect strength.** Row normalized AULC is Hybrid 0.542938, Coverage 0.562489, Ensemble 0.626849, Random 0.644749. Hybrid and Coverage beat Random in 3/3 outer seeds: promising/strong pilot evidence. Compound AULC is Hybrid 0.761203, Coverage 0.777107, Random 0.788732, Ensemble 0.802149; Hybrid/Coverage win only 2/3 seeds: suggestive only. Overall status is `PROMISING_BUT_UNFINISHED`.

**Why it cannot freeze.** There are only three outer seeds; compound evidence is not 3/3; Hybrid's mechanism lacks causal isolation; advanced batch AL has not been adequately compared; Quantile Width has offline signal but no full E2 AL curve; the Coverage representation concatenates a 128D graph block and 9D conditions after per-dimension z-scoring, allowing graph dimensionality to dominate; scaffold/general OOD is absent; and evidence exists only for the QGeoGNN surrogate.

**Is continued 4g AL worthwhile?** Yes, because row evidence is consistent and the Hybrid/Coverage pattern points to batch redundancy as a testable mechanism. The next study should be narrow enough to identify that mechanism rather than add many acquisitions.

## Track A Future Method Map

### A1. Hybrid mechanism control

`PROPOSED — NOT RUN`. Compare Random, Ensemble, uncertainty-top25%-Random-B, Hybrid(top25%-farthest-first), and Coverage. The random-within-shortlist arm isolates uncertainty filtering from diversity/de-redundancy. A1b is allowed only if A1a supports diversity, and then selects 1–2 advanced methods after relevance, feasibility, QGeoGNN compatibility, and novelty review.

### A2. Quantile Width secondary AL

Status: `offline-qualified-secondary`; full 4g AL not run. It is neither failed nor equivalent to epistemic uncertainty. It may enter a future study as a secondary baseline without altering E2's primary comparison.

### A3. Advanced batch diversity

Deep Batch Active Learning for Regression studies kernel/gradient-based batch selection including LCMD and MaxDet; Black-Box Batch Active Learning for Regression extends families including LCMD to prediction-only models. These directly optimize batch information/redundancy rather than Hybrid's heuristic uncertainty shortlist plus Euclidean farthest-first. They require representation/kernel and compute review before use.

### A4. 3D molecular-graph-aware AL

Subedi et al., *Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism* (NeurIPS 2024), combines geometry-distribution diversity with Bayesian geometric-GNN uncertainty. QGeoGNN candidates also contain chromatography conditions, so the method cannot be copied directly. A relevant adaptation would balance molecular-geometry and condition-space blocks explicitly instead of naively concatenating 128D graph and 9D condition features.

### A5. Better epistemic uncertainty

Partially Bayesian or Bayesian GNNs and ensemble variants are candidates. Allec & Ziatdinov (2025) show partially Bayesian active/transfer learning mainly with neural-network settings that do not establish direct QGeoGNN validity. Any adoption needs a graph-specific calibration and compute study.

### A6. Representation ablation

Preregister comparisons among graph-only, conditions-only, naive concat, block-balanced concat, Morgan fingerprint plus conditions, and a geometry-aware representation. Do not run during this reset.

### A7. Generalization protocol

Separate row interpolation, compound-held-out, and Bemis–Murcko scaffold OOD. Never extrapolate row-level evidence into a novel-molecule claim.

## 4g Active Learning Literature Map

| Area | Relevant work/idea | Project implication |
|---|---|---|
| Regression and batch AL | Holzmüller et al., *Deep Batch Active Learning for Regression* (JMLR 2023) | LCMD/MaxDet offer principled alternatives to heuristic farthest-first. |
| Black-box batch AL | Kirsch, *Black-Box Batch Active Learning for Regression* (TMLR) | Prediction-only kernels may reduce coupling to QGeoGNN internals. |
| 3D molecular graph AL | Subedi et al., NeurIPS 2024 | Geometry-aware diversity is relevant, but conditions require a separate balanced block. |
| Bayesian/epistemic UQ | Allec & Ziatdinov, Digital Discovery 2025 | Partially Bayesian layers may lower UQ cost; evidence is not QGeoGNN-specific. |
| Diversity-aware AL | LCMD, MaxDet, uncertainty-plus-diversity | Test only after A1a isolates whether diversity actually drives Hybrid. |

Primary sources: https://jmlr.org/papers/volume24/22-0937/22-0937.pdf ; https://arxiv.org/abs/2302.08981 ; https://papers.nips.cc/paper_files/paper/2024/hash/6462073c6bdf864ebfbbb11e80619f3e-Abstract-Conference.html ; https://doi.org/10.1039/D5DD00027K

## Transfer / Active Domain Adaptation Literature Map

### Current 4g→8g shift understanding after S1

S1 used only the analysis role of a truth-blind compound partition. Frozen-source residuals were strongly positive and V1/V2 corrections were correlated. Within-compound variation dominated between-compound clustering, while nearest 4g condition distance had only weak descriptive association with residual. Affine calibration roughly halved mean combined group-CV NRMSE relative to zero-shot; condition-aware Ridge improved zero-shot but did not improve on affine. This makes affine calibration a required T1 baseline, not proof that nonlinear or molecular adaptation is unnecessary. The reserved role remains an S1-unconsumed set, not a globally clean confirmatory test.

| Area | Relevant work/idea | Project implication |
|---|---|---|
| Small-data GNN transfer | Adaptive readouts and low-data molecular transfer | Motivates comparing head/readout, residual, and frozen-feature formulations before acquisitions. |
| Multi-fidelity inspiration | Low-fidelity molecular GNN knowledge transferred to sparse high-fidelity targets | 4g/8g are domains, not automatically fidelities; use as design inspiration, not an equivalence claim. |
| Active transfer in chemistry | Active transfer learning for reaction-condition prediction | Shows the workflow is plausible but task/model differences prevent direct performance expectations. |
| Bayesian transfer/UQ | Partially Bayesian networks initialized from pretrained weights | Candidate after deterministic adaptation baselines are established. |
| Active domain adaptation | Selection under domain shift | Track C only; defer until Track B produces a stable target formulation. |

Primary sources: https://www.nature.com/articles/s41467-024-45566-8 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC9172577/ ; https://doi.org/10.1039/D5DD00027K

## Predictor versus acquisition scope

Other acquisition methods remain a Track A priority. Broad predictor-architecture search is not timely: changing predictor and acquisition together makes an AL gain uninterpretable. Keep QGeoGNN fixed; add at most one secondary predictor robustness check only after an acquisition improves stably across row and compound/scaffold protocols.
