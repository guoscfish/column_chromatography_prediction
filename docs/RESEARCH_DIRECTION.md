# Research Direction after the Reset

## Portfolio logic

```text
                    ┌─ Track A: 4g in-domain AL (paused)
4g QGeoGNN ─────────┤  predictor regression resolution first
                    └─ Track B: 4g→8g transfer adaptation (paused)
                                      │
                                      └─ Track C: active transfer (deferred)
```

Track A and Track B are independent but related, but both are currently paused pending the R2-pruned quantile-head and UQ qualification gates. Track C depends on Track B; D42–D46 do not form an automatic route to another acquisition experiment.

## Current Predictor Gate

The E0 ladder localized the performance failure to R2 → R3 Clean. Function-preserving pruning now removes all 318,856 dead registered parameters from R2 and exactly reproduces the historical R2 checkpoint predictions and controlled retraining. The mainline is `Legacy historical → Condition Completion V2 → R2-pruned candidate baseline`. Clean is `FAILED_POINT_PERFORMANCE_ARCHITECTURE_EXPERIMENT / NOT_BASELINE`, with engineering reachability PASS and performance FAIL. The next separate test changes only the quantile head; no Clean repair, UQ, transfer or AL was run. See [predictor roadmap](roadmap/PREDICTOR.md).

## 4g In-domain AL Status

**Final strategy found? No.** Hybrid is the strongest current candidate and Coverage is a strong, simpler candidate, but neither is frozen.

**Effect strength.** Row normalized AULC is Hybrid 0.542938, Coverage 0.562489, Ensemble 0.626849, Random 0.644749. Hybrid and Coverage beat Random in 3/3 outer seeds: promising/strong pilot evidence. Compound AULC is Hybrid 0.761203, Coverage 0.777107, Random 0.788732, Ensemble 0.802149; Hybrid/Coverage win only 2/3 seeds: suggestive only. Overall status is `PROMISING_BUT_UNFINISHED`.

**Why it cannot freeze.** There are only three E2 outer seeds; compound evidence is not 3/3; scaffold/general OOD is absent; and evidence exists only for the QGeoGNN surrogate. E2 Random was FullPool-Random. A1a compared farthest-first Hybrid with Random inside the same Top25% ensemble-uncertainty shortlist and failed its diversity gate. That narrows the unsupported mechanism to farthest-first's incremental value; it does not prove uncertainty filtering ineffective. A direct paired Top25%-Random vs FullPool-Random comparison has not been run.

**Current status:** `PAUSED_AFTER_A1A / PROMISING_BUT_UNFINISHED`. Continued 4g AL may be worthwhile under a new independently justified hypothesis, but A1b/LCMD/MaxDet are not authorized from the present evidence.

## Track A Future Method Map

### A1. Hybrid mechanism control

`COMPLETED — STOPPED`. A1a compared farthest-first Hybrid against Random inside the exact same Top25% uncertainty shortlist. The diversity-mechanism gate failed, so A1b is not authorized. Because both arms shared the uncertainty prefilter, A1a did not isolate Top25% filtering against FullPool-Random.

### A2. Quantile Width secondary AL

Status: `offline-qualified-secondary`; full 4g AL not run. It is neither failed nor equivalent to epistemic uncertainty. It may enter a future study as a secondary baseline without altering E2's primary comparison.

### A3. Advanced batch diversity

Deep Batch Active Learning for Regression studies kernel/gradient-based batch selection including LCMD and MaxDet; Black-Box Batch Active Learning for Regression extends families including LCMD to prediction-only models. These directly optimize batch information/redundancy rather than Hybrid's heuristic uncertainty shortlist plus Euclidean farthest-first. They require representation/kernel and compute review before use.

### A4. 3D molecular-graph-aware AL

Subedi et al., *Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism* (NeurIPS 2024), combines geometry-distribution diversity with Bayesian geometric-GNN uncertainty. QGeoGNN candidates also contain chromatography conditions, so the method cannot be copied directly. A relevant adaptation would balance molecular-geometry and condition-space blocks explicitly instead of naively concatenating 128D graph and 9D condition features.

### A5. Better epistemic uncertainty

Partially Bayesian or Bayesian GNNs and ensemble variants are candidates. Allec & Ziatdinov (2025) show partially Bayesian active/transfer learning mainly with neural-network settings that do not establish direct QGeoGNN validity. Any adoption needs a graph-specific calibration and compute study.

### A6. Representation ablation

I0 confirmed a semantic mismatch: the predictor consumes bond length plus only four of the nine condition dimensions, while acquisition uses the full standardized 9D condition matrix. The current joint distance representation concatenates 128D graph and 9D conditions. Per-dimension z-scoring does not by itself remove possible block-dimensionality dominance, so 128:9 is a potential risk, not a demonstrated cause of historical performance.

Preregister at minimum graph-only, conditions-only, naive graph-plus-condition concatenation, and block-balanced concatenation. Morgan fingerprint plus conditions and geometry-aware variants may be secondary comparisons. Do not modify historical E2 acquisition or run a new AL study during this stage.

### A7. Generalization protocol

Separate row interpolation, target-label compound holdout, source-aware molecule holdout, and Bemis-Murcko scaffold OOD. Current 8g compound splits prevent a target test molecule from appearing in target adaptation labels, but 87/88 unique 8g compounds occur in 4g. They therefore do not establish source-plus-target novel-molecule generalization. A future source-aware protocol must remove held-out molecules from both 4g source training and 8g target adaptation before target evaluation. It is proposed, not authorized.

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

S1 used only the analysis role of a truth-blind compound partition. Zero-shot combined NRMSE was about 0.803; affine was about 0.399 ± 0.130, with V1/V2 RMSE about 7.63/11.45 mL, but affine won only 3/5 folds. Condition-aware Ridge did not improve on affine. This makes affine a strong required T1 baseline, not proof that 4g→8g transfer is solved or neural adaptation unnecessary. The reserved role remains an S1-unconsumed set, not a globally clean confirmatory test.

T1a formally tested `target_head_only` (prediction head only; pooling remains sum), `last1_head`, and `current_last2_head` alongside zero-shot, affine, and condition-Ridge residual baselines. All 180 fits and 120 contexts completed. `target_head_only` led mean normalized AULC (0.6577 versus 0.7634 for `current_last2_head`) but won only 3/5 paired seeds, so no candidate passed the stable gate. A genuine learnable readout is now a distinct T1b hypothesis requiring separate preregistration; active transfer remains deferred.

T1b-1 separated capacity from insertion-location search with a zero-initialized graph-level residual adapter after sum pooling and widths 8/16/32. All 180 Adapter fits completed. Mean normalized AULC stayed essentially flat but slightly worse than Head (Head 0.6577; r8 0.6583; r16 0.6587; r32 0.6578), and wins versus Head were only 2/5, 2/5, and 3/5. No width passed the frozen gate, so there is no tested low-capacity graph-adapter benefit in the 2,958–9,126 parameter range. The 9,126-to-93,454 gap, including possible 17k/34k/67k regions, remains untested. The result supports output-only correction as the current working baseline over these adapters only; it does not prove latent correction generally ineffective. Since T1b-1 reuses T1a's already-read row outcomes, it is developmental rather than pristine confirmation. No additional widths or T1b-2 study are authorized.

## Predictor semantic audit and V2 boundary

I0 confirmed by code tracing and perturbation that the clean legacy predictor constructs ten continuous edge features but consumes only positions 0–4. Eluent HBA, eluent LogP, loading solvent, loading mass, and loading volume do not reach the encoder. Acquisition nevertheless uses all nine condition dimensions. The audited model has 775,476 nominal requires-grad parameters, 456,620 gradient-bearing parameters, and 318,856 forward-unreachable parameters. These are implementation facts, not a finding that the historical outcomes are invalid.

Predictor V2 preserves the Legacy early condition interaction and adds the missing typed condition path. Its frozen E0 test R² is 0.889218 / 0.941563. R2-pruned now preserves that function and every retained initialization value while removing all audited dead registrations; its complete retraining history and point metrics equal R2.

The predictor sequence is `Legacy historical → Condition Completion V2 → R2-pruned candidate baseline → separate quantile-head controlled test`. The current head remains unchanged. Formal baseline/UQ qualification and any future transfer or AL require separate decisions; the next head experiment has not been executed.

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
