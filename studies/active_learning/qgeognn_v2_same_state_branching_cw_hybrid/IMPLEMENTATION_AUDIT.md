# Scientific implementation audit — exact-selector readiness

Base: latest fetched `origin/main` = `135fbb37fa63c91662f5ef5338349121cb50c49c`. This continues the pre-existing partial repair; no historical trajectory was modified. The pilot remains a development/mechanism study on previously used seeds, not independent confirmation.

## Findings in the original implementation

1. **CW was scientifically mismatched.** `_materialize_anchor_gradient` and `_extract_branch_bank` extracted ordinary scaled endpoint/q50 gradients; `select_batch` then dispatched those features to LCMD under the `center_width_lcmd` name. Historical CW instead differentiates L0-scaled center and width outputs. CountSketch acts on the concatenated output gradients; renaming or reweighting the already sketched endpoint bank is not an exact replacement. Thus the old comparison could not measure headroom over historical always-CW.
2. **The original Hybrid was not historical Hybrid.** It ranked gradient norms, retained 25%, then used gradient-space farthest-first. Historical Hybrid ranks K=3 ensemble disagreement and selects in the current member-0 latent space. Both uncertainty and geometry differed, so the old name could not support a historical-method interpretation. The proxy is removed, not silently renamed into a primary arm.
3. The unfinished repair read CW transform metadata from the wrong context structure, so it would fail at preparation or branch context creation. CW metadata is in the flat CW context; the baseline Hybrid context has no CW transform. Both code paths now compare against the correct frozen CW L0 transform.
4. The unfinished regression could compare an incoming `selected_batch.csv` against an outgoing proposal. Budget 429's true next selection is in CW-to-525 round 03; budget 653's is in CW-to-1005 round 10. The new audit checks those continuation receipts, their exact checkpoint SHA and current L/U order.
5. The old preflight file tested the separate `same_state_branching` module. New sealing requires this pilot's dedicated tests and complete exact historical regression evidence. A smoke result must be bound to the current seal SHA. Stale caches fail closed rather than being silently reused.

## Repairs and fixed method geometry

CW now calls `center_width_transform(l0_truth)` and `extract_linear_output_gradient_sketches` at every anchor and after the first branch fit. C=(V1+V2)/(2*s_C), W=(V2-V1)/s_W; both population scales come only from the seed's original 333 L0 rows and remain fixed after acquisition. The exact historical CountSketch uses 512 dimensions, `sketch_seed=outer_seed+4000037`, the same parameter/output ordering, and the same LCMD-TP routine. Current ordered L is installed as centers; current ordered U and stable tie order are retained. An explicit CW bank is required; passing a bare shared q50 bank to `select_batch` is rejected.

IVR reuses the historical q50 extractor and `conditional_batch_ivr`: L0 endpoint scales, global RMS over the original 3330-row outer training reference set, unit prior/noise, current-L conditioning, and 32 sequential covariance updates. MaxDet reuses the pure historical `conditional_gradient_maxdet`: the same q50 geometry, **current-L-only RMS**, current-L information conditioning, and sequential D-optimal updates without an ensemble gate. Representation/normalization are method definitions; same-state fairness fixes labels and checkpoint, not feature transforms.

New lineage protects exact checkpoint receipts, original states, ordered L/U, both bank hashes, graph caches, and fixed transform metadata. Diagnostics now separate common validation/count observables, `raw_gradient_*`, `cw_gradient_*`, `ivr_*`, and `maxdet_*`. Post-first-fit dispatch and cache provenance are regression-tested without training a model.

## Historical ordered-ID regression evidence

All gradient regressions start from newly extracted checkpoint banks; later audit runs verify and reuse those content-addressed artifacts. Hybrid controls re-extract member-0 latent vectors and current-U predictions, then replay the checksum-protected historical member-1/member-2 prediction CSVs exactly as the historical selector consumed them. Ensemble initialization seeds, label counts, checkpoint and acquisition-state identities are checked. No test truth is needed.

| Method | Exact states | Intersection | Ordered IDs exact | Re-extracted representation exact |
|---|---:|---|---|---|
| Center/Width-LCMD | 4/4 | 32/32 at every state | True (including order) | True |
| Historical Hybrid source control | 4/4 | 32/32 at every state | True (including order) | True |
| Kernel-IVR | 2/2 | 32/32 at every state | True (including order) | True |
| Gradient-MaxDet | 2/2 | 32/32 at every state | True (including order) | True |

CW covers seeds 157/6101 at both 429 and 653; the IVR and MaxDet regressions cover both seeds at 429. For all three gradient methods, the maximum absolute difference from the historical bank is zero after canonical reindexing. CW transform hashes, gradient hashes, checkpoint SHA256, ordered batch hashes, intersection counts and exact-match flags are in `results/cw_selection_regression_audit.csv`; parallel files cover the other methods. Hashes for selected IDs preserve order; set equality alone cannot pass.

`results/selector_identity_and_overlap_audit.csv` has 24 source-versus-proposal comparisons across eight anchors. Every same-name source reconstruction is exact. Across the 24 primary-strategy pairs, batch intersection ranges from 1 to 22 of 32; 24/24 pairs select different sets. Thus observed proposal divergence is not attributable to a failed historical identity check. This is **selection divergence**, not evidence of future performance gain.

## Historical Hybrid disposition

Historical Hybrid uses independent member initialization seeds `outer_seed+2000003+member*1000033`, K=3, and q50 columns 1/4. The uncertainty score is the Euclidean norm of population ensemble standard deviations divided by fixed L0 endpoint scales. The stable-ranked shortlist size is `max(32, ceil(0.25*|U|))`. Coreset selection uses member-0's 128D representation immediately before the output head (graph global-sum pooling plus condition residual), all current L as initial centers, and shortlist order for ties. It needs two additional member fits per acquisition state when those artifacts are absent.

**Decision B:** the four historical Hybrid anchors reproduce exactly, but all four CW anchors lack same-state member-1/member-2 checkpoints (`historical_hybrid_artifact_inventory.csv`). Same-budget models from a different trajectory have different L and cannot substitute. Therefore the complete primary Hybrid arm cannot be reconstructed from current artifacts without new training. Primary strategies are **exact CW, Kernel-IVR, Gradient-MaxDet**. Historical Hybrid remains a source trajectory and an exact source-identity control only. This choice is based on missing artifacts/method identity, not observed performance or a rule that ensembles are forbidden.

An exact Hybrid variant replacing MaxDet could be implemented in a separately authorized protocol with legal current-training/validation labels: 4 missing anchor ensembles × 2 fits = 8, plus 8 Hybrid branch-after-1 ensembles × 2 fits = 16; **24 extra fits, 72 total**. This audit neither performs nor schedules them.

## Validation, cost and readiness

- Dedicated pilot tests plus LCMD/CW, IVR, MaxDet and gradient regressions: **58 passed, zero failures/skips**, recorded in `preflight_tests.xml`. Tests cover explicit geometry, permuted current L/U, fixed L0 scaling, current-checkpoint postfit extraction, stale-cache/seal rejection, same-set/wrong-order rejection, and no test access before global freeze.
- Full historical selection audit: 12/12 exact ordered batches. Deterministic selection smoke covers all eight anchors × three primary strategies twice, without label reveals or fits.
- `results/gradient_transform_audit.csv` records 16 anchor-bank contracts and mapping hashes. `selector_audit.json` hashes the evidence and all used historical source artifacts. Historical state files remain read-only.
- The audit firewall rejects training and test reveals; recorded accesses are L0-only for transforms. Unit context checks additionally access legal validation labels. **Actual new fits = 0; test truth access count = 0.** No `--execute-seed`, `--freeze`, `--report`, or finalizer command was run.
- The obsolete seal is invalidated and preserved under `runtime/deprecated_pre_exact_audit`; `superseded_seal.json` records its SHA and reason. The replacement seal protects protocol, config, implementation, tests, lineage and immutable audit evidence; outcome CSVs remain separate from the audit evidence.
- Fixed design: 2 seeds × 2 source trajectories × 2 budgets × 3 strategies = **24 branches; 48 planned scratch fits; 0 ensemble acquisition fits**. Eight historical anchor checkpoints are reused. Endpoints remain 493/717; no performance metric or training protocol was changed to obtain a result.
- Compute estimate from eight historical anchor fit timings: mean 537.9 seconds/fit, approximately **7.17 hours of sequential fitting**, excluding gradients and selection. Using the observed min/max gives 2.97–18.33 hours; this is a planning range, not a pilot measurement. There will also be 48 post-first-fit bank extractions (24 branches × two geometries). Details are in `results/planned_compute_cost.json`.

The scientifically interpretable question is now the local-oracle headroom among these **three exact methods**, evaluated from common states. No adaptive benefit, winning strategy, or controller viability has yet been measured. After the passing audited seal/smoke, the final state is `READY_FOR_EXECUTION_AFTER_EXACT_SELECTOR_AUDIT`, and this task stops there.

The next command, only for a subsequently authorized training run, is:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_same_state_branching_cw_hybrid.py --execute-seed 157
```

The local environment uses the repository's existing OpenMP compatibility flag. One preliminary invocation encountered duplicate OpenMP initialization and another aborted; no output from either is accepted as evidence. Completed regressions use two Torch threads with one BLAS thread and demonstrate exact array/selection parity. No dependency or historical model was modified.
