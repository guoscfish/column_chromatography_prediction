# Same-state exact-selector short-rollout protocol

Revision: exact-selector scientific implementation audit of `135fbb37`, before any pilot fit or test reveal. The former seal is invalidated; its artifacts are retained under `runtime/deprecated_pre_exact_audit/`. The directory name is retained for lineage compatibility, not as the primary strategy list.

## Scientific question and fixed design

At an identical ordered `(L_t, U_t, checkpoint_t)`, do exact historical acquisition mechanisms produce different two-batch future returns, and how much local-oracle headroom remains over always-CW? This is development/mechanism evidence on reused seeds, not independent confirmation or evidence that a controller already helps.

- Outer seeds: 157 and 6101. Source trajectories: historical Center/Width-LCMD and historical Hybrid.
- Anchor budgets: 429 and 653; eight source states in total. Shared validation: 416 rows; original outer training universe: 3330 rows.
- Primary branch strategies: `center_width_lcmd`, `kernel_ivr`, `gradient_maxdet`.
- Each state has three isolated branches, each acquiring 32 then 32 rows: endpoints 493 or 717. Total: 24 branches, 48 future scratch evaluation fits, zero acquisition ensemble fits.
- All branches use the exact ordered anchor labels/model state and the same member-0 initialization/training protocol. After +32, the branch's own fitted checkpoint and updated ordered L/U drive the next acquisition. Scratch fitting remains the historical protocol; the shared checkpoint is the acquisition model, not a warm-start training change.
- Primary future outcome: `short_AULC=(NRMSE_anchor+2*NRMSE_plus32+NRMSE_plus64)/4`. Compare local oracle and always-CW within each anchor, then descriptively across seeds, budgets, and source trajectories. Do not select methods from performance during this audit.

## Exact method definitions

Same-state fairness shares L, U, checkpoint, labels, and training protocol. A representation and its normalization are part of an acquisition mechanism; methods must not be forced to share a transformed bank.

**CW:** recompute `center_width_transform(L0_truth)` with the same outer seed's 333 L0 labels, population standard deviations (ddof=0): C=(V1+V2)/(2*s_C), W=(V2-V1)/s_W. These scales remain fixed for the complete trajectory. Extract full-network linear-output gradients with `extract_linear_output_gradient_sketches` at the anchor and again from the branch-after-1 checkpoint. CountSketch dimension 512; seed=`outer_seed+4000037`; parameter/output ordering identical to the historical extractor. Call the historical `lcmd_tp_select`, installing all current ordered L as centers and current ordered U as candidates. Cluster score is the sum of squared nearest-center distances; stable center/pool order resolves ties. No ordinary endpoint-gradient substitute is accepted.

**IVR:** use `extract_q50_gradient_sketches` with L0 endpoint scales, same 512D CountSketch/seed and original L0+U0 reference order. Reuse `conditional_batch_ivr` unchanged: global RMS over all 3330 reference rows; unit prior precision and unit observation noise; covariance conditioned on current L; uniform integrated variance over the original outer universe; 32 sequential Sherman–Morrison updates; stable current-U tie order. No test features or targets enter the acquisition universe.

**MaxDet:** use the historical ordinary q50 gradient extractor and `conditional_gradient_maxdet`, with RMS scaling over current L only, current-L information conditioning, and sequential conditional D-optimal rank-one updates. This is the pure historical Gradient-MaxDet arm, without an uncertainty gate. Row reindexing preserves historical current-L and current-U order.

## Historical Hybrid decision (rule B)

The former gradient-norm-top25 plus gradient-space farthest-first arm was not historical Hybrid and is removed entirely. It is neither a primary arm nor silently retained under the Hybrid name.

Historical Hybrid uses K=3 independently initialized current-state models. Score is the Euclidean norm of the two population standard deviations of q50 predictions (columns 1 and 4), divided by fixed L0 endpoint scales. Stable descending ranking defines `max(32,ceil(0.25*|U|))` candidates. Farthest-first uses the member-0 128D representation before its linear output head (graph sum pooling plus condition residual), all current L as initial centers, and shortlist rank order for ties.

The four Hybrid source anchors have their exact ensemble artifacts and are selection-regressed as a source-identity control. The four CW source anchors lack same-label-state member-1/member-2 checkpoints; models from a Hybrid trajectory with the same budget have different labels and cannot substitute. Therefore an artifact-only complete Hybrid arm is unavailable. Apply the explicitly permitted rule B and use CW/IVR/MaxDet. This decision uses artifact completeness and method definitions, not performance. A separately authorized exact-Hybrid version could train the missing models test-blind: eight additional anchor fits plus sixteen branch-after-1 ensemble fits, increasing 48 to 72 planned fits if replacing MaxDet. Cost alone is not the reason for algorithm substitution. No such fits are run in this audit.

## Diagnostics

Common observables: current validation NRMSE, fixed two/three-step validation slopes, active-label count, candidate-pool size. Ordinary endpoint-gradient statistics use `raw_gradient_*`; CW statistics and coverage use `cw_gradient_*`; reducible variance uses `ivr_*`; marginal logdet gain uses `maxdet_*`. The ordinary q50 bank is already endpoint-scaled by L0; “raw” means before method-specific RMS normalization. Geometry-specific norms/coverage are never pooled under one feature name. No test outcome is a state diagnostic.

## Pre-execution evidence and firewall

1. Recompute anchor ordinary and CW banks from historical checkpoints; audit checkpoint SHA, ordered L/U, original universe, fixed transform, CountSketch mapping, and historical receipts.
2. Compare **ordered IDs**, not just sets, against historical CW outgoing selections. Budget 429 uses the CW-to-525 continuation's round 03 outgoing selection; budget 653 uses CW-to-1005 round 10. The anchor's `selected_batch.csv` is an incoming batch and is never a reference for this test.
3. Require four CW exact matches, four historical Hybrid source-identity exact matches, and at least one historical-state IVR and MaxDet exact match; persist full proposal lists and hashes. A mismatch blocks readiness; no “approximately equal” acceptance.
4. Produce `selector_identity_and_overlap_audit.csv` comparing each source's historical outgoing batch, its same-name exact reconstruction, and all primary proposals. Divergence is descriptive acquisition evidence, not a measured future gain.
5. Dedicated unit/regression tests, source hashes, audit evidence, protocol, config and implementation are prerequisites of the new seal. The passing smoke must reference that seal's exact SHA before execution is allowed.
6. Audit label access permits only L0/validation; the audit intercepts training and rejects test reveals even if a store is otherwise frozen. New fits=0 and test truth access=0. Source predictions/checkpoints remain read-only. No test metric or truth is read to guide any change.
7. After future execution, all branch acquisitions, checkpoints and predictions must pass one global freeze before the separate test-report action is legal. Seed 6101 requires seed 157's passing test-blind stage gate. No controller is trained.

## Stopping rule for this task

Stop at `READY_FOR_EXECUTION_AFTER_EXACT_SELECTOR_AUDIT`. Do not invoke `--execute-seed`, `--freeze`, `--report`, or the finalizer. The next separately authorized execution command is documented in README. Runtime estimate is based only on historical fit timings; no performance outcomes are used.
