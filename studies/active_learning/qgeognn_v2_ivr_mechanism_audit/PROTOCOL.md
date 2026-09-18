# IVR representation and optimizer audit

This exploratory study runs Gate A and the selection-only part of Gate B.
It creates no training fits, opens no raw label source or prediction metrics,
and never evaluates test error. Historical trajectories are already public;
this is not independent confirmation of predictive improvement.

## Fixed design

The 25 states are the existing scalar-IVR trajectories at rounds 1, 5, 10,
15 and 20 for outer seeds 157, 887, 2357, 6101 and 12203. Reuse each frozen
checkpoint, its L/U membership, the fixed 3330-row reference inputs, stored
L0 endpoint scales and a hash-verified graph cache whose labels are zero.
All selectors, including LCMD, are recomputed on this SAME state.
Historical LCMD batches from different trajectories are not controls here.

For each state, one full-network Jacobian pass creates three independent
2048D CountSketch banks. Coarsening contiguous buckets yields 1024/512/256D
maps with exactly the legacy bucket/sign assignments (verified in tests).
Float64 accumulation followed by float32 storage differs in rounding from
historical torch float32 scatter; compare the reconstructed 512D features
and initial rankings to historical cache before analysis. No dimension is
selected using test or validation outcomes.

Shared-parameter endpoint sketches use the first P buckets/signs for BOTH
endpoints; scalar sketches retain the historical independent two-block
mapping. The block bank retains both outputs. A common per-experiment RMS
normalization fixes mean information trace at one, avoiding an implicit
doubling of prior-to-observation strength. Noise and risk weights are I.
These are approximate Gaussian linearizations, not a calibrated posterior
for the complete six-output QGeoGNN training objective.

## Quantities and gates

The exact numeric rules are in config.json, sealed before extraction.
Report full reference feature spectra, entropy effective rank,
participation rank, numerical rank, raw feature condition number and the
condition number of the actual regularized precision. Rank deficiency in
a raw Gram matrix is not by itself a numerical failure of a regularized solve.

Compute initial-score Spearman, initial top-32 overlap, conditional greedy
batch overlap, LCMD overlap and same-objective risk regret for matched maps.
The main scalar reproducibility gates apply separately to cross-map 512D
comparisons and 512D versus 2048D comparisons. Block reproducibility is
tested across the three 512D maps. Other dimensions are diagnostics only.
Low index overlap may be harmless for near-ties, so objective regret is
reported too; it does not override the frozen engineering gate in this run.

At the historical mapping and 512D compare scalar forward, scalar
forward-64/backward-32, and shared-parameter two-output forward IVR.
Backward deletion uses the smallest integrated-risk increase, removes only
pending candidates, and conditions on all labeled rows. No guarantee that
forward/backward beats forward is assumed. Use direct recomputation to
check every final covariance and small-matrix tests for every update.
Integrated risk must decrease on additions and increase on removals;
individual marginal scores are NOT required to be monotone (variance
reduction is not generally submodular).

Endpoint risks and LCMD comparisons are evaluated in each method's OWN
fixed surrogate. Raw scalar and block risk magnitudes are not comparable.
Winning one's own surrogate is a mechanism check, not evidence of lower
NRMSE. No one-step NRMSE or historical test residual selects a candidate.

## Training decision

Multi-output IVR has priority only if BOTH scalar reproducibility groups,
block reproducibility, numerical checks and its mechanism gate pass.
Otherwise scalar forward/backward qualifies only with scalar stability
and its separate mechanism gate. If neither qualifies, stop before Gate C
and report the reasons. Do not silently relax gates or launch bigger sweeps.
Passing qualifies at most one candidate for a separately sealed sequential
experiment, not for COMPOUND or a claim of improved predictive performance.

The prior research note mixed no-label spectra and published-test post-hoc
correlation in one section; this study does not use that correlation.
Low effective rank alone does not justify whitening, and the earlier
cross-trajectory overlap cannot isolate a selector effect. Both distinctions
are enforced by this protocol.

## Provenance and resume

Seal config, protocol, code, test report and all reused input artifacts.
Each completed snapshot has a content-verified bank and output receipt.
Resume accepts only identical code, config and input hashes. Existing
frozen studies are read-only. Full aggregate results require all 25 states.
