# Balanced CW / IVR portfolio development protocol

Development / exposed cohort; not independent confirmation. Seeds 157 then
6101, irrespective of seed-157 performance. Exactly ten B32 acquisitions and ten
new scratch fits per seed, L333 through L653. No later budgets or quota sweep.

At each current checkpoint, extract full-network unscaled V1/V2 q50 gradients
once per row using the existing multi-transform extractor. Apply the original
L0 Center/Width transform for CW and diag(1/sV1,1/sV2) for IVR, with the original
CountSketch-512 mapping (seed + 4000037). The two compressed banks are distinct
geometries from the same derivatives, not a score or feature fusion. Floating
point distributivity differs from differentiating transformed outputs directly;
preflight checks geometric agreement and exact historical batch IDs. Do not
silently substitute the endpoint bank for CW. Reuse frozen L333 predictor and
test-X predictions, never refit L333. Re-extract its two-geometry bank once;
historical endpoint-only sketches cannot reconstruct both output slots.

Shared batch alternating greedy: zero-based even steps use CW, odd steps IVR.
CW uses largest sum of squared nearest-center distances, then farthest point;
centers are L_t plus every earlier shared pick. IVR uses fixed outer universe
R=L333 union U333, all-R RMS normalization, unit prior precision and unit noise,
C=(I+X_L'X_L)^-1, M=X_R'X_R/|R|, marginal x'C M Cx/(1+x'Cx).
Every shared pick, including CW turns, updates IVR via Sherman-Morrison. Both
experts remove every selected row. Stable current-U order breaks exact ties.
There are exactly 16 CW and 16 IVR turns, 32 distinct locations. Conditioning on
planned locations needs no labels and preserves both original objectives.

Freeze the entire selected batch and its trace before revealing its labels.
Scratch retraining uses historical preprocessing, graph cache, target scales,
initialization, deterministic shuffle, loss, Adam, validation stopping and
checkpoint rule unchanged. The next round uses its new predictor. No warm start.
All 22 curve-point checkpoints/test-X predictions and all 20 batches, gradients,
states and fits must pass recursive hash verification before any test truth
access. During execution test_truth_access_count=0. Historical artifacts read-only.

Primary comparators: pure Center/Width-LCMD and Kernel-IVR. Context comparators:
Gradient-LCMD, Hybrid, Gradient-MaxDet. Reuse only matching seeds, exact budgets,
splits, target scales and metric definitions; no baseline fits. Primary metric
is normalized trapezoidal combined NRMSE AULC 333-653. Also 333-525, 429-653,
525-653 and disjoint stage summaries 333-429,429-525,525-653. Endpoint: combined
NRMSE and V1/V2 RMSE/R2 at 653. Report per seed and arithmetic two-seed means.

Per-seed component oracle is min(CW AULC,IVR AULC). Regret = method AULC minus
this oracle (negative is allowed); report mean and maximum across seeds. Do not
clip negative regret. Historical 333-653 component means are approximately
0.670 and 0.684; tolerance 0.01 is ~1.5% of this scale and smaller than their
0.014 mean gap. Freeze absolute tolerance 0.01 for both AULC and endpoint NRMSE;
it is descriptive, not a statistical threshold. Meaningful regret improvement
means at least 0.01 smaller worst-seed regret than each fixed component.

Decision precedence, fixed before new training:
1. STOP if both seeds' primary AULC exceed the worse component by >0.01.
2. STRONG_PORTFOLIO_SIGNAL if mean primary AULC <= best component mean, every
seed is within +0.01 of its component oracle (AULC and endpoint), and maximum
regret is strictly lower than both fixed components.
3. PROMISING_PORTFOLIO if mean primary AULC <= best component mean +0.01, neither
seed deteriorates >0.01 on AULC or endpoint relative to its oracle, and maximum
regret is >=0.01 lower than both fixed components.
4. NO_CLEAR_PORTFOLIO_GAIN otherwise. Opposite material seed directions must
remain visible and preclude a robustness claim.

Diagnostics cannot change the quota: per-step proposals, CW cluster score and
distance, IVR variance before/marginal/after for all steps, gradient norm
mean/std/p50/p90/p95/max, participation-ratio rank, coverage mean/p90 before and
after, relative variance reduction, independent CW16/IVR16 overlap and union
geometry (union may have fewer than 32 points; this is not a controlled ablation),
and expert-specific coverage/variance contributions. No hidden-U/test labels or
future performance enter acquisition. Distinct selections alone do not prove
predictive complementarity. No causal claim about quota or optimization noise
without a corresponding controlled experiment.

Preflight: historical compatibility audit, selector regression tests, synthetic
shared-context/duplicate/resume/firewall tests, and real selection-only smoke.
Then seed157, engineering-only integrity check, seed6101, global freeze, unified
test reveal/report, commit. Adaptive quota may only be proposed in the final
report if justified; never run it here.
