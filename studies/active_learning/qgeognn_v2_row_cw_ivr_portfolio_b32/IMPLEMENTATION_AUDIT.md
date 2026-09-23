# Method and repository audit

Latest origin/main at initial audit: aaf3e150091ffb219d6a2ff69e31e095bb431b50.
Fetch and fast-forward check succeeded. Relevant recent commits: aaf3e15 (CW653),
db7fbee (same-state branches), e28daef (CW525 and switch development), 04efbed
(short sequential and switch setup). Existing untracked CW1005 work is outside
this study and is preserved.

Reviewed protocol/report/decision and implementation for sequential B32, IVR,
CW525, CW653, same-state branching and LCMD-to-IVR. The latter has
`DEVELOPMENT_REPORT.md` and `development/manifest.json`, not a top-level completed
five-seed FINAL_REPORT/decision: only the two-seed development cohort completed.
Relevant frozen metrics and corresponding tests were inspected. CW653 reports
AULC333-653 CW 0.669884 vs IVR 0.683845; CW wins both seeds, hence fixed CW has
zero component-oracle regret in this window. Same-state branching's coverage
expert was **Gradient-LCMD, not CW-LCMD**. Its overlap evidence motivates this
hypothesis but does not already demonstrate CW/IVR complementarity.

Historical LCMD exposes only whole-batch selection. Its loop installs L as
centers, chooses the largest cluster by sum of squared nearest-center distances,
then picks its farthest member; each pick updates all available distances using
strict '<'. Historical IVR also exposes a whole-batch API and requires L/U to
partition the reference universe. It initializes unit-prior covariance from L,
uses the all-R second moment and RMS scale, and conditions each selected location
with the same Sherman-Morrison factors q and h. Neither objective needs labels
for batch conditioning.

New incremental adapters live in `cw_ivr_portfolio.py`. Historical selector and
training source files are unchanged, preserving existing seals. Adapters expose
`propose` and `condition`; regression compares their complete pure B32 output to
the historical batch APIs and direct covariance recomputation. External picks
are mathematically equivalent to planned batch centers / observation locations.
Shared interleaving is well defined; an independent-proposal fallback is not needed.

Representation uses the existing multi-transform gradient extractor. Raw q50
endpoint derivatives are computed once per row and projected into both output
slots. The frozen endpoint transform and CW transform generate two distinct
512D banks. This is linear algebraically equivalent to the historical transformed
output differentiation but not bit-identical due to floating point accumulation.
Selection-only smoke checks relative feature error <=1e-5 and exact historical
pure CW/IVR selected IDs at both actual L333 states. The resulting shared bank is
frozen and reused during execution. No current representation is reused across
later trajectory-specific checkpoints.

L333 audit reconstructs historical preprocessing in a temporary directory,
checks graph/source/scaler hashes and split identities, training config, original
initialization state hash, loaded checkpoint state hash, all nested source round
artifacts and global binding. The existing narrowly scoped AST compatibility
audit permits only additive gradient APIs; original predictor dynamics and
numerical package versions must match. New fits enforce the original initialization
hash again. Comparator grids, splits, full preprocessing and original published
metrics are checked before freezing. Source artifact hashes are checked again
before each seed and global freeze.

The study uses a single worker lock, seed157-before-seed6101 engineering gate,
write-once state/acquisition artifacts, nested per-round hashes, trajectory hashes,
and a global 22-point barrier. A report calls recursive verification before any
test reveal. Synthetic interrupted trajectories exercise deterministic reuse and
ensure the exact round freeze exists before each label request. Global test
truth is never used to choose seed execution, allocation, or tolerance.

Diagnostic independent CW16 and IVR16 batches are computed without labels or
training; their overlap/union is not the acquisition batch. Union geometry has a
smaller budget when proposals overlap, so it is not an equal-budget ablation.
The report must distinguish duplicate avoidance from evidence of predictive gain.
