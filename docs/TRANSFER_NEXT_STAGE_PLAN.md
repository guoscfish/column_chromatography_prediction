# Transfer Next Stage Plan

> **2026-09-12 research-record addendum.** The full branch map and the strict
> distinction between the low-label matched benchmark and filtered FULL-data
> neural study are in [TRANSFER_RESEARCH_ROADMAP_2026-09-12](research/TRANSFER_RESEARCH_ROADMAP_2026-09-12.md).
> The immediate neural audit is `traditional_transfer_converged_baseline_v1`:
> P0/P1 only, five-fold inner compound GroupKFold within outer gradient-train,
> 500/80 Adam budget, and fixed-epoch full-gradient refits. Outer validation
> is not an epoch-selection set. New ROW tests are developmental confirmation
> because their identities are historically exposed.

This plan follows the Stage 2 audit. It is a development plan, not evidence that the listed experiments have been run.

## Immediate gate

1. Freeze one train-only convergence protocol for the existing filtered ROW populations, source checkpoint, seeds, scope, BN policies, and P0/P1/P2/P3 recipes.
2. Use a longer Adam budget, preferably maximum epochs 500 with patience 80, and record diagnostic checkpoints at 150, 300, and 500. Select checkpoints only from validation evidence.
3. Treat any later outer-test score as developmental confirmation because the current outer test has already been exposed in existing artifacts.
4. Promote a recipe only if it has stable endpoint and unified behavior across both columns, at least 3/5 seed wins per column, no systematic endpoint deterioration above the preregistered tolerance, and a matched comparison against paper-style under the same scales.

Do not choose different recipes for 25g and 40g from the current outer-test ranking.

## Candidate directions after the gate

### Endpoint-normalized loss

Audit and, if justified, reuse `scaled_quantile_target_loss`. A candidate objective is `L_V1 / s1^2 + L_V2 / s2^2`, with `s1` and `s2` fitted only from each training fold. This aligns the training objective with the unified normalized evaluation without using test truth.

### Hard ordered quantile head

The current soft crossing penalty does not guarantee `q10 <= q50 <= q90`. A constrained parameterization such as `q50=m`, `q10=m-softplus(d_low)`, and `q90=m+softplus(d_high)` would provide a mathematical ordering guarantee. This matters for future uncertainty and active-transfer qualification, but it is not authorized before the baseline gate.

### Center/Width neural head

Represent `C=(V1+V2)/2` and `W=V2-V1`, predict `C` and `W=softplus(raw_W)`, then reconstruct the endpoints. Existing M3/HIER evidence motivates this as a structured hypothesis, but it should be considered only after neural transfer is stable and train-only residual diagnostics show consistent cross-column benefit.

### Oracle-source diagnostic

Where matched measured 4g values exist, compare predicted-4g-to-target against measured-4g-to-target. This is a leakage-controlled diagnostic upper bound to separate source-predictor error from cross-column transfer error; it is not a deployment method.

### Physical/data identifiability

Prioritize experimental designs that separate confounded variables: exact repeats for a noise floor; same column under different flow; matched flow or linear velocity across columns; TLC Rf anchors; and measured hold-up/dead volume. These improve identifiability rather than merely increasing row count.

## Paused directions

Keep arbitrary affine/scale variants, uncontrolled latent Ridge/PLS expansion, random tiny adapters, descriptor sweeps, unsupported QGeoGNN expansion, and Active Learning paused. Existing audits do not show a stable cross-column gain large enough to justify those branches, and the transfer baseline itself is still budget-censored.

## Compound and Active Learning rule

COMPOUND remains blocked until metric comparability and convergence are resolved and one defensible neural recipe survives the matched gate. If that gate passes, run a pre-registered 25g/40g x five-seed COMPOUND developmental study comparing P0 with one selected neural recipe, retaining the frozen population, source checkpoint, train-only preprocessing, blind freeze, and one-shot score boundary. Active Learning remains blocked until a stable transfer baseline and a material label-scarcity regime are demonstrated.
