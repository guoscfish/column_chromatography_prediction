# HIER Center/Width implementation audit

## Verdict

**YES: `HIER_CW_V2` has a dimensionally inconsistent identity prior.** This is
an implementation-semantics finding, not a claim that mass-ratio scaling is a
physical law.

Let source endpoints have units mL. Define `Cs=(V1s+V2s)/2` and
`Ws=V2s-V1s`, also in mL, and let `r` be dimensionless.

## Historical HIER

For endpoint surrogate `x` (`Cs` or `Ws`), the historical code fits
`u=x/sx`, where `sx=std(x)` has mL units. Target labels are
`y/(r*sx)`, hence dimensionless. The design `[u,1,zEA]`, coefficients, prior
`[1,0,0]`, and column deviations are dimensionless. Prediction is
`r*sx*[u,1,zEA] beta`. Therefore `beta=[1,0,0]` and zero deviations gives
`r*sx*(x/sx)=r*x`. Center/Width inverse then yields `r*[V1s,V2s]`.

## Existing HIER_CW_V2

`HierarchicalBasis.transform()` produces dimensionless `c=Cs/sC` and
`w=Ws/sW`, plus dimensionless intercept and standardized interaction columns.
`_expanded()` preserves those units. `_joint_solve()` multiplies by `r` and
constructs V1/V2 design rows, but it does **not** multiply the Center block by
`sC` or Width block by `sW`. It nevertheless assigns prior value 1 to both
global slopes and compares the resulting design directly with mL truth.

Consequently, with slopes 1, intercepts/interactions/deviations zero, current
raw predictions are:

`C_hat = r*Cs/sC`, `W_hat = r*Ws/sW`,

not `r*Cs`, `r*Ws`. The missing factors are exactly `sC` and `sW`. Its design
matrix mixes dimensionless predictions with mL targets divided only by endpoint
scales; the unit-slope prior is therefore not the declared identity mapping.

The executable counterexample is
`test_current_v2_unit_slope_is_not_mass_ratio_identity` in
`tests/test_hierarchical_cw_corrected.py`. Corrected identity tests cover
Center, Width, V1/V2 inverse reconstruction, arbitrary source scales, arbitrary
mass ratios, and zero EA/deviation contributions.

## Corrected separation of questions

`HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE` preserves the historical normalized
C/W fitting and prediction equations exactly; only Center and Width
column-deviation penalties may differ. Its shared-lambda case must reproduce
historical HIER below `1e-10` for every frozen context.

`HIER_CW_ENDPOINT_ALIGNED_CORRECTED` separately changes the data objective. It
uses the same dimensionless basis but restores units before endpoint assembly:
`C=r*sC*XC*betaC`, `W=r*sW*XW*betaW`. Thus coefficient 1 again means the
mass-ratio source identity, while endpoint scales only weight loss terms.

Center and Width are treated only as operational elution-location and
elution-interval-width surrogates. No mechanistic interpretation follows from
their coefficients or penalties.
