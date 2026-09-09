# Methods note

With `V1 = C - W/2` and `V2 = C + W/2`, equal-weight squared endpoint loss is

`eV1^2 + eV2^2 = 2 eC^2 + 0.5 eW^2`.

When center and width have independent parameter blocks and regularization is scaled consistently, this objective remains separable. Merely placing the two branches in one optimizer is therefore not joint modeling and is not treated as a candidate here.

`M3_CENTER_WIDTH` fits independent Conditional-EA-equivalent center and width branches. `CENTER_MAGNITUDE` adds only `log1p(max(C_source, 0) / center_scale)` to the center slope; `center_scale` is estimated on the fit subset. `CENTER_CONDITIONED_WIDTH` instead adds only standardized source center to the width slope. Each candidate has seven coefficients and is strictly nested in the six-coefficient M3 baseline.
