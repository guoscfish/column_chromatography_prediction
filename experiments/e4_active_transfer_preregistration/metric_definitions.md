# Metrics

`NRMSE = .5*(RMSE_V1/S_V1 + RMSE_V2/S_V2)`, where `S_V1,S_V2` are frozen source-train target scales. AULC is normalized trapezoidal area over target labels, lower is better. Primary recovery uses zero-shot `E_baseline`; full-data pretrained transfer is `E_full`. Undefined or negative-denominator recovery is not coerced. `labels-to-90%-ceiling` and `-95%` use the first eligible budget.

For Protocol B, tail is `V1 > 60 OR V2 > 120`; report tail and common errors separately. These thresholds do not filter data.
