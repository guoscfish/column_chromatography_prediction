# Track B — 4g→8g Transfer Adaptation

Status: `T1_ENGINEERING_READY_FORMAL_NOT_AUTHORIZED`.

S1 is complete. T1a engineering implementation, preregistration, prepare, and smoke are authorized and complete. The formal T1 run remains blocked by `formal_authorized=false`; Track C active transfer remains deferred.

T1a primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. QGeoGNN pooling is fixed sum pooling; `target_head_only` trains the prediction head rather than a learnable graph readout.
