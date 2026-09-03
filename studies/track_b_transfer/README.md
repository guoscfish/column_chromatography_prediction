# Track B — 4g→8g Transfer Adaptation

Status: `T1_ENGINEERING_READY_FORMAL_NOT_AUTHORIZED`.

S1 is complete. T1a engineering, preregistration, and formal execution are complete: 180/180 fits and 120/120 contexts passed the completion gate, but no candidate passed the stable-improvement gate. T1b is not authorized and Track C active transfer remains deferred.

T1a primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. QGeoGNN pooling is fixed sum pooling; `target_head_only` trains the prediction head rather than a learnable graph readout.
