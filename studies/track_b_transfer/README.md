# Track B — 4g→8g Transfer Adaptation

Status: `T1_ENGINEERING_READY_FORMAL_NOT_AUTHORIZED`.

S1 is complete. T1a formal execution completed without a stable winner. T1b-1 graph-adapter capacity engineering/preregistration and 9/9 smoke fits are complete; its formal run and T1b-2 remain unauthorized. Track C active transfer remains deferred.

T1a primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. QGeoGNN pooling is fixed sum pooling; `target_head_only` trains the prediction head rather than a learnable graph readout.
