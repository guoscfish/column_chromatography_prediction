# Track B — 4g→8g Transfer Adaptation

Status: `T1B1_FORMAL_COMPLETE_NO_TESTED_LOW_CAPACITY_ADAPTER_BENEFIT`.

S1 is complete. T1a formal execution completed without a stable winner. T1b-1 then completed 180/180 Adapter fits and all 120 capacity contexts. The r=8/16/32 Adapters did not stably beat Head, establishing no benefit in the tested 2,958–9,126 trainable-parameter Adapter range. It did not cover the 9,126-to-93,454 gap; example 17k/34k/67k regions remain untested, and no expanded sweep is authorized. T1b-2 remains proposed but unauthorized; Track C active transfer remains deferred.

T1a primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. QGeoGNN pooling is fixed sum pooling; `target_head_only` trains the prediction head rather than a learnable graph readout.

I0 documents the legacy predictor's partial condition reachability and nominal-versus-gradient-bearing parameter counts. Predictor V2 is a separate `PREREGISTRATION_ONLY` proposal. Neither finding authorizes retraining T1/T1b-1 or starting active transfer.
