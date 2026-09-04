# Track B — 4g→8g Transfer Adaptation

Status: `T1B1_FORMAL_COMPLETE_NO_INTERMEDIATE_CAPACITY_BENEFIT`.

S1 is complete. T1a formal execution completed without a stable winner. T1b-1 then completed 180/180 Adapter fits and all 120 capacity contexts. The r=8/16/32 Adapters did not stably beat Head, so the tested 774-to-93,454 parameter interval contains no supported sweet spot. T1b-2 remains proposed but unauthorized; independent validation is preferred and Track C active transfer remains deferred.

T1a primary methods are `zero_shot`, `affine`, `condition_ridge_residual`, `target_head_only`, `last1_head`, and `current_last2_head`. QGeoGNN pooling is fixed sum pooling; `target_head_only` trains the prediction head rather than a learnable graph readout.
