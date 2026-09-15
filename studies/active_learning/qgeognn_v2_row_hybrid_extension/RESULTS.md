# Results — V2 Hybrid matched extension

This is a **POST-PRIMARY MATCHED EXTENSION**.  It does not amend the frozen primary decision `BEST_CURRENT_ROW_ACQUISITION`.

## Decision

`HYBRID_DOES_NOT_BEAT_RANDOM`

| Method | Mean combined NRMSE | Mean V1 RMSE | Mean V2 RMSE |
| --- | ---: | ---: | ---: |
| Random | 0.694920 | 5.136488 | 9.340196 |
| V2-Hybrid | 0.672660 | 4.999149 | 9.093374 |
| Gradient-LCMD | 0.663146 | 4.907886 | 8.894684 |

Hybrid beat its within-seed Random median in 3/5 seeds. Hybrid beat LCMD in 3/5 seeds; LCMD beat Hybrid in 2/5 seeds.

The selection and mechanism diagnostics are descriptive and did not feed back into acquisition.  Endpoint guard warnings: [].
