# Paper transfer versus Scale: descriptive only

This is not a head-to-head comparison. Paper-transfer primary runs use about 80% of legacy-filtered target rows, while the frozen Scale reference uses no target threshold and a budget of 100 labels. Neither table selected a model using the other table's test results.

| column | paper-transfer V1 RMSE (mL) | paper-transfer V2 RMSE (mL) | frozen Scale budget-100 V1 RMSE (mL) | frozen Scale budget-100 V2 RMSE (mL) |
| --- | ---: | ---: | ---: | ---: |
| 8g | 9.90 | 16.29 | 6.20 | 9.24 |
| 25g | 5.91 | 8.21 | 18.88 | 26.82 |
| 40g | 12.85 | 14.81 | 35.43 | 42.66 |
