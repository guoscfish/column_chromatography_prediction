# Loss screen report

The three fixed loss recipes were evaluated only in 5-fold canonical-SMILES GroupKFold inside each ROW outer gradient-train context. No outer validation/test endpoint was available to the screen.

Selected global recipe: **L0 / `raw_quantile`**.

| arm | column | mean relative improvement (%) | fold wins | seed wins |
| --- | --- | ---: | ---: | ---: |
| L1 | 25g | 0.682 | 21/25 | 4/5 |
| L1 | 40g | 1.202 | 21/25 | 5/5 |
| L2 | 25g | 0.428 | 21/25 | 5/5 |
| L2 | 40g | 0.270 | 15/25 | 5/5 |

A non-L0 recipe required >=1% mean improvement in both columns, >=13/25 fold wins and >=3/5 seed wins in both columns. Otherwise the simpler raw P0 recipe remains selected.
