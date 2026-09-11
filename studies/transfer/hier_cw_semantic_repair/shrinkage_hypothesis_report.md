# Shrinkage hypothesis report

Separate selection chose unequal lambdas in 4/10 protocol/seed contexts.

Mean train-only inner-CV gain over the fairly tuned shared-lambda control: 0.007%.

Outer wins: 4/20; mean outer gain -0.077%.

Weak-boundary contexts (lambda 0 or 0.01): 9/10. **WEAK_SHRINKAGE_BOUNDARY_SIGNAL**.

| lambda_C | lambda_W | contexts |
| --- | --- | --- |
| 0.000000 | 0.010000 | 2 |
| 0.010000 | 0.010000 | 5 |
| 0.010000 | 0.100000 | 1 |
| 0.100000 | 0.010000 | 1 |
| 0.100000 | 0.100000 | 1 |

Different selected values are selection evidence only. They do not establish different physical mechanisms; performance and outer stability are reported separately.
