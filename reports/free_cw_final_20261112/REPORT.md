# New CW-visible Free32 rerun

Target: five seeds × 36 accepted batches, L333 → L1485, gpt-6.1-sol / xhigh.

| Method | Seed | Completed L | Raw validation AULC |
|---|---:|---:|---:|
| new_free_cw_sol61_xhigh | 157 | 1485 | 0.4740081465660413 |
| new_free_cw_sol61_xhigh | 887 | 1485 | 0.3927819293246262 |
| new_free_cw_sol61_xhigh | 2357 | 1485 | 0.4284924886304504 |
| new_free_cw_sol61_xhigh | 6101 | 1485 | 0.5219329750335885 |
| new_free_cw_sol61_xhigh | 12203 | 1485 | 0.4537580841107515 |
| historical_free_mixed_model | 157 | 1485 | 0.47157407900469894 |
| historical_free_mixed_model | 887 | 1485 | 0.3681701752055312 |
| historical_free_mixed_model | 2357 | 1485 | 0.40625431347934665 |
| historical_free_mixed_model | 6101 | 1485 | 0.5160333594305073 |
| historical_free_mixed_model | 12203 | 1485 | 0.43376367427520096 |
| nocw | 157 | 1069 | 0.5467458536894269 |
| nocw | 887 | 1005 | 0.43023467590326364 |
| nocw | 2357 | 1133 | 0.42938866683898574 |
| nocw | 6101 | 1165 | 0.5509320888140432 |
| nocw | 12203 | 1229 | 0.45789007560439426 |
| numerical_hybrid32 | 157 | 1005 | 0.4579234871040626 |
| numerical_hybrid32 | 887 | 1005 | 0.3715562320214392 |
| numerical_hybrid32 | 2357 | 1005 | 0.401420195428162 |
| numerical_hybrid32 | 6101 | 1005 | 0.5451108417154872 |
| numerical_hybrid32 | 12203 | 1005 | 0.4160868250769183 |
| gradient_lcmd32 | 157 | 1485 | 0.41499183698228115 |
| gradient_lcmd32 | 887 | 1485 | 0.3494261954091391 |
| gradient_lcmd32 | 2357 | 1485 | 0.4124154162739323 |
| gradient_lcmd32 | 6101 | 1485 | 0.48900662455544336 |
| gradient_lcmd32 | 12203 | 1485 | 0.4194062707216367 |

Per-budget metrics: validation_curves.csv. Paired differences and seed means/std: paired_differences.csv and comparison_summary.json.
Milestones are written only after all five new seeds complete that budget. Historical Free retains its mixed-model mark.
Comparisons use each comparator's five-seed common budget. API randomness, dates and invisible backend changes remain; these are not strict causal estimates of CW.
No report, score or learning curve is supplied to selection or scientific memory. No test labels are read.
