# Final report: row column-selective latent audit

This study froze the filtered FULL-data ROW protocol and reused the source checkpoint, row identities, and corrected HIER/FULL128 implementations. A1 changes only the train-fold objective to equalize the four column×endpoint tasks. A2/A3 select one shared endpoint alpha per column from train-only OOF predictions. No compound split, Active Learning, backbone change, new physical feature, or test-guided choice was used.

## Results

column                          method   V1_rmse            V1_mae             V1_r2            V2_rmse             V2_mae             V2_r2          combined_normalized_rmse         
                                            mean      std     mean      std     mean      std      mean      std      mean      std     mean      std                     mean      std
   25g         A2_HIER_SHRINK_ORIGINAL  7.839459 2.194695 4.439637 0.402340 0.231171 0.537396 10.835658 2.584579  6.867949 0.628888 0.546654 0.198262                 0.850283 0.224837
   25g         A3_HIER_SHRINK_BALANCED  7.704371 1.812664 4.500656 0.369888 0.277315 0.455127 10.960378 2.586290  7.014879 0.569706 0.536586 0.199977                 0.843329 0.195956
   25g          BALANCED_JOINT_FULL128  7.694031 1.739046 4.535961 0.389559 0.284241 0.435882 10.996782 2.566175  7.044622 0.567072 0.533913 0.198674                 0.843451 0.190174
   25g         CORRECTED_JOINT_FULL128  7.947141 2.031434 4.615596 0.474282 0.224069 0.504049 11.025045 2.494088  7.102998 0.594159 0.532582 0.192266                 0.862903 0.210250
   25g HIER_CW_SHARED_LAMBDA_CORRECTED  8.008653 2.329468 4.521093 0.445832 0.190850 0.584754 11.030658 2.603743  7.113326 0.636417 0.530390 0.203568                 0.867743 0.235501
   25g              OLD_HIER_REFERENCE  8.131997 2.556220 4.518098 0.534028 0.151433 0.648090 11.231302 2.906421  7.213245 0.775670 0.509606 0.233511                 0.881819 0.260128
   25g          paper_style_current_v2  7.408778 1.415340 4.461541 0.301735 0.348968 0.358500 11.111435 2.498625  6.967135 0.586905 0.524934 0.197086                 0.825715 0.164470
   40g         A2_HIER_SHRINK_ORIGINAL 13.477036 3.124483 8.051793 1.066021 0.701159 0.123728 16.627654 2.422053 11.469060 1.475178 0.777807 0.041968                 1.414980 0.290120
   40g         A3_HIER_SHRINK_BALANCED 13.415814 3.179231 8.132458 1.106141 0.703416 0.124898 16.582453 2.491179 11.491245 1.649535 0.779133 0.041645                 1.409452 0.294819
   40g          BALANCED_JOINT_FULL128 13.424520 3.186559 8.165415 1.142989 0.703056 0.124987 16.559184 2.475059 11.444344 1.595604 0.779665 0.041892                 1.409629 0.294965
   40g         CORRECTED_JOINT_FULL128 13.410660 3.119507 8.018142 1.040287 0.703755 0.123996 16.550522 2.422377 11.309917 1.375498 0.779591 0.043518                 1.408153 0.289587
   40g HIER_CW_SHARED_LAMBDA_CORRECTED 14.347637 2.969474 8.937902 1.241801 0.663567 0.124699 17.435470 2.345152 12.555553 1.722679 0.756624 0.034004                 1.500312 0.274375
   40g              OLD_HIER_REFERENCE 14.269543 2.876601 8.895327 1.177351 0.667715 0.119730 17.441125 2.256112 12.568312 1.664573 0.756481 0.031705                 1.494457 0.264604
   40g          paper_style_current_v2 13.109113 2.531874 7.987241 0.477368 0.720897 0.095262 15.896779 2.196566 10.642957 0.849509 0.797198 0.034595                 1.369922 0.236785

## Alpha selections

      seed column candidate  alpha  balanced_rmse       mae  mae_guard_limit  mae_guard_pass
 536279090    25g  balanced   0.75       7.384124  5.019332         5.388417            True
 536279090    25g  original   0.50       7.476273  5.068043         5.388417            True
 536279090    40g  balanced   1.00      16.969639 11.224278        12.220995            True
 536279090    40g  original   0.75      16.781793 11.056821        12.220995            True
 769539383    25g  balanced   1.00       8.696437  5.217637         5.731786            True
 769539383    25g  original   0.50       8.922355  5.165370         5.731786            True
 769539383    40g  balanced   1.00      16.187064 10.980883        11.789777            True
 769539383    40g  original   1.00      15.591094 10.446515        11.789777            True
1362771960    25g  balanced   1.00       8.804237  5.456567         5.979111            True
1362771960    25g  original   0.75       8.947572  5.500469         5.979111            True
1362771960    40g  balanced   0.75      15.854305 10.620145        11.445637            True
1362771960    40g  original   0.75      15.637998 10.367601        11.445637            True
1425370602    25g  balanced   1.00       7.428656  5.094883         5.517217            True
1425370602    25g  original   0.50       7.564291  5.056222         5.517217            True
1425370602    40g  balanced   1.00      15.231217 10.243454        11.341135            True
1425370602    40g  original   0.75      15.211651 10.076870        11.341135            True
2767143051    25g  balanced   0.75       8.490340  5.170636         5.627604            True
2767143051    25g  original   0.50       8.662644  5.193542         5.627604            True
2767143051    40g  balanced   1.00      16.581645 10.972837        12.020224            True
2767143051    40g  original   1.00      16.127025 10.436014        12.020224            True

Interpretation must use the complete per-seed metrics and endpoint rows; no aggregate score is used to hide endpoint regressions. The retention-tail table reports filtered test strata and is descriptive only.

## Decision summary

The frozen HIER row baseline is 25g V1/V2 RMSE 8.009/11.031 and 40g 14.348/17.436 mL (five-seed means). Balanced FULL128 changes these to 7.694/10.997 and 13.425/16.559; it therefore improves both columns' combined normalized RMSE relative to corrected HIER, while the endpoint-level MAE trade-offs remain visible in `summary.csv`. A3 gives 25g 7.704/10.960 and 40g 13.416/16.582, so shrinkage does not produce a uniformly better endpoint profile than A1.

OOF alpha selections are consistently smaller for 25g (original: 0.5-0.75; balanced: 0.75-1.0) and at or near 1 for 40g (original: 0.75-1.0; balanced: 0.75-1.0). This supports column heterogeneity as a development hypothesis, but does not justify implementing column-specific latent parameters yet because the gain is modest and not uniformly better on MAE/RMSE endpoints.

On the filtered test population, retention-tail concentration is endpoint-specific rather than a universal explanation: for example, 40g V2 top-10% retention accounts for about 3-6% of SSE, while 40g V1 top-20% accounts for roughly 48-55%. The full quintile and absolute-error diagnostics are in `tail_sse_diagnostic.csv`; no rows or thresholds were changed.

Conclusion: A1 answers the objective-dominance question positively enough to retain balanced fitting as a candidate, and A2/A3 demonstrate train-only safe shrinkage with no test-guided alpha. The evidence is not yet sufficient to start the next hierarchical latent-transfer model; first preserve this audit as the current stopping point and require a separately preregistered follow-up if the column-specific latent hypothesis is pursued.
