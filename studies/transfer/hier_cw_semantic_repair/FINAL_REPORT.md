# Final report: hierarchical Center/Width semantic repair

## Decision

Q1: **YES**. Existing HIER_CW_V2 omitted Center/Width source-scale restoration, so coefficient 1 meant `r*Cs/sC` and `r*Ws/sW`, not the declared identity. See `IMPLEMENTATION_AUDIT.md` and executable tests.

Q2: **B_GRID_LOWER_BOUND_EFFECT**. The old grid ended at 0.1; the expanded preregistered grid exposes whether weaker regularization remains preferred.

Q3: unequal lambda selection occurred in 4/10 contexts; mean inner gain was 0.007%, with 4/20 outer wins and -0.077% mean outer gain. Selection difference and generalization benefit are not conflated.

Q4: endpoint-aligned corrected vs legacy-objective separate-lambda mean outer gain was -0.179% with 8/20 wins.

Q5: corrected FULL128 vs corrected endpoint HIER mean outer gain was 1.096% with 13/20 wins. Column/split detail is in `joint_full128_reaudit.md`.

Q6: **HIER_CW_SHARED_LAMBDA_CORRECTED** is the most stable corrected model: its worst context degradation versus old HIER is only 1.13%. **CORRECTED_JOINT_FULL128** has the best mean context rank but is not called most stable because its 25g-compound degradation is 6.56%. No hidden composite score is used.

## Mean metrics

| column | protocol | method | V1_rmse_mean | V1_mae_mean | V1_r2_mean | V2_rmse_mean | V2_mae_mean | V2_r2_mean | combined_normalized_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | CORRECTED_JOINT_FULL128 | 6.988102 | 4.791587 | 0.512621 | 9.322448 | 7.049757 | 0.742616 | 0.749739 |
| 25g | compound | HIER_CW_ENDPOINT_ALIGNED_CORRECTED | 6.633032 | 4.499394 | 0.565410 | 9.105921 | 6.810794 | 0.754873 | 0.717576 |
| 25g | compound | HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE | 6.604584 | 4.472327 | 0.568887 | 8.969029 | 6.734977 | 0.761409 | 0.712037 |
| 25g | compound | HIER_CW_SHARED_LAMBDA_CORRECTED | 6.594801 | 4.467481 | 0.570350 | 8.978018 | 6.747779 | 0.760927 | 0.711509 |
| 25g | compound | M3_CENTER_WIDTH_FULL | 6.624252 | 4.515934 | 0.567935 | 9.199462 | 6.893199 | 0.749903 | 0.719144 |
| 25g | compound | OLD_HIER_REFERENCE | 6.556243 | 4.463470 | 0.573731 | 8.770349 | 6.666996 | 0.771004 | 0.703553 |
| 25g | compound | conditional_EA | 7.139174 | 4.903305 | 0.500987 | 9.584352 | 7.166839 | 0.720326 | 0.767661 |
| 25g | compound | paper_style_current_v2 | 8.040766 | 5.836380 | 0.366095 | 10.959423 | 8.337980 | 0.646233 | 0.867985 |
| 25g | compound | scale_only | 7.710632 | 5.351098 | 0.416691 | 9.039011 | 6.946259 | 0.757024 | 0.798185 |
| 25g | row | CORRECTED_JOINT_FULL128 | 7.947141 | 4.615596 | 0.224069 | 11.025045 | 7.102998 | 0.532582 | 0.862903 |
| 25g | row | HIER_CW_ENDPOINT_ALIGNED_CORRECTED | 8.000920 | 4.529859 | 0.198335 | 11.023695 | 7.109033 | 0.532115 | 0.867055 |
| 25g | row | HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE | 8.021632 | 4.550239 | 0.190733 | 11.063537 | 7.130421 | 0.527981 | 0.869485 |
| 25g | row | HIER_CW_SHARED_LAMBDA_CORRECTED | 8.008653 | 4.521093 | 0.190850 | 11.030658 | 7.113326 | 0.530390 | 0.867743 |
| 25g | row | M3_CENTER_WIDTH_FULL | 8.080615 | 4.621397 | 0.174682 | 11.645691 | 7.570628 | 0.476176 | 0.888599 |
| 25g | row | OLD_HIER_REFERENCE | 8.131997 | 4.518098 | 0.151433 | 11.231302 | 7.213245 | 0.509606 | 0.881819 |
| 25g | row | conditional_EA | 8.460571 | 4.852944 | 0.056169 | 12.095896 | 7.876293 | 0.430937 | 0.929106 |
| 25g | row | paper_style_current_v2 | 7.408778 | 4.461541 | 0.348968 | 11.111435 | 6.967135 | 0.524934 | 0.825715 |
| 25g | row | scale_only | 8.754031 | 5.249917 | 0.052279 | 11.055946 | 7.305867 | 0.532995 | 0.924226 |
| 40g | compound | CORRECTED_JOINT_FULL128 | 15.149720 | 9.688142 | 0.695957 | 19.547939 | 12.937276 | 0.725959 | 1.609839 |
| 40g | compound | HIER_CW_ENDPOINT_ALIGNED_CORRECTED | 15.427865 | 9.973461 | 0.684920 | 20.103445 | 13.330879 | 0.709954 | 1.644226 |
| 40g | compound | HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE | 15.433698 | 9.958938 | 0.684749 | 20.065088 | 13.294669 | 0.711026 | 1.643738 |
| 40g | compound | HIER_CW_SHARED_LAMBDA_CORRECTED | 15.421181 | 9.948254 | 0.685296 | 20.080076 | 13.309701 | 0.710631 | 1.643076 |
| 40g | compound | M3_CENTER_WIDTH_FULL | 15.993604 | 10.400114 | 0.662364 | 21.411090 | 14.748912 | 0.671346 | 1.717746 |
| 40g | compound | OLD_HIER_REFERENCE | 15.426294 | 9.943618 | 0.685207 | 20.112998 | 13.351336 | 0.709705 | 1.644239 |
| 40g | compound | conditional_EA | 16.371568 | 10.476351 | 0.645654 | 22.261416 | 14.917865 | 0.645198 | 1.766571 |
| 40g | compound | paper_style_current_v2 | 16.239653 | 11.001171 | 0.650271 | 22.083339 | 15.389488 | 0.647213 | 1.752543 |
| 40g | compound | scale_only | 18.537280 | 12.847500 | 0.546532 | 21.144087 | 13.843434 | 0.679139 | 1.906550 |
| 40g | row | CORRECTED_JOINT_FULL128 | 13.410660 | 8.018142 | 0.703755 | 16.550522 | 11.309917 | 0.779591 | 1.408153 |
| 40g | row | HIER_CW_ENDPOINT_ALIGNED_CORRECTED | 14.338890 | 8.936851 | 0.664034 | 17.469671 | 12.581332 | 0.755687 | 1.500471 |
| 40g | row | HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE | 14.330685 | 8.936265 | 0.664514 | 17.446568 | 12.558351 | 0.756302 | 1.499223 |
| 40g | row | HIER_CW_SHARED_LAMBDA_CORRECTED | 14.347637 | 8.937902 | 0.663567 | 17.435470 | 12.555553 | 0.756624 | 1.500312 |
| 40g | row | M3_CENTER_WIDTH_FULL | 14.296370 | 8.890473 | 0.665803 | 17.405772 | 12.526943 | 0.757367 | 1.495716 |
| 40g | row | OLD_HIER_REFERENCE | 14.269543 | 8.895327 | 0.667715 | 17.441125 | 12.568312 | 0.756481 | 1.494457 |
| 40g | row | conditional_EA | 15.338437 | 9.671110 | 0.613332 | 18.516817 | 13.195458 | 0.726884 | 1.605991 |
| 40g | row | paper_style_current_v2 | 13.109113 | 7.987241 | 0.720897 | 15.896779 | 10.642957 | 0.797198 | 1.369922 |
| 40g | row | scale_only | 17.460948 | 12.332099 | 0.510906 | 18.159403 | 13.020715 | 0.734996 | 1.759777 |

All predictions, fitted coefficients, architectures, and selections were SHA256-frozen before outer truth evaluation. Compound is primary, row secondary. No Active Learning, PCA, PLS, Direct Ridge, neural adapter, or new GNN architecture was run.

Center and Width are operational elution-location and elution-interval-width surrogates only; these results do not prove physical mechanisms or a physical mass-scaling law.
