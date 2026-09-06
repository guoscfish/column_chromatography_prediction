# Next transfer residual diagnostics

Decision: `NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA`.

Two preregistered diagnostic families completed on all 120 frozen column/protocol/seed/budget contexts. The qualified source, baseline code, original results and frozen roles remain unchanged. No new architecture, fine-tuning, width search, active learning, or post-test candidate was run.

The nonlinear primary policy selects scale, affine or a train-only two-knot monotone spline using focal validation. `monotone_spline` reports the standalone family, including cases where validation rejects it. `linear_policy` uses the same validation to choose only scale or affine. Shared column affine uses mass-normalized partial pooling; local identity shrinkage is the regularization control.

The shared/independent comparison is a THREE-COLUMN PORTFOLIO at total planned budgets 90/150/210/300, with identical purchased label IDs (including all validation) for all portfolio methods. Per-column B is an allocation, not the shared model's total label cost. Actual compound budgets, discarded donor labels and allowed IDs are retained. Focal compound test/validation compounds are purged from every donor fit. Focal models are separate; no model or tuned hyperparameter is reused across focal holdouts. Row remains row interpolation.

Arithmetic source NRMSE = mean(RMSE_V1/7.8797, RMSE_V2/16.0765); AULC is its trapezoidal average over B=30..100, exactly as cross-column. `combined_normalized_rmse` is RMS, used for validation. Exact source scales are in all_predictions_frozen.json. Actual-budget AULC is also provided. Five outer seeds share one fixed source seed 42 and overlapping data; their dispersion is not five independent external cohorts.

## Frozen decision gate

{
  "decision": "NO_COMPLEXITY_JUSTIFIED_BY_CURRENT_DATA",
  "supported_conclusions": [],
  "qualifying_compound_columns": {
    "nonlinear_policy": [],
    "shared_column_affine": []
  },
  "shared_portfolio_gate": false,
  "adaptive_readout_tested": false,
  "adaptive_readout_warranted": false,
  "additional_methods_after_test": 0,
  "scope": "developmental frozen-split evidence for these two bounded families, not proof of irreducible noise",
  "source_unseen_ood_claim": false,
  "mass_flow_physical_law_claim": false
}

A candidate requires >=5% mean relative AULC improvement, negative median delta, >=4/5 seed wins against each specified reference in at least two different compound columns, with <=5% row degradation. 25g/40g must also beat head-only. Shared requires >=5% compound portfolio gain over affine, scale and local shrinkage. Test results implement a preregistered research go/no-go gate; they do not select knots, penalties or model checkpoints.

## Paired AULC (positive relative gain is better)

| index | column | protocol | method | reference | relative_mean_gain | mean_delta | median_delta | std_delta | wins | stable_material |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 25g | compound | nonlinear_policy | scale_only | 0.0327300966 | -0.0526114721 | -0.0186054581 | 0.0612708131 | 4 | False |
| 6 | 25g | compound | nonlinear_policy | affine | 0.0719211262 | -0.120490397 | -0.000780370025 | 0.34461713 | 3 | False |
| 7 | 25g | compound | nonlinear_policy | target_head_only | 0.490223929 | -1.49518834 | -1.51537128 | 0.305879391 | 5 | True |
| 14 | 25g | compound | shared_column_affine | scale_only | 0.0393402125 | -0.0632367976 | -0.119283222 | 0.100155529 | 4 | False |
| 15 | 25g | compound | shared_column_affine | affine | 0.078263419 | -0.131115722 | -0.00981520766 | 0.281206893 | 5 | True |
| 16 | 25g | compound | shared_column_affine | target_head_only | 0.49370763 | -1.50581367 | -1.51459093 | 0.348215545 | 5 | True |
| 28 | 25g | row | nonlinear_policy | scale_only | 0.0262484793 | -0.0540750193 | -0.0704393318 | 0.0551009558 | 4 | False |
| 29 | 25g | row | nonlinear_policy | affine | 0.0393805793 | -0.0822377884 | -0.00606163309 | 0.22622793 | 4 | False |
| 30 | 25g | row | nonlinear_policy | target_head_only | 0.405814534 | -1.37008104 | -1.36439865 | 0.125120186 | 5 | True |
| 37 | 25g | row | shared_column_affine | scale_only | 0.0081822481 | -0.0168564136 | -0.0730408777 | 0.192110956 | 4 | False |
| 38 | 25g | row | shared_column_affine | affine | 0.0215579909 | -0.0450191827 | -0.00625563873 | 0.071712381 | 4 | False |
| 39 | 25g | row | shared_column_affine | target_head_only | 0.394790477 | -1.33286244 | -1.415185 | 0.146694246 | 5 | True |
| 51 | 40g | compound | nonlinear_policy | scale_only | -0.0960843394 | 0.346018742 | -0.0354559688 | 0.847552494 | 3 | False |
| 52 | 40g | compound | nonlinear_policy | affine | -0.00801043667 | 0.031367662 | -0.000853296832 | 0.0816964295 | 3 | False |
| 53 | 40g | compound | nonlinear_policy | target_head_only | 0.488636734 | -3.77179058 | -4.09637042 | 1.15618806 | 5 | True |
| 60 | 40g | compound | shared_column_affine | scale_only | 0.0183977445 | -0.0662539228 | -0.109107021 | 0.147183792 | 4 | False |
| 61 | 40g | compound | shared_column_affine | affine | 0.0972726435 | -0.380905003 | -0.00979404369 | 0.783943483 | 4 | True |
| 62 | 40g | compound | shared_column_affine | target_head_only | 0.542046795 | -4.18406325 | -4.13009409 | 0.36785539 | 5 | True |
| 74 | 40g | row | nonlinear_policy | scale_only | 0.0819966674 | -0.298047088 | -0.458533845 | 0.262520095 | 4 | True |
| 75 | 40g | row | nonlinear_policy | affine | -0.00502657209 | 0.0166888845 | -0.00972667363 | 0.0670504936 | 3 | False |
| 76 | 40g | row | nonlinear_policy | target_head_only | 0.581032506 | -4.62757044 | -4.23430073 | 0.702954247 | 5 | True |
| 83 | 40g | row | shared_column_affine | scale_only | 0.113727153 | -0.413383225 | -0.444845921 | 0.18486828 | 5 | True |
| 84 | 40g | row | shared_column_affine | affine | 0.0297118437 | -0.0986472529 | 0.00396125068 | 0.244791449 | 2 | False |
| 85 | 40g | row | shared_column_affine | target_head_only | 0.59551398 | -4.74290658 | -4.23827013 | 0.815090545 | 5 | True |
| 97 | 8g | compound | nonlinear_policy | scale_only | -0.101856864 | 0.074853558 | 0.00979794465 | 0.124392673 | 1 | False |
| 98 | 8g | compound | nonlinear_policy | affine | 0.0474624947 | -0.0403474234 | -0.0420074934 | 0.0623545342 | 3 | False |
| 99 | 8g | compound | nonlinear_policy | target_head_only | -0.116875281 | 0.0847354848 | -0.0409975034 | 0.288353065 | 3 | False |
| 106 | 8g | compound | shared_column_affine | scale_only | -0.0611030631 | 0.0449040103 | 0.0582648368 | 0.0562525333 | 2 | False |
| 107 | 8g | compound | shared_column_affine | affine | 0.082693499 | -0.070296971 | -0.0107827052 | 0.0891881726 | 4 | True |
| 108 | 8g | compound | shared_column_affine | target_head_only | -0.075566002 | 0.0547859371 | -0.061208916 | 0.206307666 | 3 | False |
| 120 | 8g | row | nonlinear_policy | scale_only | -0.0437092614 | 0.0318926682 | 0.0234953547 | 0.0364182543 | 1 | False |
| 121 | 8g | row | nonlinear_policy | affine | 0.0349552732 | -0.0275843137 | -0.0244378791 | 0.03491586 | 4 | False |
| 122 | 8g | row | nonlinear_policy | target_head_only | -0.121605018 | 0.0825673748 | 0.117110041 | 0.121201335 | 1 | False |
| 129 | 8g | row | shared_column_affine | scale_only | -0.0370923568 | 0.0270646126 | 0.0378074916 | 0.0480303285 | 2 | False |
| 130 | 8g | row | shared_column_affine | affine | 0.0410734606 | -0.0324123692 | -0.0222481481 | 0.0307971444 | 4 | False |
| 131 | 8g | row | shared_column_affine | target_head_only | -0.11449427 | 0.0777393192 | 0.103635853 | 0.120097895 | 1 | False |

## Full AULC seed stability

| index | column | protocol | method | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25g | compound | affine | 1.67531299 | 0.594080973 | 1.63056727 | 0.96854132 | 2.61138909 |
| 1 | 25g | compound | linear_policy | 1.56140873 | 0.360443098 | 1.63056726 | 0.968541334 | 1.88272536 |
| 2 | 25g | compound | local_identity_shrinkage | 1.52377892 | 0.350564284 | 1.63056726 | 0.956854104 | 1.90481722 |
| 3 | 25g | compound | monotone_spline | 1.68008512 | 0.606638228 | 1.63428283 | 0.962940659 | 2.63902652 |
| 4 | 25g | compound | nonlinear_policy | 1.55482259 | 0.364194333 | 1.63428283 | 0.962838177 | 1.88272536 |
| 5 | 25g | compound | scale_only | 1.60743406 | 0.316738236 | 1.65288829 | 1.0761216 | 1.88272535 |
| 6 | 25g | compound | shared_column_affine | 1.54419727 | 0.373339116 | 1.63056726 | 0.956838378 | 1.97732866 |
| 7 | 25g | compound | target_head_only | 3.05001093 | 0.60502205 | 3.22418488 | 2.03168044 | 3.50267566 |
| 8 | 25g | row | affine | 2.08828285 | 0.513163875 | 1.8793842 | 1.55469034 | 2.90199432 |
| 9 | 25g | row | linear_policy | 2.00907789 | 0.333255735 | 1.97067375 | 1.55469034 | 2.42173134 |
| 10 | 25g | row | local_identity_shrinkage | 2.02090266 | 0.398857598 | 1.9616851 | 1.50232727 | 2.56231787 |
| 11 | 25g | row | monotone_spline | 2.09394238 | 0.533119937 | 1.8785425 | 1.54879141 | 2.94779925 |
| 12 | 25g | row | nonlinear_policy | 2.00604506 | 0.334854682 | 1.97067375 | 1.54897199 | 2.42240605 |
| 13 | 25g | row | scale_only | 2.06012008 | 0.299165715 | 1.97067375 | 1.67359826 | 2.41861304 |
| 14 | 25g | row | shared_column_affine | 2.04326367 | 0.465226136 | 1.88276216 | 1.49818564 | 2.7362808 |
| 15 | 25g | row | target_head_only | 3.37612611 | 0.422782217 | 3.30871162 | 2.91337064 | 3.90842994 |
| 16 | 40g | compound | affine | 3.9158492 | 0.980110061 | 3.86522213 | 3.05347069 | 5.5260677 |
| 17 | 40g | compound | linear_policy | 3.91584918 | 0.980110002 | 3.86522213 | 3.0534707 | 5.52606757 |
| 18 | 40g | compound | local_identity_shrinkage | 3.70447487 | 0.620122353 | 3.78231554 | 3.0534707 | 4.61861026 |
| 19 | 40g | compound | monotone_spline | 3.94601336 | 1.05462454 | 3.85601009 | 3.05210181 | 5.70324656 |
| 20 | 40g | compound | nonlinear_policy | 3.94721686 | 1.05354761 | 3.85601009 | 3.0526174 | 5.70324656 |
| 21 | 40g | compound | scale_only | 3.60119812 | 0.440966467 | 3.85624433 | 3.04285593 | 3.95179335 |
| 22 | 40g | compound | shared_column_affine | 3.5349442 | 0.3696918 | 3.74713731 | 3.04367665 | 3.85626372 |
| 23 | 40g | compound | target_head_only | 7.71900745 | 0.389243828 | 7.9863578 | 7.14898782 | 7.99846317 |
| 24 | 40g | row | affine | 3.32013233 | 0.868683733 | 2.73303461 | 2.6692586 | 4.517427 |
| 25 | 40g | row | linear_policy | 3.31827459 | 0.86548714 | 2.73303462 | 2.66925861 | 4.50813827 |
| 26 | 40g | row | local_identity_shrinkage | 3.28621915 | 0.765118028 | 2.79906346 | 2.66925861 | 4.25306398 |
| 27 | 40g | row | monotone_spline | 3.31633354 | 0.890741151 | 2.71146839 | 2.6589848 | 4.55049806 |
| 28 | 40g | row | nonlinear_policy | 3.33682122 | 0.926383357 | 2.71146839 | 2.65953193 | 4.65238935 |
| 29 | 40g | row | scale_only | 3.63486831 | 0.665990914 | 3.20555847 | 3.11806577 | 4.58824409 |
| 30 | 40g | row | shared_column_affine | 3.22148508 | 0.693123317 | 2.74518078 | 2.67321985 | 3.98126884 |
| 31 | 40g | row | target_head_only | 7.96439166 | 1.12202577 | 8.21740488 | 6.77543614 | 9.46306517 |
| 32 | 8g | compound | affine | 0.850090658 | 0.304561943 | 0.811644277 | 0.517879698 | 1.33989843 |
| 33 | 8g | compound | linear_policy | 0.767353952 | 0.229986491 | 0.681312017 | 0.535854853 | 1.14233872 |
| 34 | 8g | compound | local_identity_shrinkage | 0.728161315 | 0.183770609 | 0.65697151 | 0.542724254 | 1.02208564 |
| 35 | 8g | compound | monotone_spline | 0.862361764 | 0.330475756 | 0.814946042 | 0.517567663 | 1.40217117 |
| 36 | 8g | compound | nonlinear_policy | 0.809743235 | 0.319306463 | 0.681312017 | 0.538091111 | 1.35572519 |
| 37 | 8g | compound | scale_only | 0.734889677 | 0.211296843 | 0.681312023 | 0.528293167 | 1.06533692 |
| 38 | 8g | compound | shared_column_affine | 0.779793687 | 0.242242965 | 0.701695571 | 0.517879699 | 1.15393447 |
| 39 | 8g | compound | target_head_only | 0.72500775 | 0.0836987042 | 0.770599806 | 0.579088615 | 0.773965867 |
| 40 | 8g | row | affine | 0.789131686 | 0.104427666 | 0.773612253 | 0.66592492 | 0.953310967 |
| 41 | 8g | row | linear_policy | 0.760188582 | 0.118734112 | 0.730533761 | 0.640617281 | 0.955268667 |
| 42 | 8g | row | local_identity_shrinkage | 0.754220229 | 0.0916786918 | 0.745809931 | 0.635478055 | 0.890902853 |
| 43 | 8g | row | monotone_spline | 0.790238594 | 0.104809934 | 0.773327077 | 0.664614523 | 0.952806626 |
| 44 | 8g | row | nonlinear_policy | 0.761547372 | 0.1179935 | 0.730507727 | 0.640509585 | 0.955268667 |
| 45 | 8g | row | scale_only | 0.729654704 | 0.1362811 | 0.718436968 | 0.61701423 | 0.955268656 |
| 46 | 8g | row | shared_column_affine | 0.756719316 | 0.102589675 | 0.717231108 | 0.665924923 | 0.931062818 |
| 47 | 8g | row | target_head_only | 0.678979997 | 0.113004129 | 0.652608607 | 0.538254952 | 0.825467811 |

## Portfolio total-cost AULC

| protocol | method | normalized_aulc / mean | normalized_aulc / std | actual_budget_normalized_aulc / mean | actual_budget_normalized_aulc / std |
| --- | --- | --- | --- | --- | --- |
| compound | affine | 2.14708428 | 0.427707124 | 2.15469761 | 0.437759136 |
| compound | linear_policy | 2.08153729 | 0.334498294 | 2.08760843 | 0.343549324 |
| compound | local_identity_shrinkage | 1.9854717 | 0.21861093 | 1.98982973 | 0.22341877 |
| compound | monotone_spline | 2.16282008 | 0.457462099 | 2.17129298 | 0.468477433 |
| compound | nonlinear_policy | 2.10392756 | 0.360915059 | 2.11179728 | 0.371078895 |
| compound | scale_only | 1.98117395 | 0.102906718 | 1.98185089 | 0.103473284 |
| compound | shared_column_affine | 1.95297838 | 0.107475005 | 1.95549938 | 0.109103299 |
| compound | target_head_only | 3.83134204 | 0.184572219 | 3.83184287 | 0.184010845 |
| row | affine | 2.06584896 | 0.190798599 | 2.06584896 | 0.190798599 |
| row | linear_policy | 2.02918035 | 0.207351656 | 2.02918035 | 0.207351656 |
| row | local_identity_shrinkage | 2.02044735 | 0.180319982 | 2.02044735 | 0.180319982 |
| row | monotone_spline | 2.06683817 | 0.197076696 | 2.06683817 | 0.197076696 |
| row | nonlinear_policy | 2.03480455 | 0.225316739 | 2.03480455 | 0.225316739 |
| row | scale_only | 2.1415477 | 0.143561872 | 2.1415477 | 0.143561872 |
| row | shared_column_affine | 2.00715602 | 0.147816358 | 2.00715602 | 0.147816358 |
| row | target_head_only | 4.00649926 | 0.264026674 | 4.00649926 | 0.264026674 |

## Budget 100 point metrics (mean over five seeds)

| index | column | protocol | method | V1_r2 | V1_rmse | V1_mae | V2_r2 | V2_rmse | V2_mae | normalized_rmse | combined_normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25g | compound | affine | 0.836770811 | 14.1437998 | 8.84463892 | 0.872567103 | 20.1297115 | 12.6080769 | 1.52354782 | 1.54864835 |
| 1 | 25g | compound | linear_policy | 0.830663 | 14.4609917 | 9.89382522 | 0.8669037 | 20.6202155 | 14.1841208 | 1.55893038 | 1.58397009 |
| 2 | 25g | compound | local_identity_shrinkage | 0.841729036 | 13.9518854 | 8.90875243 | 0.877833418 | 19.7581037 | 12.4900954 | 1.49981251 | 1.52508751 |
| 3 | 25g | compound | monotone_spline | 0.836679294 | 14.1454853 | 8.84321359 | 0.873762953 | 20.0477875 | 12.3854633 | 1.52110683 | 1.54695369 |
| 4 | 25g | compound | nonlinear_policy | 0.835252619 | 14.2750216 | 9.56381085 | 0.875750766 | 19.986563 | 13.2398593 | 1.52742234 | 1.55484938 |
| 5 | 25g | compound | scale_only | 0.822533722 | 14.7586126 | 10.3311545 | 0.856383991 | 21.3405612 | 15.0078261 | 1.60021945 | 1.62391476 |
| 6 | 25g | compound | shared_column_affine | 0.836447293 | 14.1559593 | 8.8477823 | 0.876662023 | 19.843129 | 12.5049031 | 1.51540632 | 1.54215699 |
| 7 | 25g | compound | target_head_only | 0.483927452 | 25.3517097 | 14.5006417 | 0.335982835 | 46.0397653 | 28.9216564 | 3.04057619 | 3.04623844 |
| 8 | 25g | row | affine | 0.746508321 | 18.6082205 | 9.71050769 | 0.816426673 | 25.2226077 | 14.3620431 | 1.96523109 | 2.0058811 |
| 9 | 25g | row | linear_policy | 0.743278725 | 18.7295409 | 9.95143166 | 0.81023567 | 25.6370508 | 14.5490281 | 1.98581913 | 2.02532713 |
| 10 | 25g | row | local_identity_shrinkage | 0.746422364 | 18.6123694 | 9.91575478 | 0.811642996 | 25.5261753 | 15.1517525 | 1.9749357 | 2.01435302 |
| 11 | 25g | row | monotone_spline | 0.745727566 | 18.6357129 | 9.68808684 | 0.817847614 | 25.1391795 | 13.9286382 | 1.96438088 | 2.00597664 |
| 12 | 25g | row | nonlinear_policy | 0.742533729 | 18.7557793 | 9.93551172 | 0.811318039 | 25.5773274 | 14.155828 | 1.9856266 | 2.02592023 |
| 13 | 25g | row | scale_only | 0.737948806 | 18.8817811 | 10.6871391 | 0.791517025 | 26.8216001 | 15.8154919 | 2.03232046 | 2.06554354 |
| 14 | 25g | row | shared_column_affine | 0.746701702 | 18.6048521 | 9.90030903 | 0.815343953 | 25.2871746 | 14.4894677 | 1.96702546 | 2.00728546 |
| 15 | 25g | row | target_head_only | 0.411279517 | 28.2771172 | 16.0440159 | 0.270070207 | 50.3619845 | 30.6782405 | 3.36063307 | 3.36908536 |
| 16 | 40g | compound | affine | 0.732092815 | 33.5744582 | 20.0755742 | 0.791624055 | 40.9652489 | 23.0829021 | 3.40452282 | 3.51085807 |
| 17 | 40g | compound | linear_policy | 0.732092814 | 33.5744583 | 20.0755743 | 0.791624055 | 40.9652489 | 23.082902 | 3.40452282 | 3.51085807 |
| 18 | 40g | compound | local_identity_shrinkage | 0.732092814 | 33.5744583 | 20.0755743 | 0.791624055 | 40.9652489 | 23.082902 | 3.40452282 | 3.51085807 |
| 19 | 40g | compound | monotone_spline | 0.732688376 | 33.5434612 | 20.1037057 | 0.791639205 | 40.9101628 | 22.2834717 | 3.40084267 | 3.50723096 |
| 20 | 40g | compound | nonlinear_policy | 0.732644075 | 33.5462682 | 20.1011022 | 0.791682446 | 40.9057433 | 22.2741236 | 3.40088334 | 3.50734447 |
| 21 | 40g | compound | scale_only | 0.715105317 | 34.7390702 | 24.3340597 | 0.774938119 | 42.7094038 | 28.5145868 | 3.53266817 | 3.64032324 |
| 22 | 40g | compound | shared_column_affine | 0.733294213 | 33.5385647 | 19.9534299 | 0.793385546 | 40.87521 | 22.7799382 | 3.39944489 | 3.50651209 |
| 23 | 40g | compound | target_head_only | -0.169431673 | 70.985732 | 47.326328 | -0.261411375 | 102.751069 | 71.6015569 | 7.70005533 | 7.8113061 |
| 24 | 40g | row | affine | 0.791185708 | 32.0110551 | 18.8613486 | 0.85100921 | 37.0238547 | 20.6804159 | 3.18273543 | 3.30378792 |
| 25 | 40g | row | linear_policy | 0.791185708 | 32.0110552 | 18.8613486 | 0.851009209 | 37.0238547 | 20.6804159 | 3.18273543 | 3.30378793 |
| 26 | 40g | row | local_identity_shrinkage | 0.779221618 | 32.9108217 | 19.9986015 | 0.850403158 | 37.0609797 | 20.6929087 | 3.24098432 | 3.37417228 |
| 27 | 40g | row | monotone_spline | 0.790654466 | 32.0555302 | 18.8597729 | 0.854474579 | 36.5933763 | 19.847591 | 3.17216914 | 3.29791437 |
| 28 | 40g | row | nonlinear_policy | 0.790654466 | 32.0555302 | 18.8597729 | 0.854474579 | 36.5933763 | 19.847591 | 3.17216914 | 3.29791437 |
| 29 | 40g | row | scale_only | 0.743871923 | 35.4278899 | 24.6393799 | 0.803438868 | 42.6606947 | 28.5279499 | 3.57486197 | 3.69238945 |
| 30 | 40g | row | shared_column_affine | 0.788581231 | 32.2191405 | 19.0723971 | 0.847622459 | 37.3755403 | 21.052303 | 3.20687726 | 3.32789724 |
| 31 | 40g | row | target_head_only | -0.0826711366 | 73.4771817 | 46.4122625 | -0.178794124 | 105.724222 | 71.1574211 | 7.95061795 | 8.06910193 |
| 32 | 8g | compound | affine | 0.804183626 | 7.39319806 | 3.88424684 | 0.899444814 | 9.17605206 | 4.72780545 | 0.754518801 | 0.778987648 |
| 33 | 8g | compound | linear_policy | 0.85316155 | 6.53021883 | 3.78195958 | 0.909083101 | 8.72036903 | 4.56068988 | 0.685586542 | 0.701667092 |
| 34 | 8g | compound | local_identity_shrinkage | 0.876619764 | 5.99423034 | 3.25531751 | 0.909457919 | 8.59407834 | 4.7462885 | 0.647647847 | 0.660439503 |
| 35 | 8g | compound | monotone_spline | 0.802935092 | 7.41042782 | 3.84470394 | 0.898759787 | 9.19559025 | 4.72110357 | 0.75621977 | 0.780803912 |
| 36 | 8g | compound | nonlinear_policy | 0.85316155 | 6.53021883 | 3.78195958 | 0.909083101 | 8.72036903 | 4.56068988 | 0.685586542 | 0.701667092 |
| 37 | 8g | compound | scale_only | 0.853161546 | 6.5302189 | 3.78195963 | 0.9090831 | 8.72036909 | 4.56068992 | 0.685586548 | 0.701667099 |
| 38 | 8g | compound | shared_column_affine | 0.848160917 | 6.58064061 | 3.50054084 | 0.91025167 | 8.60091247 | 4.4102681 | 0.68507078 | 0.702221322 |
| 39 | 8g | compound | target_head_only | 0.882364372 | 5.85147673 | 3.04107529 | 0.886658567 | 9.88319044 | 4.70007943 | 0.678682519 | 0.684303024 |
| 40 | 8g | row | affine | 0.842078527 | 6.6066767 | 3.56043672 | 0.86721644 | 9.49291375 | 5.64921101 | 0.714465268 | 0.730171965 |
| 41 | 8g | row | linear_policy | 0.841634601 | 6.60992373 | 3.71453116 | 0.871784876 | 9.35503443 | 5.65747541 | 0.710383083 | 0.726555853 |
| 42 | 8g | row | local_identity_shrinkage | 0.850606799 | 6.47345125 | 3.54957387 | 0.868345082 | 9.46017794 | 5.63623111 | 0.704993385 | 0.720136631 |
| 43 | 8g | row | monotone_spline | 0.841265436 | 6.63411193 | 3.55750775 | 0.867512822 | 9.4901048 | 5.63139068 | 0.716118796 | 0.732302288 |
| 44 | 8g | row | nonlinear_policy | 0.840740539 | 6.63890187 | 3.71603116 | 0.871632945 | 9.3661006 | 5.66565362 | 0.712566049 | 0.729174344 |
| 45 | 8g | row | scale_only | 0.852852013 | 6.19854078 | 3.82488382 | 0.874623506 | 9.24361438 | 5.51975048 | 0.680813669 | 0.690482992 |
| 46 | 8g | row | shared_column_affine | 0.843979483 | 6.57417779 | 3.50426759 | 0.870673891 | 9.36510515 | 5.4930557 | 0.708428054 | 0.724553387 |
| 47 | 8g | row | target_head_only | 0.863370088 | 6.06985454 | 3.34305143 | 0.878758554 | 9.12294668 | 5.29794852 | 0.668895025 | 0.678905355 |

## Validation selections

| index | column | protocol | nonlinear_policy | contexts |
| --- | --- | --- | --- | --- |
| 0 | 25g | compound | affine | 1 |
| 1 | 25g | compound | monotone_spline | 12 |
| 2 | 25g | compound | scale_only | 7 |
| 3 | 25g | row | affine | 1 |
| 4 | 25g | row | monotone_spline | 11 |
| 5 | 25g | row | scale_only | 8 |
| 6 | 40g | compound | affine | 5 |
| 7 | 40g | compound | monotone_spline | 15 |
| 8 | 40g | row | affine | 1 |
| 9 | 40g | row | monotone_spline | 18 |
| 10 | 40g | row | scale_only | 1 |
| 11 | 8g | compound | affine | 1 |
| 12 | 8g | compound | monotone_spline | 4 |
| 13 | 8g | compound | scale_only | 15 |
| 14 | 8g | row | affine | 3 |
| 15 | 8g | row | monotone_spline | 5 |
| 16 | 8g | row | scale_only | 12 |

| index | column | protocol | shared_strength | contexts |
| --- | --- | --- | --- | --- |
| 0 | 25g | compound | 0 | 14 |
| 1 | 25g | compound | 0.1 | 4 |
| 2 | 25g | compound | 1 | 2 |
| 3 | 25g | row | 0 | 7 |
| 4 | 25g | row | 0.1 | 3 |
| 5 | 25g | row | 1 | 10 |
| 6 | 40g | compound | 0 | 6 |
| 7 | 40g | compound | 0.1 | 6 |
| 8 | 40g | compound | 1 | 8 |
| 9 | 40g | row | 0 | 7 |
| 10 | 40g | row | 0.1 | 5 |
| 11 | 40g | row | 1 | 8 |
| 12 | 8g | compound | 0 | 8 |
| 13 | 8g | compound | 0.1 | 6 |
| 14 | 8g | compound | 1 | 6 |
| 15 | 8g | row | 0 | 9 |
| 16 | 8g | row | 0.1 | 1 |
| 17 | 8g | row | 1 | 10 |

## Interpretation limits and data

Negative results do not establish an irreducible noise floor, prove affine is the true physical mapping, or identify sum readout as the residual cause. Adaptive readout was not tested and is not warranted by exclusion alone. A successful diagnostic only motivates independent confirmation of that bounded mechanism. Only three observed column/flow combinations exist, so mass and flow effects cannot be separated; no new-specification extrapolation or physical law is claimed. Target-compound holdout is not source-unseen OOD. These frozen tests have historical exposure: all results are developmental, not pristine confirmatory evidence.

Additional data most useful for a stronger conclusion: independent compound/batch validation; matched conditions with replicated measurements; crossed mass/flow settings; greater high-q50 tail coverage; and genuinely source-unseen molecules for OOD. Existing data were sufficient to run the two diagnostics, but cannot identify all four hypotheses uniquely.

Every per-seed/per-budget R²/RMSE/MAE/NRMSE is in all_metrics.csv; all point-metric mean/std/median/min/max are in aggregate_metrics.csv. paired_aulc.csv includes linear-policy and local-shrinkage controls. contexts/* contains train/validation/test IDs, donor purges, all validation candidate scores, slopes, knots, coefficients and truth-free frozen predictions. No further method is appended after this evaluation.
