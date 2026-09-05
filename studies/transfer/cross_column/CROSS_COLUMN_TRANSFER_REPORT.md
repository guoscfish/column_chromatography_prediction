# Cross-column transfer report

We test whether transfer across column specifications is primarily a low-dimensional calibration problem or requires condition-/representation-dependent adaptation.

Decision: `LOW_DIMENSIONAL_COLUMN_CALIBRATION_SUPPORTED`. Primary next route: `ACTIVE_CALIBRATION`.

Affine is strongly better than zero-shot in all six column/protocol contexts (zero-shot wins 0/5 seeds in each). Condition Ridge passes the directional 4/5 gate in three contexts, but its mean relative AULC gains are only 1.6-2.7%; no context reaches a 5% material gain. It worsens both 8g protocols. Head-only is 0/5 against affine in every 25g/40g context, so the conditional last2 trigger for those columns is not met. The evidence supports low-dimensional calibration, not a residual-transfer or new column-context model as the primary line.

Normalization uses fixed source-train scales V1=7.8797 mL and V2=16.0765 mL. The historical 8g row study remains separate (affine AULC 0.570); the present row and compound estimates use new matched five-seed protocols and a 20% test set.

For 8g target-compound holdout, AULC is 0.850 for affine, 0.725 for head-only, and 0.930 for last2. Affine improves strongly over zero-shot (1.507) but is not first; scale-only and the descriptive mass-ratio baseline are better. The matched affine row-to-compound change is +0.061 AULC.

At budget 100, 25g affine RMSE is 18.6/25.2 mL under row split and 14.1/20.1 mL under compound split. For 40g the corresponding values are 32.0/37.0 and 33.6/41.0 mL. Thus trend R2 remains useful while absolute error grows materially with column size.

Budget-100 affine slopes increase monotonically with column mass (approximately 2.5/1.8 at 8g, 6.4-6.7/5.0-5.1 at 25g, and 13.3/8.7-9.0 at 40g for V1/V2). Intercepts are nonzero and become more negative. Learned scales are of the same order as mass ratios but do not equal them, especially for 40g V1. Condition associations are modest and descriptive; repeated-compound variance remains visible, but neither condition Ridge nor shallow neural adaptation supplies a large, cross-column incremental gain.

Target-compound holdout means no target-domain label for a held-out compound; it does not imply that the source predictor never saw that compound. The audit finds too few source-unseen compounds, so `SOURCE_UNSEEN_MOLECULE_OOD_NOT_ESTIMABLE_WITH_CURRENT_DATA`.

The mass-ratio result is `DESCRIPTIVE_PHYSICAL_BASELINE_ONLY`. R2 describes trend fit; RMSE/MAE and normalized RMSE remain necessary for absolute-error interpretation.

## AULC

| column | protocol | method | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | scale_only | 1.60743406 | 0.316738236 | 1.65288829 | 1.0761216 | 1.88272535 |
| 25g | compound | affine_condition_residual | 1.64869445 | 0.605080533 | 1.57808318 | 0.930768743 | 2.60165543 |
| 25g | compound | affine | 1.67531299 | 0.594080973 | 1.63056727 | 0.96854132 | 2.61138909 |
| 25g | compound | column_mass_ratio_scaling | 2.33858006 | 0.163481602 | 2.36717695 | 2.06941686 | 2.51572306 |
| 25g | compound | target_head_only | 3.05001093 | 0.605022051 | 3.22418488 | 2.03168044 | 3.50267566 |
| 25g | compound | zero_shot | 4.8893521 | 0.748075157 | 5.06052768 | 3.66282153 | 5.5498219 |
| 25g | row | affine_condition_residual | 2.05277886 | 0.521280836 | 1.85686876 | 1.4989562 | 2.8782689 |
| 25g | row | scale_only | 2.06012008 | 0.299165715 | 1.97067375 | 1.67359826 | 2.41861304 |
| 25g | row | affine | 2.08828285 | 0.513163875 | 1.8793842 | 1.55469034 | 2.90199432 |
| 25g | row | column_mass_ratio_scaling | 2.65897972 | 0.412264425 | 2.61604699 | 2.09080882 | 3.15948796 |
| 25g | row | target_head_only | 3.37612611 | 0.422782214 | 3.30871162 | 2.91337065 | 3.90842994 |
| 25g | row | zero_shot | 5.13725331 | 0.622917583 | 5.0338546 | 4.51204916 | 5.99464796 |
| 40g | compound | scale_only | 3.60119812 | 0.440966467 | 3.85624433 | 3.04285593 | 3.95179335 |
| 40g | compound | affine_condition_residual | 3.89948221 | 0.934746108 | 3.78028011 | 3.09793891 | 5.47098375 |
| 40g | compound | affine | 3.9158492 | 0.980110061 | 3.86522213 | 3.05347069 | 5.5260677 |
| 40g | compound | column_mass_ratio_scaling | 4.33018078 | 0.352938875 | 4.48401978 | 3.90348174 | 4.72480709 |
| 40g | compound | target_head_only | 7.71900745 | 0.389243832 | 7.9863578 | 7.14898781 | 7.99846317 |
| 40g | compound | zero_shot | 9.63588292 | 0.463474485 | 9.88578205 | 9.02083662 | 10.0929731 |
| 40g | row | affine_condition_residual | 3.23063906 | 0.84148974 | 2.66199959 | 2.60209007 | 4.35550056 |
| 40g | row | affine | 3.32013233 | 0.868683733 | 2.73303461 | 2.6692586 | 4.517427 |
| 40g | row | scale_only | 3.63486831 | 0.665990914 | 3.20555847 | 3.11806577 | 4.58824409 |
| 40g | row | column_mass_ratio_scaling | 4.37155674 | 0.479894737 | 4.11248207 | 3.92708578 | 4.9332497 |
| 40g | row | target_head_only | 7.96439166 | 1.12202577 | 8.21740487 | 6.77543614 | 9.46306518 |
| 40g | row | zero_shot | 9.89456022 | 1.28610812 | 10.1267008 | 8.48018396 | 11.5203116 |
| 8g | compound | column_mass_ratio_scaling | 0.682404006 | 0.0810823318 | 0.650623932 | 0.612755726 | 0.803673812 |
| 8g | compound | target_head_only | 0.725007752 | 0.0836987066 | 0.770599815 | 0.579088612 | 0.773965865 |
| 8g | compound | scale_only | 0.734889677 | 0.211296843 | 0.681312023 | 0.528293167 | 1.06533692 |
| 8g | compound | affine | 0.850090658 | 0.304561943 | 0.811644277 | 0.517879698 | 1.33989843 |
| 8g | compound | affine_condition_residual | 0.903987548 | 0.316627274 | 0.881215146 | 0.516914627 | 1.34027275 |
| 8g | compound | last2 | 0.930188224 | 0.104252299 | 0.920855796 | 0.821934574 | 1.09896013 |
| 8g | compound | zero_shot | 1.50681521 | 0.105339592 | 1.50520513 | 1.3866882 | 1.64745873 |
| 8g | row | target_head_only | 0.67898 | 0.113004128 | 0.652608607 | 0.538254958 | 0.825467813 |
| 8g | row | scale_only | 0.729654704 | 0.1362811 | 0.718436968 | 0.61701423 | 0.955268656 |
| 8g | row | column_mass_ratio_scaling | 0.736537497 | 0.0812203476 | 0.709292893 | 0.65285034 | 0.828092752 |
| 8g | row | affine | 0.789131686 | 0.104427666 | 0.773612253 | 0.66592492 | 0.953310967 |
| 8g | row | affine_condition_residual | 0.815300856 | 0.106709113 | 0.820314926 | 0.66215518 | 0.951753582 |
| 8g | row | zero_shot | 1.5131718 | 0.202649825 | 1.505668 | 1.26347878 | 1.75978414 |

## Paired AULC versus affine

| column | protocol | method | reference | mean_delta | median_delta | std_delta | wins | seeds | stable_improvement | affine_mean | relative_mean_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | affine_condition_residual | affine | -0.0266185397 | -0.0304540258 | 0.0204194717 | 5 | 5 | True | 1.67531299 | -0.0158886966 |
| 25g | compound | column_mass_ratio_scaling | affine | 0.663267072 | 0.728601756 | 0.452472958 | 1 | 5 | False | 1.67531299 | 0.395906363 |
| 25g | compound | scale_only | affine | -0.0678789247 | 0.10758028 | 0.372048089 | 1 | 5 | False | 1.67531299 | -0.0405171602 |
| 25g | compound | target_head_only | affine | 1.37469795 | 1.51459091 | 0.53228094 | 0 | 5 | False | 1.67531299 | 0.820561862 |
| 25g | compound | zero_shot | affine | 3.21403911 | 3.2870623 | 0.635284459 | 0 | 5 | False | 1.67531299 | 1.91847084 |
| 25g | row | affine_condition_residual | affine | -0.0355039948 | -0.034088141 | 0.0138663806 | 5 | 5 | True | 2.08828285 | -0.0170015258 |
| 25g | row | column_mass_ratio_scaling | affine | 0.57069687 | 0.613286902 | 0.192376139 | 0 | 5 | False | 2.08828285 | 0.273285235 |
| 25g | row | scale_only | affine | -0.028162769 | 0.0730408735 | 0.25545036 | 1 | 5 | False | 2.08828285 | -0.0134860893 |
| 25g | row | target_head_only | affine | 1.28784326 | 1.35868031 | 0.195703972 | 0 | 5 | False | 2.08828285 | 0.616699627 |
| 25g | row | zero_shot | affine | 3.04897045 | 3.09265364 | 0.21041511 | 0 | 5 | False | 2.08828285 | 1.46003711 |
| 40g | compound | affine_condition_residual | affine | -0.0163669854 | -0.0550839447 | 0.0774448474 | 3 | 5 | False | 3.9158492 | -0.00417967714 |
| 40g | compound | column_mass_ratio_scaling | affine | 0.41433158 | 0.697034378 | 0.806971825 | 1 | 5 | False | 3.9158492 | 0.10580887 |
| 40g | compound | scale_only | affine | -0.31465108 | 0.0237552397 | 0.766620794 | 2 | 5 | False | 3.9158492 | -0.0803532169 |
| 40g | compound | target_head_only | affine | 3.80315825 | 4.09551712 | 1.07857459 | 0 | 5 | False | 3.9158492 | 0.971221836 |
| 40g | compound | zero_shot | affine | 5.72003372 | 5.96736593 | 1.17165144 | 0 | 5 | False | 3.9158492 | 1.46073902 |
| 40g | row | affine_condition_residual | affine | -0.0894932721 | -0.071035015 | 0.0441404127 | 5 | 5 | True | 3.32013233 | -0.0269547305 |
| 40g | row | column_mass_ratio_scaling | affine | 1.0514244 | 1.2252791 | 0.413431313 | 0 | 5 | False | 3.32013233 | 0.316681474 |
| 40g | row | scale_only | affine | 0.314735972 | 0.443321764 | 0.207902916 | 0 | 5 | False | 3.32013233 | 0.094796213 |
| 40g | row | target_head_only | affine | 4.64425933 | 4.23827009 | 0.71797357 | 0 | 5 | False | 3.32013233 | 1.39881753 |
| 40g | row | zero_shot | affine | 6.57442789 | 6.14756598 | 0.902678952 | 0 | 5 | False | 3.32013233 | 1.98017044 |
| 8g | compound | affine_condition_residual | affine | 0.0538968898 | 0.00512325683 | 0.11103291 | 1 | 5 | False | 0.850090658 | 0.0634013435 |
| 8g | compound | column_mass_ratio_scaling | affine | -0.167686652 | -0.0925940189 | 0.266981611 | 4 | 5 | True | 0.850090658 | -0.197257375 |
| 8g | compound | last2 | affine | 0.0800975657 | 0.123563678 | 0.201562522 | 1 | 5 | False | 0.850090658 | 0.0942223807 |
| 8g | compound | scale_only | affine | -0.115200981 | -0.11608907 | 0.104739091 | 4 | 5 | True | 0.850090658 | -0.135516113 |
| 8g | compound | target_head_only | affine | -0.125082906 | -0.0376784126 | 0.261759469 | 3 | 5 | False | 0.850090658 | -0.147140667 |
| 8g | compound | zero_shot | affine | 0.656724547 | 0.693560855 | 0.233192178 | 0 | 5 | False | 0.850090658 | 0.772534718 |
| 8g | row | affine_condition_residual | affine | 0.0261691706 | 0.00641201182 | 0.0465113624 | 2 | 5 | False | 0.789131686 | 0.0331619818 |
| 8g | row | column_mass_ratio_scaling | affine | -0.0525941889 | -0.0643193604 | 0.0791544173 | 4 | 5 | True | 0.789131686 | -0.0666481778 |
| 8g | row | scale_only | affine | -0.0594769819 | -0.0489106901 | 0.0663201073 | 4 | 5 | True | 0.789131686 | -0.0753701606 |
| 8g | row | target_head_only | affine | -0.110151685 | -0.121003646 | 0.129564619 | 4 | 5 | True | 0.789131686 | -0.139585936 |
| 8g | row | zero_shot | affine | 0.724040111 | 0.732055742 | 0.222059447 | 0 | 5 | False | 0.789131686 | 0.91751494 |

## Budget 100 point and absolute-error metrics

| column | protocol | method | V1_r2 | V1_rmse | V1_mae | V2_r2 | V2_rmse | V2_mae | normalized_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | affine | 0.836770811 | 14.1437998 | 8.84463892 | 0.872567103 | 20.1297115 | 12.6080769 | 1.52354782 |
| 25g | compound | affine_condition_residual | 0.850641302 | 13.5046565 | 8.28455807 | 0.873539204 | 20.0273467 | 12.6369185 | 1.47980761 |
| 25g | compound | scale_only | 0.822533722 | 14.7586126 | 10.3311545 | 0.856383991 | 21.3405612 | 15.0078261 | 1.60021945 |
| 25g | compound | target_head_only | 0.483927451 | 25.3517097 | 14.5006417 | 0.335982835 | 46.0397653 | 28.9216564 | 3.04057619 |
| 25g | compound | zero_shot | -0.604762006 | 44.2282607 | 31.8553543 | -0.423101348 | 66.9705248 | 48.4121393 | 4.8893521 |
| 25g | row | affine | 0.746508321 | 18.6082205 | 9.71050769 | 0.816426673 | 25.2226077 | 14.3620431 | 1.96523109 |
| 25g | row | affine_condition_residual | 0.762485228 | 18.0124579 | 9.4589708 | 0.819257736 | 25.0188221 | 14.2962131 | 1.92108927 |
| 25g | row | scale_only | 0.737948806 | 18.8817811 | 10.6871391 | 0.791517025 | 26.8216001 | 15.8154919 | 2.03232046 |
| 25g | row | target_head_only | 0.411279518 | 28.2771172 | 16.0440159 | 0.270070207 | 50.3619845 | 30.6782405 | 3.36063307 |
| 25g | row | zero_shot | -0.591105057 | 46.4425703 | 33.0887743 | -0.426794629 | 70.4235421 | 50.0072221 | 5.13725331 |
| 40g | compound | affine | 0.732092815 | 33.5744582 | 20.0755742 | 0.791624055 | 40.9652489 | 23.0829021 | 3.40452282 |
| 40g | compound | affine_condition_residual | 0.747213025 | 32.772767 | 19.3728409 | 0.788266122 | 41.2315244 | 24.2566018 | 3.3619334 |
| 40g | compound | scale_only | 0.715105317 | 34.7390702 | 24.3340597 | 0.774938119 | 42.7094038 | 28.5145868 | 3.53266817 |
| 40g | compound | target_head_only | -0.169431671 | 70.985732 | 47.326328 | -0.261411376 | 102.751069 | 71.6015569 | 7.70005533 |
| 40g | compound | zero_shot | -0.927453183 | 91.1545781 | 66.89399 | -0.832483183 | 123.844192 | 91.7486047 | 9.63588292 |
| 40g | row | affine | 0.791185708 | 32.0110551 | 18.8613486 | 0.85100921 | 37.0238547 | 20.6804159 | 3.18273543 |
| 40g | row | affine_condition_residual | 0.810965272 | 30.4730553 | 17.5089853 | 0.852863651 | 36.7735023 | 20.7157574 | 3.0773561 |
| 40g | row | scale_only | 0.743871923 | 35.4278899 | 24.6393799 | 0.803438868 | 42.6606947 | 28.5279499 | 3.57486197 |
| 40g | row | target_head_only | -0.0826711386 | 73.4771817 | 46.4122625 | -0.178794125 | 105.724223 | 71.1574211 | 7.95061795 |
| 40g | row | zero_shot | -0.765020853 | 93.6984719 | 65.8649181 | -0.702334216 | 126.971258 | 91.315141 | 9.89456022 |
| 8g | compound | affine | 0.804183626 | 7.39319806 | 3.88424684 | 0.899444814 | 9.17605206 | 4.72780545 | 0.754518801 |
| 8g | compound | affine_condition_residual | 0.801570999 | 7.429443 | 3.96126557 | 0.899569106 | 9.17845738 | 4.82831538 | 0.756893515 |
| 8g | compound | last2 | 0.738296088 | 8.62585414 | 4.73569992 | 0.841072392 | 11.729807 | 6.50771989 | 0.912161446 |
| 8g | compound | scale_only | 0.853161546 | 6.5302189 | 3.78195963 | 0.9090831 | 8.72036909 | 4.56068992 | 0.685586548 |
| 8g | compound | target_head_only | 0.882364373 | 5.85147672 | 3.04107528 | 0.886658563 | 9.88319062 | 4.70007944 | 0.678682523 |
| 8g | compound | zero_shot | 0.333329941 | 13.9731807 | 9.15795927 | 0.53489168 | 19.9398137 | 13.2427402 | 1.50681521 |
| 8g | row | affine | 0.842078527 | 6.6066767 | 3.56043672 | 0.86721644 | 9.49291375 | 5.64921101 | 0.714465268 |
| 8g | row | affine_condition_residual | 0.836786479 | 6.71641708 | 3.64438855 | 0.865626006 | 9.55740507 | 5.67266542 | 0.723434553 |
| 8g | row | scale_only | 0.852852013 | 6.19854078 | 3.82488382 | 0.874623506 | 9.24361438 | 5.51975048 | 0.680813669 |
| 8g | row | target_head_only | 0.863370089 | 6.06985455 | 3.34305143 | 0.878758555 | 9.12294661 | 5.29794847 | 0.668895023 |
| 8g | row | zero_shot | 0.304230785 | 14.3150482 | 9.49952309 | 0.495649221 | 19.4467006 | 13.6199451 | 1.5131718 |

## Budget 100 affine coefficients

| column | protocol | target | slope / mean | slope / std | intercept / mean | intercept / std |
| --- | --- | --- | --- | --- | --- | --- |
| 25g | compound | V1 | 6.66040434 | 0.63417858 | -8.6836062 | 1.95642858 |
| 25g | compound | V2 | 5.13591766 | 0.550379646 | -19.1547385 | 8.04279346 |
| 25g | row | V1 | 6.40344316 | 0.147654233 | -7.68934379 | 1.0544544 |
| 25g | row | V2 | 4.96363124 | 0.391377404 | -17.4673108 | 4.0547412 |
| 40g | compound | V1 | 13.254245 | 0.531344274 | -20.2887725 | 4.74061384 |
| 40g | compound | V2 | 8.9976973 | 0.621670636 | -37.9083788 | 13.6465068 |
| 40g | row | V1 | 13.3198964 | 0.655699234 | -20.5607893 | 3.20522188 |
| 40g | row | V2 | 8.7457586 | 0.628743647 | -32.8978199 | 6.66544853 |
| 8g | compound | V1 | 2.47777421 | 0.512717242 | -2.36648872 | 2.96862151 |
| 8g | compound | V2 | 1.76989824 | 0.224313771 | -0.194899607 | 2.99270623 |
| 8g | row | V1 | 2.54245275 | 0.166811022 | -2.61634507 | 1.14385594 |
| 8g | row | V2 | 1.88693873 | 0.12085796 | -1.41170322 | 2.96881344 |

## Learned scale versus mass ratio

| column | protocol | target | learned_scale | mass_ratio | learned_minus_mass_ratio |
| --- | --- | --- | --- | --- | --- |
| 25g | compound | V1 | 5.79512108 | 6.25 | -0.454878922 |
| 25g | compound | V2 | 4.26758657 | 6.25 | -1.98241343 |
| 25g | row | V1 | 5.58985485 | 6.25 | -0.660145148 |
| 25g | row | V2 | 4.14637776 | 6.25 | -2.10362224 |
| 40g | compound | V1 | 11.3525068 | 10 | 1.35250678 |
| 40g | compound | V2 | 7.42692095 | 10 | -2.57307905 |
| 40g | row | V1 | 11.3127998 | 10 | 1.31279975 |
| 40g | row | V2 | 7.35060962 | 10 | -2.64939038 |
| 8g | compound | V1 | 2.26742243 | 2 | 0.26742243 |
| 8g | compound | V2 | 1.7587534 | 2 | -0.241246596 |
| 8g | row | V1 | 2.34479367 | 2 | 0.344793666 |
| 8g | row | V2 | 1.85327387 | 2 | -0.146726131 |

## Row-to-compound AULC gap

| column | method | compound | row | compound_minus_row | compound_vs_row_percent |
| --- | --- | --- | --- | --- | --- |
| 25g | affine | 1.67531299 | 2.08828285 | -0.412969865 | -19.7755713 |
| 25g | affine_condition_residual | 1.64869445 | 2.05277886 | -0.40408441 | -19.6847512 |
| 25g | column_mass_ratio_scaling | 2.33858006 | 2.65897972 | -0.320399663 | -12.0497219 |
| 25g | scale_only | 1.60743406 | 2.06012008 | -0.452686021 | -21.9737686 |
| 25g | target_head_only | 3.05001093 | 3.37612611 | -0.326115177 | -9.65944891 |
| 25g | zero_shot | 4.8893521 | 5.13725331 | -0.247901212 | -4.82555944 |
| 40g | affine | 3.9158492 | 3.32013233 | 0.595716865 | 17.9425639 |
| 40g | affine_condition_residual | 3.89948221 | 3.23063906 | 0.668843151 | 20.7031222 |
| 40g | column_mass_ratio_scaling | 4.33018078 | 4.37155674 | -0.0413759569 | -0.946481069 |
| 40g | scale_only | 3.60119812 | 3.63486831 | -0.0336701876 | -0.926311072 |
| 40g | target_head_only | 7.71900745 | 7.96439166 | -0.245384214 | -3.08101641 |
| 40g | zero_shot | 9.63588292 | 9.89456022 | -0.258677302 | -2.61433855 |
| 8g | affine | 0.850090658 | 0.789131686 | 0.0609589726 | 7.72481624 |
| 8g | affine_condition_residual | 0.903987548 | 0.815300856 | 0.0886866918 | 10.8777872 |
| 8g | column_mass_ratio_scaling | 0.682404006 | 0.736537497 | -0.0541334904 | -7.34972633 |
| 8g | last2 | 0.930188224 | NA | NA | NA |
| 8g | scale_only | 0.734889677 | 0.729654704 | 0.00523497314 | 0.717458972 |
| 8g | target_head_only | 0.725007752 | 0.67898 | 0.0460277516 | 6.77895543 |
| 8g | zero_shot | 1.50681521 | 1.5131718 | -0.00635659113 | -0.420083902 |

## Affine residual associations

| column | protocol | target | feature | pearson | spearman |
| --- | --- | --- | --- | --- | --- |
| 25g | compound | V1 | PE_fraction | 0.256107448 | 0.360550357 |
| 25g | compound | V1 | loading_amount_ul | -0.0429470535 | -0.0886066906 |
| 25g | compound | V1 | loading_solvent_code | 0.110101848 | 0.185440814 |
| 25g | compound | V1 | loading_solvent_volume_ul | -0.040773482 | -0.0392646977 |
| 25g | compound | V1 | source_prediction | -0.0923025961 | -0.084771937 |
| 25g | compound | V2 | PE_fraction | -0.0241012944 | -0.104286052 |
| 25g | compound | V2 | loading_amount_ul | -0.01302019 | 0.0401962455 |
| 25g | compound | V2 | loading_solvent_code | 0.0664644605 | 0.0940821807 |
| 25g | compound | V2 | loading_solvent_volume_ul | -0.208311357 | -0.337361565 |
| 25g | compound | V2 | source_prediction | -0.167784064 | -0.287750056 |
| 25g | row | V1 | PE_fraction | 0.195356938 | 0.251098536 |
| 25g | row | V1 | loading_amount_ul | -0.113460856 | -0.142387945 |
| 25g | row | V1 | loading_solvent_code | 0.136630521 | 0.16863655 |
| 25g | row | V1 | loading_solvent_volume_ul | -0.0393894166 | -0.0618725781 |
| 25g | row | V1 | source_prediction | -0.140524599 | -0.215170642 |
| 25g | row | V2 | PE_fraction | 0.0681076039 | -0.0344707765 |
| 25g | row | V2 | loading_amount_ul | -0.0776084801 | -0.00666316885 |
| 25g | row | V2 | loading_solvent_code | 0.0855242182 | 0.0784326171 |
| 25g | row | V2 | loading_solvent_volume_ul | -0.111604456 | -0.21952738 |
| 25g | row | V2 | source_prediction | 0.0614292718 | -0.23498014 |
| 40g | compound | V1 | PE_fraction | 0.159697667 | 0.27305506 |
| 40g | compound | V1 | loading_amount_ul | NA | NA |
| 40g | compound | V1 | loading_solvent_code | 0.113825589 | 0.167410304 |
| 40g | compound | V1 | loading_solvent_volume_ul | -0.0460790393 | -0.0239582972 |
| 40g | compound | V1 | source_prediction | -0.247651355 | -0.215217265 |
| 40g | compound | V2 | PE_fraction | 0.000981218362 | -0.144335726 |
| 40g | compound | V2 | loading_amount_ul | NA | NA |
| 40g | compound | V2 | loading_solvent_code | 0.0377499329 | -0.0111432907 |
| 40g | compound | V2 | loading_solvent_volume_ul | -0.213698504 | -0.284915054 |
| 40g | compound | V2 | source_prediction | -0.155993427 | -0.34172506 |
| 40g | row | V1 | PE_fraction | 0.338304102 | 0.454928096 |
| 40g | row | V1 | loading_amount_ul | NA | NA |
| 40g | row | V1 | loading_solvent_code | 0.0892231757 | 0.115750196 |
| 40g | row | V1 | loading_solvent_volume_ul | -0.0464070186 | -0.0150454605 |
| 40g | row | V1 | source_prediction | 0.21271729 | 0.016808431 |
| 40g | row | V2 | PE_fraction | 0.092335819 | 0.0230990397 |
| 40g | row | V2 | loading_amount_ul | NA | NA |
| 40g | row | V2 | loading_solvent_code | 0.0425819705 | -0.00905029168 |
| 40g | row | V2 | loading_solvent_volume_ul | -0.174240919 | -0.198982082 |
| 40g | row | V2 | source_prediction | 0.117818549 | -0.181974861 |
| 8g | compound | V1 | PE_fraction | 0.0544986666 | 0.108818698 |
| 8g | compound | V1 | loading_amount_ul | NA | NA |
| 8g | compound | V1 | loading_solvent_code | 0.0669907114 | 0.0622900395 |
| 8g | compound | V1 | loading_solvent_volume_ul | 0.0353406909 | 0.0781037078 |
| 8g | compound | V1 | source_prediction | -0.104032733 | -0.264665101 |
| 8g | compound | V2 | PE_fraction | -0.00975481509 | -0.0884782113 |
| 8g | compound | V2 | loading_amount_ul | NA | NA |
| 8g | compound | V2 | loading_solvent_code | 0.0789956755 | 0.0728950598 |
| 8g | compound | V2 | loading_solvent_volume_ul | 0.0220671391 | 0.0298947982 |
| 8g | compound | V2 | source_prediction | 0.124759814 | -0.102980643 |
| 8g | row | V1 | PE_fraction | 0.0310766727 | 0.0888977469 |
| 8g | row | V1 | loading_amount_ul | NA | NA |
| 8g | row | V1 | loading_solvent_code | 0.0257269005 | 0.0467154434 |
| 8g | row | V1 | loading_solvent_volume_ul | 0.0404805134 | 0.0535028428 |
| 8g | row | V1 | source_prediction | -0.277934467 | -0.313988765 |
| 8g | row | V2 | PE_fraction | -0.110271642 | -0.171773465 |
| 8g | row | V2 | loading_amount_ul | NA | NA |
| 8g | row | V2 | loading_solvent_code | 0.00239932078 | -0.00596548318 |
| 8g | row | V2 | loading_solvent_volume_ul | 0.0478670137 | 0.00568400983 |
| 8g | row | V2 | source_prediction | -0.218864858 | -0.322137014 |

## Affine residual compound structure

| column | protocol | target | mean | std |
| --- | --- | --- | --- | --- |
| 25g | compound | V1 | 0.304213147 | 0.0885115696 |
| 25g | compound | V2 | 0.419134662 | 0.0997604605 |
| 25g | row | V1 | 0.784190578 | 0.0859613362 |
| 25g | row | V2 | 0.754724171 | 0.0805742732 |
| 40g | compound | V1 | 0.29991633 | 0.0658724004 |
| 40g | compound | V2 | 0.383211703 | 0.0288665264 |
| 40g | row | V1 | 0.64221019 | 0.143016597 |
| 40g | row | V2 | 0.701156779 | 0.135352015 |
| 8g | compound | V1 | 0.249126002 | 0.0753262011 |
| 8g | compound | V2 | 0.278356241 | 0.0819966691 |
| 8g | row | V1 | 0.622440952 | 0.112267373 |
| 8g | row | V2 | 0.611280778 | 0.117398706 |

All simple fits use gradient-train labels only. Ridge alpha selection is nested and group-isolated; neural checkpoint selection uses validation only; test labels are read after fits freeze. No active acquisition or predictor architecture search was run.
