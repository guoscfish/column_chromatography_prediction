# Pre-model physical audit supplement

This supplement precedes all model fitting and uses exactly audit_training_ids.json. Ratio is the median of observed target/source-train matched values (source exact repeats averaged), with source denominator floor 0.5 mL. These are descriptive paired ratios, not fitted calibration coefficients. Center is not peak apex; width is not peak variance.

| column | matching | dimension | level | target | rows | compounds | ratio_rows | median_ratio | ratio_q25 | ratio_q75 | packing_mass_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8g | exact | overall | all | V1 | 52 | 10 | 52 | 1.9790228940738306 | 1.6610600322004734 | 2.304476629806185 | 2.0 |
| 8g | exact | overall | all | V2 | 52 | 10 | 52 | 1.7642771214895734 | 1.5761890641975063 | 1.989435107376284 | 2.0 |
| 8g | exact | overall | all | center | 52 | 10 | 52 | 1.8700096675838478 | 1.6619831940394174 | 2.0584267596258274 | 2.0 |
| 8g | exact | overall | all | width | 52 | 10 | 50 | 1.5882922043040226 | 1.3197608009137936 | 1.8161796611180026 | 2.0 |
| 8g | relaxed | overall | all | V1 | 57 | 11 | 57 | 2.0085106382978726 | 1.6774193548387095 | 2.3328631875881523 | 2.0 |
| 8g | relaxed | overall | all | V2 | 57 | 11 | 57 | 1.771406431615653 | 1.6089328332765092 | 2.0 | 2.0 |
| 8g | relaxed | overall | all | center | 57 | 11 | 57 | 1.916911045943304 | 1.6784506609283738 | 2.1045891141942366 | 2.0 |
| 8g | relaxed | overall | all | width | 57 | 11 | 55 | 1.5783132530120483 | 1.251237033638318 | 1.8288662638442936 | 2.0 |
| 25g | exact | overall | all | V1 | 0 | 0 | 0 | nan | nan | nan | 6.25 |
| 25g | exact | overall | all | V2 | 0 | 0 | 0 | nan | nan | nan | 6.25 |
| 25g | exact | overall | all | center | 0 | 0 | 0 | nan | nan | nan | 6.25 |
| 25g | exact | overall | all | width | 0 | 0 | 0 | nan | nan | nan | 6.25 |
| 25g | relaxed | overall | all | V1 | 40 | 8 | 40 | 4.599012366611209 | 3.975124968645859 | 7.301168173340798 | 6.25 |
| 25g | relaxed | overall | all | V2 | 40 | 8 | 40 | 3.733603652259034 | 3.005130014435337 | 4.379003898820844 | 6.25 |
| 25g | relaxed | overall | all | center | 40 | 8 | 40 | 3.8816850642036496 | 3.4128497604655745 | 5.202884312061528 | 6.25 |
| 25g | relaxed | overall | all | width | 40 | 8 | 39 | 2.830607476635514 | 2.2769083905822716 | 3.2419042359211723 | 6.25 |
| 40g | exact | overall | all | V1 | 0 | 0 | 0 | nan | nan | nan | 10.0 |
| 40g | exact | overall | all | V2 | 0 | 0 | 0 | nan | nan | nan | 10.0 |
| 40g | exact | overall | all | center | 0 | 0 | 0 | nan | nan | nan | 10.0 |
| 40g | exact | overall | all | width | 0 | 0 | 0 | nan | nan | nan | 10.0 |
| 40g | relaxed | overall | all | V1 | 54 | 10 | 54 | 8.067858971441574 | 6.985796178343949 | 11.20784507894592 | 10.0 |
| 40g | relaxed | overall | all | V2 | 54 | 10 | 54 | 6.089573860936821 | 4.750001639236771 | 6.906836131762789 | 10.0 |
| 40g | relaxed | overall | all | center | 54 | 10 | 54 | 7.050755954927972 | 5.271525309355159 | 7.912777763650043 | 10.0 |
| 40g | relaxed | overall | all | width | 54 | 10 | 53 | 3.7387116195063226 | 2.732009925558313 | 4.923236514522821 | 10.0 |

Full EA, volumetric-flow and prespecified loading-mass bins (<=50, 50–100, >100 mg) are in training_center_width_scales.csv. No matching at identical flow is available for 25g/40g, so their relaxed ratios do not isolate scale from flow.

Canonical flow census:

| column | flow_ml_min | canonical_rows | compounds |
| --- | --- | --- | --- |
| 4g | 4 | 8 | 3 |
| 4g | 5 | 371 | 31 |
| 4g | 6 | 12 | 3 |
| 4g | 8 | 11 | 3 |
| 4g | 10 | 3761 | 217 |
| 8g | 10 | 574 | 88 |
| 25g | 15 | 490 | 78 |
| 40g | 30 | 529 | 80 |

Raw and canonical censuses are distinct. Current branch context exposes source flow variation already present in the inherited source training data; original V2 did not explicitly consume flow. Thus improvements cannot uniquely distinguish new source-flow information from cross-column conditioning. No feature ablation is appended.
