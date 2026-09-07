# Physical metadata audit

Metadata census uses every raw row without target-label reads. Normalization and center/width evidence below uses only source-train and the frozen first compound-seed budget100 gradient-train rows, not a union of seed labels. Full outcome distributions are deferred until global prediction freeze; audit labels are previously purchased developmental data.

| column | rows | compounds | invalid_smiles | column_specs | packing_mass_g | EA_min | EA_max | loading_mass_min_mg | loading_mass_max_mg | loading_solvent_min_ul | loading_solvent_max_ul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4g | 4243 | 217 | 14 | Silica-CS 4g | 4.0 | 0.009900990099009901 | 1.0 | 30.0 | 327.3 | 200 | 2000 |
| 8g | 574 | 88 | 0 | Silica-CS 4g+4g | 8.0 | 0.0196078431372549 | 1.0 | 86.0 | 173.70000000000002 | 300 | 1000 |
| 25g | 569 | 78 | 0 | Silica-CS 25g | 25.0 | 0.0196078431372549 | 1.0 | 48.75 | 352.2 | 300 | 900 |
| 40g | 531 | 80 | 0 | Silica-CS 40g | 40.0 | 0.0196078431372549 | 1.0 | 94.6 | 235.0 | 200 | 900 |

## All flow values

| column | flow_ml_min | rows | fraction | compounds |
| --- | --- | --- | --- | --- |
| 4g | 4 | 8 | 0.0018854584020740043 | 3 |
| 4g | 5 | 371 | 0.08743813339618195 | 31 |
| 4g | 6 | 12 | 0.0028281876031110063 | 3 |
| 4g | 8 | 11 | 0.002592505302851756 | 3 |
| 4g | 10 | 3841 | 0.9052557152957813 | 217 |
| 8g | 10 | 574 | 1.0 | 88 |
| 25g | 15 | 569 | 1.0 | 78 |
| 40g | 30 | 531 | 1.0 | 80 |

## Raw pairing (direction source → target)

| source | target | shared_compounds | exact_rows | relaxed_rows |
| --- | --- | --- | --- | --- |
| 4g | 8g | 87 | 412 | 429 |
| 4g | 25g | 77 | 0 | 318 |
| 4g | 40g | 80 | 0 | 424 |
| 8g | 25g | 48 | 0 | 247 |
| 8g | 40g | 43 | 0 | 271 |
| 25g | 40g | 58 | 0 | 299 |

Exact matching uses canonical molecule, rational EA composition, loading solvent, density, sample volume, loading-solvent volume and flow; relaxed ignores flow only. Pair coverage is metadata availability, not free source-test labels. 8g is explicitly 4g+4g, with unspecified physical connection geometry. Source/target canonical eligibility differs from raw census (especially 25g); inherited ledgers remain unchanged.

Units: packing_mass_g [g] is nominal label mass; flow_ml_min [mL/min] is volumetric flow; loading_mass_mg = density[g/mL] × sample_volume[uL]; loading_mass_mg_per_g [mg/g]; loading_solvent_ul_per_g [uL/g]; flow_per_g [mL/min/g] is an engineering proxy. V1/V2/center/width are mL; dividing by mass gives mL/g, not dimensionless and not CV. Center is an elution-window position proxy, not peak apex; width is an elution-window/dispersion proxy, not measured peak variance.
