# Cross-column target data audit

No V1/V2 threshold is applied to any target column.

## 8g

Raw/valid rows: 574/574; compounds: 88; source-train overlap: 87/88; source-unseen rows: 7.

Invalid label rows: 0; repeated exact-condition groups: 7; exact source-condition overlap rows: 483.

Verified/raw use: 574 verified pairs, 0 raw fallbacks, 0/0 t1/t2 mismatches. Retained t1>t2 warnings: 2.

V1/V2 (mL): {'count': 574, 'mean': 17.390171312427412, 'std': 17.717642228742026, 'min': 1.7583333333333333, 'median': 10.804166666666667, 'max': 138.38333333333333}; {'count': 574, 'mean': 31.112601626016264, 'std': 29.18756324355625, 'min': 6.1, 'median': 21.924999999999997, 'max': 248.90833333333333}. Flow: {'count': 574, 'mean': 10.0, 'std': 0.0, 'min': 10.0, 'median': 10.0, 'max': 10.0}.

PE/EA: {'0/1': 90, '1/1': 90, '2/1': 88, '5/1': 86, '10/1': 79, '20/1': 76, '50/1': 65}. Loading solvent: {'DCM': 368, 'PE': 206}. Loading amount: {'count': 574, 'mean': 100.0, 'std': 0.0, 'min': 100.0, 'median': 100.0, 'max': 100.0}. Loading-solvent volume: {'count': 574, 'mean': 326.30662020905925, 'std': 99.49935791178379, 'min': 300.0, 'median': 300.0, 'max': 1000.0}.

Column specs: {'Silica-CS 4g+4g': 574}. Graphs reused/generated/failed: 87/1/0.

OOD status: `SOURCE_UNSEEN_MOLECULE_OOD_NOT_ESTIMABLE_WITH_CURRENT_DATA`.

## 25g

Raw/valid rows: 569/490; compounds: 78; source-train overlap: 77/78; source-unseen rows: 19.

Invalid label rows: 79; repeated exact-condition groups: 3; exact source-condition overlap rows: 0.

Verified/raw use: 563 verified pairs, 6 raw fallbacks, 0/0 t1/t2 mismatches. Retained t1>t2 warnings: 2.

V1/V2 (mL): {'count': 490, 'mean': 38.68992346938776, 'std': 35.101021417574366, 'min': 7.2875, 'median': 23.85, 'max': 240.5875}; {'count': 490, 'mean': 63.9477806122449, 'std': 55.680048958122036, 'min': 22.0375, 'median': 42.56875, 'max': 371.8125}. Flow: {'count': 490, 'mean': 15.0, 'std': 0.0, 'min': 15.0, 'median': 15.0, 'max': 15.0}.

PE/EA: {'0/1': 80, '1/1': 79, '2/1': 76, '10/1': 70, '20/1': 70, '5/1': 66, '50/1': 49}. Loading solvent: {'DCM': 395, 'PE': 95}. Loading amount: {'count': 490, 'mean': 118.87755102040816, 'std': 47.166819926360255, 'min': 50.0, 'median': 100.0, 'max': 300.0}. Loading-solvent volume: {'count': 490, 'mean': 324.2857142857143, 'std': 95.7266900190542, 'min': 300.0, 'median': 300.0, 'max': 900.0}.

Column specs: {'Silica-CS 25g': 490}. Graphs reused/generated/failed: 77/1/0.

OOD status: `SOURCE_UNSEEN_MOLECULE_OOD_NOT_ESTIMABLE_WITH_CURRENT_DATA`.

## 40g

Raw/valid rows: 531/529; compounds: 80; source-train overlap: 80/80; source-unseen rows: 0.

Invalid label rows: 2; repeated exact-condition groups: 7; exact source-condition overlap rows: 0.

Verified/raw use: 531 verified pairs, 0 raw fallbacks, 0/0 t1/t2 mismatches. Retained t1>t2 warnings: 2.

V1/V2 (mL): {'count': 529, 'mean': 76.2890359168242, 'std': 75.26089559394622, 'min': 21.975, 'median': 45.15, 'max': 529.7}; {'count': 529, 'mean': 112.28993383742912, 'std': 105.21691697602954, 'min': 34.375, 'median': 72.575, 'max': 745.025}. Flow: {'count': 529, 'mean': 30.0, 'std': 0.0, 'min': 30.0, 'median': 30.0, 'max': 30.0}.

PE/EA: {'0/1': 82, '1/1': 82, '2/1': 82, '5/1': 79, '10/1': 76, '20/1': 70, '50/1': 58}. Loading solvent: {'DCM': 426, 'PE': 103}. Loading amount: {'count': 529, 'mean': 100.0, 'std': 0.0, 'min': 100.0, 'median': 100.0, 'max': 100.0}. Loading-solvent volume: {'count': 529, 'mean': 331.0018903591682, 'std': 109.67901685757407, 'min': 200.0, 'median': 300.0, 'max': 900.0}.

Column specs: {'Silica-CS 40g': 529}. Graphs reused/generated/failed: 80/0/0.

OOD status: `SOURCE_UNSEEN_MOLECULE_OOD_NOT_ESTIMABLE_WITH_CURRENT_DATA`.

