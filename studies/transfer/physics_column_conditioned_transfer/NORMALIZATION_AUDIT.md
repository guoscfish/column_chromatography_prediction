# Normalization audit

PHYSICAL_NORMALIZATION_PARTIALLY_SUPPORTED

Training-only cross-column CV of means:

| target | normalization | cross_column_mean_cv |
| --- | --- | --- |
| V1 | raw | 0.8463479679011675 |
| V1 | mass_normalized | 0.14873250719093722 |
| V2 | raw | 0.7509059205506977 |
| V2 | mass_normalized | 0.26454992928318694 |
| center | raw | 0.7868160293309551 |
| center | mass_normalized | 0.2202068174418959 |
| width | raw | 0.6061165240552783 |
| width | mass_normalized | 0.4269906169948595 |

Compound/EA/flow matched comparisons:

| column | target | normalization | matching | groups | mean_absolute_difference | median_relative_discrepancy |
| --- | --- | --- | --- | --- | --- | --- |
| 8g | V1 | raw | canonical_smiles | 13 | 10.200255030487227 | 0.963559545814629 |
| 8g | V1 | raw | canonical_smiles+EA_bin | 35 | 8.92112564419707 | 0.8974675033617211 |
| 8g | V1 | raw | canonical_smiles+Flow mL/min | 13 | 10.009436094950798 | 0.921775629610786 |
| 8g | V1 | mass_normalized | canonical_smiles | 13 | 0.5390103298404381 | 0.17489722507708116 |
| 8g | V1 | mass_normalized | canonical_smiles+EA_bin | 35 | 0.5735187031883461 | 0.18099547511312214 |
| 8g | V1 | mass_normalized | canonical_smiles+Flow mL/min | 13 | 0.49675117162249516 | 0.17489722507708116 |
| 25g | V1 | raw | canonical_smiles | 15 | 32.74123452134413 | 4.387740844080234 |
| 25g | V1 | raw | canonical_smiles+EA_bin | 39 | 30.38687922417089 | 3.498573466476463 |
| 25g | V1 | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 25g | V1 | mass_normalized | canonical_smiles | 15 | 0.40608163361298205 | 0.15519777931991677 |
| 25g | V1 | mass_normalized | canonical_smiles+EA_bin | 39 | 0.5443815116210949 | 0.32872643324355516 |
| 25g | V1 | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | V1 | raw | canonical_smiles | 13 | 70.12883965900801 | 8.335444805599957 |
| 40g | V1 | raw | canonical_smiles+EA_bin | 37 | 59.541241805929296 | 6.620267260579065 |
| 40g | V1 | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | V1 | mass_normalized | canonical_smiles | 13 | 0.41635859598562763 | 0.18520997208830237 |
| 40g | V1 | mass_normalized | canonical_smiles+EA_bin | 37 | 0.5754078561109811 | 0.2639384615384614 |
| 40g | V1 | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 8g | V2 | raw | canonical_smiles | 13 | 15.585993505083094 | 0.7768706290422552 |
| 8g | V2 | raw | canonical_smiles+EA_bin | 35 | 13.835970323988183 | 0.6953873914514491 |
| 8g | V2 | raw | canonical_smiles+Flow mL/min | 13 | 15.32362133849266 | 0.731222030981067 |
| 8g | V2 | mass_normalized | canonical_smiles | 13 | 0.8915429443116654 | 0.16744386927973656 |
| 8g | V2 | mass_normalized | canonical_smiles+EA_bin | 35 | 1.025382419002955 | 0.19902569340771586 |
| 8g | V2 | mass_normalized | canonical_smiles+Flow mL/min | 13 | 0.9800956955966147 | 0.1908890242806204 |
| 25g | V2 | raw | canonical_smiles | 15 | 50.31981713325947 | 3.6448882278314887 |
| 25g | V2 | raw | canonical_smiles+EA_bin | 39 | 47.03761941376524 | 2.5531324345757334 |
| 25g | V2 | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 25g | V2 | mass_normalized | canonical_smiles | 15 | 1.2433468277962434 | 0.25681788354696183 |
| 25g | V2 | mass_normalized | canonical_smiles+EA_bin | 39 | 1.356554156530198 | 0.4333415601921287 |
| 25g | V2 | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | V2 | raw | canonical_smiles | 13 | 102.03276152690046 | 5.122412871597339 |
| 40g | V2 | raw | canonical_smiles+EA_bin | 37 | 90.60690904378403 | 5.054744525547445 |
| 40g | V2 | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | V2 | mass_normalized | canonical_smiles | 13 | 1.7211733621698788 | 0.38775871284026614 |
| 40g | V2 | mass_normalized | canonical_smiles+EA_bin | 37 | 1.6181601264413767 | 0.4156539828378262 |
| 40g | V2 | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 8g | center | raw | canonical_smiles | 13 | 12.893124267785165 | 0.8280666734084603 |
| 8g | center | raw | canonical_smiles+EA_bin | 35 | 11.24234758726723 | 0.7851102385986104 |
| 8g | center | raw | canonical_smiles+Flow mL/min | 13 | 12.66652871672173 | 0.8267744256172413 |
| 8g | center | mass_normalized | canonical_smiles | 13 | 0.6344165943410093 | 0.1816395590001092 |
| 8g | center | mass_normalized | canonical_smiles+EA_bin | 35 | 0.711650837670927 | 0.19444272669181462 |
| 8g | center | mass_normalized | canonical_smiles+Flow mL/min | 13 | 0.6657555459416673 | 0.1816395590001092 |
| 25g | center | raw | canonical_smiles | 15 | 41.530525827301794 | 3.8926823570635984 |
| 25g | center | raw | canonical_smiles+EA_bin | 39 | 38.71224931896806 | 2.9846625766871173 |
| 25g | center | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 25g | center | mass_normalized | canonical_smiles | 15 | 0.7484659863563683 | 0.2171708228698241 |
| 25g | center | mass_normalized | canonical_smiles+EA_bin | 39 | 0.9122577201155325 | 0.3943481228668942 |
| 25g | center | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | center | raw | canonical_smiles | 13 | 86.08080059295425 | 6.108932676518884 |
| 40g | center | raw | canonical_smiles+EA_bin | 37 | 75.07407542485667 | 5.832256251756111 |
| 40g | center | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | center | mass_normalized | canonical_smiles | 13 | 0.9806260307907462 | 0.3071785076098343 |
| 40g | center | mass_normalized | canonical_smiles+EA_bin | 37 | 0.9750246873293749 | 0.33619501854795975 |
| 40g | center | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 8g | width | raw | canonical_smiles | 13 | 5.451214665072057 | 0.5484575551574349 |
| 8g | width | raw | canonical_smiles+EA_bin | 35 | 5.806315768226484 | 0.5735190231103038 |
| 8g | width | raw | canonical_smiles+Flow mL/min | 13 | 5.379661434018051 | 0.5484575551574349 |
| 8g | width | mass_normalized | canonical_smiles | 13 | 0.7562201799157584 | 0.33125630312635074 |
| 8g | width | mass_normalized | canonical_smiles+EA_bin | 35 | 0.84954709724799 | 0.311436024162548 |
| 8g | width | mass_normalized | canonical_smiles+Flow mL/min | 13 | 0.767072393326989 | 0.35205566097406693 |
| 25g | width | raw | canonical_smiles | 15 | 17.578582611915348 | 2.3051371982669697 |
| 25g | width | raw | canonical_smiles+EA_bin | 39 | 16.650740189594355 | 1.914740626605033 |
| 25g | width | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 25g | width | mass_normalized | canonical_smiles | 15 | 0.9973205212635884 | 0.47117804827728493 |
| 25g | width | mass_normalized | canonical_smiles+EA_bin | 39 | 1.0205423030287615 | 0.5336414997431947 |
| 25g | width | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | width | raw | canonical_smiles | 13 | 31.903921867892464 | 2.8871892925430207 |
| 40g | width | raw | canonical_smiles+EA_bin | 37 | 31.065667237854736 | 2.7351080078551164 |
| 40g | width | raw | canonical_smiles+Flow mL/min | 0 | nan | nan |
| 40g | width | mass_normalized | canonical_smiles | 13 | 1.5419058654444684 | 0.6112810707456979 |
| 40g | width | mass_normalized | canonical_smiles+EA_bin | 37 | 1.3498913737194986 | 0.6264891992144883 |
| 40g | width | mass_normalized | canonical_smiles+Flow mL/min | 0 | nan | nan |

Within-column CV is algebraically unchanged by division by a constant mass. Cross-column location discrepancy, not within-column CV, is the relevant test. Matching is descriptive and composition/flow/geometry remain confounded. Partial support permits one fixed mass-normalized arm; it does not validate void-volume proportionality. Full-data outcome census is post-freeze only.
