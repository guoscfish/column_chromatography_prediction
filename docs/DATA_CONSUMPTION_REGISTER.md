# Dataset and outcome consumption register

The machine-readable authority is `docs/data_consumption_register.json`. This page explains how to interpret it; it is not a claim that every function retained under `application/` was executed.

| Dataset | Training use found | Validation/model selection found | Held-out test truth read | Other outcome inspection | Developmental reuse | Pristine confirmation |
|---|---:|---:|---:|---:|---:|---:|
| 4g | yes | yes | yes | yes | yes, disclose prior use | no |
| 8g | yes | yes | yes | yes | yes, disclose prior use | no |
| 25g | no current-study artifact found | no | no | repository data audit | yes | no |
| 40g | no current-study artifact found | no | no | repository data audit | yes | no |
| C18 | no current-study artifact found | no | no | repository data audit | yes | no |
| CN | no current-study artifact found | no | no | repository data audit | yes | no |
| NH2 | no current-study artifact found | no | no | repository data audit | yes | no |
| DCM | no current-study artifact found | no | no | repository data audit | yes | no |

`test_truth_read=false` for the six audit-only datasets means that no formal held-out test evaluation was found. It does not mean their outcomes are unseen: `experiments/data_audit/` read label values to calculate missingness, ordering, thresholds, and other summaries. They can support future developmental work, but none is a pristine confirmatory dataset in its current committed form.

The 8g outcomes have been used across G0, E1/E4, S1, T1, T1b-1, and I0. Further selection on the same rows is developmental evidence and must not be described as independent external confirmation. S1's reserved role was unconsumed by S1, but it is not globally pristine after other 8g studies.

## Split terminology

The current target compound split is a **target-label compound holdout**: a held-out molecule does not occur in target adaptation labels for that split. It is not source-plus-target molecule OOD. I0 found that 87 of the 88 unique 8g compounds already occur in 4g, so the pretrained source predictor has generally seen those molecular identities.

A future novel-molecule protocol would need a source-aware holdout: remove selected compounds from 4g source training and 8g adaptation, then evaluate only their target rows. A separately preregistered Bemis-Murcko scaffold holdout could test a stronger OOD boundary. Neither protocol is authorized here.
