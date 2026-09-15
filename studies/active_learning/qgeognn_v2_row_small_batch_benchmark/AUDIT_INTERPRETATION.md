# Retrospective Label-Free Audit

Population: the existing B=333 study's five development outer-training sets.
Only canonical X, label-scrubbed graphs, frozen L0 checkpoints, gradient sketches
and selected ID tables were read. No formal test metric or test-target access
was needed. No acquisition algorithm or hyperparameter was changed.

## Exact-X and Sketch Duplicates

| Seed | Outer rows | Unique model inputs | Duplicate groups | Excess duplicate rows | Mixed-X sketch groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| 73 | 3330 | 3112 | 209 | 218 | 0 |
| 311 | 3330 | 3113 | 204 | 217 | 0 |
| 1297 | 3330 | 3103 | 210 | 227 | 0 |
| 4093 | 3330 | 3104 | 209 | 226 | 0 |
| 8191 | 3330 | 3110 | 206 | 220 | 0 |

All exact gradient duplicates are explained by identical actual model-input
tensors. Their group counts and excess-row counts match exactly. There is no
observed different-X/same-sketch group requiring a CountSketch-collision or
local-symmetry/saturation explanation. This is evidence about exact collapse in
these five finite datasets, not proof that approximate collisions cannot occur.

Across the five LCMD batches, redundancy relative to L0 or an earlier selected
row is 0/333 in every seed. Random controls select 3-12 such redundant rows per
333 (0.90%-3.60%). Membership in any X-duplicate group is a different measure:
LCMD selects some members but does not acquire multiple already represented
copies. Avoiding redundant inputs is a plausible contributor, not by itself a
quantitative explanation of the previously reported performance gain.

The frozen B=333 LCMD algorithm was rerun on cached features for all five seeds.
All five ordered 333-ID selections match the existing artifacts exactly. Only
selection was replayed; no predictor was retrained and no formal metric read.

## Norm Mechanisms

Mean of the five within-U0 Spearman correlations (descriptive):

| Variable | Mean rho | Seed range |
| --- | ---: | --- |
| Predicted V2 q50 | 0.8931 | 0.8786 to 0.9024 |
| Predicted V1 q50 | 0.7668 | 0.6925 to 0.8467 |
| Nearest-L0 gradient distance | 0.6397 | 0.5847 to 0.6759 |
| Eluent logP | 0.5203 | 0.2603 to 0.6988 |
| Atom count | 0.1937 | 0.0552 to 0.2707 |
| Bond count | 0.2085 | 0.0594 to 0.3254 |
| MolWt | -0.1480 | -0.2346 to -0.0604 |
| Nearest-L0 condition distance | 0.0119 | -0.0456 to 0.0720 |

Across selected rows (five LCMD arms versus 25 Random controls), means are:

| Covariate | LCMD | Random |
| --- | ---: | ---: |
| Sketch gradient norm | 206.5137 | 119.9898 |
| Atom count | 12.0553 | 11.4489 |
| MolWt | 178.6460 | 182.4353 |
| Predicted V1 q50 | 11.9644 | 8.2478 |
| Predicted V2 q50 | 26.2335 | 17.3757 |
| Nearest-L0 condition distance | 0.2302 | 0.1986 |
| Nearest-L0 gradient distance | 125.9293 | 60.4803 |

The strongest label-free pattern is joint model sensitivity/gradient-space
novelty and **predicted retention-tail targeting**, with an eluent-condition
association. A simple "larger molecular weight" explanation is not supported:
MolWt correlates negatively and its selected mean is lower for LCMD. Atom/bond
count effects are modest. Direct condition novelty has near-zero norm
correlation even though its selected mean is somewhat higher for LCMD.

These are associations from model predictions and the sketch norm. They do not
establish true high retention, baseline error, causal mechanism, or small-batch
superiority. True-label/error diagnostics remain deferred. Correlated eluent
descriptor columns are not independent mechanisms; constant hydrogen-donor
values have undefined rho and are explicitly flagged. No diagnostic here is a
selection threshold or tuning signal.

Source file hashes and the no-test-access statement are recorded in
`results/retrospective_audit_provenance.json`. Full group and correlation tables
are the three requested audit CSVs, with selection profiles in a fourth CSV.
