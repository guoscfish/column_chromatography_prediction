# Implementation audit: controlled shared-backbone PCGrad

Audited before implementation on 2026-09-14 against repository commit
`584eeb5`. The repository-wide search found discussion of PCGrad as a future
control, but no executed or implemented experiment scientifically equivalent to
the intervention defined here. The study may therefore proceed without
duplicating historical evidence.

## Frozen reference and sole candidate

`A2_ADAM_FROZEN` is the completed `MULTITASK_COLUMN_FILM` result in
`studies/transfer/column_conditioned_multitask`. It is historical, immutable,
and will be reused rather than retrained. Its qualified source SHA256 is
`fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0` and
its outer split-manifest SHA256 is
`33e079a2b7250a82b040605bddb372646ab424684268b5b0c63df6366194a0a3`.

`A2_PCGRAD` is the only new candidate. It must retain the complete A2
architecture and training protocol: qualified 4g initialization, separate
copied 4g/25g/40g heads, 16-dimensional categorical task embedding,
zero-initialized FiLM after node blocks 3 and 4, existing condition completion,
sum pooling, six quantile outputs, raw quantile loss, equal-task batches, Adam,
the existing learning rates/weight decay, 500-epoch maximum, patience 80, and
the same joint inner selector. No A2-Adam fit may be repeated.

No historical transfer or calibration architecture may be renamed or rerun as
a new method. This excludes zero-shot/scaling/affine/Conditional-EA,
Center/Width variants, target-head and shallow/full fine-tuning, latent
Ridge/PLS/adapters, adaptive or condition-query readout, physical scaling, and
legacy geometry descriptors. The study adds no extra head, adapter, readout,
loss, task weighting, architecture size, 8g data, or post-result candidate.

## Mechanism authority

The completed A2 diagnostic measured gradients only on the shared late
molecular backbone. The exact authority is
`MultiTaskQGeoGNN.shared_gradient_parameters()`, whose registered prefixes are:

- `backbone.convs.3`
- `backbone.convs.4`
- `backbone.convs_bond_angle.3`
- `backbone.convs_bond_float.3`
- `backbone.convs_bond_embeding.3`
- `backbone.convs_angle_float.3`

The completed evidence found persistent negative task-gradient cosine,
especially 4g versus 40g (A2 mean `-0.1868`, median `-0.2170`, negative on
76.13% of diagnostic observations). Median task-gradient magnitude ratios were
1.32-2.38, all below the preregistered severe-imbalance threshold of 10.
PCGrad is therefore motivated specifically by observed conflict on the shared
late backbone. GradNorm and other magnitude-balancing methods are outside the
current hypothesis.

PCGrad will replace gradients only on that measured parameter tuple. Heads,
task embedding, FiLM generators, condition-completion parameters, and every
other trainable parameter retain the ordinary gradient of
`mean(L_4g, L_25g, L_40g)`. Frozen early parameters and BatchNorm policy remain
unchanged. The task-order permutation is derived deterministically from outer
seed, inner fold, epoch, and optimization step; diagnostic calculations may not
mutate optimizer gradients.

## Data, interpretation, and exclusions

The new candidate must reuse the completed inner identities, global
canonical-SMILES fold grouping, target gradient-train populations, source
supervision, train-only metric scales, and outer-gradient-train 80th-percentile
tail thresholds exactly. Historical A2 comparisons are prohibited unless all
identity, scale, source, architecture, loss, epoch-selection, and protocol
checks pass. Outer validation/test truth stays unavailable unless both the
COMPOUND and ROW inner gates authorize a global prediction freeze.

The column embedding remains categorical task identity only. No column
diameter, length, density, bed volume, mass scaling, flow-derived quantity, or
other physical descriptor is introduced, and no physical meaning is assigned
to the embedding. Target-compound holdout is not source-unseen molecular OOD;
the inherited per-column COMPOUND outer overlap limitation remains if outer
confirmation becomes eligible.

GradNorm, CAGrad, NashMTL, uncertainty weighting, MMoE, adapters, additional
FiLM layers, embedding sweeps, attention/readout/head changes, loss/LR/batch
searches, 8g, calibration, physical scaling, and Active Learning are explicitly
excluded. A failed gate terminates the study without another method.
