# Clean-QGeoGNN design

Model variant: `qgeognn_clean_fusion_v1`.

Status: `IMPLEMENTATION_PREFLIGHT_COMPLETE / FORMAL_PERFORMANCE_UNQUALIFIED`. It is distinct from `qgeognn_condition_complete_v2`, which remains a controlled historical ablation.

## Architecture

The molecular branch retains the useful QGeoGNN idea: atom/bond topology, bond lengths, bond-angle geometry, and molecule-derived descriptors are processed by a geometry-aware graph backbone. It produces a 128D pooled molecular representation, followed by `Linear(128, 64)` and `LayerNorm` to produce `z_mol`.

The condition branch consumes the typed sample-level fields `ea_fraction`, `loading_solvent`, `loading_mass_mg`, and `loading_solvent_volume_ul`. The solvent vocabulary is `PE/EA/DCM` with embedding width 4. Three normalized continuous values plus the embedding pass through `Linear(7, 32)`, GELU, `Linear(32, 64)`, and `LayerNorm` to produce `z_cond`.

`concat(z_mol, z_cond)` is a 128D fused representation consumed by a monotonic six-output quantile head for V1 and V2. The head enforces within-target `q10 <= q50 <= q90` but does not enforce `V1 <= V2`.

Conditions are graph/sample-level causes, not chemical properties of individual bonds. They therefore enter once per sample and are not repeated over every bond only to be pooled back to graph level.

The 64/64 projections control representation width, numerical scale, and fusion capacity. Equal latent width does not imply equal scientific importance. Future benchmark diagnostics, not dimensionality alone, must assess modality contribution.

## Fixed engineering defaults

- graph hidden width: 128
- molecular projection width: 64
- loading-solvent embedding width: 4
- condition raw continuous width: 3
- condition hidden width: 32
- condition output width: 64
- activation: GELU
- graph pooling: sum
- configured target-loss weights: V1 = 1.0, V2 = 1.0

These are fixed implementation defaults, not claims of optimality. No width, activation, learning-rate, optimizer, dropout, or loss sweep is authorized during preflight.

## Reachability and diagnostics contract

The clean backbone must not register known-dead `NN_descriptor`, unused outer batch normalizations, or redundant final-layer geometry modules. A real multi-fixture forward/backward must yield zero forward-unreachable trainable parameters.

Utilities must expose molecular/condition latent L2 norms, molecular-projection and condition-encoder gradient norms, V1/V2 gradient contribution, condition permutation evaluation, and molecule-only plus condition-only/disabled modes. Fixture magnitudes are diagnostics only and must not be used to tune the architecture.

Real observations include some `V1 > V2` rows in both 4g and 8g data. Until label quality is audited, a hard cross-target order would encode an unsupported assumption.

## Checkpoint contract

A clean checkpoint must include model variant, full input schema and hash, condition schema hash, normalization statistics and fit-ID hash, molecular/condition/fusion/quantile-head configs, nominal/requires-grad/gradient-bearing parameter counts, git commit SHA, source split hash, training config hash, and state dict. Loading must reject schema or configuration mismatch rather than infer semantics from an unannotated state dict.
