# I0 — QGeoGNN Predictor Semantic Audit

Status: `AUDIT_COMPLETE`. I0 is an engineering and scientific-mechanism audit, not a new predictor performance experiment. It does not authorize training, alter the legacy forward path, select architecture from test outcomes, or change any historical checkpoint or result.

The audit traces dataframe fields through `build_model_data`, `Data.edge_attr`, `GINNodeEmbedding`, and `BondFloatRBF`; empirically perturbs each constructed continuous edge feature; performs a real forward/backward parameter reachability audit; quantifies effective-input collisions in canonical 4g and no-threshold 8g data; and measures cross-target V1/V2 ordering diagnostics. Machine-readable facts are in the JSON and CSV files in this directory.

Confirmed implementation boundary: the clean legacy path constructs ten continuous edge features, while its five-name `BondFloatRBF` consumes only bond length and the first four eluent descriptors. Eluent HBA, eluent LogP, loading-solvent code, loading amount (`Density × V`), and loading-solvent volume are constructed but forward-unreachable. This is legacy implementation evidence, not a declaration that historical experiments are invalid.

The acquisition path separately uses a complete 9D condition matrix and concatenates it with a standardized 128D graph embedding. That creates a predictor/acquisition semantic mismatch; the 128:9 block-size imbalance is recorded as a potential Euclidean-distance dominance risk, not a confirmed performance cause.
