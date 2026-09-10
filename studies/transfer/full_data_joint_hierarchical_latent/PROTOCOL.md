# Protocol

Populations and all outer identities are copied exactly from `full_data_baseline_finalization`. Only outer gradient-train labels enter fitting, normalization, PCA, PLS, or inner-CV selection. Compound inner folds group canonical SMILES. HIER and latent terms are fitted simultaneously in the endpoint-space objective. Predictions are frozen before test truth is read.
