# Corrected FULL128 re-audit

The model uses the complete frozen 128D QGeoGNN representation without PCA. Its corrected C/W basis restores source scales before endpoint assembly.

For every inner fold, forcing latent coefficients to zero reproduced corrected endpoint HIER below 1e-10 before latent fitting continued.

Across all outer contexts/seeds, mean relative NRMSE gain is 1.096% with 13/20 wins.

- 25g compound: mean gain -4.391%, wins 0/5
- 25g row: mean gain 0.101%, wins 3/5
- 40g compound: mean gain 2.188%, wins 5/5
- 40g row: mean gain 6.486%, wins 5/5
