# Row Column-Selective Latent Audit

This study is frozen to the existing filtered FULL-data ROW protocol: the qualified source checkpoint, filtered population, row split identities, five outer seeds, and target gradient-train/validation/test roles are reused byte-for-byte. Only 25g and 40g are included.

`BALANCED_JOINT_FULL128` keeps the corrected FULL128 basis and latent grid, but weights each training row by the inverse squared standard deviation fitted within that training fold for its own column and endpoint. The four column-by-endpoint losses are therefore equally represented. No validation or test truth is used for normalization.

`A2_HIER_SHRINK_ORIGINAL` and `A3_HIER_SHRINK_BALANCED` use one alpha per column, shared by V1/V2, from `{0, .25, .5, .75, 1}`. Every OOF prediction is produced by a model fitted without that OOF row; lambda selection is performed inside the corresponding OOF training fold. Alpha selection minimizes OOF RMSE subject to a 2% MAE guard against the OOF HIER prediction. Test predictions are frozen before test truth is read.

The retention diagnostic is descriptive on the frozen filtered test population. It reports retention quintiles, top-10% and top-20% retention SSE shares, and error concentration by absolute-error quantile. It does not change filtering or model selection.

The study deliberately does not implement compound split, Active Learning, measured-4g anchors, new backbone layers, new physical metadata, or a new hyperparameter grid.
