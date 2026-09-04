# Active-learning roadmap

Status: predictor regression unresolved; 4g in-domain active learning is paused with historical evidence retained; active transfer is `DEFERRED`.

## Known representation issues

Historical joint acquisition concatenates a standardized 128D graph embedding with a standardized 9D condition vector and applies Euclidean distance. This has three known risks:

1. after per-dimension scaling, a 128D molecular block can contribute much more expected squared distance than a 9D condition block;
2. six PE/EA chemical descriptors encode a binary mixture with approximately one intrinsic degree of freedom and may repeatedly weight the same composition axis;
3. the acquisition representation includes condition dimensions that the audited legacy predictor does not consume, creating a predictor/acquisition semantic mismatch.

These are identified risks, not post-hoc proof of why a historical acquisition did or did not work.

## Future direction

If active learning is reopened after predictor regression resolution and qualification, use a non-redundant typed condition representation and preregister block-aware distance. Candidate concepts include:

`d = lambda * d_molecule + (1 - lambda) * d_condition`

or a dimension-normalized block distance. No value of `lambda`, distance family, or condition metric is selected here. Historical E2/A1a acquisition artifacts and conclusions must remain unchanged.
