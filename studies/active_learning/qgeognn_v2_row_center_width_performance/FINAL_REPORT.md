# Center/Width Gradient-LCMD performance

Decision: `CENTER_WIDTH_NO_GAIN`.

| Seed | L0 | Raw B32 | CW B32 | CW - Raw |
| --- | ---: | ---: | ---: | ---: |
| 73 | 0.906007199 | 0.906475434 | 0.880519431 | -0.025956003 |
| 311 | 0.837025901 | 0.776701062 | 0.831850932 | +0.055149870 |
| 1297 | 0.708525189 | 0.599436736 | 0.694614892 | +0.095178156 |
| 4093 | 0.653848884 | 0.566153981 | 0.540125997 | -0.026027983 |
| 8191 | 0.774394930 | 0.695035176 | 0.696938421 | +0.001903245 |

CW wins 2/5. Mean paired improvement (Raw-CW): -0.020049457; median: -0.001903245; paired delta sample std: 0.053491303.

| Method | NRMSE mean | median | sample std | V1 mean RMSE | V2 mean RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CW | 0.728809935 | 0.696938421 | 0.133622548 | 5.715026 | 10.469717 |
| L0 | 0.775960421 | 0.774394930 | 0.100127851 | 6.104230 | 11.184222 |
| Raw | 0.708760478 | 0.695035176 | 0.138007069 | 5.534392 | 10.263536 |

Endpoint mean RMSE changes CW relative to Raw: V1 +3.26%, V2 +2.01%.
The frozen endpoint guard allows at most 2% deterioration on either endpoint.

Both endpoints improve for seeds 73 and 4093 and deteriorate for seeds 311 and
1297. Seed 8191 has a small trade-off: V1 RMSE rises from 5.288676 to 5.328252,
while V2 RMSE falls from 10.804566 to 10.783674. Across seeds, both endpoint
means worsen, so this is not an average benefit masked by an endpoint trade-off.

Relative to L0, CW reduces mean combined NRMSE by 0.047150486, versus
0.067199943 for Raw. CW improves over L0 in all five seeds, but the matched
comparison provides no advantage over Raw at the same 365-label budget.
These are L0 error reductions, not fractions of a full-data gap closed.

Full-data gap_closed: NOT_COMPARABLE. Existing final qualification uses seeds 42/525/1101 and full-train normalization, not these L0-normalized development splits. No new reference was trained.

All five Raw and L0 metric vectors reproduce the historical archive within 1e-12. Only five CW scratch fits were run. Acquisition IDs and all predictions were frozen before test truth. All results are exploratory development evidence, not independent confirmation or a multi-round learning curve. The active budget is 365/3330 (10.96%); including shared validation it is 781/4163 (18.76%).

The 30 scoped tests passed. Five development preflights additionally verified
split hashes, preprocessing/checkpoint provenance, deterministic B32 selection,
exact historical Raw selection, and exact innovation CW selection. No predictor
or historical acquisition implementation bug was found or changed.

The five CW fits used 1,134.997 seconds of measured training/prediction time in
total (18.92 minutes); this excludes preflight, tests and reporting and is not a
controlled compute-speed comparison. See execution_audit.csv for initialization,
checkpoint, prediction hashes, epochs and per-seed timing. Runtime checkpoints
remain in this worktree's gitignored runtime directory; committed metrics and
audit receipts remain available on the new branch.

## NEXT_STEP_RECOMMENDATION

This does not establish a reliable improvement in label efficiency from CW output-aware gradient geometry. Do not add CW variants or advance CW plus Target-IVR on this evidence. Reassess target-aware acquisition using the existing Raw gradient representation in a separately preregistered study.
