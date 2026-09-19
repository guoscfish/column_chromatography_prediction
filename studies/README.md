# Studies

The [current research status](../docs/NEXT_STAGE_DECISION.md) owns project decisions. Start with one topic; each study keeps its original protocol, result tables and scientific limitations.

| Topic | Current focus | Index |
| --- | --- | --- |
| Predictor | Qualified standalone QGeoGNN-V2 and its source checkpoint | [Predictor](predictor/README.md) |
| 4g active learning | Label efficiency; matched-budget batch/adaptivity control | [Active learning](active_learning/README.md) |
| Cross-column transfer | Matched low-label baselines and completed transfer controls | [Transfer](transfer/README.md) |

## Reading a result

Read the final report and decision first, then the protocol and aggregate tables. Per-fit directories, split manifests and hashes support audit; they are not the starting point.

Earlier study names and paths are retained because code, protocols and hashes refer to them. Topic indexes identify which results supersede earlier work. Historical `track_*` directories and [early experiments](../experiments/INDEX.md) are evidence stores.

Some completed one-off runners have been retired. Their results remain here; [the retirement registry](../docs/repository/RETIREMENTS.json) links each old script to its record and recovery commit. Historical commands refer to that original version of the code, not necessarily the current checkout.
