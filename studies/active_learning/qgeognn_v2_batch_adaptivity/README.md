# Matched-budget batch/adaptivity control

See `PROTOCOL.md` for the frozen 333+320 design and `seal.json` for source hashes.
Static LCMD, Adaptive B32x10 and nested Random share a 653-active-label endpoint.
OneShot-B320 shares the exact Static final model; its intermediate curve is not
invented. B64/B160 are not included in this first control.

Preparation and execution:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish pytest -q tests/active_learning_v2/test_efficiency_reporting.py tests/active_learning_v2/test_batch_adaptivity.py tests/active_learning_v2/test_sequential_b32.py --junitxml=/tmp/qgeognn_adaptivity_preflight.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_batch_adaptivity.py --prepare /tmp/qgeognn_adaptivity_preflight.xml
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_qgeognn_v2_batch_adaptivity.py --run
```

`--run` resumes only compatible artifacts, uses at most two seed workers, freezes
all five trajectories and then performs the authorized post-freeze evaluation.
`--execute-seed SEED` runs one trajectory without opening test truth. `--report`
requires all trajectories to be complete and verified. Runtime checkpoints are
local, gitignored artifacts; selections, seals, metrics and audits are retained.
