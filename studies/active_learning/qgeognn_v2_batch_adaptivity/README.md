# 研究记录：qgeognn_v2_batch_adaptivity

本页合并该研究原先分散的说明、方法、审计或资源说明；不新增训练、标签访问或研究结论。

以下按原文件分节合并，日期、方法与数值保留其历史语境；其中的后续建议不代表当前待办。当前进度以[研究状态](../../../docs/NEXT_STAGE_DECISION.md)为准。

<a id="记录-1"></a>

## 记录 1：`README.md`

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

<a id="记录-2"></a>

## 记录 2：`EXECUTION_NOTES.md`

# Execution notes

The first invocation started 2026-09-18 after acquisition freeze at 10:45 UTC.
It exited before all trajectories completed. Six new fits (seeds 157 and 887,
rounds 2 through 4) were already complete and are resumed with artifact checks.
No new test evaluation had occurred. Partial round-5 attempts, where present,
are preserved in `runtime/seed_*/incomplete_attempts` and restarted from the
registered initialization.

Completed-fit timing excludes work lost in incomplete attempts, which did not
write a final fit audit. That overhead is unmeasured, not zero. Calendar elapsed
time in `execution_time.json` covers the successful invocation only and must not
be presented as the full elapsed duration including the first invocation.
