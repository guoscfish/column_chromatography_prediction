# Resource and recovery audit

The actual preflight verified all five local L333 checkpoint, prediction, history and fit-audit artifacts against frozen hashes. It called the original `verify_freeze()` successfully. The preflight reads no targets. The original sealed commit is `b1f54b5132f671ab1c6068127fe8cc313df5b3ac`; current task HEAD is `f27c1bbca2296ea6ac38b713cc9ec1eec8153be4`. This is not a source drift: the original freeze stores the commit at its creation and verifies its listed bytes/dependencies, and all those checks passed.

`resource_preflight.json` records 24 GiB RAM, 10 logical CPUs, arm64, ~199 GiB free disk, five verified initial states, the process inventory and lock probes. Existing old study monitor processes use little CPU and have been left alone. No QGeoGNN training was active at preflight; no original/new formal lock was held. The historical same-protocol LCMD fits have 105 downstream audits: 56,371.1262 training seconds, median 474.624 s, maximum 1434.271 s. A descriptive two-arm extrapolation is 112,742.2523 s / 31.3173 h. This is an estimate, not a guarantee for different trajectories or current system load. There is no demonstrated checkpoint/hardware blocker to starting the full experiment.

| Finding | Source and actual setting | Intended behavior / impact | Verification / evidence |
| --- | --- | --- | --- |
| Original all-seed gate correctly prevents formal AL | original `epig_formal_runner.py:58–79`; status must be `QUALIFIED` and all five seed families present | Does not evaluate efficacy when two seeds fail; old gate remains unchanged | Original freeze and decision hashes verified; source inspection, direct byte verification |
| Original fit cap counts attempts | original `epig_formal_runner.py:82–119`; `len(attempts) >= 210` | A single interruption can prevent 210 successes; fixed only in independent exploratory runner | v2 ledger separates target 210 successful fits from maximum 231 attempts and three attempts/key; unit test exhausts 231 failed attempts with zero successes |
| Original wall cap includes inactive periods | original `epig_formal_runner.py:101–106`; start+40h−current wall time | Pause/resume can consume all budget while no compute runs | New monotonic active-time tracking, 2-second durable heartbeat, conservative heartbeat tail after process death; fake-clock resume test proves idle gap excluded |
| Original training protocol is retained | `benchmark_protocol.py:25–43`; V2, Adam0.001, batch2048,1000 epochs,patience100, deterministic seeds | Preserves actual learning comparator | Original `SequentialSeedContext.training_config` and `fit_from_same_initialization` are used directly; new checks require458952 parameters, exact initialization and training/validation identities |
| Generic context defaults prediction X to test role | `sequential_runner.py:213–253`, especially225 | Generic reuse could accidentally include test inputs in prediction path | New runner passes legal current U explicitly; callback receives NaN-masked validation/test input rows and only current-L targets |
| Label access precedes a failed training attempt | original selection/reveal helper plus downstream fit | Must retain evidence even if a fit then fails | New independent immutable request/outcome files are written for every attempted reveal; aggregate audit persists before fitting; a denied access test verifies durable request and failure outcome |
| Cached fits require more than existence | original `runner.py:157–219` and new `fit_round` | Reusing wrong state/checkpoint would confound trajectories | New receipts bind ordered L/U, actual revealed truth hash, freeze, config, fixed scales and all four fit artifact hashes; repeat succeeds without another attempt; corrupted truth/checkpoint fails |
| A returned fit is not yet a success | new `fit_round` | Architecture/hash/metric failure must not count toward210 | All validations and immutable receipt writes are inside the ledger success operation; invalid-parameter-count test leaves zero successes |
| Numerical/protocol errors require diagnosis | new ledger `reserve` | Cannot consume retries automatically to hide implementation errors | Previously failed non-I/O/non-interruption errors block automatic retry; finite-budget safety test |
| Posterior cache must remain bound to selected batch | new `verify_auxiliary_receipt` | Resume must not silently replace the probability model/acquisition | Verifies model.pkl, fit_audit.json, acquisition.json and complete receipt before label access; deliberate corruption test |

The new resource envelope is 72 cumulative active QGeoGNN hours and 12 auxiliary hours (84 total), target210 successful downstream fits, maximum231 neural attempts, maximum300 auxiliary successes/330 attempts, and one worker/two threads. Offline C/D and reference diagnostics conservatively reserve65 auxiliary fit/attempt slots and2.5h. These are reservations, not claimed actual costs. The main callback therefore has235 successful/265 attempted auxiliary slots and9.5h. Actual offline cost is reported separately. Any individual operation is limited to4h. No efficacy-based stop is implemented.

Reproduce the read-only preflight with the existing fish Python:

```bash
/Users/fish/miniforge3/envs/fish/bin/python studies/active_learning/epig_prediction_oriented_v2_20261117/resource_preflight.py
```

Hardware/process inspection can be blocked by the filesystem sandbox; that failure is explicit in the output and is not evidence that training resources are absent. The preserved current preflight obtained a read-only approved hardware/process inventory.

The focused runner suite has **16 passing tests**, preserved in `tests.xml`. No new QGeoGNN fit, acquisition or U/test-label reveal was performed by this audit agent. Actual full-budget execution is coordinated by the root agent after independent QA and the new decision freeze.
