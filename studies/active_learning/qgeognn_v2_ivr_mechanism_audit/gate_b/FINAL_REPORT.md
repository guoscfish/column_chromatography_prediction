# Gate B: batch optimizer and multi-output information

Completed: selection-only audits on all 25 fixed states. No one-step
retraining or predictive-error evaluation was performed.

## Fixed mathematics

`J_x = [S grad(V1/s1); S grad(V2/s2)] / a`, using the SAME S for both
endpoints and common `a^2 = mean_R ||J_raw||_F^2`.
`C = (I + sum_L J_i.T J_i)^(-1)` and `M = mean_R J_z.T J_z`.
`risk = tr(M C)`; unit noise and endpoint weights remain frozen.
`delta(x) = tr((I + J_x C J_x.T)^(-1) J_x C M C J_x.T)`.
`C_new = C - C J_x.T (I + J_x C J_x.T)^(-1) J_x C`.

The BAIT-style optimizer selects 64 then removes 32 pending candidates.
A removal uses `C_new = C + C J_x.T (I - J_x C J_x.T)^(-1) J_x C`.
At each step it minimizes the risk increase; labeled rows are never removed.
There is no theorem here that forward/backward dominates forward greedy.

## Results

Positive forward/backward gains: 7/25.
Median forward/backward relative batch gain: 0.000000%.
Maximum relative batch gain: 0.130935%.
Minimum relative batch gain: -0.072376%.
Multi-output IVR beats LCMD on its OWN surrogate: 25/25.

Winning the objective a selector explicitly optimizes is a mechanism
sanity check, not evidence that it improves test NRMSE.
Scalar and block risk magnitudes cannot be compared directly.

## Training decision

Frozen decision: `STOP_BEFORE_GATE_C`; candidate: `None`.
No complete sequential run or COMPOUND experiment was launched here.
Do not interpret a stability-gate failure as a negative predictive trial.
See ../decision.json for each fixed criterion and ../results for paired data.
