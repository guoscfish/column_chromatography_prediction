# E4-A2a bounded engineering smoke

The bounded command is:

`conda run -n fish python scripts/run_e4_a2a_engineering_smoke.py --protocol A --outer-seed 42 --engineering-smoke`

It accepts only Protocol A and outer seed 42, uses L0=30 (22 gradient-train +
8 fixed validation), and reuses the frozen E4 predictor/acquisition functions.
The exact partition gate covers seeds 42/525/1101. The completed smoke includes
three-member source initialization and parameter freeze audits, all five
Round-0 acquisition dry-runs, Random and Hybrid 30→40 retraining, strict Round-1
source reset, and exact resume identity. No test metrics are generated or used
for the gate, and formal training has not started.
