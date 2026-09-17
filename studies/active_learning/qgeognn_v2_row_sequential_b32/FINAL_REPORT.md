# Final Report — Sequential B=32 Active-Learning Efficiency

## Primary answers

Random, V2-Hybrid, and Gradient-LCMD mean normalized AULC values are
0.624260, 0.479354, and 0.479539, respectively.
The preregistered decision is **NO_CLEAR_SEQUENTIAL_AL_GAIN**.

T_R30 is 0.543301. Labels-to-target are Random
1005.00, Hybrid
504.47, and LCMD
472.33 active labels.

Hybrid saves 500.53 new experiments
(74.48%). LCMD saves
532.67 new experiments
(79.27%).

The secondary T_80 is 0.437402; labels-to-T_80 are Random
>1005 / censored, Hybrid
643.32, and LCMD
651.19.

Endpoint RMSE, MAE, and R2 are secondary predictor diagnostics in
`results/endpoint_learning_curves.csv`; they do not determine the active-learning winner.
