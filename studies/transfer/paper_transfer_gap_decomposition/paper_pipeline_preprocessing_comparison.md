# Paper pipeline preprocessing comparison

| item | paper/supplement evidence | released code | prior reconstruction | aligned diagnostic |
| --- | --- | --- | --- | --- |
| target normalization | fit scope not identified | whole reader-compatible target before split/filter | frozen 4g source scaler | released target-full semantics |
| transfer scope | pretrained transfer described; exact module freeze map unavailable | transfer branch loads source then optimizes `model.parameters()` | 242,216-parameter last-layer/adapter/head scope | all 458,952 parameters of qualified model |
| output head | quantile outputs; positivity parameterization unspecified | `Linear(128,6) + ReLU`, eval clamp | monotonic quantile head | qualified six-output linear+ReLU head |
| endpoint loss | quantile objective | `loss_V1 + 0.5*loss_V2` | `loss_V1 + loss_V2` | released 1:0.5 weighting |
| learning rate | Adam/transfer details incomplete | Adam 1e-4, weight decay 1e-5 | Adam 1e-4, weight decay 1e-5 | same as release |
| scheduler | StepLR described | StepLR(50,0.5) instantiated but never stepped | no StepLR | instantiated and deliberately not stepped |
| duration | source training reports 1500 epochs; target-transfer duration not recovered | 500 target epochs | cap 500 with patience 100 | exactly 500, validation-only checkpoint |
| batch size | 2048 | 2048 | 2048 | 2048 |
| column fields | transfer narrative refers to column specification | global `Use_column_info=False` | appended adapter inputs | no column fields |

The public release does not contain enough state to reconstruct the paper's exact source checkpoint, data snapshot, split map, or complete author training procedure. Therefore this study is explicitly not an exact reproduction.
