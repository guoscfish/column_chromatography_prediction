# 稳定 API 与运行边界

```text
scripts/run_al.py → al/workflow
                       ├─ predictor → application/QGeoGNN.py
                       ├─ al/representations → al/acquisitions
                       ├─ al/agents / al/evaluation
                       └─ infra
```

| 包 | 唯一职责 |
| --- | --- |
| [predictor](../src/qgeognn_al/predictor/) | 模型、图/条件输入、预处理、训练函数与原始六输出推理 |
| [representations](../src/qgeognn_al/al/representations/) | 从允许的模型/输入计算latent128、q50 gradient CountSketch512、CW与fused640；绑定有序L/U和checkpoint |
| [acquisitions](../src/qgeognn_al/al/acquisitions/) | Random / LCMD / MaxDet / IVR / Hybrid纯数值选点，不读取真值 |
| [agents](../src/qgeognn_al/al/agents/) | 候选白名单、预算/ID合法化、已提供计划和数据包；无provider执行 |
| [workflow](../src/qgeognn_al/al/workflow/) | 唯一round状态机、持久freeze、有限标签能力、恢复 |
| [evaluation](../src/qgeognn_al/al/evaluation/) | 分目标误差、Combined NRMSE与AULC |
| [infra](../src/qgeognn_al/infra/) | 身份、hash、分区、只读资源、工件 |

## 当前可运行入口

```sh
python -m scripts.run_al --config configs/offline_synthetic.json --output /private/tmp/qgeognn_al_demo
```

CLI只接受 `schema_version=2 / execution_mode=offline_synthetic`；预算2→4→6。`SyntheticBackend.fit`只生成工程fixture receipt，表示为latent-kind合成向量。它不开放真实标签loader、训练开关、LLM provider或自动Agent计划。`QGeoGNNBackend`虽然有调用训练/推理的实现，但默认 `allow_training=False`，完整真实训练/选模循环未资格化。

## 组合表示与acquisition

实际科研先用授权模型/输入生成表示，冻结其ID/顺序/模型身份，再调用纯选点API。下面只是128D合成向量契约示例，不是新的latent质量实验：

```python
import numpy as np
from src.qgeognn_al.al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.al.acquisitions import select
from src.qgeognn_al.infra.contracts import BaselineProvenance

auth = Authorization(("L0",), ("U0", "U1", "U2"))
x_l = np.zeros((1, 128))
x_l[0, 0] = 0.5
x_u = np.zeros((3, 128))
x_u[:, 0] = [1., 2., 3.]
snapshot = FeatureSnapshot(
    authorization=auth, train_ids=auth.train_ids, candidate_ids=auth.candidate_ids,
    train_features=x_l, candidate_features=x_u,
    config=RepresentationConfig("latent", 128),
    predictor=BaselineProvenance("0"*40, "1"*64, "2"*64, "3"*64, "4"*64),
)
for method in ("random", "lcmd", "maxdet", "ivr"):
    chosen = select(snapshot, acquisition=method, budget=2, seed=1234)
    print(method, chosen.selected_ids)
```

真实表示的提取函数在对应模块中；`FeatureSnapshot`验证L/U互斥、有序ID、维数、有限值和不可变hash。Hybrid还必须显式传入同seed/同L的K3预测、候选ID与target scales，缺失失败关闭；它不是上例中可伪造的无条件选项。方法的历史来源与状态见 [method_registry](../configs/method_registry.json)。

| 科学合同 | 保持的语义 |
| --- | --- |
| 时间状态 | 每轮由当前checkpoint重算特征；中心和fused块RMS用当前L。L0 target sigma、CW变换及CountSketch映射固定。 |
| IVR | 固定原L0+U0 reference成员及顺序，每轮按ID对齐当前特征；固定reference不等于冻结predictor。 |
| Random | 有效acquisition seed显式传入；逐轮choice与原U0一次permutation的fixed trajectory不同，round从0开始。 |
| Hybrid | top ceil(25% U) ensemble disagreement后latent k-center；真实K3供应方还须保证成员checkpoint/初始化来源。 |
| LLM协议 | Free/CW/NoCW/V3/V3.2是历史冻结协议，新通用计划合同不证明旧Prompt、重试或随机输出一致。 |

## Freeze与恢复

`RoundWorkflow`按当前L拟合→validate→feedback→表示/选样→排他持久化freeze→受限reveal→提交L/U转移→下一轮拟合；达到预算后仍验证最终模型。schema v2绑定seed、数据/划分、初始化、协议、拟合和有序身份。

恢复从L0与逐轮freeze/reveal receipt重建状态；重复/越权ID、跨seed复用、标签/配置漂移及重算hash后注入标签均拒绝。started存在而receipt缺失的拟合或揭示不自动重复；receipt已落盘可安全恢复。工件使用exclusive create、fsync与atomic replace；hash是完整性绑定，不是签名认证。

[资源登记](../configs/protected_artifacts.json) 和 [数据消费](../configs/data_consumption_register.json) 控制数据角色；传入文件路径不构成标签读取授权。历史源码与完整研究按 [archive配置](../configs/archive_manifest.json) 恢复，不能直接用新树替换旧实验完整源码hash universe。当前验收见 [FRAMEWORK_V2_ACCEPTANCE](FRAMEWORK_V2_ACCEPTANCE.md)。
