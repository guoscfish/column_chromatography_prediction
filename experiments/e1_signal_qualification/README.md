# E1：Acquisition Signal Qualification

## 状态与Gate决定

本阶段不运行主动学习闭环，也不修改Gate 0 Predictor。Primary error固定为训练尺度标准化的V1/V2绝对误差均值；raw Quantile Width与全局conformal inflation后的width只算同一种ranking。

- Ensemble进入E2主uncertainty：**True**（相对Quantile Width赢12/16个关键聚合比较）。
- Quantile Width qualified：**True**。
- Latent Distance支持Coverage主线：**True**。
- E2 uncertainty策略：**ensemble**。

12/16恰好达到预注册门槛，不解释为Ensemble普遍占优：row/full与row/common的8项比较全部由Ensemble胜出；compound/full中Quantile Width赢Spearman、AUROC与enrichment，Ensemble仅在AUSE上略优；compound/common中Ensemble赢AUROC、enrichment与AUSE，但Spearman较低。E2因此使用Ensemble作为唯一主uncertainty信号，同时保留Quantile Width为secondary/legacy诊断，不增加为第五个主策略。

Tail只作failure/mechanism slice，不凭小样本结果触发方法入选。完整逐样本结果在`uq_predictions.csv.gz`，每个run保留member q50、真实误差、三种signal、tail与compound identity；128维embedding及拼接conditions表示保存在各run的`reference_embeddings.npz`。

## 关键切片均值

```text
split_mode  slice          signal  spearman_mean  hard_error_auroc_mean  enrichment_mean  ause_mean
  compound common        ensemble       0.412299               0.883333         5.000000   0.034692
  compound common latent_distance       0.448181               0.794444         5.000000   0.039056
  compound common  quantile_width       0.483132               0.791111         2.777778   0.039034
  compound common          random      -0.073912               0.412222         0.000000   0.084243
  compound   full        ensemble       0.431889               0.825855         3.888889   0.037079
  compound   full latent_distance       0.501241               0.845085         6.111111   0.037898
  compound   full  quantile_width       0.530915               0.830128         5.555556   0.038078
  compound   full          random      -0.044244               0.513889         1.111111   0.110972
       row common        ensemble       0.509159               0.878889         5.555556   0.028600
       row common latent_distance       0.487264               0.884444         3.888889   0.027580
       row common  quantile_width       0.345568               0.726667         3.333333   0.058228
       row common          random       0.090112               0.454444         1.666667   0.097020
       row   full        ensemble       0.570699               0.904612         6.666667   0.032117
       row   full latent_distance       0.557530               0.948637         7.777778   0.027214
       row   full  quantile_width       0.376914               0.757862         6.111111   0.071010
       row   full          random       0.107988               0.550314         1.666667   0.133426
```

## Ensemble独立性

K=3成员使用相同4g训练split和相同8g target train/validation；seed42复用Gate 0 anchor，seed525/1101从不同4g随机初始化重新训练source后再迁移。只改变target seed在当前full-batch、无dropout路径下不会形成真正ensemble，因此没有采用该伪独立方案。

## 限制

- 每个row测试切片只有59个样本、compound测试切片只有58个样本；seed级置信区间很宽，E1只决定方法入口，不给出稳定效应量结论。
- tail每个run仅2--3个样本，compound/tail的Spearman不可定义。
- row seed1101的两个新增Ensemble成员分别在epoch 496/500和498/500取得validation最优，接近训练上限；这不触发事后延长epoch，但在E2中继续作为收敛审计项。

## 图表

- `plots/signal_error_rank_scatter.png`
- `plots/risk_coverage.png`
- `plots/hard_error_enrichment.png`
- `plots/signal_slice_heatmap.png`
