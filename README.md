# Intelligent Column Chromatography Prediction Model

## 当前研究主线

本项目当前研究问题是：在已有 4g 预训练 QGeoGNN 的前提下，通过主动学习选择最有价值的新实验标签并逐轮重新训练模型，是否能在有限标签预算下提高 V1/V2 预测能力；主实验进一步检验主动选择少量 8g 标签能否比 Random 更快接近全量 8g 迁移模型。

主指标不是 SQ，而是 V1/V2 的 normalized RMSE 学习曲线与 AULC；主结论口径是 `labels-to-90%-ceiling` 和相对 Random 的标签节省。SQ 只在最后作为色谱推荐的 downstream utility 验证。

严格执行顺序：

```text
Gate 0 Predictor qualification
  → E1 acquisition-signal qualification
  → E2 4g active-learning closed loop
  → E4 4g→8g active transfer (main experiment)
  → E5 downstream SQ utility
```

Gate 0 与 E1 已完成并冻结。E2 row 三seed正式pilot也已完成：平均normalized AULC为Hybrid 0.543、Coverage 0.562、Ensemble 0.627、Random 0.645；Coverage和Hybrid对Random均3/3 seeds胜出，Ensemble为2/3且paired CI跨0。当前阶段是row机制审计与compound seed42 preflight，完整compound pilot尚未启动；详细证据见 [E2 row](experiments/e2_4g_active_learning/README.md)。

## Citation
If you use this work in your research, please cite:
```bibtex
@misc{wu2024intelligentchemicalpurificationtechnique,
      title={Intelligent Chemical Purification Technique Based on Machine Learning}, 
      author={Wenchao Wu and Hao Xu and Dongxiao Zhang and Fanyang Mo},
      year={2024},
      eprint={2404.09114},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.09114}, 
}
```

## Environment Configuration

下列版本是原项目的 legacy/recommended 环境（Python 3.9）。本仓库新增实验层会在每个正式实验目录保存实际 `environment.json`；本次 G0-1/G0-2 使用 conda `fish`、Python 3.11.14、PyTorch 2.10.0、PyG 2.7.0，并在 CPU 上运行。

**Legacy Python Version**: 3.9

### Core Dependencies
| Package | Version | Installation Command |
|---------|---------|----------------------|
| RDKit   | 2023.9.2 | `conda install -c conda-forge rdkit` |
| PyTorch | 2.1.0   | `pip install torch==2.1.0` |
| Mordred | 1.2.0   | `pip install mordred==1.2.0` |
| pandas  | 2.1.4   | `pip install pandas==2.1.4` |

### Recommended Installation
```bash
# Create conda environment
conda create -n chromatography python=3.9
conda activate chromatography

# Install core packages
conda install -c conda-forge rdkit==2023.9.2
pip install torch==2.1.0 pandas==2.1.4 mordred==1.2.0
```

## Experiment workflow in this repository

The original model code is kept under `application/`. The reproducible experiment layer is separate:

- `EXPERIMENT_PLAN.md`: stage order, gates, frozen protocol and current progress.
- `experiments/METHOD_DECISION_REGISTER.md`: evidence, unresolved issues and decisions that must not be inferred from the paper.
- `experiments/e0_4g_baseline/`: frozen 4g baseline.
- `experiments/e0_8g_transfer/`: original single-seed 4g→8g transfer matrix.
- `experiments/e0_3b_controls/`: three-seed robustness and transfer-range controls.
- `experiments/e0_3c_loss_controls/`: paired loss-weight and target-scaling controls.
- `experiments/d04_conformer_selection/`: paired first-conformer versus lowest-energy control.
- `scripts/run_g0_1_quantile_monotonicity.py`: Gate 0-1 legacy independent quantiles versus structurally monotonic quantiles.
- `experiments/g0_1_quantile_monotonicity/`: finalized G0-1 paired outputs; validation predictions are retained for G0-2 calibration.
- `scripts/run_g0_2_interval_calibration.py`: validation-only per-target split-conformal interval scaling and independent test reporting.
- `experiments/g0_2_interval_calibration/`: G0-2 factors, calibration curves, test metrics and calibrated predictions.
- `scripts/run_g0_3_threshold_sensitivity.py`: paired legacy-threshold/no-threshold training with tail-stratified row and compound splits.
- `experiments/g0_3_threshold_sensitivity/`: G0-3 tail error, calibration, width-error and high-uncertainty diagnostics.
- `scripts/run_g0_4_paper_style_transfer.py`: paired last2/full/paper-style transfer qualification with explicit column inputs.
- `experiments/g0_4_paper_style_transfer/`: finalized G0-4 checkpoints, validation-only decision, calibrated predictions and slice metrics.
- `experiments/g0_4_paper_style_transfer_random_init_diagnostic/`: superseded diagnostic showing why new column adapters must preserve the transferred source function at initialization.
- `experiments/PREDICTOR_FREEZE.md`: Gate 0 scientific Predictor freeze contract; AL results must not be used to reopen it.
- `scripts/al_engine.py`: shared frozen `fit/predict`, stable identity and resumable AL-state primitives.
- `scripts/run_d28_engineering_checks.py`: 10k+ batched-inference, resume and paired-partition qualification.
- `experiments/d28_al_engineering/`: finalized D28 audit and E2/E4 row/compound L0/U0/test partitions.
- `experiments/e1_signal_qualification/`: E1 signal qualification, per-sample scores, metrics and figures.
- `experiments/e2_random_smoke/`: source-free Random reveal/retrain/resume chain smoke; not a scientific AL result.
- `scripts/run_e2_random_smoke.py`: reproducible E2 source-free Random chain smoke entry point.
- `scripts/al_acquisition.py`: deterministic Coverage、Ensemble top-score、Hybrid Top-25%+farthest-first与signal agreement primitives.
- `scripts/run_e2_4g_active_learning.py`: E2 Round-0 diagnostics、四策略row pilot、paired AULC与绘图入口。
- `experiments/e2_4g_active_learning/`: finalized E2 row outputs and mechanism audit.
- `experiments/e2_4g_compound_preflight/`: compound seed42 Round-0 and Round-1 acquisition-only preflight; not a full pilot.
- `scripts/run_e0_8g_controls.py`: shared entry point for E0-3b/E0-3c controls.
- `scripts/qgeognn_graphs.py`: the single deterministic conformer/graph implementation used by 4g and 8g.
- `scripts/run_d04_conformer_selection.py`: the D04 cache, source-training and transfer pipeline.

The current experiments use the conda environment `fish`. Finalized experiment directories are protected from accidental overwrite; use a new output directory for a reproduction run, for example:

```bash
conda run --no-capture-output -n fish python scripts/run_e0_8g_controls.py \
  --study loss_controls \
  --output-dir experiments/reproductions/e0_3c_repeat
```

Gate 0-1 的正式入口为：

```bash
conda run --no-capture-output -n fish python \
  scripts/run_g0_1_quantile_monotonicity.py
```
