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

Gate 0 的四项科学对照已经完成。G0-3 按 validation-only 规则选择 no-threshold，保留全部574行8g数据和22条tail难例；G0-4的paper-style在paired validation仅赢1/6，平均比last2+head恶化15.3%，full fine-tune恶化29.9%，因此冻结`last2+head`。Paper-style虽在独立test平均小幅改善1.3%，但test不参与结构选择，不能据此反向改选。科学Predictor配置见 [PREDICTOR_FREEZE.md](experiments/PREDICTOR_FREEZE.md)，完整证据见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)、[G0-1](experiments/g0_1_quantile_monotonicity/README.md)、[G0-2](experiments/g0_2_interval_calibration/README.md)、[G0-3](experiments/g0_3_threshold_sensitivity/README.md) 与 [G0-4](experiments/g0_4_paper_style_transfer/README.md)。进入E1前仍需完成10k+分块推理、索引一致性和round-resume工程检查。

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
