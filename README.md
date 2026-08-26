# Intelligent Column Chromatography Prediction Model

## 当前研究主线

本项目当前研究问题是：D45 测得的 8g single-label oracle marginal utility 究竟是稳定的 candidate 属性，还是显著受到 target fine-tuning stochasticity 与有限 58-row test sampling 的影响。D46-A 只审计 oracle target 的可靠性，不开发新 acquisition。

主指标不是 SQ，而是 V1/V2 的 normalized RMSE 学习曲线与 AULC；主结论口径是 `labels-to-90%-reference` 和相对 Random 的标签节省。SQ 只在最后作为色谱推荐的 downstream utility 验证。

严格执行顺序：

```text
Gate 0 Predictor qualification
  → E1 acquisition-signal qualification
  → E2 4g active-learning closed loop
  → E4 4g→8g active transfer (main experiment)
  → E5 downstream SQ utility
```

Gate 0 与 E1 已完成并冻结。E2 row 支持 Hybrid/Coverage，compound split 仅为 suggestive。E4 Protocol A 的 active acquisition 相对 Random 为 null；仅降低 L0 50→30 的 E4-A2a formal 仍为 null，因此 headroom hypothesis 未获支持。D43/D44 的 transfer-aware gates 均失败。当前 D45 只做 post-hoc oracle marginal-utility diagnostic，不改变任何历史正式结论，也不把同一 test partition 继续当作无污染 confirmatory test。

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
- `experiments/e2_compound_failure_audit/`: D38R corrected post-hoc audit; compound localization remains inconclusive and does not alter frozen E2 results.
- `experiments/e4_active_transfer_preregistration/`: E4 partition/source compatibility evidence.
- `experiments/e4_protocol_a_engineering_smoke/`: Protocol A seed42 engineering-only smoke; no scientific conclusion and no formal pilot.
- `experiments/e4_protocol_a_formal/`: completed three-seed Protocol A formal pilot; pretrained Random had the best mean normalized AULC, so active-acquisition evidence is null.
- `experiments/e4_a2a_low_budget_formal/`: completed L0=30 sensitivity; evidence is null and the headroom hypothesis is not supported.
- `scripts/run_d45_oracle_marginal_utility.py`: D45 smoke/bounded post-hoc single-label oracle diagnostic.
- `experiments/d45_oracle_marginal_utility/`: D45 compact outputs, audits, decision, and figures; no confirmatory evidence.
- `scripts/run_d46_oracle_utility_reliability.py`: D46-A smoke/bounded oracle-target reliability audit.
- `experiments/d46_oracle_utility_reliability/`: D46 preflight, reliability outputs, predictions, audits, and decision; post-hoc only.
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

Earlier finalized experiments used the conda environment `fish`; D45 was executed in `chromatography` and records its exact package versions in `environment.json`. Finalized experiment directories are protected from accidental overwrite; use a new output directory for a reproduction run, for example:

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
