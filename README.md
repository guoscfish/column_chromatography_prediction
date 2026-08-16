# Intelligent Column Chromatography Prediction Model

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
**Python Version**: 3.9

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
- `scripts/run_e0_8g_controls.py`: shared entry point for E0-3b/E0-3c controls.
- `scripts/qgeognn_graphs.py`: the single deterministic conformer/graph implementation used by 4g and 8g.
- `scripts/run_d04_conformer_selection.py`: the D04 cache, source-training and transfer pipeline.

The current experiments use the conda environment `fish`. Finalized experiment directories are protected from accidental overwrite; use a new output directory for a reproduction run, for example:

```bash
conda run --no-capture-output -n fish python scripts/run_e0_8g_controls.py \
  --study loss_controls \
  --output-dir experiments/reproductions/e0_3c_repeat
```
