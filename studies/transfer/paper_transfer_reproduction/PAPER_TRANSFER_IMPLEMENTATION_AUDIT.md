# Paper-transfer implementation audit

## Scope and classification

This study answers one constrained question: what are the row-split test RMSE and
MAE of the paper-aligned 4g-to-25g and 4g-to-40g transfer procedure? It does not
compare new transfer methods, alter the backbone, or use a test result to select an
adaptation depth.

Classification: `PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION`.

The official released repository contains a runnable 25g function but it defaults to
`direct_train`; changing it to transfer loads all 4g parameters and gives no freeze
map or validation-only checkpoint rule. There is no runnable `QGeoGNN_transfer_40g`
function or released 40g checkpoint, although a 40g checkpoint path is mentioned in
a comment. Therefore this cannot be called a direct execution of the authors'
25g/40g training path. The reconstruction deliberately reuses the established 8g
paper-style implementation without tuning it for either new target.

## Paper evidence

The published-paper record used by the repository is Wu et al., *Intelligent column
chromatography prediction model based on automation and machine learning*, Chem 11
(2025), 102598, DOI `10.1016/j.chempr.2025.102598`. The accessible arXiv precursor
is `2404.09114`. The precursor's transfer section and Figure 4 were inspected during
this audit; numerical Figure 4 reference values below follow the task's published
paper record, not values selected from a reproduction.

| Item | Explicitly supported by paper | Repository evidence / reconstruction choice |
| --- | --- | --- |
| Source | 4g model trained on the large dataset | `experiments/e0_4g_baseline/checkpoints/best.pt`, SHA256 `7b9e3d0d4c8036c738ef220802e7ee46bc6ab8261cc541fb7d194e8c17044323` |
| Target | 8g, 25g, and 40g columns use transfer learning | Primary targets are 25g and 40g |
| Split | Secondary-data random row train/validation/test split, about 80/10/10 | Five fixed random-row splits: 42, 525, 1101, 2025, 2026 |
| Column information | New column specification is added at model input | Append `column_dia`, `column_len`, `column_den` to every Graph-G edge |
| Pretraining | Original-network parameters are transferred | Load every shape-compatible 4g parameter |
| Learning rate | `1e-4` | Adam, `lr=1e-4`, `weight_decay=1e-5`, matching G0-4 |
| Output layer | Output layer is updated for new columns | Existing monotonic q10/q50/q90 head is installed from source q50 and increments |
| Checkpoint rule | Validation is monitored before final test evaluation | Validation-only minimum combined normalized RMSE; test is read after selection |
| Transfer trainable scope | Not specified | New column RBF/linear adapters, last two GNN layers, and output head; this is the frozen 8g G0-4 implementation choice |
| Direct control | Figure 4 contrasts direct training with transfer | Same column-input QGeoGNN from random initialization, full parameter training, same split/scaler/loss/test |

The paper does **not** give an exact random seed, batch size for these targets,
epoch count, early-stopping patience, per-layer freeze map, exact physical column
geometry values, or 25g/40g RMSE/MAE. None is represented as paper fact here.

## Original code audit

- `application/QGeoGNN.py:1603-1667` builds 8g data with `V1 <= 60` and
  `V2 <= 120`; `QGeoGNN_transfer_8g` uses an 80/10/10 row split.
- `application/QGeoGNN.py:1670-1734` builds 25g data with the same thresholds.
- `application/QGeoGNN.py:1737-1801` builds 40g data with `V1 <= 150` and
  `V2 <= 200`.
- `QGeoGNN_transfer_25g` sets `transfer_mode='direct_train'` at
  `application/QGeoGNN.py:2584-2587`. Its transfer branch loads a 4g checkpoint at
  line 2680, updates all parameters at `lr=1e-4`, evaluates test R2 during fitting,
  and hard-codes a test checkpoint epoch. It is evidence of intent, not an auditable
  primary protocol.
- No `QGeoGNN_transfer_40g` training function is present. The only 40g model
  reference is the commented inference path at `application/QGeoGNN.py:3889`.

## Reused 8g implementation

`scripts/run_g0_4_paper_style_transfer.py` is the fixed implementation reference.
It loads the 4g source, adds the three column features, zero-initializes only newly
introduced column linear adapters to preserve the compatible source function,
installs the monotonic head, and trains new adapters plus last two GNN layers and
the head. Its 3 row-split paper-style test means are V1 R2 `0.7612`, V2 R2 `0.8057`,
V1 RMSE `9.90 mL`, and V2 RMSE `16.29 mL`. Artifact checksums for its summary,
slice metrics, and predictions were recomputed before this extension.

## Frozen primary protocol

`scripts/studies/run_paper_transfer_reproduction_25g_40g.py` implements the
primary protocol. It uses legacy-filtered data, source-only preprocessing, identical
row split assignments for direct and transfer within each column/seed, the fixed
G0-4 architecture choices above, and no test-driven modification. `no_threshold`
is a separately named optional sensitivity protocol and is never used to choose the
primary implementation.
