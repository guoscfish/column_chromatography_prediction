from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.studies.run_paper_transfer_gap_decomposition import _release_loss
from scripts.studies.run_paper_transfer_reproduction_25g_40g import quantile_target_loss


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/transfer/paper_transfer_gap_decomposition"
RECON = ROOT / "studies/transfer/paper_transfer_reproduction"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"


def test_release_loss_is_exact_one_to_half_weighting():
    true = torch.tensor([[1.0, 4.0], [2.0, 6.0]])
    prediction = torch.tensor([[.5, 1.5, 2.5, 3., 4.5, 7.], [.7, 2.1, 3., 5., 6.5, 8.]])
    expected = quantile_target_loss(true[:, 0], prediction[:, :3]) + .5 * quantile_target_loss(true[:, 1], prediction[:, 3:])
    torch.testing.assert_close(_release_loss(true, prediction), expected, atol=0, rtol=0)


def test_release_scheduler_and_head_contract_are_recorded_exactly():
    audit = json.loads((STUDY / "audit.json").read_text())
    assert audit["release_scheduler"] == "StepLR(50,0.5) instantiated but scheduler.step() absent"
    assert audit["release_loss"] == "loss_V1 + 0.5*loss_V2"
    assert audit["release_head"] == "Linear(128,6)+ReLU; evaluation clamp"
    assert audit["release_column_info"] == "Use_column_info=False"
    assert audit["exact_paper_reproduction"] is False


def test_release_aligned_splits_reuse_reconstruction_identities_exactly():
    frozen = pd.read_csv(RECON / "split_manifest.csv")
    for column in ("25g", "40g"):
        for seed in (42, 525, 1101, 2025, 2026):
            result = json.loads((STUDY / f"runtime/runs/{column}/seed_{seed}/result.json").read_text())
            block = frozen.loc[(frozen.column == column) & (frozen.protocol == "legacy_filtered") & (frozen.seed == seed)]
            payload = block.sort_values("canonical_index")[["sample_id", "split"]].astype(str).to_dict("records")
            from src.qgeognn_al.training.predictor import stable_hash
            assert result["split_ids_hash"] == stable_hash(payload)
            assert result["selection"] == "validation_only_at_20_epoch_cadence"


def test_all_ten_release_aligned_runs_use_qualified_source_and_full_500_epochs():
    metrics = pd.read_csv(STUDY / "release_code_aligned_metrics.csv")
    assert len(metrics) == 10 and metrics.groupby("column").seed.nunique().eq(5).all()
    assert (metrics.epochs_run == 500).all()
    assert (metrics.trainable_parameters == 458952).all()
    assert metrics.source_checkpoint_sha256.eq(SOURCE_SHA256).all()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    assert np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()


def test_track_a_outputs_and_remaining_gap_boundary():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert protocol["classification"] == "RELEASE_CODE_ALIGNED_REPRODUCTION_DIAGNOSTIC"
    assert protocol["exact_reproduction_claim"] is False
    assert protocol["test_during_training"] is False
    assert protocol["duration"] == [500]
    comparison = pd.read_csv(STUDY / "paper_r2_comparison.csv")
    assert (comparison[["V1_gap", "V2_gap"]] < 0).all().all()
