from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.studies import run_clean_4g_baseline_qualification as qualification
from src.qgeognn_al.historical.clean_schema import CleanConditionBatch, fit_clean_condition_normalization


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/predictor/clean_4g_baseline_qualification"


def test_qualification_dataset_contract_is_exact() -> None:
    data = qualification.load_qualification_data()
    assert len(data) == 4163
    assert data["canonical_smiles"].nunique() == 217
    assert data["V1_ml"].max() <= 60
    assert data["V2_ml"].max() <= 120
    assert qualification.sha256_file(qualification.RAW_SOURCE) == qualification.EXPECTED_SOURCE_SHA


def test_row_splits_are_reproducible_and_have_frozen_counts() -> None:
    data = qualification.load_qualification_data()
    for seed in qualification.SEEDS:
        first = qualification.generate_split_table(data, "row", seed)
        second = qualification.generate_split_table(data, "row", seed)
        pd.testing.assert_frame_equal(first, second)
        assert qualification.split_audit(first)["row_counts"] == {"train": 3330, "validation": 416, "test": 417}


def test_compound_splits_are_reproducible_and_group_disjoint() -> None:
    data = qualification.load_qualification_data()
    for seed in qualification.SEEDS:
        first = qualification.generate_split_table(data, "compound", seed)
        second = qualification.generate_split_table(data, "compound", seed)
        pd.testing.assert_frame_equal(first, second)
        audit = qualification.split_audit(first)
        assert audit["compound_counts"] == {"train": 173, "validation": 22, "test": 22}
        assert audit["compound_overlap_counts"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
        assert audit["compound_overlap_assertion"] == "REQUIRED_ZERO_AND_PASS"


def test_committed_split_hashes_and_dataset_manifest() -> None:
    manifest_path = STUDY / "splits/split_manifest.json"
    assert manifest_path.exists(), "splits must be frozen before formal execution"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["retained_rows"] == 4163
    assert manifest["unique_compounds"] == 217
    assert manifest["dataset_source_sha256"] == qualification.EXPECTED_SOURCE_SHA
    assert len(manifest["splits"]) == 6
    for item in manifest["splits"]:
        assert qualification.sha256_file(ROOT / item["path"]) == item["sha256"]
    assert qualification.sha256_file(ROOT / manifest["qualification_dataset_manifest"]) == manifest["qualification_dataset_manifest_sha256"]
    # The frozen manifest records the original generator, before its Clean
    # imports were moved to the historical namespace. Verify that immutable
    # source, plus the still-identical split-generation function, without
    # rewriting historical provenance hashes to today's source.
    import hashlib
    import subprocess
    frozen_source = subprocess.check_output(["git", "show", "433c926:" + manifest["generation_code"]], cwd=ROOT)
    assert hashlib.sha256(frozen_source).hexdigest() == manifest["generation_code_sha256"]
    current_source = (ROOT / manifest["generation_code"]).read_text()
    old_generation = frozen_source.decode().split("def generate_split_table", 1)[1].split("def split_audit", 1)[0]
    new_generation = current_source.split("def generate_split_table", 1)[1].split("def split_audit", 1)[0]
    assert new_generation == old_generation


def test_normalization_fits_only_each_training_partition() -> None:
    data = qualification.load_qualification_data()
    for mode in qualification.MODES:
        for seed in qualification.SEEDS:
            split = pd.read_csv(STUDY / f"splits/{mode}_seed_{seed}.csv")
            normalized_split = split[["sample_id", "split"]].copy()
            normalized_split["split"] = normalized_split["split"].replace({"validation": "valid"})
            normalization = fit_clean_condition_normalization(data, normalized_split)
            assert normalization.fit_row_count == int(split["split"].eq("train").sum())
            assert normalization.validation_rows_used == 0
            assert normalization.test_rows_used == 0
            assert normalization.target_8g_rows_used == 0


def test_runner_has_no_8g_data_dependency_and_no_test_during_training() -> None:
    protocol = json.loads((STUDY / "protocol.json").read_text())
    usage = json.loads((STUDY / "data_usage.json").read_text())
    assert protocol["forbidden_data"] == ["8g"]
    assert protocol["training"]["test_during_training"] is False
    assert usage["8g_rows_used"] == 0
    source = inspect.getsource(qualification.run_one)
    loop_source = source.split("for epoch in range", 1)[1].split("pd.DataFrame(history)", 1)[0]
    assert 'indices["test"]' not in loop_source
    assert '"validation_' in loop_source


def test_six_run_config_is_complete_and_validation_selected() -> None:
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert protocol["model"] == "qgeognn_clean_fusion_v1"
    assert protocol["preflight_revision"] == 2
    assert protocol["split_modes"] == ["row", "compound"]
    assert protocol["seeds"] == [42, 525, 1101]
    assert protocol["formal_run_count"] == 6
    assert protocol["training"]["checkpoint_selection"] == "minimum_validation_combined_normalized_RMSE"
    assert protocol["forbidden_models"] == ["legacy_retrain", "condition_completion_v2", "clean_contract_mlp", "paper_reference_ann"]


def test_condition_permutation_changes_only_condition_alignment() -> None:
    conditions = CleanConditionBatch(
        continuous=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        loading_solvent=torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        sample_ids=("a", "b", "c", "d"),
    )
    first, order = qualification.permute_condition_batch(conditions, 900042)
    second, second_order = qualification.permute_condition_batch(conditions, 900042)
    assert np.array_equal(order, second_order)
    assert torch.equal(first.continuous, second.continuous)
    assert sorted(order.tolist()) == [0, 1, 2, 3]
    assert torch.equal(first.continuous, conditions.continuous[order])
    assert torch.equal(first.loading_solvent, conditions.loading_solvent[order])


def test_condition_disabled_mode_and_metric_schema() -> None:
    fixture = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]).numpy()
    truth = np.asarray([[2.0, 5.0], [3.0, 6.0]])
    metrics = qualification.metric_bundle(truth, fixture, {"V1": 1.0, "V2": 1.0})
    required = {
        "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2",
        "combined_normalized_rmse", "V1_q10_q90_coverage", "V2_q10_q90_coverage",
        "V1_interval_width", "V2_interval_width", "V1_mean_pinball_loss", "V2_mean_pinball_loss",
        "within_target_quantile_crossing_rate", "q50_V1_gt_q50_V2_rate", "V1_q90_gt_V2_q10_rate",
    }
    assert required.issubset(metrics)
    assert "condition_disabled" in inspect.getsource(qualification.run_one)


def test_artifact_manifest_hashes_compact_files(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}\n")
    (tmp_path / "b.csv").write_text("x\n1\n")
    manifest = qualification.build_artifact_manifest(tmp_path, ["a.json", "b.csv"])
    assert manifest["runtime_checkpoint_committed"] is False
    assert [item["path"] for item in manifest["files"]] == ["a.json", "b.csv"]
    for item in manifest["files"]:
        assert item["sha256"] == qualification.sha256_file(tmp_path / item["path"])
