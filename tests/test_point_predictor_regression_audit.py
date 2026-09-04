from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

from scripts.studies import run_point_predictor_regression_audit as audit


ROOT = Path(__file__).resolve().parents[1]


def test_e0_split_is_exactly_reused_without_generation() -> None:
    assert audit.FROZEN_SPLIT == ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
    assert audit.sha256_file(audit.FROZEN_SPLIT) == audit.EXPECTED_SPLIT_SHA
    source = inspect.getsource(audit)
    assert "RandomState" not in source
    assert "make_split" not in source


def test_e0_split_counts_and_threshold_domain_are_exact() -> None:
    data, _, indices = audit.load_frozen_inputs()
    assert {role: len(values) for role, values in indices.items()} == audit.EXPECTED_COUNTS
    assert len(data) == 4163
    assert data["V1_ml"].max() <= 60
    assert data["V2_ml"].max() <= 120


def test_split_sample_ids_cover_data_once() -> None:
    data, split, _ = audit.load_frozen_inputs()
    assert not data["sample_id"].duplicated().any()
    assert not split["sample_id"].duplicated().any()
    assert set(data["sample_id"].astype(str)) == set(split["sample_id"].astype(str))


def test_all_variants_use_the_same_frozen_split_hash() -> None:
    configs = [audit.variant_config(variant) for variant in audit.VARIANTS]
    assert {config["split_sha256"] for config in configs} == {audit.EXPECTED_SPLIT_SHA}
    assert all(config["split_generation_called"] is False for config in configs)


def test_variant_contracts_are_recorded_correctly() -> None:
    configs = {variant: audit.variant_config(variant) for variant in audit.VARIANTS}
    assert configs["R0_LEGACY_E0_EXACT"]["weight_decay"] == 1e-5
    assert configs["R0_LEGACY_E0_EXACT"]["shuffle"] is False
    assert configs["R0_LEGACY_E0_EXACT"]["loss_weights"]["V2"] == 0.5
    for variant in audit.VARIANTS[1:]:
        assert configs[variant]["weight_decay"] == 0
        assert configs[variant]["shuffle"] == "deterministic_each_epoch"
        assert configs[variant]["loss_weights"]["V2"] == 1.0
    assert len({config["config_hash"] for config in configs.values()}) == 4


def test_test_split_is_absent_from_training_and_checkpoint_selection_loop() -> None:
    source = inspect.getsource(audit.run_variant)
    loop = source.split("for epoch in range", 1)[1].split("pd.DataFrame(history)", 1)[0]
    assert 'indices["test"]' not in loop
    assert 'indices["validation"]' in loop
    assert "validation_selection_score" in loop


def test_scaler_and_condition_normalization_contract_is_train_only() -> None:
    for variant in audit.VARIANTS:
        config = audit.variant_config(variant)
        assert config["normalization_fit_role"] == "train_only"
        assert config["validation_rows_used_for_normalization"] == 0
        assert config["test_rows_used_for_normalization"] == 0
        assert config["8g_rows_used"] == 0


def test_train_metrics_are_post_checkpoint_metrics_from_train_indices() -> None:
    source = inspect.getsource(audit.run_variant)
    post_checkpoint = source.split("reloaded.load_state_dict", 1)[1]
    assert 'for role in ("train", "validation", "test")' in post_checkpoint
    assert "evaluate(" in post_checkpoint
    assert "indices[role]" in post_checkpoint


def test_point_metric_schema_includes_combined_normalized_rmse() -> None:
    truth = np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=float)
    prediction = np.asarray([
        [0.0, 1.0, 2.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ])
    result = audit.metrics(truth, prediction, {"V1": 1.0, "V2": 2.0})
    assert {
        "V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae",
        "combined_normalized_rmse", "all_outputs_finite",
    } == set(result)
    assert result["all_outputs_finite"] is True


def test_runner_imports_no_8g_transfer_or_active_learning_module() -> None:
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("8g" in name.lower() for name in imported)
    assert not any("transfer" in name.lower() or "active" in name.lower() for name in imported)


def test_r0_gate_precedes_later_variants() -> None:
    source = inspect.getsource(audit.execute)
    assert source.index("r0_sanity") < source.index("R0_SANITY_CHECK_FAILED")
    assert audit.R0_TOLERANCE == 0.03
