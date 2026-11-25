from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from src.qgeognn_al.models import predictor_checkpoint
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import stage_state
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import backend as module


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def public_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("public_assets_only")
    paths = module.MODEL_FILES | {
        module.CANONICAL.as_posix(), module.GRAPH_CACHE.as_posix(),
        (module.SEQUENTIAL_STUDY / "protocol.json").as_posix(),
        (module.SPLITS / "split_manifest.json").as_posix(),
        *((module.SPLITS / f"row_seed_{seed}.csv").as_posix() for seed in module.SEEDS),
        "dataset/dataset_4g.csv",
        "experiments/e0_4g_baseline/data_metadata.json",
        "studies/active_learning/qgeognn_v2_row_small_batch_benchmark/protocol.json",
        "studies/active_learning/pure_llm_v3_20261124/PROMPT_PURE_LLM_V3.md",
    }
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(ROOT / name)
    return root


@pytest.fixture(scope="module")
def resources(public_root, tmp_path_factory):
    return module.TrainingBackend(public_root, tmp_path_factory.mktemp("qualified_resources"), 157)


@pytest.mark.parametrize("seed", module.SEEDS)
def test_independent_full_A7_initial_packet_with_mock_predictions(seed, public_root, tmp_path, resources):
    data = resources if seed == resources.seed else module.TrainingBackend(public_root, tmp_path, seed)
    # This temporary root contains only published data/graph/config and Python files.
    assert not list(public_root.rglob("selections")) and not list(public_root.rglob("feedback.json"))
    assert len(data.atom) == len(data.angle) == 4163
    assert data.preprocessing["target_scale_fit_rows"] == 333
    assert {row["roles"] for row in data.initial_label_audit} == {"l0", "validation"}
    assert {row["purpose"] for row in data.initial_label_audit} == {"initial_fit"}
    predictions = np.broadcast_to([1., 2., 3., 4., 5., 6.], (len(data.u0_ids), 6))
    state = data.state_from_predictions(predictions, "a" * 64)
    stage = stage_state(public_root, state, arm="A7", memory_mode="M2")
    assert state.messages == state.history == []
    assert state.packet["validation_test_records"] == 0
    assert len(stage.context.candidates) == 2997
    assert len(stage.context.observed) == 333
    assert stage.memory.snapshot()["hypotheses"] == {}
    assert not set(data.roles["validation"] + data.roles["test"]) & set(stage.context.all_row_ids.values())
    for role, rows in (("C", stage.context.candidates), ("O", stage.context.observed)):
        assert all(key.startswith(f"{role}-") for key in rows)
        assert all({"flow", "purity_raw", "CAS", "name"} <= card["conditions"].keys() for card in rows.values())
    assert all("V1_ml" not in row and "V2_ml" not in row for row in stage.context.candidates.values())
    assert "cw_distance_percentile" not in stage.messages[-1]["content"]
    assert "high_error_measured_examples" not in stage.messages[-1]["content"]
    assert "premeasurement_error" not in stage.messages[-1]["content"]


def test_preflight_has_no_fresh_fit_fallback(resources, monkeypatch, tmp_path):
    @contextmanager
    def forbidden():
        pytest.fail("preflight tried to enter CUDA mode or fit without a qualified anchor")
        yield

    monkeypatch.setattr(resources, "_cuda", forbidden)
    with pytest.raises(FileNotFoundError, match="qualifying existing"):
        resources.initial_state(None, tmp_path)
    assert not (tmp_path / "best.pt").exists()


def test_prepared_cache_replay_and_artifact_tampering(resources, public_root, tmp_path, monkeypatch):
    replay = module.TrainingBackend(public_root, resources.cache_dir, 157)
    assert replay.cache_receipt_sha256 == resources.cache_receipt_sha256
    assert replay.initialization_hash == resources.initialization_hash
    for name in ("resource_receipt.json", "scrubbed_graphs.pt", "scaler.json"):
        (tmp_path / name).symlink_to(resources.cache_dir / name)

    def tampered(path):
        return "1" * 64 if path == tmp_path / "scrubbed_graphs.pt" else file_digest(path)

    monkeypatch.setattr(module, "file_digest", tampered)
    with pytest.raises(ValueError, match="graph_sha256"):
        module.TrainingBackend(public_root, tmp_path, 157)


def test_foreign_or_duplicate_acquisition_cannot_access_label_store(resources, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an invalid acquisition opened the label store")

    monkeypatch.setattr(module, "RestrictedLabelStore", forbidden)
    partial = resources.u0_ids[:31]
    with pytest.raises(ValueError, match="32 fresh"):
        resources.reveal_batch((), partial + (resources.roles["validation"][0],))
    with pytest.raises(ValueError, match="32 fresh"):
        resources.reveal_batch((), partial + (partial[0],))


def test_verified_mock_cuda_anchor_does_not_retrain_and_replays(resources, monkeypatch, tmp_path):
    anchor = tmp_path / "anchor"
    frozen = anchor / "rounds/random/round_00"
    model_dir = frozen / "model"
    model_dir.mkdir(parents=True)
    environment = {"cuda": "mock CUDA", "packages": {"torch": "test fixture"}}
    b = resources
    checkpoint = model_dir / "best.pt"
    model = module.build_predictor(b.normalization)
    module.torch.save(predictor_checkpoint(model, preprocessing=b.preprocessing,
                      training_config=b.training_config, provenance={"mock": True}), checkpoint)
    checkpoint_state = module.runner.state_dict_hash(model)
    initialization = module.initialization_seed(b.seed)
    checkpoint_sha = file_digest(checkpoint)
    initial_l_hash, initial_u_hash = module.ids_hash(b.l0_ids), module.ids_hash(b.u0_ids)
    fit_file = model_dir / "fit_audit.json"
    metrics = module.point_metrics(b.validation_truth, np.ones((416, 6), dtype=np.float32),
                                   b.preprocessing["target_scales"])
    once(anchor / "run_contract.json", {
        "kind": module.REFERENCE_KIND, "method": "random", "outer_seed": b.seed,
        "canonical_source_sha256": b.source_sha256, "canonical_graph_cache_sha256": b.graph_sha256,
        "training": module.TRAINING_CONFIG, "reference_preregistration_commit": module.REFERENCE_PREREGISTRATION,
        "reference_initial_L": "common_original_L333_per_seed", "batch_size": 32, "selection_rounds": 36,
        "budgets": [333 + 32 * r for r in range(37)], "code_and_config_sha256": b.code_and_split_hashes,
        **environment,
    })
    once(fit_file, {
        "fit_contract_hash": "b" * 64, "train_ids_hash": initial_l_hash,
        "validation_ids_hash": module.ids_hash(b.roles["validation"]), "train_rows": 333, "validation_rows": 416,
        "initialization_seed": initialization, "initialization_hash": b.initialization_hash,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_state_hash": checkpoint_state,
        "test_labels_used_for_fit_or_checkpoint_selection": 0,
        "best_validation_combined_normalized_rmse": metrics["combined_normalized_rmse"],
        "best_epoch": 1, "epochs_run": 1, "training_seconds": 0.0,
    })
    once(frozen / "contract.json", {
        "input": {"outer_seed": b.seed, "method": "random", "round": 0, "active_label_count": 333,
            "unlabeled_count": 2997, "L_t_ids_hash": initial_l_hash, "U_t_ids_hash": initial_u_hash,
            "ordered_L_t_hash": module.stable_hash(list(b.l0_ids)),
            "ordered_U_t_hash": module.stable_hash(list(b.u0_ids)), "model_initialization_seed": initialization,
            "training_config": b.training_config, "endpoint_scales": b.preprocessing["target_scales"],
            "endpoint_scales_fit_ids_hash": initial_l_hash, "test_truth_access_count": 0},
        "status": "FROZEN_BEFORE_TEST_TRUTH", "test_truth_access_count": 0,
        "initialization_hash": b.initialization_hash, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_hash": checkpoint_state, "evaluation_fit_contract_hash": "b" * 64,
        "files": {"model/best.pt": checkpoint_sha, "model/fit_audit.json": file_digest(fit_file)},
    })
    calls = []

    def mocked_infer(_model, identities):
        calls.append(tuple(identities))
        return np.ones((len(identities), 6), dtype=np.float32)

    def forbidden_fit(**kwargs):
        pytest.fail("qualified L333 anchor tried to train")

    @contextmanager
    def mocked_cuda():
        yield environment

    monkeypatch.setattr(b, "_infer", mocked_infer)
    monkeypatch.setattr(b, "_cuda", mocked_cuda)
    monkeypatch.setattr(module.runner, "fit_from_same_initialization", forbidden_fit)
    fit_dir = tmp_path / "reusable_state"
    state = b.initial_state(anchor, fit_dir)
    assert calls == [b.u0_ids, b.roles["validation"]]
    assert file_digest(fit_dir / "best.pt") == state.checkpoint == checkpoint_sha
    assert state == b.initial_state(None, fit_dir)
    assert calls == [b.u0_ids, b.roles["validation"]]
    checkpoint.write_text("tampered synthetic source")
    with pytest.raises(ValueError, match="model/best.pt"):
        b.initial_state(anchor, tmp_path / "unqualified_reuse")


def test_optimizer_complete_receipt_replays_under_exact_fit_contract(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    calls = []
    contract = {"synthetic": True, "seed": 157}

    def mocked_train(*args, **kwargs):
        calls.append(1)
        checkpoint.write_text("synthetic checkpoint weights")
        return [{"epoch": 1, "validation_selection_score": 0.1}], 1

    monkeypatch.setattr(module.runner, "train_source", mocked_train)
    args = (None, None, None, None, None, None, None, checkpoint)
    with module._preserve_completed_training(tmp_path, contract):
        result = module.runner.train_source(*args)
    with module._preserve_completed_training(tmp_path, contract):
        assert module.runner.train_source(*args) == result
    with module._preserve_completed_training(tmp_path, {"synthetic": True, "seed": 887}):
        with pytest.raises(ValueError, match="contract"):
            module.runner.train_source(*args)
    assert len(calls) == 1
