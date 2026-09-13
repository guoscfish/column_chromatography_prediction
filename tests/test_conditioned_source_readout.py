"""Unit contracts for the residual conditioned-source readout modules."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool

from src.qgeognn_al.transfer.adaptive_readout import (
    ResidualAttentionReadout,
    build_conditioned_readout_model,
    full_chromatographic_condition_inputs,
)
from src.qgeognn_al.transfer.source_augmentation import (
    SourcePredictionAndRepresentationFusion,
    SourcePredictionFusion,
    attach_source_features,
    build_or_load_source_cache,
)
from src.qgeognn_al.transfer import adaptation as a
from scripts.studies import run_conditioned_source_readout as runner


def _inputs():
    torch.manual_seed(7)
    nodes = torch.randn(7, 128)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    fixed = global_add_pool(nodes, batch)
    conditions = torch.randn(2, 8)
    return nodes, batch, fixed, conditions


@pytest.mark.parametrize("kind", ("adaptive", "condition_query"))
def test_attention_readout_is_permutation_invariant(kind):
    nodes, batch, fixed, conditions = _inputs()
    torch.manual_seed(11)
    readout = ResidualAttentionReadout(kind=kind, heads=4)
    with torch.no_grad():
        readout.gate.fill_(1.0)
    order = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    original = readout(fixed, nodes, batch, conditions if kind == "condition_query" else None)
    permuted = readout(fixed, nodes[order], batch[order], conditions if kind == "condition_query" else None)
    torch.testing.assert_close(original, permuted, rtol=0, atol=1e-6)


def test_zero_gate_is_exact_fixed_sum_identity():
    nodes, batch, fixed, _ = _inputs()
    readout = ResidualAttentionReadout(kind="adaptive", heads=4)
    assert float(readout.gate.detach()) == 0.0
    output = readout(fixed, nodes, batch)
    torch.testing.assert_close(output, fixed, rtol=0, atol=0)


def test_attention_is_batch_independent():
    nodes, batch, fixed, _ = _inputs()
    readout = ResidualAttentionReadout(kind="adaptive", heads=4)
    with torch.no_grad():
        readout.gate.fill_(1.0)
    one_nodes, one_batch, one_fixed = nodes[:3], torch.zeros(3, dtype=torch.long), fixed[:1]
    standalone = readout(one_fixed, one_nodes, one_batch)
    combined = readout(fixed, nodes, batch)
    torch.testing.assert_close(standalone[0], combined[0], rtol=0, atol=1e-6)


def test_attention_has_gradient_flow_and_no_nan():
    nodes, batch, fixed, conditions = _inputs()
    readout = ResidualAttentionReadout(kind="condition_query", heads=4)
    with torch.no_grad():
        readout.gate.fill_(1.0)
    loss = readout(fixed, nodes, batch, conditions).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert readout.key.weight.grad is not None and torch.isfinite(readout.key.weight.grad).all()
    assert readout.condition_encoder[0].weight.grad is not None
    assert readout.gate.grad is not None


def test_attention_inference_is_deterministic():
    nodes, batch, fixed, _ = _inputs()
    readout = ResidualAttentionReadout(kind="adaptive", heads=4).eval()
    with torch.no_grad():
        readout.gate.fill_(1.0)
        first = readout(fixed, nodes, batch)
        second = readout(fixed, nodes, batch)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


class _TinyBackbone(nn.Module):
    def forward(self, atom, angle):
        return atom.x


class _TinyConditions(nn.Module):
    def typed_inputs(self, atom):
        return atom.condition

    def full_condition_inputs(self, atom):
        return atom.full_condition

    def forward(self, atom):
        return torch.zeros((atom.condition.shape[0], 128), device=atom.x.device), atom.condition


class _TinySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _TinyBackbone()
        self.condition_branch = _TinyConditions()
        self.head = nn.Sequential(nn.Linear(128, 6), nn.ReLU())

    def extract_representation(self, atom, angle):
        return global_add_pool(self.backbone(atom, angle), atom.batch) + self.condition_branch(atom)[0]

    def forward(self, atom, angle):
        output = self.head(self.extract_representation(atom, angle))
        return output if self.training else torch.clamp(output, min=0, max=1e8)


def _tiny_batch():
    torch.manual_seed(19)
    graphs = [
        Data(x=torch.randn(3, 128), condition=torch.randn(1, 8)),
        Data(x=torch.randn(4, 128), condition=torch.randn(1, 8)),
    ]
    for graph in graphs:
        graph.full_condition = torch.randn(1, 12)
    return Batch.from_data_list(graphs), Batch.from_data_list([Data() for _ in graphs])


def test_wrapped_model_has_epoch_zero_source_identity():
    source = _TinySource().eval()
    wrapped = build_conditioned_readout_model(source, kind="condition_query", heads=4).eval()
    atom, angle = _tiny_batch()
    with torch.no_grad():
        expected = source(atom, angle)
        actual = wrapped(atom, angle)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_source_fusion_is_zero_impact_then_receives_gradient():
    atom, _ = _tiny_batch()
    atom.source_prediction_q50 = torch.randn(2, 2)
    atom.source_representation = torch.randn(2, 128)
    fusion = SourcePredictionFusion({"V1": 10.0, "V2": 20.0})
    zero = fusion(atom)
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
    zero.sum().backward()
    assert fusion.output.weight.grad is not None
    both = SourcePredictionAndRepresentationFusion({"V1": 10.0, "V2": 20.0})
    torch.testing.assert_close(both(atom), torch.zeros((2, 128)), rtol=0, atol=0)


def test_full_condition_query_contract_includes_all_audited_semantics():
    class Conditions(nn.Module):
        def typed_inputs(self, atom):
            # [HBA, LogP, loading amount, loading volume, solvent embedding]
            return atom.typed
    atom = Data(
        x=torch.zeros(4, 2),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        edge_attr=torch.arange(52, dtype=torch.float32).reshape(4, 13),
        batch=torch.tensor([0, 0, 1, 1]),
        typed=torch.arange(16, dtype=torch.float32).reshape(2, 8),
    )
    values = full_chromatographic_condition_inputs(atom, Conditions())
    assert values.shape == (2, 12)
    # It retains six eluent values and omits duplicate HBA/LogP from typed.
    torch.testing.assert_close(values[:, 6:], atom.typed[:, 2:])


def test_loss_recipes_are_explicit_and_l0_matches_p0():
    true_1 = torch.tensor([2.0, 4.0])
    true_2 = torch.tensor([8.0, 12.0])
    prediction = torch.tensor([[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], [3.0, 4.0, 5.0, 11.0, 12.0, 13.0]])
    scales = {"V1": 2.0, "V2": 4.0}
    l0 = a.target_loss_recipe(true_1, true_2, prediction, scales, recipe="raw_quantile")
    expected = a.quantile_target_loss(true_1, prediction[:, :3]) + a.quantile_target_loss(true_2, prediction[:, 3:])
    torch.testing.assert_close(l0, expected)
    l1 = a.target_loss_recipe(true_1, true_2, prediction, scales, recipe="endpoint_normalized_quantile")
    l2 = a.target_loss_recipe(true_1, true_2, prediction, scales, recipe="q50_weak_quantile")
    assert torch.isfinite(l1) and torch.isfinite(l2)
    with pytest.raises(ValueError):
        a.target_loss_recipe(true_1, true_2, prediction, {"V1": 0.0, "V2": 1.0}, recipe="q50_weak_quantile")


def test_additional_trainable_prefixes_are_opt_in():
    class Scope(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.convs = nn.ModuleList([nn.Linear(2, 2) for _ in range(5)])
            self.condition_branch = nn.Linear(2, 2)
            self.head = nn.Linear(2, 6)
            self.adaptive_readout = nn.Linear(2, 2)
            self.source_augmentation = nn.Linear(2, 2)
    model = Scope()
    a.configure_trainable(model, "historical_shallow", additional_prefixes=("adaptive_readout.", "source_augmentation."))
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("backbone.convs.4.") for name in names)
    assert any(name.startswith("condition_branch.") for name in names)
    assert any(name.startswith("head.") for name in names)
    assert any(name.startswith("adaptive_readout.") for name in names)
    assert any(name.startswith("source_augmentation.") for name in names)
    assert not any(name.startswith("backbone.convs.0.") for name in names)


class _CacheSource(nn.Module):
    model_variant = "cache-test-source"

    def forward(self, atom, angle):
        graph = global_add_pool(atom.x, atom.batch)
        return torch.cat([graph[:, :1]] * 6, dim=1)

    def extract_representation(self, atom, angle):
        return global_add_pool(atom.x, atom.batch)


def test_source_cache_is_label_free_keyed_and_batches(tmp_path):
    torch.manual_seed(31)
    atoms = [Data(x=torch.randn(2, 128), y=torch.tensor([[999.0, 999.0]])),
             Data(x=torch.randn(3, 128), y=torch.tensor([[999.0, 999.0]]))]
    angles = [Data(), Data()]
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"frozen source bytes")
    kwargs = dict(cache_root=tmp_path / "cache", column="25g", sample_ids=["a", "b"],
                  feature_table_sha256="f" * 64, source_checkpoint_path=checkpoint,
                  source_model=_CacheSource(), atom_data=atoms, angle_data=angles,
                  source_target_scales={"V1": 1.0, "V2": 2.0})
    cache = build_or_load_source_cache(**kwargs)
    assert cache["target_labels_read"] is False
    assert tuple(cache["source_prediction_q50"].shape) == (2, 2)
    cached = build_or_load_source_cache(**kwargs)
    assert cached["sample_ids"] == ["a", "b"]
    attach_source_features(atoms, cache)
    batch = Batch.from_data_list(atoms)
    assert tuple(batch.source_prediction_q50.shape) == (2, 2)
    assert tuple(batch.source_representation.shape) == (2, 128)
    with pytest.raises(RuntimeError):
        build_or_load_source_cache(**{**kwargs, "sample_ids": ["b", "a"]})
    with pytest.raises(RuntimeError):
        build_or_load_source_cache(**{**kwargs, "feature_table_sha256": "e" * 64})


def test_clustered_bootstrap_uses_compound_clusters():
    truth = torch.tensor([[1.0, 2.0], [1.5, 2.5], [4.0, 5.0], [4.5, 5.5]]).numpy()
    candidate = torch.tensor([[0., 1., 0., 0., 2., 0.], [0., 1.5, 0., 0., 2.5, 0.],
                              [0., 4., 0., 0., 5., 0.], [0., 4.5, 0., 0., 5.5, 0.]]).numpy()
    reference = candidate.copy(); reference[:, 1] += .5; reference[:, 4] += .5
    values = runner._clustered_bootstrap_delta(truth, candidate, reference, ["a", "a", "b", "b"], seed=7, draws=40)
    assert len(values) == 4
    assert {item["cluster_count"] for item in values} == {2}
    assert all(item["ci95_high"] < 0 for item in values)


def _write_negative_finalization_prerequisites(study, *, gates=None, candidate=None):
    study.mkdir(parents=True)
    for name in ("PREREGISTRATION.md", "IMPLEMENTATION_AUDIT.md", "LOSS_SCREEN_REPORT.md"):
        (study / name).write_text("completed\n", encoding="utf-8")
    (study / "LOSS_SCREEN.csv").write_text("arm\nL0\n", encoding="utf-8")
    (study / "INNER_SCREEN_RESULTS.csv").write_text("arm\nR0\n", encoding="utf-8")
    (study / "run_manifest.csv").write_text("column\n25g\n", encoding="utf-8")
    runner.write_json(study / "protocol.json", {"study": "test"})
    gates = {"R1": False, "R2": False} if gates is None else gates
    decision = {
        "screened_arms": ["R0", "R1", "R2"],
        "summary": [
            {"arm": "R1", "column": "25g", "mean_relative_improvement_pct": -1.0,
             "fold_wins": 10, "seed_wins": 2, "n_folds": 25, "n_seeds": 5},
            {"arm": "R1", "column": "40g", "mean_relative_improvement_pct": -2.0,
             "fold_wins": 9, "seed_wins": 1, "n_folds": 25, "n_seeds": 5},
            {"arm": "R2", "column": "25g", "mean_relative_improvement_pct": 0.2,
             "fold_wins": 12, "seed_wins": 2, "n_folds": 25, "n_seeds": 5},
            {"arm": "R2", "column": "40g", "mean_relative_improvement_pct": 0.1,
             "fold_wins": 11, "seed_wins": 2, "n_folds": 25, "n_seeds": 5},
        ],
        "gates": gates,
        "next_arm": None,
        "selected_candidate_if_screen_stops": candidate,
        "selection_boundary": "ROW inner GroupKFold only; no outer validation/test truth",
    }
    runner.write_json(study / "inner_screen_decision.json", decision)
    return decision


def test_negative_finalization_writes_required_unscored_artifacts_without_truth(tmp_path, monkeypatch):
    study = tmp_path / "study"
    decision = _write_negative_finalization_prerequisites(study)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "_verified_final_readout_decision", lambda: decision)

    def truth_must_not_be_read(*args, **kwargs):
        raise AssertionError("negative finalization must not read outer truth")

    monkeypatch.setattr(runner, "_read_authorized_truth", truth_must_not_be_read)
    stopping = runner.finalize_negative_outcome()
    assert stopping["status"] == "NO_MATERIAL_ARCHITECTURE_GAIN"
    assert stopping["outer_test_truth_read"] is False
    assert stopping["outer_predictions_created"] is False
    for name, columns in (
        ("ROW_RESULTS.csv", runner.RESULT_COLUMNS),
        ("COMPOUND_RESULTS.csv", runner.RESULT_COLUMNS),
        ("PAIRED_COMPARISON.csv", runner.PAIRED_COLUMNS),
        ("BOOTSTRAP_CI.csv", runner.BOOTSTRAP_COLUMNS),
    ):
        frame = pd.read_csv(study / name)
        assert frame.empty
        assert list(frame.columns) == list(columns)
    hashes = json.loads((study / "prediction_hashes.json").read_text(encoding="utf-8"))
    assert hashes["stopping_decision"]["outer_test_truth_read"] is False
    report = (study / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "NO_MATERIAL_ARCHITECTURE_GAIN" in report
    assert "no outer-test score was read" in report
    # The terminal action is resumable without rewriting real score evidence.
    assert runner.finalize_negative_outcome() == stopping


def test_negative_finalization_refuses_a_screen_with_an_authorized_candidate(tmp_path, monkeypatch):
    study = tmp_path / "study"
    decision = _write_negative_finalization_prerequisites(
        study, gates={"R1": True, "R2": False}, candidate="R1",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "_verified_final_readout_decision", lambda: decision)
    with pytest.raises(RuntimeError, match="only when no readout arm passed|selected or authorized"):
        runner.finalize_negative_outcome()
    assert not (study / "OUTER_TEST_STOPPING_DECISION.json").exists()


def test_negative_finalization_never_replaces_scored_evidence(tmp_path, monkeypatch):
    study = tmp_path / "study"
    decision = _write_negative_finalization_prerequisites(study)
    runner.write_frame(study / "ROW_RESULTS.csv", pd.DataFrame([
        {column: "evidence" for column in runner.RESULT_COLUMNS},
    ], columns=runner.RESULT_COLUMNS))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "_verified_final_readout_decision", lambda: decision)
    with pytest.raises(RuntimeError, match="refusing to replace existing scored artifact"):
        runner.finalize_negative_outcome()
    assert not (study / "OUTER_TEST_STOPPING_DECISION.json").exists()
    assert not (study / "prediction_hashes.json").exists()


def test_staged_readout_append_preserves_completed_rows_and_contract(tmp_path):
    run = tmp_path / "inner_screen" / "row" / "25g" / "seed_7"
    run.mkdir(parents=True)
    arms = ("R0", "R1", "R2")
    recipes = {arm: "raw_quantile" for arm in arms}
    summary = pd.DataFrame([
        {"phase": "inner_screen", "column": "25g", "protocol": "row", "outer_seed": 7,
         "inner_fold": fold, "arm": arm, "loss_recipe": "raw_quantile", "validation_score": fold + 0.1}
        for arm in arms for fold in range(5)
    ])
    history = pd.DataFrame([{"arm": arm, "epoch": 1, "train_loss": 0.1} for arm in arms])
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    runner.write_frame(summary_path, summary)
    runner.write_frame(history_path, history)
    runner.write_json(run / "context_audit.json", {"frozen": True})
    protocol_sha256 = "p" * 64
    runner.write_json(run / "context_complete.json", {
        "protocol_sha256": protocol_sha256, "phase": "inner_screen", "column": "25g",
        "outer_protocol": "row", "outer_seed": 7, "arms": list(arms), "recipes": recipes,
        "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
        "outer_validation_or_test_labels_used": False,
    })
    original_summary_bytes = summary_path.read_bytes()
    original_history_bytes = history_path.read_bytes()

    resumed = runner._load_completed_inner_context(
        run=run, phase="inner_screen", column="25g", outer_protocol="row", seed=7,
        requested_arms=("R3",), requested_recipes={"R3": "raw_quantile"}, protocol_sha256=protocol_sha256,
    )
    assert resumed is not None
    assert resumed["missing_arms"] == ("R3",)
    with pytest.raises(RuntimeError, match="different loss recipes"):
        runner._load_completed_inner_context(
            run=run, phase="inner_screen", column="25g", outer_protocol="row", seed=7,
            requested_arms=("R2",), requested_recipes={"R2": "q50_weak_quantile"}, protocol_sha256=protocol_sha256,
        )

    r3_summary = summary.loc[summary.arm.eq("R0")].copy()
    r3_summary["arm"] = "R3"
    r3_history = history.loc[history.arm.eq("R0")].copy()
    r3_history["arm"] = "R3"
    runner._append_frame_without_rewriting_existing_rows(
        summary_path, runner._normalize_appended_frame(r3_summary, existing_columns=resumed["summary"].columns),
    )
    runner._append_frame_without_rewriting_existing_rows(
        history_path, runner._normalize_appended_frame(r3_history, existing_columns=resumed["history"].columns),
    )
    assert summary_path.read_bytes().startswith(original_summary_bytes)
    assert history_path.read_bytes().startswith(original_history_bytes)
    combined = pd.read_csv(summary_path)
    assert len(combined) == 20
    pd.testing.assert_frame_equal(combined.iloc[:15].reset_index(drop=True), summary)

    payload = json.loads((run / "context_complete.json").read_text(encoding="utf-8"))
    payload.update({
        "arms": [*arms, "R3"], "recipes": {**recipes, "R3": "raw_quantile"},
        "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
    })
    runner.write_json(run / "context_complete.json", payload)
    already_complete = runner._load_completed_inner_context(
        run=run, phase="inner_screen", column="25g", outer_protocol="row", seed=7,
        requested_arms=("R3",), requested_recipes={"R3": "raw_quantile"}, protocol_sha256=protocol_sha256,
    )
    assert already_complete is not None
    assert already_complete["missing_arms"] == ()


def test_staged_inner_fit_trains_only_the_missing_readout_arm(monkeypatch, tmp_path):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "STUDY", study)
    runner.write_json(study / "protocol.json", {})
    run = runner._context_dir("inner_screen", "row", "25g", 7)
    run.mkdir(parents=True)
    arms = ("R0", "R1", "R2")
    columns = [
        "phase", "column", "protocol", "outer_seed", "inner_fold", "arm", "loss_recipe",
        "n_inner_train", "n_inner_validation", "inner_train_group_sha256", "inner_validation_group_sha256",
        "inner_train_scales", "best_epoch", "epochs_run", "validation_score", "runtime_seconds",
        "trainable_parameter_count", "total_parameter_count", "all_outputs_finite", "combined_nrmse",
    ]
    summary_rows = []
    history_rows = []
    for arm in arms:
        for fold in range(5):
            summary_rows.append({
                "phase": "inner_screen", "column": "25g", "protocol": "row", "outer_seed": 7,
                "inner_fold": fold, "arm": arm, "loss_recipe": "raw_quantile",
                "n_inner_train": 4, "n_inner_validation": 1, "inner_train_group_sha256": "train",
                "inner_validation_group_sha256": "valid", "inner_train_scales": "{\"V1\": 1.0, \"V2\": 1.0}",
                "best_epoch": 1, "epochs_run": 1, "validation_score": 0.2, "runtime_seconds": 0.0,
                "trainable_parameter_count": 3, "total_parameter_count": 4,
                "all_outputs_finite": True, "combined_nrmse": 0.1,
            })
            history_rows.append({
                "phase": "inner_screen", "column": "25g", "protocol": "row", "outer_seed": 7,
                "inner_fold": fold, "arm": arm, "loss_recipe": "raw_quantile", "epoch": 1.0,
            })
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    runner.write_frame(summary_path, pd.DataFrame(summary_rows, columns=columns))
    runner.write_frame(history_path, pd.DataFrame(history_rows))
    audit = {"frozen": True}
    runner.write_json(run / "context_audit.json", audit)
    runner.write_json(run / "context_complete.json", {
        "protocol_sha256": runner.sha(study / "protocol.json"), "phase": "inner_screen", "column": "25g",
        "outer_protocol": "row", "outer_seed": 7, "arms": list(arms),
        "recipes": {arm: "raw_quantile" for arm in arms},
        "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
        "outer_validation_or_test_labels_used": False,
    })
    old_summary = summary_path.read_bytes()
    built_arms = []
    source_feature_calls = []
    context = {
        "positions": {"gradient_train": list(range(5))},
        "frame": pd.DataFrame({"canonical_smiles": ["a", "b", "c", "d", "e"]}),
        "atoms": [object()] * 5, "angles": [object()] * 5, "audit": audit,
    }
    monkeypatch.setattr(runner, "prepare", lambda: {})
    monkeypatch.setattr(runner, "prepare_context", lambda *_: context)
    monkeypatch.setattr(runner, "_ensure_source_features", lambda *_: source_feature_calls.append(True))
    monkeypatch.setattr(runner.a, "fit_target_scales", lambda *_: {"V1": 1.0, "V2": 1.0})

    def build(arm, _context, *, initialization_seed):
        built_arms.append(arm)
        return object(), ()

    monkeypatch.setattr(runner, "build_arm_model", build)
    monkeypatch.setattr(runner.a, "train_target_adaptation", lambda *args, **kwargs: SimpleNamespace(
        best_epoch=1, epochs_run=1, validation_score=0.2, trainable_parameters=3,
        total_parameters=4, history=({"epoch": 1.0},),
    ))
    monkeypatch.setattr(runner.a, "predict_point", lambda *args, **kwargs: (
        np.array([[1.0, 1.0]]), np.zeros((1, 6)), np.array([0]),
    ))
    monkeypatch.setattr(runner, "point_metrics", lambda *args, **kwargs: {
        "all_outputs_finite": True, "combined_nrmse": 0.1,
    })

    runner._inner_fit_context(
        phase="inner_screen", column="25g", outer_protocol="row", seed=7,
        arms=("R3",), recipes={"R3": "raw_quantile"},
    )
    assert built_arms == ["R3"] * 5
    assert source_feature_calls == [True]
    assert summary_path.read_bytes().startswith(old_summary)
    combined = pd.read_csv(summary_path)
    assert set(combined.arm) == {"R0", "R1", "R2", "R3"}
    assert len(combined) == 20
    completed = json.loads((run / "context_complete.json").read_text(encoding="utf-8"))
    assert completed["arms"] == ["R0", "R1", "R2", "R3"]


def _write_staged_append_fixture(study):
    """Create a completed R0--R2 context and the deterministic R3 postimage."""

    runner.write_json(study / "protocol.json", {"study": "journal-recovery-test"})
    run = runner._context_dir("inner_screen", "row", "25g", 7)
    run.mkdir(parents=True)
    arms = ("R0", "R1", "R2")
    summary = pd.DataFrame([
        {"phase": "inner_screen", "column": "25g", "protocol": "row", "outer_seed": 7,
         "inner_fold": fold, "arm": arm, "loss_recipe": "raw_quantile", "validation_score": fold + 0.1}
        for arm in arms for fold in range(5)
    ])
    history = pd.DataFrame([
        {"arm": arm, "epoch": 1, "train_loss": 0.1}
        for arm in arms
    ])
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    audit_path = run / "context_audit.json"
    completion_path = run / "context_complete.json"
    runner.write_frame(summary_path, summary)
    runner.write_frame(history_path, history)
    runner.write_json(audit_path, {"frozen": True})
    protocol_sha256 = runner.sha(study / "protocol.json")
    runner.write_json(completion_path, {
        "protocol_sha256": protocol_sha256, "phase": "inner_screen", "column": "25g",
        "outer_protocol": "row", "outer_seed": 7, "arms": list(arms),
        "recipes": {arm: "raw_quantile" for arm in arms},
        "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
        "outer_validation_or_test_labels_used": False,
    })
    additions_summary = summary.loc[summary.arm.eq("R0")].copy()
    additions_summary["arm"] = "R3"
    additions_history = history.loc[history.arm.eq("R0")].copy()
    additions_history["arm"] = "R3"
    completion_identity = {
        "protocol_sha256": protocol_sha256, "phase": "inner_screen", "column": "25g",
        "outer_protocol": "row", "outer_seed": 7, "arms": [*arms, "R3"],
        "recipes": {**{arm: "raw_quantile" for arm in arms}, "R3": "raw_quantile"},
        "outer_validation_or_test_labels_used": False,
    }
    return {
        "run": run,
        "summary_path": summary_path,
        "history_path": history_path,
        "completion_path": completion_path,
        "additions_summary": additions_summary,
        "additions_history": additions_history,
        "completion_identity": completion_identity,
    }


@pytest.mark.parametrize("installed_before_interruption", (1, 2), ids=("after-summary", "after-history"))
def test_staged_inner_append_recovers_interrupted_postimages_without_refitting(
        tmp_path, monkeypatch, installed_before_interruption):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "STUDY", study)
    fixture = _write_staged_append_fixture(study)
    run = fixture["run"]
    summary_path = fixture["summary_path"]
    history_path = fixture["history_path"]
    prefix_summary = summary_path.read_bytes()
    prefix_history = history_path.read_bytes()
    original_install = runner._install_snapshot
    installed = []

    def interrupt_after_boundary(snapshot, target):
        installed.append(target.name)
        if len(installed) > installed_before_interruption:
            raise OSError("simulated append interruption")
        original_install(snapshot, target)

    monkeypatch.setattr(runner, "_install_snapshot", interrupt_after_boundary)
    with pytest.raises(OSError, match="simulated append interruption"):
        runner._commit_staged_inner_append_unlocked(
            run=run, completed={}, additions_summary=fixture["additions_summary"],
            additions_history=fixture["additions_history"], completion_identity=fixture["completion_identity"],
        )
    assert installed == (["inner_cv_summary.csv", "inner_cv_history.csv"]
                         if installed_before_interruption == 1
                         else ["inner_cv_summary.csv", "inner_cv_history.csv", "context_complete.json"])
    assert (run / runner.INNER_APPEND_JOURNAL_NAME).exists()

    # The second invocation is a process-restart model: recovery must return
    # the committed R3 context before it reaches label/context preparation.
    monkeypatch.setattr(runner, "_install_snapshot", original_install)
    monkeypatch.setattr(runner, "prepare", lambda: {})
    prepared_contexts = []

    def fail_if_refit(*_args, **_kwargs):
        prepared_contexts.append(True)
        raise AssertionError("recovery must not prepare or refit R3")

    monkeypatch.setattr(runner, "prepare_context", fail_if_refit)
    monkeypatch.setattr(runner, "build_arm_model", fail_if_refit)
    assert runner._inner_fit_context(
        phase="inner_screen", column="25g", outer_protocol="row", seed=7,
        arms=("R3",), recipes={"R3": "raw_quantile"},
    ) == run
    assert prepared_contexts == []

    summary = pd.read_csv(summary_path)
    assert summary_path.read_bytes().startswith(prefix_summary)
    assert history_path.read_bytes().startswith(prefix_history)
    r3_rows = summary.loc[summary.arm.eq("R3")]
    assert len(r3_rows) == 5
    assert set(r3_rows.inner_fold.astype(int)) == set(range(5))
    assert not summary.duplicated(["phase", "column", "protocol", "outer_seed", "inner_fold", "arm"]).any()
    completed = json.loads(fixture["completion_path"].read_text(encoding="utf-8"))
    assert completed["arms"] == ["R0", "R1", "R2", "R3"]
    assert completed["summary_sha256"] == runner.sha(summary_path)
    assert completed["history_sha256"] == runner.sha(history_path)
    assert not (run / runner.INNER_APPEND_JOURNAL_NAME).exists()
    assert not list((run / runner.INNER_APPEND_SNAPSHOT_DIRECTORY).glob("transaction-*"))


def test_staged_inner_append_rejects_a_malformed_journal_without_mutating_live_context(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "STUDY", study)
    fixture = _write_staged_append_fixture(study)
    run = fixture["run"]
    summary_before = fixture["summary_path"].read_bytes()
    history_before = fixture["history_path"].read_bytes()
    completion_before = fixture["completion_path"].read_bytes()
    journal_path = run / runner.INNER_APPEND_JOURNAL_NAME
    runner.write_json(journal_path, {
        "schema_version": 1,
        "kind": "INNER_APPEND",
        "base": {},
        "snapshots": {},
    })

    with pytest.raises(RuntimeError, match="invalid base hashes"):
        runner._load_completed_inner_context(
            run=run, phase="inner_screen", column="25g", outer_protocol="row", seed=7,
            requested_arms=("R3",), requested_recipes={"R3": "raw_quantile"},
            protocol_sha256=runner.sha(study / "protocol.json"),
        )

    assert fixture["summary_path"].read_bytes() == summary_before
    assert fixture["history_path"].read_bytes() == history_before
    assert fixture["completion_path"].read_bytes() == completion_before
    assert journal_path.exists()


def _six_output_frame(ids, *, offset=0.0):
    values = np.arange(len(ids), dtype=float) + float(offset)
    return pd.DataFrame({
        "sample_id": list(ids), "V1_q10": values - .1, "V1_q50": values, "V1_q90": values + .1,
        "V2_q10": values + 1.0, "V2_q50": values + 1.1, "V2_q90": values + 1.2,
    })


def _write_gzip_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def _write_selected_candidate_decision(study, candidate="R1"):
    runner.write_json(study / "inner_screen_decision.json", {
        "screened_arms": ["R0", "R1", "R2"],
        "gates": {"R1": candidate == "R1", "R2": candidate == "R2"},
        "next_arm": None,
        "selected_candidate_if_screen_stops": candidate,
        "selection_boundary": "ROW inner GroupKFold only; no outer validation/test truth",
    })


def _candidate_provenance(candidate="R1", recipe="raw_quantile"):
    return {
        "candidate": candidate,
        "inner_screen_decision": "inner_screen_decision.json",
        "inner_screen_decision_sha256": "d" * 64,
        "stage": "initial",
        "stage_decision": {"path": "INNER_SCREEN_INITIAL_DECISION.json", "sha256": "s" * 64},
        "loss_selection": {
            "path": "loss_selection.json",
            "sha256": "l" * 64,
            "selected_loss_arm": "L0",
            "selected_loss_recipe": recipe,
        },
    }


def _terminal_candidate_decision(candidate="R1", recipe="raw_quantile"):
    provenance = _candidate_provenance(candidate, recipe)
    return {
        "stage": provenance["stage"],
        "stage_decision": provenance["stage_decision"],
        "loss_selection": provenance["loss_selection"],
        "screened_arms": ["R0", candidate],
        "gates": {candidate: True},
        "next_arm": None,
        "selected_candidate_if_screen_stops": candidate,
        "selection_boundary": runner.READOUT_SELECTION_BOUNDARY,
    }


def _write_initial_readout_contexts(study, *, scores=None):
    scores = {"R0": 1.0, "R1": 1.03, "R2": .95} if scores is None else scores
    arms = runner.INITIAL_READOUT_ARMS
    recipe = "raw_quantile"
    runner.write_json(study / "protocol.json", {
        "architecture_continuation_rule": {"test": True},
        "loss_selection_rule": "test loss rule",
    })
    for column in runner.COLUMNS:
        for seed in runner.SEEDS:
            run = runner._context_dir("loss_screen", "row", column, seed)
            rows = [
                {
                    "phase": "loss_screen", "column": column, "protocol": "row", "outer_seed": seed,
                    "inner_fold": fold, "arm": arm, "loss_recipe": loss_recipe,
                    "validation_score": score,
                }
                for arm, loss_recipe, score in (
                    ("L0", "raw_quantile", 1.0),
                    ("L1", "endpoint_normalized_quantile", 1.02),
                    ("L2", "q50_weak_quantile", 1.04),
                )
                for fold in range(5)
            ]
            summary = pd.DataFrame(rows)
            history = pd.DataFrame([{"arm": arm, "epoch": 1} for arm in runner.LOSS_ARMS])
            summary_path = run / "inner_cv_summary.csv"
            history_path = run / "inner_cv_history.csv"
            runner.write_frame(summary_path, summary)
            runner.write_frame(history_path, history)
            runner.write_json(run / "context_audit.json", {"context": f"loss-{column}-{seed}"})
            runner.write_json(run / "context_complete.json", {
                "protocol_sha256": runner.sha(study / "protocol.json"),
                "phase": "loss_screen", "column": column, "outer_protocol": "row", "outer_seed": seed,
                "arms": list(runner.LOSS_ARMS), "recipes": dict(runner.LOSS_ARMS),
                "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
                "outer_validation_or_test_labels_used": False,
            })
    runner.aggregate_loss_screen()
    for column in runner.COLUMNS:
        for seed in runner.SEEDS:
            run = runner._context_dir("inner_screen", "row", column, seed)
            rows = [
                {
                    "phase": "inner_screen", "column": column, "protocol": "row", "outer_seed": seed,
                    "inner_fold": fold, "arm": arm, "loss_recipe": recipe,
                    "validation_score": scores[arm],
                }
                for arm in arms for fold in range(5)
            ]
            summary = pd.DataFrame(rows)
            history = pd.DataFrame([{"arm": arm, "epoch": 1} for arm in arms])
            summary_path = run / "inner_cv_summary.csv"
            history_path = run / "inner_cv_history.csv"
            runner.write_frame(summary_path, summary)
            runner.write_frame(history_path, history)
            runner.write_json(run / "context_audit.json", {"context": f"{column}-{seed}"})
            runner.write_json(run / "context_complete.json", {
                "protocol_sha256": runner.sha(study / "protocol.json"),
                "phase": "inner_screen", "column": column, "outer_protocol": "row", "outer_seed": seed,
                "arms": list(arms), "recipes": {arm: recipe for arm in arms},
                "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
                "outer_validation_or_test_labels_used": False,
            })


def test_readout_state_machine_rejects_out_of_order_public_launches(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    launches = []
    monkeypatch.setattr(runner, "_inner_fit_context", lambda **kwargs: launches.append(kwargs))

    with pytest.raises(ValueError, match="exactly R0 R1 R2"):
        runner.run_readout_context("25g", runner.SEEDS[0], ("R0", "R3"))
    with pytest.raises(ValueError, match="exactly R0 R1 R2"):
        runner.aggregate_readout_screen(("R0", "R3"))
    with pytest.raises(RuntimeError, match="missing persisted initial readout decision/evidence"):
        runner.run_readout_context("25g", runner.SEEDS[0], ("R3",))

    _write_initial_readout_contexts(study)
    initial = runner.aggregate_readout_screen(runner.INITIAL_READOUT_ARMS)
    assert initial["next_arm"] == "R3"
    with pytest.raises(RuntimeError, match="missing persisted r3 readout decision/evidence"):
        runner.run_readout_context("25g", runner.SEEDS[0], ("R4",))
    assert launches == []


def test_r3_launch_rejects_tampered_initial_stage_decision(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    _write_initial_readout_contexts(study)
    runner.aggregate_readout_screen(runner.INITIAL_READOUT_ARMS)
    decision_path = study / "INNER_SCREEN_INITIAL_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["gates"]["R2"] = False
    runner.write_json(decision_path, decision)
    monkeypatch.setattr(runner, "_inner_fit_context", lambda **kwargs: (_ for _ in ()).throw(AssertionError("R3 must not fit")))

    with pytest.raises(RuntimeError, match="persisted initial readout decision is mutable"):
        runner.run_readout_context("25g", runner.SEEDS[0], ("R3",))


def test_r3_launch_rejects_tampered_initial_stage_evidence(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    _write_initial_readout_contexts(study)
    runner.aggregate_readout_screen(runner.INITIAL_READOUT_ARMS)
    evidence_path = study / "INNER_SCREEN_INITIAL_RESULTS.csv"
    evidence = pd.read_csv(evidence_path)
    evidence.loc[0, "validation_score"] = 999.0
    runner.write_frame(evidence_path, evidence)
    monkeypatch.setattr(runner, "_inner_fit_context", lambda **kwargs: (_ for _ in ()).throw(AssertionError("R3 must not fit")))

    with pytest.raises(RuntimeError, match="readout evidence no longer matches"):
        runner.run_readout_context("25g", runner.SEEDS[0], ("R3",))


def test_terminal_stage_reaggregation_recovers_missing_final_pointer(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    _write_initial_readout_contexts(study, scores={"R0": 1.0, "R1": 1.03, "R2": 1.02})
    decision = runner.aggregate_readout_screen(runner.INITIAL_READOUT_ARMS)
    assert decision["next_arm"] is None
    pointer = study / runner.FINAL_READOUT_DECISION_NAME
    pointer.unlink()

    recovered = runner.aggregate_readout_screen(runner.INITIAL_READOUT_ARMS)
    assert recovered == decision
    assert json.loads(pointer.read_text(encoding="utf-8"))["stage"] == "initial"


def test_final_context_rejects_wrong_loss_recipe_or_selection_phase(monkeypatch):
    provenance = _candidate_provenance("R1")
    monkeypatch.setattr(runner, "_selected_candidate_provenance", lambda _: provenance)

    with pytest.raises(RuntimeError, match="loss recipe does not match"):
        runner._final_context_provenance("row", "R1", "endpoint_normalized_quantile", "inner_screen")
    with pytest.raises(RuntimeError, match="selection phase does not match"):
        runner._final_context_provenance("row", "R1", "raw_quantile", "compound_selection")


def test_final_context_rejects_tampered_completed_artifacts_before_resume(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    runner.write_json(study / "protocol.json", {"study": "test"})
    provenance = _candidate_provenance("R1")
    monkeypatch.setattr(runner, "_final_context_provenance", lambda *args, **kwargs: provenance)
    monkeypatch.setattr(runner, "_formal_selection", lambda *args, **kwargs: 3)
    monkeypatch.setattr(runner, "_roles", lambda *_: {
        "gradient_train": ["g"], "validation": ["v"], "test": ["t"],
    })
    monkeypatch.setattr(runner, "prepare_context", lambda *_: (_ for _ in ()).throw(AssertionError("resume must not refit")))
    run = runner._final_dir("row", "25g", 7, "R1")
    run.mkdir(parents=True)
    final_pt = run / "final.pt"
    final_pt.write_bytes(b"checkpoint")
    validation = run / "validation_predictions_blind.csv.gz"
    test = run / "test_predictions_blind.csv.gz"
    runner.write_frame(validation, _six_output_frame(["v"]))
    runner.write_frame(test, _six_output_frame(["t"]))
    runner.write_json(run / "final_fit_complete.json", {
        "protocol_sha256": runner.sha(study / "protocol.json"), "selection_phase": "inner_screen",
        "outer_protocol": "row", "column": "25g", "outer_seed": 7, "arm": "R1",
        "loss_recipe": "raw_quantile", "selected_epoch": 3, "candidate_provenance": provenance,
        "outer_validation_or_test_labels_used": False, "final_pt_sha256": runner.sha(final_pt),
        "validation_prediction_sha256": runner.sha(validation), "test_prediction_sha256": runner.sha(test),
    })
    assert runner.run_final_context("25g", "row", 7, "R1", "raw_quantile", selection_phase="inner_screen") == run
    final_pt.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="final fit completion digest assertion failed"):
        runner.run_final_context("25g", "row", 7, "R1", "raw_quantile", selection_phase="inner_screen")


def test_write_frame_round_trips_gzip_artifact(tmp_path):
    path = tmp_path / "prediction.csv.gz"
    expected = _six_output_frame(["a", "b"])
    runner.write_frame(path, expected)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    pd.testing.assert_frame_equal(pd.read_csv(path), expected)


def test_atomic_writers_reserve_unique_same_directory_temporary_paths(tmp_path, monkeypatch):
    seen = []
    original = runner._atomic_temporary_path

    def record(path):
        temporary = original(path)
        seen.append(temporary)
        return temporary

    monkeypatch.setattr(runner, "_atomic_temporary_path", record)
    json_path = tmp_path / "shared.json"
    csv_path = tmp_path / "shared.csv"
    runner.write_json(json_path, {"worker": "one"})
    runner.write_frame(csv_path, pd.DataFrame({"arm": ["R0"]}))
    runner._append_frame_without_rewriting_existing_rows(csv_path, pd.DataFrame({"arm": ["R1"]}))

    assert len(seen) == 3
    assert len(set(seen)) == len(seen)
    assert all(path.parent == tmp_path for path in seen)
    assert all(not path.exists() for path in seen)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"worker": "one"}
    assert pd.read_csv(csv_path).arm.tolist() == ["R0", "R1"]


def test_full_data_reference_reorders_ids_and_verifies_frozen_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    role_hashes = {"gradient_train": "gradient-train", "validation": "validation", "test": "test"}
    monkeypatch.setattr(runner, "_frozen_role_hashes", lambda *_: role_hashes)
    study = tmp_path / "studies/transfer/full_data_baseline_finalization"
    prediction = study / "runtime/contexts/25g/row/seed_7/predictions_blind.csv.gz"
    _write_gzip_frame(prediction, pd.DataFrame({
        "sample_id": ["b", "a"], "paper_style_current_v2_V1": [20.0, 10.0],
        "paper_style_current_v2_V2": [200.0, 100.0],
    }))
    frozen = prediction.parent / "frozen.json"
    runner.write_json(frozen, {
        "contract": {
            "column": "25g", "protocol": "row", "seed": 7,
            "train_ids_hash": role_hashes["gradient_train"],
            "validation_ids_hash": role_hashes["validation"],
            "test_ids_hash": role_hashes["test"],
            "test_labels_used_for_fit_normalization_or_selection": 0,
            "donor_outer_test_rows_used": 0,
        },
        "files": {prediction.name: runner.sha(prediction)},
    })
    runner.write_json(study / "prediction_freeze_manifest.json", {
        "files": {str(frozen.relative_to(study)): runner.sha(frozen)},
    })
    actual = runner._reference_prediction("25g", "row", 7, "paper_style_current_v2", ["a", "b"])
    np.testing.assert_allclose(actual[:, [1, 4]], [[10.0, 100.0], [20.0, 200.0]])
    prediction.write_bytes(b"changed after freeze")
    with pytest.raises(RuntimeError, match="reference prediction digest mismatch"):
        runner._reference_prediction("25g", "row", 7, "paper_style_current_v2", ["a", "b"])


def test_hier_reference_reorders_ids_and_verifies_frozen_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    study = tmp_path / "studies/transfer/hier_cw_semantic_repair"
    prediction = study / "runtime/contexts/40g/compound/seed_7/predictions_blind.csv.gz"
    _write_gzip_frame(prediction, pd.DataFrame({
        "sample_id": ["b", "a"], "HIER_CW_SHARED_LAMBDA_CORRECTED_V1": [4.0, 3.0],
        "HIER_CW_SHARED_LAMBDA_CORRECTED_V2": [40.0, 30.0],
    }))
    roles = {
        "25g": {"gradient_train": ["25g-train"], "validation": ["25g-validation"], "test": ["25g-test"]},
        "40g": {"gradient_train": ["40g-train"], "validation": ["40g-validation"], "test": ["40g-test"]},
    }
    monkeypatch.setattr(runner, "_roles", lambda column, *_: roles[column])
    audit = prediction.with_name("fit_audit.json")
    runner.write_json(audit, {
        "column": "40g", "protocol": "compound", "seed": 7,
        "train_ids_sha256": runner.stable_hash(["25g-train", "40g-train"]),
        "test_ids_sha256": runner.stable_hash(["40g-test"]),
        "outer_validation_used": False,
        "test_truth_used": False,
    })
    runner.write_json(study / "prediction_freeze_manifest.json", {
        "files": {
            str(prediction.relative_to(tmp_path)): runner.sha(prediction),
            str(audit.relative_to(tmp_path)): runner.sha(audit),
        },
    })
    actual = runner._reference_prediction("40g", "compound", 7, "HIER_CW_SHARED_LAMBDA_CORRECTED", ["a", "b"])
    np.testing.assert_allclose(actual[:, [1, 4]], [[3.0, 30.0], [4.0, 40.0]])
    runner.write_json(study / "prediction_freeze_manifest.json", {"files": {}})
    with pytest.raises(RuntimeError, match="reference prediction digest mismatch"):
        runner._reference_prediction("40g", "compound", 7, "HIER_CW_SHARED_LAMBDA_CORRECTED", ["a", "b"])


def test_prediction_freeze_hashes_checkpoint_and_both_blind_prediction_roles(tmp_path, monkeypatch):
    study = tmp_path / "studies/transfer/conditioned_source_readout"
    study.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "COLUMNS", ("25g",))
    monkeypatch.setattr(runner, "SEEDS", (7,))
    runner.write_json(study / "protocol.json", {"study": "test"})
    _write_selected_candidate_decision(study, "R1")
    provenance = _candidate_provenance("R1")
    monkeypatch.setattr(runner, "_selected_candidate_provenance", lambda candidate: provenance)
    monkeypatch.setattr(runner, "_formal_selection", lambda *args, **kwargs: 3)
    monkeypatch.setattr(runner, "_roles", lambda *_: {
        "gradient_train": ["g"], "validation": ["v"], "test": ["t"],
    })
    protocol_sha256 = runner.sha(study / "protocol.json")
    for arm in ("R0", "R1"):
        run = runner._final_dir("row", "25g", 7, arm)
        run.mkdir(parents=True)
        final_pt = run / "final.pt"
        final_pt.write_bytes(f"checkpoint-{arm}".encode())
        validation = run / "validation_predictions_blind.csv.gz"
        test = run / "test_predictions_blind.csv.gz"
        runner.write_frame(validation, _six_output_frame(["v"]))
        runner.write_frame(test, _six_output_frame(["t"]))
        runner.write_json(run / "final_fit_complete.json", {
            "protocol_sha256": protocol_sha256, "outer_protocol": "row", "column": "25g",
            "outer_seed": 7, "arm": arm, "loss_recipe": "raw_quantile", "selection_phase": "inner_screen",
            "selected_epoch": 3, "candidate_provenance": provenance,
            "outer_validation_or_test_labels_used": False,
            "final_pt_sha256": runner.sha(final_pt),
            "validation_prediction_sha256": runner.sha(validation),
            "test_prediction_sha256": runner.sha(test),
        })
    manifest = runner.finalize_prediction_freeze("row", ("R0", "R1"))
    assert set(manifest["files"]) == runner._expected_final_freeze_files("row", ("R0", "R1"))
    assert manifest["candidate_provenance"]["candidate"] == "R1"
    runner._assert_frozen_for_score("row", ("R0", "R1"))
    (runner._final_dir("row", "25g", 7, "R1") / "validation_predictions_blind.csv.gz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="frozen final artifact digest changed"):
        runner._assert_frozen_for_score("row", ("R0", "R1"))
    # Rewriting a changed artifact and its local completion hash cannot replace
    # the first global freeze after the score boundary has been crossed.
    candidate_run = runner._final_dir("row", "25g", 7, "R1")
    replacement = _six_output_frame(["v"], offset=10.0)
    runner.write_frame(candidate_run / "validation_predictions_blind.csv.gz", replacement)
    completion_path = candidate_run / "final_fit_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["validation_prediction_sha256"] = runner.sha(candidate_run / "validation_predictions_blind.csv.gz")
    runner.write_json(completion_path, completion)
    manifest_bytes = (study / "ROW_PREDICTION_FREEZE_MANIFEST.json").read_bytes()
    hashes_bytes = (study / "prediction_hashes.json").read_bytes()
    with pytest.raises(RuntimeError, match="prediction-freeze hash entry drift"):
        runner.finalize_prediction_freeze("row", ("R0", "R1"))
    assert (study / "ROW_PREDICTION_FREEZE_MANIFEST.json").read_bytes() == manifest_bytes
    assert (study / "prediction_hashes.json").read_bytes() == hashes_bytes


def test_score_refuses_candidate_outside_the_frozen_inner_selection(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    _write_selected_candidate_decision(study, "R1")
    monkeypatch.setattr(runner, "_verified_final_readout_decision", lambda: _terminal_candidate_decision("R1"))
    monkeypatch.setattr(runner, "_assert_frozen_for_score", lambda *_: (_ for _ in ()).throw(AssertionError("freeze check should not run")))
    monkeypatch.setattr(runner, "_read_authorized_truth", lambda *_: (_ for _ in ()).throw(AssertionError("truth must not be read")))
    with pytest.raises(RuntimeError, match="does not match the frozen ROW inner-screen selection"):
        runner.score("row", "R2")


def test_inner_aggregation_and_formal_selection_reject_tampered_completion(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "COLUMNS", ("25g",))
    monkeypatch.setattr(runner, "SEEDS", (7,))
    runner.write_json(study / "protocol.json", {"study": "test"})
    run = runner._context_dir("loss_screen", "row", "25g", 7)
    run.mkdir(parents=True)
    summary = pd.DataFrame([
        {"phase": "loss_screen", "column": "25g", "protocol": "row", "outer_seed": 7,
         "inner_fold": fold, "arm": "L0", "loss_recipe": "raw_quantile", "best_epoch": fold + 1}
        for fold in range(5)
    ])
    summary_path = run / "inner_cv_summary.csv"
    history_path = run / "inner_cv_history.csv"
    runner.write_frame(summary_path, summary)
    runner.write_frame(history_path, pd.DataFrame(columns=["arm"]))
    runner.write_json(run / "context_audit.json", {"frozen": True})
    complete_path = run / "context_complete.json"
    complete = {
        "protocol_sha256": runner.sha(study / "protocol.json"), "phase": "loss_screen", "column": "25g",
        "outer_protocol": "row", "outer_seed": 7, "arms": ["L0"], "recipes": {"L0": "raw_quantile"},
        "summary_sha256": runner.sha(summary_path), "history_sha256": runner.sha(history_path),
        "outer_validation_or_test_labels_used": False,
    }
    runner.write_json(complete_path, complete)
    assert len(runner._collect_inner("loss_screen", "row", ("L0",), {"L0": "raw_quantile"})) == 5
    assert runner._formal_selection("25g", "row", 7, "R0", "raw_quantile", phase="loss_screen") == 3
    complete["outer_seed"] = 8
    runner.write_json(complete_path, complete)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner._collect_inner("loss_screen", "row", ("L0",), {"L0": "raw_quantile"})
    complete["outer_seed"] = 7
    runner.write_json(complete_path, complete)
    summary_path.write_bytes(summary_path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="summary or history digest mismatch"):
        runner._formal_selection("25g", "row", 7, "R0", "raw_quantile", phase="loss_screen")


def test_compound_selection_matches_r0_to_the_frozen_global_loss(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(runner, "_selected_candidate_provenance", lambda candidate: _candidate_provenance(candidate, "endpoint_normalized_quantile"))
    monkeypatch.setattr(runner, "_assert_row_continuation_authorized", lambda candidate: {"candidate": candidate})

    def capture_inner_fit(**kwargs):
        captured.update(kwargs)
        return tmp_path / "compound-selection"

    monkeypatch.setattr(runner, "_inner_fit_context", capture_inner_fit)
    assert runner.run_compound_selection_context("25g", 769539383, "R2") == tmp_path / "compound-selection"
    assert captured == {
        "phase": "compound_selection",
        "column": "25g",
        "outer_protocol": "compound",
        "seed": 769539383,
        "arms": ("R0", "R2"),
        "recipes": {
            "R0": "endpoint_normalized_quantile",
            "R2": "endpoint_normalized_quantile",
        },
    }


def _row_gate_results(candidate="R1", *, endpoint_multiplier=1.01):
    """Five matched formal scores per column with exactly four candidate wins."""

    rows = []
    for column in runner.COLUMNS:
        for index, seed in enumerate(runner.SEEDS):
            p0_combined = 1.0 + .02 * index
            candidate_combined = p0_combined * (.95 if index < 4 else 1.0)
            for method, combined, multiplier in (("P0", p0_combined, 1.0), (candidate, candidate_combined, endpoint_multiplier)):
                v1 = 10.0 + index
                v2 = 20.0 + index
                rows.append({
                    "protocol": "row", "column": column, "outer_seed": seed, "method": method,
                    "n_test": 12, "metric_scale_authority": "outer_gradient_train_ddof0",
                    "V1_r2": .5, "V1_rmse": v1 * multiplier, "V1_mae": (v1 / 2) * multiplier,
                    "V2_r2": .5, "V2_rmse": v2 * multiplier, "V2_mae": (v2 / 2) * multiplier,
                    "combined_normalized_rmse": combined, "all_outputs_finite": True,
                })
    return pd.DataFrame(rows, columns=runner.RESULT_COLUMNS)


def test_clustered_bootstrap_estimate_is_the_observed_full_sample_delta():
    truth = np.zeros((4, 2), dtype=float)
    candidate = np.zeros((4, 6), dtype=float)
    reference = np.zeros((4, 6), dtype=float)
    # The full-sample V1 ΔRMSE is sqrt((0 + 0 + 9 + 9) / 4) - 1.  With seed 7
    # and one draw the bootstrap sample is not representative, so this also
    # distinguishes the observed estimate from a bootstrap-mean estimate.
    candidate[:, 1] = [0.0, 0.0, 3.0, 3.0]
    reference[:, 1] = 1.0
    values = runner._clustered_bootstrap_delta(truth, candidate, reference, ["a", "a", "b", "b"], seed=7, draws=1)
    v1_rmse = next(item for item in values if item["endpoint"] == "V1" and item["metric"] == "delta_rmse")
    assert v1_rmse["estimate"] == pytest.approx(np.sqrt(4.5) - 1.0)


def test_row_continuation_requires_four_seed_wins_and_endpoint_nonregression():
    passed = runner.evaluate_row_continuation("R1", _row_gate_results())
    assert passed["status"] == "ROW_CONTINUATION_PASSED"
    assert passed["compound_confirmation_authorized"] is True
    assert all(passed["columns"][column]["combined_nrmse"]["seed_wins"] == 4 for column in runner.COLUMNS)
    assert all(passed["columns"][column]["combined_nrmse"]["mean_gain_pct"] >= 3.0 for column in runner.COLUMNS)

    endpoint_failed = runner.evaluate_row_continuation("R1", _row_gate_results(endpoint_multiplier=1.021))
    assert endpoint_failed["status"] == "ROW_CONTINUATION_FAILED"
    assert endpoint_failed["criteria"]["no_endpoint_mean_rmse_or_mae_regression_above_limit"] is False
    assert endpoint_failed["compound_confirmation_authorized"] is False

    seed_failed_result = _row_gate_results()
    for column in runner.COLUMNS:
        for seed in runner.SEEDS[-2:]:
            p0 = seed_failed_result.loc[(seed_failed_result.column.eq(column)) & (seed_failed_result.outer_seed.eq(seed)) & (seed_failed_result.method.eq("P0")), "combined_normalized_rmse"]
            seed_failed_result.loc[(seed_failed_result.column.eq(column)) & (seed_failed_result.outer_seed.eq(seed)) & (seed_failed_result.method.eq("R1")), "combined_normalized_rmse"] = p0.iloc[0] * 1.02
    seed_failed = runner.evaluate_row_continuation("R1", seed_failed_result)
    assert seed_failed["criteria"]["seed_wins_both_columns"] is False
    assert seed_failed["compound_confirmation_authorized"] is False

    gain_failed_result = _row_gate_results()
    gain_failed_result.loc[gain_failed_result.method.eq("R1"), "combined_normalized_rmse"] = gain_failed_result.loc[gain_failed_result.method.eq("P0"), "combined_normalized_rmse"].to_numpy() * .99
    gain_failed = runner.evaluate_row_continuation("R1", gain_failed_result)
    assert gain_failed["criteria"]["mean_combined_gain_both_columns"] is False
    assert gain_failed["compound_confirmation_authorized"] is False


def test_row_continuation_decision_round_trips_with_its_frozen_row_manifest(tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setattr(runner, "STUDY", study)
    manifest_reference = {"path": "ROW_PREDICTION_FREEZE_MANIFEST.json", "sha256": "f" * 64}
    monkeypatch.setattr(runner, "_row_prediction_freeze_reference", lambda _: manifest_reference)
    result = _row_gate_results()
    runner.write_frame(study / "ROW_RESULTS.csv", result)

    persisted = runner._persist_row_continuation_decision("R1", result)
    assert persisted["row_prediction_freeze_manifest"] == manifest_reference
    assert runner._load_row_continuation_decision("R1") == persisted


def test_compound_score_is_blocked_before_any_outer_truth_read(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(runner, "STUDY", study)
    truth_read = []
    monkeypatch.setattr(runner, "_read_authorized_truth", lambda *_: truth_read.append(True))
    with pytest.raises(RuntimeError, match="COMPOUND is blocked"):
        runner.score("compound", "R1")
    assert truth_read == []


def test_final_report_surfaces_gate_means_wins_and_clustered_ci(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "_row_prediction_freeze_reference", lambda _: {
        "path": "ROW_PREDICTION_FREEZE_MANIFEST.json", "sha256": "f" * 64,
    })
    result = _row_gate_results()
    row_path = study / "ROW_RESULTS.csv"
    runner.write_frame(row_path, result)
    decision = runner._persist_row_continuation_decision("R1", result)
    paired = []
    bootstrap = []
    for column in runner.COLUMNS:
        for seed in runner.SEEDS:
            p0 = result.loc[(result.column.eq(column)) & (result.outer_seed.eq(seed)) & (result.method.eq("P0"))].iloc[0]
            candidate = result.loc[(result.column.eq(column)) & (result.outer_seed.eq(seed)) & (result.method.eq("R1"))].iloc[0]
            paired.append({
                "protocol": "row", "column": column, "outer_seed": seed, "candidate": "R1", "reference": "P0",
                "candidate_minus_reference_V1_rmse": candidate.V1_rmse - p0.V1_rmse,
                "candidate_minus_reference_V2_rmse": candidate.V2_rmse - p0.V2_rmse,
                "candidate_minus_reference_V1_mae": candidate.V1_mae - p0.V1_mae,
                "candidate_minus_reference_V2_mae": candidate.V2_mae - p0.V2_mae,
                "candidate_minus_reference_combined_nrmse": candidate.combined_normalized_rmse - p0.combined_normalized_rmse,
            })
            bootstrap.append({
                "protocol": "row", "column": column, "outer_seed": seed, "candidate": "R1", "reference": "P0",
                "endpoint": "V1", "metric": "delta_rmse", "estimate": -0.2,
                "ci95_low": -0.4, "ci95_high": -0.1, "bootstrap_draws": 40, "cluster_count": 8,
            })
    runner.write_frame(study / "PAIRED_COMPARISON.csv", pd.DataFrame(paired, columns=runner.PAIRED_COLUMNS))
    runner.write_frame(study / "BOOTSTRAP_CI.csv", pd.DataFrame(bootstrap, columns=runner.BOOTSTRAP_COLUMNS))
    monkeypatch.setattr(runner, "_read_authorized_truth", lambda *_: (_ for _ in ()).throw(AssertionError("report must not read outer truth")))
    runner.generate_final_report("R1")
    report = (study / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert decision["status"] in report
    assert "mean ± std" in report
    assert "wins / 5" in report
    assert "observed full-sample Δ" in report
    assert "clustered 95% CI" in report


def test_score_preflight_rejects_reference_before_any_outer_truth_read(tmp_path, monkeypatch):
    study = tmp_path / "study"
    frozen = tmp_path / "frozen"
    sample_id = "test-sample"
    runner.write_frame(frozen / "filtered_features_25g.csv", pd.DataFrame({
        "sample_id": [sample_id], "canonical_smiles": ["C"],
    }))
    monkeypatch.setattr(runner, "STUDY", study)
    monkeypatch.setattr(runner, "FROZEN", frozen)
    monkeypatch.setattr(runner, "COLUMNS", ("25g",))
    monkeypatch.setattr(runner, "SEEDS", (7,))
    monkeypatch.setattr(runner, "_selected_candidate_provenance", lambda candidate: _candidate_provenance(candidate))
    frozen_calls = []
    monkeypatch.setattr(runner, "_assert_frozen_for_score", lambda protocol, arms: frozen_calls.append((protocol, tuple(arms))))
    monkeypatch.setattr(runner, "prepare_context", lambda *_: {
        "roles": {"test": [sample_id]}, "outer_scales": {"V1": 1.0, "V2": 1.0},
    })
    final_arms = []

    def final_prediction(_column, _protocol, _seed, arm, test_ids):
        final_arms.append((arm, list(test_ids)))
        return np.zeros((len(test_ids), 6), dtype=float)

    monkeypatch.setattr(runner, "_final_prediction", final_prediction)
    reference_calls = []

    def fail_reference(column, protocol, seed, reference, test_ids):
        reference_calls.append((column, protocol, seed, reference, list(test_ids)))
        raise RuntimeError("historical reference preflight failed")

    monkeypatch.setattr(runner, "_reference_prediction", fail_reference)
    truth_reads = []
    monkeypatch.setattr(runner, "_read_authorized_truth", lambda *args: truth_reads.append(args))

    with pytest.raises(RuntimeError, match="historical reference preflight failed"):
        runner.score("row", "R1")

    assert frozen_calls == [("row", ("R0", "R1"))]
    assert final_arms == [("R0", [sample_id]), ("R1", [sample_id])]
    assert reference_calls == [("25g", "row", 7, "paper_style_current_v2", [sample_id])]
    assert truth_reads == []
