"""Contracts for the isolated A2 shared-late-backbone PCGrad intervention."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.studies import run_column_conditioned_multitask as base
from scripts.studies import run_column_conditioned_pcgrad as run
from src.qgeognn_al.transfer.pcgrad import (
    deterministic_pcgrad_seed,
    gradient_geometry,
    pcgrad_backward,
    project_conflicting_gradients,
    task_gradient_tuples,
)


@pytest.fixture(scope="module")
def real_batch():
    torch.set_num_threads(1)
    specification = base.protocol_payload()
    source = base.task_data("4g", "compound", base.SEEDS[0], specification)
    atom, angle, _ = base.make_batch({"4g": source}, {"4g": list(range(12))})
    tasks = torch.arange(12) % 3
    return atom, angle, tasks


@pytest.fixture(scope="module")
def compound_data():
    data = run.load_data("compound", run.SEEDS[0])
    return data, base.fold_schedules(data)


def losses_for(model, atom, angle, tasks):
    prediction = model(atom, angle, tasks)
    return {task: base.target_loss_recipe(atom.y[tasks.eq(index), 0], atom.y[tasks.eq(index), 1],
                                          prediction[tasks.eq(index)], {}, recipe="raw_quantile")
            for index, task in enumerate(run.TASKS)}


def parameter_grads(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()}


def test_candidate_architecture_is_completed_a2_architecture():
    torch.manual_seed(42)
    historical = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    torch.manual_seed(42)
    candidate = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    assert run.architecture_signature(candidate) == run.architecture_signature(historical)
    assert tuple(candidate.state_dict()) == tuple(historical.state_dict())
    for name, value in historical.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)


def test_trainable_parameter_identities_equal_completed_a2():
    first = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    second = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    assert run.trainable_signature(first) == run.trainable_signature(second)
    assert any(name.startswith("heads.") for name in run.trainable_signature(first)[1])
    assert any(name.startswith("film_generators.") for name in run.trainable_signature(first)[1])


def test_pcgrad_scope_is_previous_diagnostic_authority():
    model = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    _, names = run.shared_signature(model)
    expected_prefixes = ("backbone.convs.3.", "backbone.convs.4.", "backbone.convs_bond_angle.3.",
                         "backbone.convs_bond_float.3.", "backbone.convs_bond_embeding.3.",
                         "backbone.convs_angle_float.3.")
    assert names
    assert all(name.startswith(expected_prefixes) for name in names)
    assert not any(name.startswith(("heads.", "column_embedding.", "film_generators.", "condition_branch.")) for name in names)
    lookup = {id(parameter): name for name, parameter in model.named_parameters()}
    assert names == tuple(lookup[id(parameter)] for parameter in model.shared_gradient_parameters())


def test_nonshared_trainables_receive_ordinary_mean_loss_gradient(real_batch):
    atom, angle, tasks = real_batch
    torch.manual_seed(7)
    ordinary = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    torch.manual_seed(7)
    candidate = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    for model in (ordinary, candidate):
        base.set_training_mode(model)
    torch.stack(tuple(losses_for(ordinary, atom, angle, tasks).values())).mean().backward()
    expected = parameter_grads(ordinary)
    losses = losses_for(candidate, atom, angle, tasks)
    pcgrad_backward(losses, candidate.shared_gradient_parameters(), permutation_seed=11)
    actual = parameter_grads(candidate)
    shared_names = set(run.shared_signature(candidate)[1])
    checked_prefixes = ("heads.", "column_embedding.", "film_generators.", "condition_branch.")
    for name, parameter in candidate.named_parameters():
        if parameter.requires_grad and name not in shared_names:
            assert actual[name] is not None
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    assert all(any(name.startswith(prefix) for name in actual if actual[name] is not None) for prefix in checked_prefixes)


def test_only_shared_late_backbone_gradients_are_replaced(real_batch):
    atom, angle, tasks = real_batch
    torch.manual_seed(9)
    ordinary = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    torch.manual_seed(9)
    candidate = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    for model in (ordinary, candidate):
        base.set_training_mode(model)
    torch.stack(tuple(losses_for(ordinary, atom, angle, tasks).values())).mean().backward()
    pcgrad_backward(losses_for(candidate, atom, angle, tasks), candidate.shared_gradient_parameters(), permutation_seed=37)
    expected, actual = parameter_grads(ordinary), parameter_grads(candidate)
    shared = set(run.shared_signature(candidate)[1])
    assert any(not torch.allclose(actual[name], expected[name]) for name in shared)
    for name, parameter in candidate.named_parameters():
        if parameter.requires_grad and name not in shared:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)


def test_nonconflicting_synthetic_gradients_remain_unchanged():
    gradients = {"4g": (torch.tensor([1., 0.]),), "25g": (torch.tensor([2., 0.]),), "40g": (torch.tensor([3., 1.]),)}
    projected, diagnostics = project_conflicting_gradients(gradients, permutation_seed=1)
    for task in gradients:
        torch.testing.assert_close(projected[task][0], gradients[task][0])
    assert diagnostics.directed_projections_triggered == 0


def test_exactly_opposing_synthetic_gradients_project_to_zero():
    gradients = {"left": (torch.tensor([1., 0.]),), "right": (torch.tensor([-1., 0.]),)}
    projected, diagnostics = project_conflicting_gradients(gradients, permutation_seed=5)
    torch.testing.assert_close(projected["left"][0], torch.zeros(2), atol=1e-10, rtol=0)
    torch.testing.assert_close(projected["right"][0], torch.zeros(2), atol=1e-10, rtol=0)
    assert diagnostics.directed_projections_triggered == 2


def test_deterministic_permutation_and_projection():
    assert deterministic_pcgrad_seed(1, 2, 3, 4) == deterministic_pcgrad_seed(1, 2, 3, 4)
    assert deterministic_pcgrad_seed(1, 2, 3, 4) != deterministic_pcgrad_seed(1, 2, 3, 5)
    gradients = {"a": (torch.tensor([1., -2.]),), "b": (torch.tensor([-2., 1.]),), "c": (torch.tensor([1., 1.]),)}
    first, first_diagnostic = project_conflicting_gradients(gradients, permutation_seed=123)
    second, second_diagnostic = project_conflicting_gradients(gradients, permutation_seed=123)
    assert first_diagnostic == second_diagnostic
    for task in gradients:
        torch.testing.assert_close(first[task][0], second[task][0], rtol=0, atol=0)


def test_zero_norm_gradient_is_finite():
    gradients = {"a": (torch.zeros(3),), "b": (torch.tensor([1., -1., 0.]),), "c": (torch.tensor([-1., 1., 0.]),)}
    projected, diagnostics = project_conflicting_gradients(gradients, permutation_seed=3)
    assert all(torch.isfinite(value).all() for values in projected.values() for value in values)
    assert np.isfinite(diagnostics.projected_original_update_norm_ratio)


def test_existing_a2_artifact_identities_and_protocol_match():
    policy = run.verify_reference()
    assert policy["source_sha256"] == base.SOURCE_SHA
    assert policy["split_manifest_sha256"] == base.SPLIT_SHA
    results, _, identities = run.historical_frames("compound")
    assert len(results.loc[results.column.isin(base.TASKS[1:])]) == 50
    assert set(identities.protocol) == {"compound"}


def test_global_target_compound_inner_overlap_is_zero(compound_data):
    data, folds = compound_data
    for fold, (train, valid) in enumerate(folds, 1):
        run.assert_context_comparable(data, train, valid, "compound", run.SEEDS[0], fold)
        train_groups = {str(data[column]["frame"].iloc[index].canonical_smiles) for column in run.TASKS[1:] for index in train[column]}
        valid_groups = {str(data[column]["frame"].iloc[index].canonical_smiles) for column in run.TASKS[1:] for index in valid[column]}
        assert not train_groups & valid_groups


def test_outer_truth_unavailable_before_prediction_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "STUDY", tmp_path)
    called = []
    monkeypatch.setattr(base, "_read_authorized_truth", lambda *args: called.append(args))
    with pytest.raises(RuntimeError, match="before global prediction freeze"):
        run.score()
    assert not called


def test_pcgrad_diagnostics_do_not_mutate_optimizer_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    losses = {"4g": parameter.sum(), "25g": 2 * parameter.sum(), "40g": -parameter.sum()}
    raw = task_gradient_tuples(losses, [parameter])
    before = {task: tuple(value.clone() for value in values) for task, values in raw.items()}
    gradient_geometry(raw)
    project_conflicting_gradients(raw, permutation_seed=4)
    assert parameter.grad is None
    for task in raw:
        torch.testing.assert_close(raw[task][0], before[task][0], rtol=0, atol=0)


def test_deterministic_candidate_inference(real_batch):
    atom, angle, tasks = real_batch
    torch.manual_seed(42)
    model = base.build_multitask_model(base.source_path(), base.ARMS["A2"]).eval()
    with torch.no_grad():
        first = model(atom, angle, tasks)
        second = model(atom, angle, tasks)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()


def test_frozen_prefix_equivalence_remains_under_pcgrad(real_batch):
    atom, angle, tasks = real_batch
    uncached = atom.clone()
    del uncached.frozen_nodes
    del uncached.frozen_edges
    torch.manual_seed(13)
    cached = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    torch.manual_seed(13)
    full = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    for model in (cached, full):
        base.set_training_mode(model)
    cached_losses = losses_for(cached, atom, angle, tasks)
    full_losses = losses_for(full, uncached, angle, tasks)
    pcgrad_backward(cached_losses, cached.shared_gradient_parameters(), permutation_seed=99)
    pcgrad_backward(full_losses, full.shared_gradient_parameters(), permutation_seed=99)
    for (left_name, left), (right_name, right) in zip(cached.named_parameters(), full.named_parameters()):
        assert left_name == right_name
        if left.requires_grad:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-5)
