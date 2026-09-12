"""Contracts specific to the train-only-selected converged P0/P1 study."""
from copy import deepcopy
import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from src.qgeognn_al.transfer import adaptation as a
from scripts.studies import run_traditional_transfer_converged_baseline_v1 as runner


@pytest.fixture
def tiny_fixed():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.convs = nn.ModuleList([nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)) for _ in range(5)])
            self.condition_branch = nn.Linear(2, 2)
            self.head = nn.Linear(2, 6)

        def forward(self, atom, angle):
            return self.head(self.backbone.convs[4](atom.x) + self.condition_branch(atom.x))

    torch.manual_seed(23)
    atoms = [Data(x=torch.tensor([[i / 10, (i % 3) / 2]], dtype=torch.float32),
                  y=torch.tensor([[i + 1.0, i * 2.0 + 3.0]])) for i in range(8)]
    return Tiny(), atoms, [Data(x=torch.zeros(1, 1)) for _ in atoms]


def test_fixed_epoch_refit_has_no_validation_or_test_input(tiny_fixed):
    signature = inspect.signature(a.train_target_adaptation_fixed_epochs)
    assert "validation_indices" not in signature.parameters
    assert "test_indices" not in signature.parameters
    model, atoms, angles = tiny_fixed
    fit = a.train_target_adaptation_fixed_epochs(
        model, atoms, angles, range(6), 3, mode="historical_shallow",
        config={"learning_rate": .01, "batch_size": 32, "bn_policy": "current"},
    )
    assert fit.epochs_run == 3
    assert len(fit.history) == 3
    assert all(np.isfinite(row["train_loss"]) for row in fit.history)


def test_fixed_p1_stage_b_receives_stage_a_final_state(tiny_fixed, monkeypatch):
    model, atoms, angles = tiny_fixed
    original = a.train_target_adaptation_fixed_epochs
    observed = []

    def wrapped(*args, **kwargs):
        if observed:
            observed.append(deepcopy(args[0].state_dict()))
        result = original(*args, **kwargs)
        if not observed:
            observed.append(deepcopy(args[0].state_dict()))
        return result

    monkeypatch.setattr(a, "train_target_adaptation_fixed_epochs", wrapped)
    a.train_staged_target_adaptation_fixed_epochs(
        model, atoms, angles, range(6), 2, 2, stage_b_mode="historical_shallow",
        stage_a_config={"learning_rate": .01, "batch_size": 32, "bn_policy": "current"},
        stage_b_config={"learning_rate": .01, "batch_size": 32, "bn_policy": "current"},
    )
    assert len(observed) == 2
    assert all(torch.equal(observed[0][name], observed[1][name]) for name in observed[0])


def test_groupkfold_selection_is_group_disjoint_and_rounds_half_up():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    for train, valid in runner.GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups):
        assert not set(groups[train]) & set(groups[valid])
    summary = pd.DataFrame([
        {"method": "P0", "stage": "B", "best_epoch": epoch} for epoch in (10, 11, 12, 13, 14)
    ] + [
        {"method": "P1", "stage": stage, "best_epoch": epoch}
        for stage, values in (("A", (10, 11, 12, 13, 14)), ("B", (20, 21, 22, 23, 24))) for epoch in values
    ])
    selected = runner._selected_epochs(summary, "25g", 1).set_index(["method", "stage"])
    assert int(selected.loc[("P0", "B"), "selected_epoch"]) == 12
    assert runner.round_half_up(12.5) == 13


def test_protocol_forbids_outer_validation_selection_and_uses_shared_denominator():
    protocol = runner._protocol()
    assert protocol["inner_cv"]["outer_validation_used_for_selection"] is False
    assert "outer gradient_train only" in protocol["scale_authority"]
    source = inspect.getsource(runner.score)
    assert 'audit["outer_gradient_train_scales"]' in source
    assert "for method in METHODS" in source


def test_protocol_carries_correct_validation_metadata_semantics():
    source = inspect.getsource(runner.prepare_context)
    assert '"gradient_fit_rows"' in source
    assert '"validation_selection_rows": 0' in source
    assert '"test_truth_rows_used_for_fit": 0' in source
    assert '"test_truth_rows_used_for_selection": 0' in source
