from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from src.qgeognn_al.active_learning_v2.benchmark_protocol import OLD_STUDY, load_features
from src.qgeognn_al.active_learning_v2.gradient_features import (
    extract_linear_output_gradient_sketches,
    extract_linear_output_gradient_sketches_many,
    extract_q50_gradient_sketches,
)
from src.qgeognn_al.active_learning_v2.gradient_transforms import (
    center_width_transform,
    output_whitening_transform,
    power_normalize_gradients,
    raw_output_transform,
)
from src.qgeognn_al.active_learning_v2.innovation_screen import (
    METHODS,
    gradient_norm_topb,
    reveal_l0_targets_only,
    select_innovation_methods,
    validate_screen_seed,
)
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.protocol import make_row_protocol
from src.qgeognn_al.active_learning_v2.runner import make_label_scrubbed_graphs
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE


class TinySixOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 6)

    def forward(self, atom, angle):
        del angle
        return self.linear(atom.x.reshape(-1, 1).float())


def _tiny_graphs(count=12):
    atom = [
        Data(x=torch.tensor([[float(index + 1)]]), y=torch.zeros((1, 2)),
             canonical_position=torch.tensor(index))
        for index in range(count)
    ]
    angle = [Data(x=torch.zeros((1, 1))) for _ in atom]
    return atom, angle


def test_generic_scaled_output_gradient_matches_authoritative_raw_and_lcmd() -> None:
    torch.manual_seed(91)
    model = TinySixOutput()
    atom, angle = _tiny_graphs()
    scales = (2.5, 4.0)
    kwargs = dict(model=model, atom_data=atom, angle_data=angle, indices=list(range(12)),
                  dimension=32, sketch_seed=129)
    raw = extract_q50_gradient_sketches(**kwargs, scales=scales)
    transform, _ = raw_output_transform(scales)
    generic = extract_linear_output_gradient_sketches(**kwargs, output_transform=transform)
    np.testing.assert_allclose(generic.features, raw.features, rtol=1e-6, atol=1e-6)
    raw_selected = lcmd_tp_select(raw.features[4:], raw.features[:4], 4).selected_pool_positions
    generic_selected = lcmd_tp_select(generic.features[4:], generic.features[:4], 4).selected_pool_positions
    assert np.array_equal(raw_selected, generic_selected)


def test_multi_transform_single_pass_matches_generic_extractors() -> None:
    torch.manual_seed(92)
    model = TinySixOutput()
    atom, angle = _tiny_graphs(5)
    transforms = {
        "scaled": np.diag([0.4, 0.25]),
        "mixed": np.array([[0.25, 0.25], [-0.5, 0.5]]),
    }
    kwargs = dict(model=model, atom_data=atom, angle_data=angle, indices=list(range(5)),
                  dimension=24, sketch_seed=511)
    combined = extract_linear_output_gradient_sketches_many(**kwargs, output_transforms=transforms)
    for name, transform in transforms.items():
        separate = extract_linear_output_gradient_sketches(**kwargs, output_transform=transform)
        np.testing.assert_allclose(combined[name].features, separate.features, rtol=1e-6, atol=1e-6)
        assert combined[name].audit["multi_transform_single_backward_pass"] is True


def test_historical_seed73_scaled_gradient_and_selected_id_regression() -> None:
    runtime = OLD_STUDY / "runtime/seed_73"
    split = pd.read_csv(OLD_STUDY / "splits/row_seed_73.csv")
    preprocessing = json.loads((runtime / "preprocessing.json").read_text())["preprocessing"]
    data = load_features()
    graph_cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    atom, angle = make_label_scrubbed_graphs(data, graph_cache, preprocessing["scaler"])
    with np.load(runtime / "gradient_features.npz") as cache:
        historical = cache["features"]
        canonical_indices = cache["canonical_indices"]
    fixture_positions = np.r_[np.arange(20), np.arange(333, 373)]
    fixture_indices = canonical_indices[fixture_positions]
    scales = tuple(preprocessing["target_scales"][target] for target in ("V1", "V2"))
    transform, _ = raw_output_transform(scales)
    generic = extract_linear_output_gradient_sketches(
        load_predictor_checkpoint(runtime / "baseline_l0/best.pt"), atom, angle, fixture_indices,
        transform, dimension=512, sketch_seed=4_000_110,
    ).features
    expected = historical[fixture_positions]
    np.testing.assert_allclose(generic, expected, rtol=2e-5, atol=2e-5)
    expected_positions = lcmd_tp_select(expected[20:], expected[:20], 10).selected_pool_positions
    generic_positions = lcmd_tp_select(generic[20:], generic[:20], 10).selected_pool_positions
    expected_ids = data.iloc[fixture_indices[20:][expected_positions]].sample_id.tolist()
    generic_ids = data.iloc[fixture_indices[20:][generic_positions]].sample_id.tolist()
    assert generic_ids == expected_ids


def test_power_transforms_audit_zero_rows_and_are_deterministic() -> None:
    features = np.array([[0.0, 0.0], [3.0, 4.0], [8.0, 6.0]])
    direction = power_normalize_gradients(features, 0.0)
    tempered_a = power_normalize_gradients(features, 0.5)
    tempered_b = power_normalize_gradients(features.copy(), 0.5)
    assert direction.audit["zero_norm_rows"] == 1
    assert direction.audit["zero_norm_indices"] == [0]
    assert direction.audit["zero_norm_policy"] == "retain_zero_vector"
    assert np.array_equal(direction.features[0], np.zeros(2))
    np.testing.assert_allclose(np.linalg.norm(direction.features[1:], axis=1), 1.0)
    assert np.array_equal(tempered_a.features, tempered_b.features)
    assert np.isfinite(tempered_a.features).all()


def test_whitening_is_finite_symmetric_and_center_width_formula_is_exact() -> None:
    targets = np.array([[1.0, 4.0], [2.0, 8.0], [4.0, 9.0], [7.0, 15.0]])
    transform, audit = output_whitening_transform(targets, (2.0, 5.0))
    inverse_sqrt = np.asarray(audit["inverse_sqrt_covariance"])
    assert np.isfinite(transform).all()
    np.testing.assert_allclose(inverse_sqrt, inverse_sqrt.T, atol=1e-12)
    assert min(audit["eigenvalues_after_flooring"]) >= audit["applied_eigenvalue_floor"]
    cw, cw_audit = center_width_transform(targets)
    center_scale = np.std((targets[:, 0] + targets[:, 1]) / 2, ddof=0)
    width_scale = np.std(targets[:, 1] - targets[:, 0], ddof=0)
    np.testing.assert_allclose(
        cw,
        [[0.5 / center_scale, 0.5 / center_scale], [-1 / width_scale, 1 / width_scale]],
    )
    assert cw_audit["scale_ddof"] == 0


def test_norm_topb_has_canonical_stable_ties() -> None:
    pool = np.array([[3.0, 4.0], [-3.0, -4.0], [2.0, 0.0], [1.0, 0.0]])
    assert gradient_norm_topb(pool, 3).tolist() == [0, 1, 2]
    assert np.array_equal(gradient_norm_topb(pool, 3), gradient_norm_topb(pool.copy(), 3))


def test_all_innovation_methods_are_unique_and_deterministic() -> None:
    rng = np.random.default_rng(2026)
    args = dict(
        raw_gradients=rng.normal(size=(60, 16)),
        latent_features=rng.normal(size=(60, 8)),
        output_whitened_gradients=rng.normal(size=(60, 16)),
        center_width_gradients=rng.normal(size=(60, 16)),
        l0_count=10,
        batch_size=32,
    )
    first = select_innovation_methods(**args)
    second = select_innovation_methods(**args)
    assert tuple(first.positions) == METHODS
    for method in METHODS:
        assert len(first.positions[method]) == len(np.unique(first.positions[method])) == 32
        assert np.array_equal(first.positions[method], second.positions[method])


def test_latent_lcmd_and_gradient_farthest_first_return_unique_rows() -> None:
    rng = np.random.default_rng(15)
    args = dict(
        raw_gradients=rng.normal(size=(50, 7)),
        latent_features=rng.normal(size=(50, 5)),
        output_whitened_gradients=rng.normal(size=(50, 7)),
        center_width_gradients=rng.normal(size=(50, 7)),
        l0_count=10,
        batch_size=20,
    )
    result = select_innovation_methods(**args)
    for method in ("latent_lcmd", "gradient_farthest_first"):
        assert len(result.positions[method]) == len(set(result.positions[method])) == 20


def test_l0_only_transforms_ignore_mutated_u0_and_test_truth(tmp_path) -> None:
    identities = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(50)],
        "canonical_smiles": [f"m{i % 9}" for i in range(50)],
    })
    partition = make_row_protocol(identities, 73)
    original = identities.assign(V1_ml=np.arange(50, dtype=float) + 1,
                                 V2_ml=np.arange(50, dtype=float) * 2 + 3)
    mutated = original.copy()
    hidden = partition.role.isin(("u0", "validation", "test")).to_numpy()
    mutated.loc[hidden, ["V1_ml", "V2_ml"]] = 1e12
    paths = []
    for name, frame in (("original", original), ("mutated", mutated)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    first, audit_first = reveal_l0_targets_only(paths[0], partition)
    second, audit_second = reveal_l0_targets_only(paths[1], partition)
    assert np.array_equal(first, second)
    assert audit_first == audit_second
    np.testing.assert_allclose(output_whitening_transform(first, (2.0, 4.0))[0],
                               output_whitening_transform(second, (2.0, 4.0))[0])
    np.testing.assert_allclose(center_width_transform(first)[0], center_width_transform(second)[0])


def test_confirmation_seed_is_rejected_before_truth_access() -> None:
    with pytest.raises(PermissionError, match="confirmation"):
        validate_screen_seed(157)
    with pytest.raises(ValueError, match="registered development"):
        validate_screen_seed(29)
