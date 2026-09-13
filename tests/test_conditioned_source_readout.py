"""Unit contracts for the residual conditioned-source readout modules."""

from __future__ import annotations

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
