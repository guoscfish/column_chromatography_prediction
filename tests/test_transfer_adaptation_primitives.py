import torch
from torch import nn

from src.qgeognn_al.transfer.adaptation import (
    build_discriminative_optimizer,
    l2_sp_penalty,
    parameter_drift,
    fit_target_scales,
    quantile_target_loss,
    scaled_quantile_target_loss,
)


class TinyTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.convs = nn.ModuleList([nn.Linear(2, 2) for _ in range(5)])
        self.head = nn.Linear(2, 6)


def test_scaled_loss_preserves_quantile_contract():
    true = torch.tensor([1.0, 2.0])
    pred = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    expected = (quantile_target_loss(true, pred) - torch.mean((true - pred[:, 1]) ** 2)) / 2
    expected = expected + torch.mean((true - pred[:, 1]) ** 2) / 4
    assert torch.allclose(scaled_quantile_target_loss(true, pred, 2.0), expected)


def test_discriminative_groups_and_l2_sp_are_source_anchored():
    model = TinyTransfer()
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = build_discriminative_optimizer(model, head_lr=3e-4, late_lr=2e-5, early_lr=1e-5)
    assert [group["name"] for group in optimizer.param_groups] == ["head", "late", "early"]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 2e-5, 1e-5]
    source = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert float(l2_sp_penalty(model, source)) == 0.0
    with torch.no_grad():
        model.head.weight.add_(1.0)
    assert parameter_drift(model, source) > 0
    assert float(l2_sp_penalty(model, source)) > 0


def test_target_scales_use_explicit_training_indices_only():
    class Row:
        def __init__(self, y):
            self.y = torch.tensor(y)
    rows = [Row([1.0, 10.0]), Row([3.0, 14.0]), Row([99.0, 999.0])]
    scales = fit_target_scales(rows, [0, 1])
    assert scales == {"V1": 1.0, "V2": 2.0}
