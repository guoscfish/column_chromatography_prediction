"""Native rsLoRA and float64 full-batch exact GP MLL engineering preflight.

This is an implementation scaffold, not a validated scientific GP trainer.
It has no model downloads and does not depend on PEFT or GPyTorch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import time

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.2
    target_ratio: float = 0.25
    from_top: bool = True


class RSLoRALinear(nn.Module):
    """Frozen W plus alpha/sqrt(r) BA, with standard zero-B initialization."""

    def __init__(self, base: nn.Linear, settings: LoRAConfig):
        super().__init__()
        if settings.rank < 1 or settings.alpha <= 0 or not 0 <= settings.dropout < 1:
            raise ValueError("Invalid rsLoRA settings")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(settings.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, settings.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = settings.alpha / math.sqrt(settings.rank)
        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, inputs):
        adaptation = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + self.scale * adaptation


def target_layer_names(model, settings=LoRAConfig()):
    """Match upstream: all linear modules in the top rounded layer fraction."""
    if not 0 < settings.target_ratio <= 1:
        raise ValueError("target_ratio must be in (0, 1]")
    found = []
    for name, module in model.named_modules():
        matched = re.search(r"(?:block|layers)\.(\d+)\.", name)
        if matched and isinstance(module, nn.Linear):
            found.append((int(matched.group(1)), name))
    indices = sorted({index for index, _ in found}, reverse=settings.from_top)
    if not indices:
        raise ValueError("No audited transformer block/layers linear modules found")
    chosen = set(indices[:max(1, round(len(indices) * settings.target_ratio))])
    return [name for index, name in found if index in chosen]


def attach_native_lora(model, settings=LoRAConfig()):
    """Install rsLoRA without adding or updating any base parameter."""
    if any(isinstance(module, RSLoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already attached")
    selected = target_layer_names(model, settings)
    model.requires_grad_(False)
    for name in selected:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, RSLoRALinear(getattr(parent, leaf), settings))
    return {"backend": "native_torch_rsLoRA", "settings": asdict(settings),
            "target_modules": selected,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}


class Projection(nn.Module):
    """GOLLuM linear+ELU, Xavier weights and bias=0.01; no active dropout."""

    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, dtype=torch.float64)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.01)

    def forward(self, inputs):
        return F.elu(self.linear(inputs.to(torch.float64)))


class ExactMLLHead(nn.Module):
    """Isotropic Matérn-5/2 and Gaussian likelihood with a constant mean."""

    def __init__(self, noise_floor=1e-4, jitter=1e-8):
        super().__init__()
        if noise_floor <= 0 or jitter <= 0:
            raise ValueError("Positive noise floor and jitter are required")
        self.log_lengthscale = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_signal = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_noise_excess = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.mean = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.noise_floor, self.jitter = float(noise_floor), float(jitter)
        self.last_diagnostics = {}

    @property
    def noise_variance(self):
        return self.log_noise_excess.exp() + self.noise_floor

    def covariance(self, features):
        x = features.to(torch.float64)
        if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
            raise ValueError("Exact MLL requires at least two finite feature rows")
        distance = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
        radius = math.sqrt(5) * distance / self.log_lengthscale.exp()
        kernel = self.log_signal.exp() * (1 + radius + radius.square() / 3) * torch.exp(-radius)
        eye = torch.eye(len(x), dtype=torch.float64, device=x.device)
        return kernel + eye * (self.noise_variance + self.jitter)

    def forward(self, features, targets):
        if targets.ndim != 1 or len(targets) != len(features) or not torch.isfinite(targets).all():
            raise ValueError("Exact MLL targets must be one finite value per full training row")
        covariance = self.covariance(features)
        lower, info = torch.linalg.cholesky_ex(covariance)
        if int(info.item()) != 0:
            raise FloatingPointError(f"Cholesky failure: info={int(info.item())}")
        residual = targets.to(torch.float64) - self.mean
        alpha = torch.cholesky_solve(residual[:, None], lower)[:, 0]
        nll = 0.5 * residual.dot(alpha) + lower.diagonal().log().sum() + 0.5 * len(targets) * math.log(2 * math.pi)
        if not torch.isfinite(nll):
            raise FloatingPointError("Nonfinite full-batch exact marginal likelihood")
        self.last_diagnostics = {"n_train": len(targets), "cholesky_info": int(info.item()),
                                 "jitter": self.jitter, "noise_variance": float(self.noise_variance.detach()),
                                 "signal_variance": float(self.log_signal.exp().detach()),
                                 "lengthscale": float(self.log_lengthscale.exp().detach()),
                                 "full_nll": float(nll.detach()), "gp_dtype": "float64"}
        # GPyTorch ExactMarginalLogLikelihood also normalizes by the row count.
        return nll / len(targets)


def fit_exact_mll(representation, inputs, targets, gp, *, projection=None,
                  steps=3, ft_lr=0.002, gp_lr=0.2, clip_norm=1.0, weight_decay=1e-3):
    """Retain every row/embedding graph and rebuild one NxN covariance per step.

    No minibatch MSE or minibatch GP objective is used. This bounded preflight
    intentionally reports step_limit, not convergence or calibrated probability.
    """
    if steps < 1 or clip_norm <= 0 or ft_lr <= 0 or gp_lr <= 0:
        raise ValueError("Positive training limits are required")
    projection = projection if projection is not None else nn.Identity()
    feature_parameters = [p for module in (representation, projection) for p in module.parameters() if p.requires_grad]
    gp_parameters = list(gp.parameters())
    if set(map(id, feature_parameters)) & set(map(id, gp_parameters)):
        raise ValueError("Optimizer parameter groups must be disjoint")
    groups = [{"params": gp_parameters, "lr": gp_lr, "weight_decay": weight_decay}]
    if feature_parameters:
        groups.insert(0, {"params": feature_parameters, "lr": ft_lr, "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    standardized = (targets.double() - targets.double().mean()) / targets.double().std(unbiased=True).clamp_min(1e-12)
    history, start = [], time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        features = projection(representation(inputs)).to(dtype=torch.float64)
        loss = gp(features, standardized)
        loss.backward()
        all_parameters = feature_parameters + gp_parameters
        gradients = {"feature": float(sum((p.grad.detach().double().square().sum().cpu() for p in feature_parameters if p.grad is not None), torch.tensor(0.)).sqrt()),
                     "gp": float(sum((p.grad.detach().double().square().sum().cpu() for p in gp_parameters if p.grad is not None), torch.tensor(0.)).sqrt())}
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in all_parameters):
            raise FloatingPointError("Nonfinite MLL parameter gradient")
        norm_before = torch.nn.utils.clip_grad_norm_(all_parameters, clip_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        history.append({"step": step + 1, "mean_nll_before_step": float(loss.detach()),
                        "gradient_norm_before_clip": float(norm_before), "gradient_groups": gradients,
                        "diagnostics_before_step": dict(gp.last_diagnostics)})
    return {"status": "step_limit_preflight_not_convergence", "steps": steps,
            "optimizer": "AdamW", "ft_lr": ft_lr, "gp_lr": gp_lr,
            "weight_decay": weight_decay, "clip_norm": clip_norm,
            "clip_position": "after_backward_before_every_optimizer_step",
            "scheduler": "StepLR(gamma=0.95, step_size=1)", "full_batch": True,
            "target_standardization": "training_only_sample_std_ddof1",
            "seconds": time.perf_counter() - start, "history": history}
