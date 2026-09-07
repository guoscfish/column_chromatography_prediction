"""Matched target-only and source-anchored transfer of the qualified V2.

No change to geometry, sum pooling, condition completion, six-output head or
loss. Source replay holds BN running statistics fixed, but retains gradients.
"""
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn
from torch_geometric.nn import global_add_pool

from .baseline import set_training_mode
from ..models import build_predictor

METHODS = {
    "standard_shallow_finetune": ("shallow", False),
    "standard_full_finetune": ("full", False),
    "source_anchored_shallow": ("shallow", True),
    "source_anchored_full": ("full", True),
}


class SourceAnchoredTransfer(nn.Module):
    def __init__(self, source, scope):
        super().__init__()
        if scope not in ("shallow", "full"):
            raise ValueError(scope)
        # RBF primitives own non-leaf construction tensors, so reconstruct the
        # exact qualified module and load its state instead of deepcopying it.
        copied = build_predictor(source.condition_branch.normalization)
        copied.load_state_dict(source.state_dict(), strict=True)
        self.backbone = copied.backbone
        self.condition_branch = copied.condition_branch
        self.source_head = deepcopy(source.head)
        self.target_head = deepcopy(source.head)
        self.scope = scope
        prefixes = (("backbone.", "condition_branch.", "target_head.") if scope == "full"
                    else ("backbone.convs.4.", "condition_branch.", "target_head."))
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
        self.eval()

    def extract_representation(self, atom, angle):
        pooled = global_add_pool(self.backbone(atom, angle), atom.batch)
        residual, _ = self.condition_branch(atom)
        return pooled + residual

    def forward(self, atom, angle, task="target"):
        if task not in ("target", "source"):
            raise ValueError(task)
        head = self.target_head if task == "target" else self.source_head
        output = head(self.extract_representation(atom, angle))
        return output if self.training else output.clamp(0, 1e8)

    def training_target(self):
        set_training_mode(self)
        self.source_head.eval()

    @contextmanager
    def replay_mode(self):
        """Eval BN only: do not detach or disable source-loss autograd."""
        states = [(m, m.training) for m in self.modules()
                  if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        try:
            for module, _ in states:
                module.eval()
            yield
        finally:
            for module, training in states:
                module.train(training)

    def parameter_inventory(self):
        selected = {n: p.numel() for n, p in self.named_parameters() if p.requires_grad}
        return {"trainable_parameters": sum(selected.values()),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_named_parameters": selected}

    def parameter_drift(self, initial):
        result = {}
        for prefix in ("backbone", "condition_branch", "target_head", "source_head"):
            squared = sum(float(torch.square(p.detach().double()-initial[n].double()).sum())
                          for n, p in self.named_parameters() if n.startswith(prefix+"."))
            result[prefix+"_l2_drift"] = squared ** .5
        return result
