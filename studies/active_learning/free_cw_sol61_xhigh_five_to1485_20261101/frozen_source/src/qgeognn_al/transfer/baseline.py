"""Adaptation scopes for the standalone predictor; no acquisition or head replacement."""
import torch


def configure_trainable(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "head_only":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    elif mode == "last2":
        prefixes = ("head.", "backbone.convs.3.", "backbone.convs.4.",
                    "backbone.convs_bond_angle.3.", "backbone.convs_bond_float.3.",
                    "backbone.convs_bond_embeding.3.", "backbone.convs_angle_float.3.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
    elif mode != "full":
        raise ValueError(mode)
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def set_training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                module.eval()
