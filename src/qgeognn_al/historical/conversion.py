"""Explicit migration from the retained pruned diagnostic checkpoint schema."""
from collections import OrderedDict

import torch


def original_key(key):
    if key.startswith("backbone."):
        return "legacy_model.gnn_node." + key[len("backbone."):]
    if key.startswith("head."):
        return "legacy_model.graph_pred_linear." + key[len("head."):]
    return key


def convert_pruned_to_standalone(state, standalone):
    expected = standalone.state_dict()
    keys = {original_key(k) for k in expected}
    missing, unexpected = sorted(keys - set(state)), sorted(set(state) - keys)
    if missing or unexpected:
        raise ValueError(f"pruned migration missing={missing}, unexpected={unexpected}")
    converted = OrderedDict()
    for key, template in expected.items():
        value = state[original_key(key)]
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(f"migration shape/dtype mismatch: {key}")
        converted[key] = value.detach().clone()
    report = {"status": "PASS", "missing_keys": missing, "unexpected_keys": unexpected,
              "key_mapping": {original_key(k): k for k in converted},
              "retained_values_bitwise_equal": all(torch.equal(v, state[original_key(k)]) for k, v in converted.items())}
    return converted, report
