"""Bounded offline engineering checks; never a scientific performance benchmark."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import torch
from torch import nn

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))
from lora_adaptation import (ExactMLLHead, LoRAConfig, Projection, attach_native_lora,
                             fit_exact_mll)
from text_embeddings import (EncoderConfig, LocalTextEncoder, MODEL_REVISIONS,
                              masked_average_pool)


class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, hidden):
        return hidden + torch.tanh(self.linear(hidden)) / 4


class ToyEncoder(nn.Module):
    """Random small network for plumbing only; no pretrained language knowledge."""
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(32, 8)
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([ToyBlock(8) for _ in range(4)])
        self.requires_grad_(False)

    def forward(self, inputs):
        hidden = self.tokens(inputs["input_ids"])
        for block in self.encoder.block:
            hidden = block(hidden)
        return masked_average_pool(hidden, inputs["attention_mask"]).double()


def _parameter_digest(model, *, frozen_only=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not frozen_only or not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _group_snapshot(module, kind):
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()
            if ("lora_" in name) == (kind == "lora") and parameter.requires_grad}


def run_gradient_variant(base, inputs, targets, *, variant, seed, steps=3, hidden_size=8):
    torch.manual_seed(seed)
    encoder = copy.deepcopy(base)
    lora = None
    if "lora" in variant:
        # The zero-dropout fixture makes finite differences deterministic.
        lora = attach_native_lora(encoder, LoRAConfig(dropout=0.0))
    # Keep projection initialization identical across both projected variants.
    torch.manual_seed(seed + 10000)
    projection = Projection(hidden_size, min(64, hidden_size)) if "projection" in variant else None
    gp = ExactMLLHead()
    frozen_digest = _parameter_digest(encoder, frozen_only=True)
    lora_before = _group_snapshot(encoder, "lora")
    projection_before = {} if projection is None else {name: p.detach().clone() for name, p in projection.named_parameters()}
    gp_before = {name: p.detach().clone() for name, p in gp.named_parameters()}
    trace = fit_exact_mll(encoder, inputs, targets, gp, projection=projection, steps=steps)
    groups = {}
    for group_name, module, snapshot in (("lora", encoder, lora_before),
                                          ("projection", projection, projection_before),
                                          ("gp", gp, gp_before)):
        if module is None or not snapshot:
            groups[group_name] = {"applicable": False}
            continue
        current = dict(module.named_parameters())
        groups[group_name] = {
            "applicable": True,
            "gradients": {name: {"finite": current[name].grad is not None and bool(torch.isfinite(current[name].grad).all()),
                                 "norm": None if current[name].grad is None else float(current[name].grad.norm()),
                                 "max_parameter_update": float((current[name].detach() - old).abs().max())}
                          for name, old in snapshot.items()},
        }
    checks = {
        "frozen_base_bitwise_unchanged": frozen_digest == _parameter_digest(encoder, frozen_only=True),
        "frozen_base_has_no_gradients": all(p.grad is None for p in encoder.parameters() if not p.requires_grad),
        "full_covariance_rows_each_step": all(item["diagnostics_before_step"]["n_train"] == len(targets) for item in trace["history"]),
        "cholesky_success_each_step": all(item["diagnostics_before_step"]["cholesky_info"] == 0 for item in trace["history"]),
    }
    for name, group in groups.items():
        if group["applicable"]:
            values = list(group["gradients"].values())
            checks[f"{name}_all_gradients_finite"] = all(item["finite"] for item in values)
            checks[f"{name}_nonzero_gradient"] = any(item["norm"] > 0 for item in values)
            checks[f"{name}_updates"] = any(item["max_parameter_update"] > 0 for item in values)
    if not all(checks.values()):
        raise AssertionError(f"Gradient-flow failure: {checks}")
    return {"variant": variant, "seed": seed, "lora": lora,
            "checks": checks, "parameter_groups": groups, "training": trace}


def toy_preflight(steps=3):
    torch.manual_seed(20261106)
    base = ToyEncoder()
    tokens = {"input_ids": torch.tensor([[2, 4, 5, 0], [3, 5, 7, 2], [4, 1, 0, 0],
                                          [7, 8, 9, 0], [11, 2, 6, 4], [12, 1, 0, 0],
                                          [13, 5, 2, 0], [14, 3, 8, 1]]),
              "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0],
                                               [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0],
                                               [1, 1, 1, 0], [1, 1, 1, 1]])}
    targets = torch.tensor([-0.7, 0.3, -0.2, 1.2, 0.8, -0.1, 0.1, -1.1], dtype=torch.float64)
    variants = [run_gradient_variant(base, tokens, targets, variant=variant, seed=seed, steps=steps)
                for seed in (17, 29)
                for variant in ("frozen", "projection", "lora", "lora_projection")]
    return {"status": "passed_engineering_preflight", "record_kind": "toy_random_network_not_T5_or_performance_evidence",
            "n_rows": len(targets), "variants": variants,
            "performance_claims_allowed": False, "pretrained_weights_loaded": False}


def pretrained_preflight(model_id, kind, revision, steps=3):
    settings = EncoderConfig(model_id_or_path=model_id, kind=kind, revision=revision)
    try:
        encoder = LocalTextEncoder(settings)
    except (OSError, ValueError, ImportError) as error:
        return {"status": "blocked_offline_model_load", "record_kind": "pretrained_engineering_preflight_not_executed",
                "model": model_id, "requested_revision": revision or MODEL_REVISIONS.get(model_id),
                "exception_type": type(error).__name__, "reason": str(error),
                "local_files_only": True, "model_downloads": 0,
                "pretrained_weights_loaded": False, "performance_claims_allowed": False}
    # These artificial labels check gradients only and are never scientific data.
    texts = ["Molecule (canonical SMILES): CCO", "Molecule (canonical SMILES): c1ccccc1",
             "Molecule (canonical SMILES): CC(=O)O", "Molecule (canonical SMILES): CCN"]
    tokens = encoder.tokenize(texts)
    targets = torch.tensor([-1.2, 0.5, 0.8, -0.1], dtype=torch.float64)
    first, second = encoder.encode_frozen(texts), encoder.encode_frozen(texts)
    results = [run_gradient_variant(encoder, tokens, targets, variant=variant, seed=17,
                                    steps=steps, hidden_size=encoder.hidden_size)
               for variant in ("frozen", "projection", "lora", "lora_projection")]
    return {"status": "passed_pretrained_engineering_preflight", "model_metadata": encoder.metadata,
            "frozen_repeat_bitwise_equal": bool((first == second).all()),
            "record_kind": "artificial_targets_gradient_flow_only", "variants": results,
            "pretrained_weights_loaded": True, "performance_claims_allowed": False}


def resource_audit():
    packages = ["torch", "transformers", "peft", "gpytorch", "botorch", "sentencepiece",
                "safetensors", "numpy", "scipy", "pytest"]
    cache_paths = {Path.home() / ".cache/huggingface/hub"}
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(variable):
            path = Path(os.environ[variable]).expanduser()
            cache_paths.add(path / "hub" if variable == "HF_HOME" else path)
    inventory = []
    for cache in sorted(cache_paths):
        if cache.exists():
            for model in sorted(cache.glob("models--*")):
                weights = [p for p in model.glob("snapshots/*/*") if p.is_file() and p.suffix in {".safetensors", ".bin"}]
                inventory.append({"cache": str(cache), "model": model.name,
                                  "weight_files": [{"path": str(p), "bytes": p.stat().st_size} for p in weights]})
    return {"audit_date": "2026-11-06", "python": sys.executable,
            "python_version": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) if importlib.util.find_spec(name) else None for name in packages},
            "device": {"cuda_available": torch.cuda.is_available(), "mps_built": torch.backends.mps.is_built(),
                       "mps_available": torch.backends.mps.is_available(), "torch_threads": torch.get_num_threads()},
            "disk_free_gib": shutil.disk_usage(STUDY_DIR).free / 2**30,
            "model_cache_inventory": inventory, "model_downloads": 0,
            "dependency_installs": 0, "paid_api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--kind", choices=("t5", "qwen3_embedding"), default="t5")
    parser.add_argument("--revision")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.toy == bool(args.model_id):
        parser.error("Choose exactly one of --toy or --model-id")
    if not 2 <= args.steps <= 10:
        parser.error("Engineering preflight requires 2 to 10 steps")
    torch.set_num_threads(1)
    start = time.perf_counter()
    receipt = toy_preflight(args.steps) if args.toy else pretrained_preflight(args.model_id, args.kind, args.revision, args.steps)
    receipt["resource_audit"] = resource_audit()
    receipt["elapsed_seconds"] = time.perf_counter() - start
    # macOS ru_maxrss is bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt["process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output),
                      "seconds": receipt["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
