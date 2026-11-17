"""Offline encoders with audited GOLLuM T5 and Qwen3 embedding pooling."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


STUDY_DIR = Path(__file__).resolve().parent
LEGACY_ENCODER = STUDY_DIR.parent / "representation_gp_benchmark_20261104" / "llm_embeddings.py"
MODEL_REVISIONS = {
    "google-t5/t5-base": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}


def render_experiment_text(record, variant="conditions", template="A"):
    """Reuse the audited feature-only renderer, including both volume fields."""
    spec = importlib.util.spec_from_file_location("gollum_legacy_text_renderer", LEGACY_ENCODER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_experiment_text(record, variant=variant, template=template)


def masked_average_pool(hidden, attention_mask):
    """Match upstream average_pool: zero padding, sum, divide by token count."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden-state and attention-mask shapes disagree")
    counts = attention_mask.sum(dim=1)
    if (counts <= 0).any():
        raise ValueError("Empty token sequences cannot be pooled")
    return hidden.masked_fill(~attention_mask[..., None].bool(), 0.0).sum(1) / counts[:, None]


def last_token_pool(hidden, attention_mask):
    """Pool the final nonpadding token with either padding side."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden-state and attention-mask shapes disagree")
    indices = torch.arange(attention_mask.shape[1], device=hidden.device)
    last = indices.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(1).values
    if (last < 0).any():
        raise ValueError("Empty token sequences cannot be pooled")
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


def embedding_identity(row_ids, texts, metadata):
    """Bind ordered rows, exact text, revision, pooling and numeric policy."""
    if len(row_ids) != len(texts) or len(set(map(str, row_ids))) != len(row_ids):
        raise ValueError("Embedding identity requires unique, aligned row IDs")
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("Embedding text must be nonempty")
    payload = {"row_ids": list(map(str, row_ids)), "texts": list(texts), "encoder": metadata}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EncoderConfig:
    model_id_or_path: str = "google-t5/t5-base"
    kind: str = "t5"
    revision: str | None = None
    max_tokens: int = 512
    device: str = "cpu"
    chunk_size: int = 4


class LocalTextEncoder(nn.Module):
    """Load pretrained local files only; forward preserves the adaptation graph.

    T5 uses its encoder, dropout disabled and unnormalized masked mean pooling.
    Qwen3 uses its last nonpadding EOS state followed by L2 normalization.
    Chunking only limits transformer forward size; exact MLL still uses all rows.
    """

    def __init__(self, settings: EncoderConfig):
        super().__init__()
        if settings.kind not in {"t5", "qwen3_embedding"}:
            raise ValueError("Only audited T5 and Qwen3 embedding architectures are allowed")
        if settings.max_tokens < 1 or settings.chunk_size < 1:
            raise ValueError("Token and chunk limits must be positive")
        # Both flags and environment guards prevent implicit Hub access.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel

        self.settings = settings
        local_path = Path(settings.model_id_or_path).is_dir()
        revision = settings.revision or (None if local_path else MODEL_REVISIONS.get(settings.model_id_or_path))
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        if revision is not None:
            kwargs["revision"] = revision
        start = time.perf_counter()
        config = AutoConfig.from_pretrained(settings.model_id_or_path, **kwargs)
        expected = "t5" if settings.kind == "t5" else "qwen3"
        if config.model_type != expected:
            raise ValueError(f"Expected architecture {expected}; found {config.model_type}")
        if settings.kind == "t5":
            # This is the explicit GOLLuM encoder configuration, not T5 generation.
            config.dropout_rate = 0.0
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_id_or_path, use_fast=True, **kwargs)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer has no padding or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" if settings.kind == "t5" else "left"
        model_class = T5EncoderModel if settings.kind == "t5" else AutoModel
        self.model, info = model_class.from_pretrained(
            settings.model_id_or_path, config=config, dtype=torch.float32,
            use_safetensors=True, output_loading_info=True, **kwargs)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError("Incomplete pretrained weights; random initialization is forbidden")
        unexpected = info.get("unexpected_keys", [])
        allowed = lambda name: name == "lm_head.weight" or (
            settings.kind == "t5" and name.startswith("decoder."))
        if any(not allowed(name) for name in unexpected):
            raise ValueError(f"Unaudited unused checkpoint keys: {unexpected}")
        self.model = self.model.to(settings.device).float()
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = int(config.d_model if settings.kind == "t5" else config.hidden_size)
        self.context_limit = int(getattr(config, "max_position_embeddings",
                                         getattr(config, "n_positions", 512)))
        self.eos_ids = {int(x) for x in (self.tokenizer.eos_token_id, config.eos_token_id) if x is not None}
        self.metadata = {
            "model": settings.model_id_or_path, "requested_revision": revision,
            "loaded_commit": getattr(config, "_commit_hash", None), "kind": settings.kind,
            "local_files_only": True, "trust_remote_code": False,
            "pooling": "attention_masked_mean" if settings.kind == "t5" else "last_nonpadding_eos_l2",
            "hidden_size": self.hidden_size, "max_tokens": settings.max_tokens,
            "truncation": False, "chat_template": False, "generate_calls": 0,
            "padding_side": self.tokenizer.padding_side, "forward_dtype": "float32",
            "gp_output_dtype": "float64", "gp_output_device": "cpu", "device": settings.device,
            "base_dropout": float(config.dropout_rate) if settings.kind == "t5" else float(config.attention_dropout),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "loading_info": info, "load_seconds": time.perf_counter() - start,
        }

    def tokenize(self, texts):
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("A nonempty text sequence is required")
        raw = self.tokenizer(list(texts), padding=False, truncation=False, add_special_tokens=True)
        lengths = [len(ids) for ids in raw["input_ids"]]
        if max(lengths) > min(self.settings.max_tokens, self.context_limit):
            raise ValueError("Input exceeds the frozen token limit; truncation is forbidden")
        if self.settings.kind == "qwen3_embedding" and any(ids[-1] not in self.eos_ids for ids in raw["input_ids"]):
            raise ValueError("The audited Qwen3 tokenizer must append EOS before last-token pooling")
        batch = self.tokenizer(list(texts), padding=True, truncation=False,
                               add_special_tokens=True, return_tensors="pt")
        return {key: batch[key].to(self.settings.device) for key in ("input_ids", "attention_mask")}

    def forward(self, tokens):
        chunks = []
        for start in range(0, len(tokens["input_ids"]), self.settings.chunk_size):
            part = {key: value[start:start + self.settings.chunk_size] for key, value in tokens.items()}
            if self.settings.kind == "qwen3_embedding":
                part["use_cache"] = False
            hidden = self.model(**part).last_hidden_state
            if self.settings.kind == "t5":
                pooled = masked_average_pool(hidden, part["attention_mask"])
            else:
                pooled = F.normalize(last_token_pool(hidden, part["attention_mask"]), p=2, dim=1)
            # Casting preserves autograd; no detach or NumPy in adaptation.
            chunks.append(pooled.to(device="cpu", dtype=torch.float64))
        result = torch.cat(chunks, dim=0)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite encoder output")
        return result

    def encode_frozen(self, texts):
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Frozen extraction cannot be used after enabling LoRA")
        self.eval()
        with torch.no_grad():
            return self(self.tokenize(texts)).cpu().numpy()
