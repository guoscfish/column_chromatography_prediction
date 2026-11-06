"""Strictly offline pretrained hidden-state encodings and lossless templates."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import time
from pathlib import Path

import numpy as np


CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent",
              "Volume of loading solvent/ul")
MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


def _number(value, field, *, allow_zero=False):
    if isinstance(value, (bool, type(None))):
        raise ValueError(f"Missing or invalid {field}")
    text = str(value).strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric {field}") from exc
    if not parsed.is_finite() or parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"Nonpositive or nonfinite {field}")
    return text


def render_experiment_text(record, variant="conditions", template="A"):
    """Render feature-only records without rounding or confusing sample/solvent volume.

    Records contain canonical_smiles (or smiles), and optionally a conditions dict.
    Rejecting extra keys prevents accidental inclusion of response-derived fields.
    """
    allowed = {"canonical_smiles", "smiles", "conditions", *CONDITIONS}
    if set(record) - allowed:
        raise ValueError(f"Unknown feature keys: {sorted(set(record) - allowed)}")
    smiles = record.get("canonical_smiles", record.get("smiles"))
    if not isinstance(smiles, str) or not smiles.strip() or any(c in smiles for c in "\r\n"):
        raise ValueError("A nonempty canonical SMILES string is required")
    if "canonical_smiles" in record and "smiles" in record and record["smiles"] != smiles:
        raise ValueError("Conflicting SMILES keys")
    if variant not in {"smiles_only", "conditions"} or template not in {"A", "B"}:
        raise ValueError("Unknown variant or template")
    if variant == "smiles_only":
        return f"Molecule (canonical SMILES): {smiles}" if template == "A" else f"Canonical SMILES of molecule: {smiles}"
    condition = record.get("conditions", {k: record[k] for k in CONDITIONS if k in record})
    if set(condition) != set(CONDITIONS):
        raise ValueError("Exactly the five recorded experimental conditions are required")
    if "conditions" in record and any(k in record for k in CONDITIONS):
        raise ValueError("Mixed nested and flat condition definitions")
    ratio = str(condition["PE/EA"]).strip()
    parts = ratio.split("/")
    if len(parts) != 2:
        raise ValueError("PE/EA must preserve the recorded a/b ratio")
    _number(parts[0], "PE/EA petroleum-ether component", allow_zero=True)
    _number(parts[1], "PE/EA ethyl-acetate component")
    density = _number(condition["Density g/ml"], "density")
    sample_volume = _number(condition["V/ul"], "sample volume")
    solvent_volume = _number(condition["Volume of loading solvent/ul"], "loading solvent volume")
    solvent = condition["loading solvent"]
    if solvent not in {"PE", "EA", "DCM"}:
        raise ValueError("Unknown loading-solvent category")
    # density [g/mL] times sample volume [uL] equals sample mass [mg].
    mass = str(Decimal(density) * Decimal(sample_volume))
    if template == "A":
        return (f"Molecule (canonical SMILES): {smiles}\n"
                "Stationary phase: silica gel\nColumn size: 4 g\n"
                f"Mobile phase petroleum ether/ethyl acetate (PE/EA), recorded ratio: {ratio}\n"
                f"Sample density: {density} g/mL\nSample loading volume: {sample_volume} microliters\n"
                f"Derived sample loading mass (density times sample volume): {mass} mg\n"
                f"Loading solvent: {solvent}\nLoading solvent volume: {solvent_volume} microliters")
    return (f"Canonical SMILES of molecule: {smiles}\n"
            "A 4 g column containing silica gel is used.\n"
            f"The recorded petroleum ether/ethyl acetate mobile-phase ratio (PE/EA) is {ratio}.\n"
            f"The sample has density {density} g/mL and is loaded at a volume of {sample_volume} μL.\n"
            f"Its derived loading mass, density multiplied by sample volume, is {mass} mg.\n"
            f"The loading solvent is {solvent}, with a loading-solvent volume of {solvent_volume} μL.")


class LocalFrozenTextEncoder:
    """Load existing files only; freeze all parameters and never generate text."""

    def __init__(self, model_path_or_id=MODEL_ID, *, max_tokens=512, revision=None):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
        start = time.perf_counter()
        self.torch = torch
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        if revision is not None:
            kwargs["revision"] = revision
        config = AutoConfig.from_pretrained(str(model_path_or_id), **kwargs)
        if config.model_type not in {"qwen2", "t5"}:
            raise ValueError("Only audited Qwen2 and T5 architectures are supported")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path_or_id), use_fast=True, **kwargs)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither padding nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        model_class = T5EncoderModel if config.model_type == "t5" else AutoModel
        self.model, loading_info = model_class.from_pretrained(
            str(model_path_or_id), dtype=torch.float32, use_safetensors=True,
            output_loading_info=True, **kwargs)
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
            raise ValueError("Incomplete or incompatible pretrained weights; random initialization is forbidden")
        # A causal-LM checkpoint can contain an unused output head when loaded as a base model.
        allowed_unused = {"lm_head.weight"} if config.model_type == "qwen2" else set()
        if set(loading_info.get("unexpected_keys", [])) - allowed_unused:
            raise ValueError("Unexpected pretrained weight keys require an explicit architecture audit")
        loading_info = {key: sorted(value) if isinstance(value, set) else value
                        for key, value in loading_info.items()}
        self.model = self.model.to("cpu")
        self.model.eval()
        self.model.requires_grad_(False)
        torch.use_deterministic_algorithms(True)
        self.hidden_size = int(config.d_model if config.model_type == "t5" else config.hidden_size)
        self.context_limit = int(getattr(config, "max_position_embeddings", self.max_tokens))
        self.metadata = {
            "model": str(model_path_or_id), "model_type": config.model_type,
            "model_commit": getattr(config, "_commit_hash", None),
            "requested_revision": revision, "local_files_only": True,
            "trust_remote_code": False, "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "hidden_size": self.hidden_size, "model_dtype": "float32", "output_dtype": "float64",
            "device": "cpu", "pooling": "last_hidden_state_attention_masked_mean",
            "max_tokens": self.max_tokens, "truncation": False, "padding_side": "right",
            "chat_template": False, "generate_calls": 0,
            "frozen_parameters": True, "load_seconds": time.perf_counter()-start,
            "loading_info": loading_info,
            "model_kind": "chat_decoder_pretrained_hidden_state_exploratory" if config.model_type == "qwen2" else "pretrained_text_encoder",
        }

    def encode_texts(self, texts, *, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts or any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("Nonempty text sequence required")
        torch = self.torch
        chunks, token_counts = [], []
        start = time.perf_counter()
        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset:offset+batch_size])
            raw = self.tokenizer(batch, padding=False, truncation=False, add_special_tokens=True)
            lengths = [len(ids) for ids in raw["input_ids"]]
            if max(lengths) > min(self.max_tokens, self.context_limit):
                raise ValueError("Input exceeds frozen token limit; truncation is forbidden")
            encoded = self.tokenizer(batch, padding=True, truncation=False,
                                     add_special_tokens=True, return_tensors="pt")
            with torch.inference_mode():
                arguments = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
                if self.metadata["model_type"] != "t5":
                    arguments["use_cache"] = False
                hidden = self.model(**arguments).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(torch.float64)
                pooled = (hidden.to(torch.float64)*mask).sum(1)/mask.sum(1)
            if tuple(pooled.shape) != (len(batch), self.hidden_size) or not torch.isfinite(pooled).all():
                raise ValueError("Unexpected or nonfinite pretrained embedding")
            chunks.append(pooled.cpu().numpy())
            token_counts.extend(lengths)
        result = np.concatenate(chunks).astype(np.float64, copy=False)
        self.metadata["last_encoding"] = {
            "count": len(texts), "tokens": sum(token_counts), "token_min": min(token_counts),
            "token_max": max(token_counts), "seconds": time.perf_counter()-start,
            "texts_sha256": hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest(),
        }
        return result


def model_file_manifest(model_directory):
    """Hash local model/tokenizer files once when freezing the encoding cache."""
    result = {}
    for path in sorted(Path(model_directory).iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            result[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if not result or not any(k.endswith(".safetensors") for k in result):
        raise ValueError("No complete local safetensors model was found")
    return result
