"""Content-verified caches; incompatible or partial entries are never reused."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file
from ..training.predictor import atomic_json
from .protocol import stable_hash


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def verify_cache(path, contract):
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".contract.json")
    if not path.exists() and not receipt.exists():
        return False
    if not path.exists() or not receipt.exists():
        raise RuntimeError(f"incomplete cache: {path}")
    saved = json.loads(receipt.read_text())
    if saved["contract_hash"] != stable_hash(contract) or saved["sha256"] != sha256_file(path):
        raise RuntimeError(f"cache contract/content mismatch: {path}")
    return True


def seal_cache(path, contract):
    path = Path(path)
    atomic_json(path.with_suffix(path.suffix + ".contract.json"), {
        "contract": contract, "contract_hash": stable_hash(contract), "sha256": sha256_file(path),
    })
