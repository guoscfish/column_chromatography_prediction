"""Study-local identity, feature-only input, and immutable receipts.

No function in this module accepts or reads validation/test targets.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/epig_matplotlib")

import numpy as np
import pandas as pd

from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, ids_hash, stable_hash
from qgeognn_al.active_learning_v2.benchmark_protocol import load_features, FEATURE_COLUMNS
from qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE

BASE = ROOT / "studies/active_learning/qgeognn_v2_row_sequential_b32"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"immutable receipt differs: {path}")
    else:
        dump(path, value)


def partition(seed):
    path = BASE / f"splits/row_seed_{int(seed)}.csv"
    from qgeognn_al.active_learning_v2.protocol import validate_row_protocol
    split = pd.read_csv(path, dtype={"sample_id": str})
    validate_row_protocol(split)
    if set(split.outer_seed) != {int(seed)}:
        raise ValueError("split seed drift")
    return split


def training_inputs(seed, *, feature_frame=None):
    """Read only L333 labels, with the original store and a persisted audit."""
    frame = load_features() if feature_frame is None else feature_frame
    split = partition(seed)
    if frame.sample_id.tolist() != split.sample_id.tolist():
        raise ValueError("source/split identity drift")
    if not np.array_equal(split.canonical_index.to_numpy(int), np.arange(len(frame))):
        raise ValueError("split canonical row order drift")
    indices = split.loc[split.role.eq("l0"), "canonical_index"].to_numpy(int)
    store = RestrictedLabelStore(SOURCE_DATA, split)
    truth = store.reveal(frame.iloc[indices].sample_id, "initial_fit").astype(np.float64)
    record = {"seed": seed, "calls": store.audit, "training_rows": len(indices),
              "training_ids_hash": ids_hash(frame.iloc[indices].sample_id),
              "validation_labels_read": 0, "future_U_labels_read": 0,
              "test_labels_read": 0, "legacy_store_float32_then_float64": True}
    write_once(HERE / f"label_access/qualification_seed_{seed}.json", record)
    return frame, split, indices, truth


def feature_matrix():
    """29 raw, response-independent coordinates; all scaling remains fold-local."""
    import torch
    from qgeognn_al.data import eluent_descriptor
    baseline = read(BASE / "protocol.json")
    if digest(SOURCE_DATA) != baseline["source_sha256"] or digest(SOURCE_GRAPH_CACHE) != baseline["graph_cache_sha256"]:
        raise RuntimeError("canonical source/cache drift")
    frame = load_features()
    if set(frame) != set(FEATURE_COLUMNS):
        raise PermissionError("feature schema contains unexpected columns")
    graphs = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    molecular = np.vstack([graphs[s]["descriptor"] for s in frame.canonical_smiles]).astype(np.float64)
    eluents = np.vstack([eluent_descriptor(v) for v in frame["PE/EA"]]).astype(np.float64)
    solvent = np.column_stack([frame["loading solvent"].eq(v).to_numpy(float) for v in ("PE", "EA", "DCM")])
    if not np.all(solvent.sum(1) == 1):
        raise ValueError("unknown loading solvent")
    amount = frame["Density g/ml"].to_numpy(float) * frame["V/ul"].to_numpy(float)
    volume = frame["Volume of loading solvent/ul"].to_numpy(float)
    if np.any(amount < 0) or np.any(volume < 0):
        raise ValueError("negative amount/volume")
    x = np.column_stack([molecular, eluents, solvent, amount, volume, np.log1p(amount), np.log1p(volume)])
    if x.shape != (4163, 29) or not np.isfinite(x).all():
        raise ValueError(f"nonfinite or unexpected raw features: {x.shape}")
    names = ([f"cached_molecular_descriptor_{i}" for i in range(16)] +
             [f"eluent_descriptor_{i}" for i in range(6)] + ["solvent_PE", "solvent_EA", "solvent_DCM",
              "loading_amount_density_times_ul", "loading_solvent_volume_ul", "log1p_amount", "log1p_volume"])
    receipt = {"shape": list(x.shape), "dtype": str(x.dtype), "columns": names,
               "source_sha256": digest(SOURCE_DATA), "graph_cache_sha256": digest(SOURCE_GRAPH_CACHE),
               "features_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
               "labels_read": 0, "standardization": "none here; each surrogate fold fits its own transform"}
    write_once(HERE / "features.json", receipt)
    return frame, x


def reference_positions(seed, distribution, allowed_positions, row_ids, size):
    values = [int(i) for i in allowed_positions]
    if len(set(values)) != len(values):
        raise ValueError("duplicate reference population")
    ranked = sorted(values, key=lambda i: hashlib.sha256(f"epig_v1:{seed}:{distribution}:{row_ids[i]}".encode()).digest())
    return np.asarray(ranked[:min(int(size), len(ranked))], dtype=int)


def implementation_paths():
    paths = list((ROOT / "src/qgeognn_al").rglob("*.py"))
    paths += list((ROOT / "application").glob("*.py"))
    # The later reporting program is sealed separately; it cannot change decisions.
    paths += [p for p in HERE.glob("*.py") if p.name != "report_study.py"]
    paths += [HERE / "config.json", HERE / "EXPERIMENT_PROTOCOL.md"]
    paths += [ROOT / "tests/active_learning_v2" / name for name in ("test_epig.py", "test_epig_models.py", "test_epig_study.py")]
    return sorted(set(paths))


def dependency_paths():
    paths = [BASE / "protocol.json"]
    for seed in read(HERE / "config.json")["seeds"]:
        paths += [BASE / f"splits/row_seed_{seed}.csv"]
        runtime = BASE / f"runtime/seed_{seed}"
        paths += [runtime / "context.json", runtime / "scaler.json",
                  runtime / "lcmd/round_00/input_contract.json"]
        paths += [runtime / f"lcmd/round_00/model/{name}" for name in ("best.pt", "fit_audit.json", "history.csv", "predictions.csv.gz")]
    return paths


def freeze():
    path = HERE / "freeze.json"
    if path.exists():
        return verify_freeze()
    if list((HERE / "qualification").glob("*/complete.json")):
        raise RuntimeError("cannot preregister after qualification scores")
    test = read(HERE / "test_receipt.json")
    if test["exit_code"] != 0:
        raise RuntimeError("passing mathematical/engineering tests required")
    record = {"schema": "epig-freeze-v1", "created_unix": time.time(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "files": {str(p.relative_to(ROOT)): digest(p) for p in implementation_paths()},
              "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in dependency_paths()},
              "test_receipt_sha256": digest(HERE / "test_receipt.json"),
              "source_sha256": digest(SOURCE_DATA), "graph_cache_sha256": digest(SOURCE_GRAPH_CACHE),
              "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn")},
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "results_before_freeze": {"qualification": 0, "new_gnn_fits": 0, "new_acquisitions": 0}}
    write_once(path, record)
    return record


def verify_freeze():
    record = read(HERE / "freeze.json")
    for relative, expected in {**record["files"], **record["dependencies"]}.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"frozen implementation/config drift: {relative}")
    if digest(SOURCE_DATA) != record["source_sha256"] or digest(SOURCE_GRAPH_CACHE) != record["graph_cache_sha256"]:
        raise RuntimeError("data dependency drift")
    current = {name: importlib.metadata.version(name) for name in record["packages"]}
    if current != record["packages"]:
        raise RuntimeError("frozen scientific environment drift")
    return record
