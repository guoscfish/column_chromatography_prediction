"""Frozen scientific constants and preparation for the small-batch study."""

import importlib.metadata
import json
import platform
import subprocess

import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .protocol import ids_hash, make_row_protocol, stable_hash


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_small_batch_benchmark"
OLD_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_lcmd"
BASE_COMMIT = "8081e8653fd0539554ca42b01482f1493a0dc3fa"
COMMIT_MESSAGE = "study: preregister realistic-batch V2 row active learning benchmark"
DEVELOPMENT_SEEDS = (73, 311, 1297, 4093, 8191)
CONFIRMATION_SEEDS = (157, 887, 2357, 6101, 12203)
ALL_SEEDS = CONFIRMATION_SEEDS + DEVELOPMENT_SEEDS
FEATURE_COLUMNS = ("sample_id", "canonical_smiles", "Density g/ml", "V/ul", "loading solvent",
                   "Volume of loading solvent/ul", "PE/EA")
TRAINING_CONFIG = {
    "model_variant": "qgeognn_v2", "optimizer": "Adam", "learning_rate": 0.001,
    "weight_decay": 0.0, "batch_size": 2048, "maximum_epochs": 1000, "patience": 100,
    "shuffle": "deterministic_each_epoch", "loss_weights": {"V1": 1.0, "V2": 1.0},
    "checkpoint_selection": "validation_combined_normalized_rmse", "test_during_training": False,
    "architecture_changed": False, "head_changed": False, "condition_branch_changed": False,
}


def seed_config(seed, member=0, smoke=False):
    return {
        **TRAINING_CONFIG, "seed": int(seed) + 3_000_017 + member * 1_000_033,
        "outer_seed": int(seed), "cpu_threads": 2,
        **({"maximum_epochs": 2, "patience": 2} if smoke else {}),
    }


def initialization_seed(seed, member=0):
    return int(seed) + 2_000_003 + member * 1_000_033


def sketch_seed(seed):
    return int(seed) + 4_000_037


def label_budget(active, validation, outer_train, total):
    return {"active_label_count": active, "active_label_fraction_of_outer_train": active / outer_train,
            "shared_validation_label_count": validation, "total_observed_label_count": active + validation,
            "total_observed_fraction_of_full_dataset": (active + validation) / total}


def load_features(source=SOURCE_DATA):
    frame = pd.read_csv(source, usecols=list(FEATURE_COLUMNS)).reset_index(drop=True)
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("invalid canonical identities")
    return frame


def code_hashes():
    files = list((ROOT / "src/qgeognn_al").rglob("*.py"))
    files += list((ROOT / "application").glob("*.py"))
    files += [ROOT / "scripts/studies/run_qgeognn_v2_4g_row_small_batch_benchmark.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn")}


def protocol_record():
    return {
        "study": STUDY.name, "base_commit": BASE_COMMIT, "primary_batch": 32,
        "secondary_batches": [16, 64], "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS), "random_controls": 5, "ensemble_K": 3,
        "training": TRAINING_CONFIG, "sketch_dimension": 512, "std_ddof": 0,
        "primary_decision_cohort": "confirmation", "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE), "code_hashes": code_hashes(),
        "packages": package_versions(), "python": platform.python_version(),
    }


def prepare(study=STUDY):
    if (study / "primary_results_freeze.json").exists():
        raise RuntimeError("cannot prepare over completed formal results")
    record = protocol_record()
    qualified = json.loads((ROOT / "studies/predictor/final_4g_qualification/protocol.json").read_text())
    if qualified["model_variant"] != "qgeognn_v2" or qualified["source_sha256"] != record["source_sha256"]:
        raise RuntimeError("canonical source qualification mismatch")
    data = load_features()
    if len(data) != 4163 or data.canonical_smiles.nunique() != qualified["compounds"]:
        raise RuntimeError("canonical row/compound count drift")
    (study / "splits").mkdir(parents=True, exist_ok=True)
    (study / "results").mkdir(exist_ok=True)
    splits = []
    for seed in ALL_SEEDS:
        split = make_row_protocol(data, seed)
        split["study"] = study.name
        path = study / "splits" / f"row_seed_{seed}.csv"
        split.to_csv(path, index=False)
        splits.append({"outer_seed": seed, "sha256": sha256_file(path),
                       "role_counts": split.role.value_counts().to_dict(),
                       **{f"{role}_ids_hash": ids_hash(split.loc[split.role.eq(role), "sample_id"])
                          for role in ("l0", "u0", "validation", "test")}})
    atomic_json(study / "protocol.json", record)
    atomic_json(study / "splits/split_manifest.json", {"protocol_hash": stable_hash(record), "splits": splits})
    atomic_json(study / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "threads": 2,
        "packages": package_versions(),
        "base_commit": BASE_COMMIT,
    })
    pd.DataFrame([{"batch_size": batch, **label_budget(333 + batch, 416, 3330, 4163)}
                  for batch in (0, 16, 32, 64)]).to_csv(study / "results/label_budget_accounting.csv", index=False)


def assert_formal_authorized(commit, study=STUDY):
    """Require the actual committed preregistration and all frozen input hashes."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("show", "-s", "--format=%s", commit) != COMMIT_MESSAGE:
        raise RuntimeError("formal execution requires Commit A")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    manifest_path = study / "artifact_manifest.json"
    committed = git("show", f"{commit}:{manifest_path.relative_to(ROOT)}")
    manifest = json.loads(manifest_path.read_text())
    if json.loads(committed) != manifest or manifest["phase"] != "PHASE_1_PREREGISTERED":
        raise RuntimeError("preregistration manifest differs from Commit A")
    for relative, digest in manifest["files"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"preregistered artifact drift: {relative}")
    if json.loads((study / "protocol.json").read_text()) != protocol_record():
        raise RuntimeError("frozen code/data/protocol drift")
    return git("rev-parse", commit)
