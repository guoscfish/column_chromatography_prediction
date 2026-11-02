"""Read-only LCMD lineage inspection and sealing of a separate continuation."""

from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import build_predictor, load_predictor_checkpoint, validate_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json, seed_everything
from .benchmark_protocol import initialization_seed, sketch_seed
from .cache import array_hash, verify_cache
from .gradient_features import extract_q50_gradient_sketches, state_dict_hash
from .ivr_study import exclusive_lock, now, read_json
from .protocol import ids_hash, stable_hash
from .runner import hashlib_sha
from .sequential_acquisition import validate_trajectory_transition
from .sequential_protocol import ACTIVE_LABEL_BUDGETS, CONFIRMATION_SEEDS, STUDY as BASELINE, package_versions
from .sequential_runner import SequentialSeedContext, _assert_protected, _partition, _write_json_once
from .strategy_policy import FixedSwitchPolicy


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_lcmd_to_ivr_b32"
IVR = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
POLICY = FixedSwitchPolicy()
METHOD = "lcmd_to_ivr"
ENTRY = ROOT / "scripts/studies/run_qgeognn_v2_row_lcmd_to_ivr_b32.py"
GRADIENT_MODULE = "src/qgeognn_al/active_learning_v2/gradient_features.py"
PHASES = (("early", 333, 525), ("middle", 525, 653), ("late", 653, 1005))


def hashes(paths):
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}


def switch_round():
    return ACTIVE_LABEL_BUDGETS.index(POLICY.switch_active_labels)


def assert_partition(labeled, unlabeled, outer):
    l, u, o = list(labeled), list(unlabeled), list(outer)
    if (len(set(l)) != len(l) or len(set(u)) != len(u) or len(set(o)) != len(o)
            or set(l) & set(u) or set(l) | set(u) != set(o)):
        raise RuntimeError("L/U must uniquely partition the frozen outer universe")


def advance(labeled, unlabeled, selected, outer):
    assert_partition(labeled, unlabeled, outer)
    if len(selected) != 32:
        raise RuntimeError("each continuation acquisition must contain exactly 32 rows")
    selected_set = set(selected)
    next_l = list(labeled) + list(selected)
    next_u = [i for i in unlabeled if i not in selected_set]
    validate_trajectory_transition(labeled, unlabeled, selected, next_l, next_u)
    assert_partition(next_l, next_u, outer)
    return next_l, next_u


def reconstruct(seed):
    """Replay frozen ID transitions, never a model fit or historical acquisition."""
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("seed outside the frozen five-seed cohort")
    schedule = read_json(BASELINE / "protocol.json")["active_label_budgets"]
    if schedule != list(ACTIVE_LABEL_BUDGETS):
        raise RuntimeError("source protocol budget schedule mismatch")
    r_switch = schedule.index(POLICY.switch_active_labels)
    partition = _partition(seed)
    role = lambda name: partition.loc[partition.role.eq(name), "canonical_index"].astype(int).tolist()
    labeled, unlabeled = role("l0"), role("u0")
    outer = labeled + unlabeled
    ids = lambda indices: partition.iloc[indices].sample_id.astype(str).tolist()
    base = BASELINE / "runtime" / f"seed_{seed}/lcmd"
    global_record = read_json(BASELINE / "global_pre_test_freeze.json")
    trajectory = read_json(base / "trajectory_freeze.json")
    if trajectory["test_truth_access_count"] != 0 or trajectory["final_active_labels"] != schedule[-1]:
        raise RuntimeError("source trajectory is not frozen")
    paths = [BASELINE / "protocol.json", BASELINE / "global_pre_test_freeze.json",
             BASELINE / "splits/split_manifest.json", BASELINE / f"splits/row_seed_{seed}.csv",
             base / "trajectory_freeze.json"]
    for r in range(r_switch + 1):
        directory = base / f"round_{r:02d}"
        record = read_json(directory / "contract.json")
        expected_sha = global_record["entries"][f"seed_{seed}/lcmd/round_{r:02d}"]["round_contract_sha256"]
        if sha256_file(directory / "contract.json") != expected_sha:
            raise RuntimeError("source round differs from global freeze")
        inputs = read_json(directory / "input_contract.json")
        if (record["input"] != inputs or inputs["round"] != r or inputs["method"] != "lcmd"
                or inputs["outer_seed"] != seed or inputs["active_label_count"] != schedule[r]
                or record["test_truth_access_count"] != 0):
            raise RuntimeError("source round identity/budget mismatch")
        # State remains usable even when a large model/gradient cache has been removed.
        state_files = {k: v for k, v in record["files"].items()
                       if k.startswith(("labeled_", "unlabeled_", "input_contract", "selected_batch"))}
        _assert_protected(directory, state_files)
        for name, values in (("labeled", labeled), ("unlabeled", unlabeled)):
            if (np.load(directory / f"{name}_indices.npy").tolist() != values
                    or np.load(directory / f"{name}_ids.npy").astype(str).tolist() != ids(values)):
                raise RuntimeError("LCMD prefix reconstruction mismatch")
        assert_partition(labeled, unlabeled, outer)
        if (inputs["L_t_ids_hash"] != ids_hash(ids(labeled))
                or inputs["U_t_ids_hash"] != ids_hash(ids(unlabeled))
                or inputs["ordered_L_t_hash"] != stable_hash(ids(labeled))
                or inputs["ordered_U_t_hash"] != stable_hash(ids(unlabeled))):
            raise RuntimeError("LCMD state hash mismatch")
        paths += [directory / "contract.json", *[directory / k for k in state_files]]
        if r < r_switch:
            selection_path = directory / "acquisition_artifacts/selected_next_batch.csv"
            if sha256_file(selection_path) != record["files"]["acquisition_artifacts/selected_next_batch.csv"]:
                raise RuntimeError("source selected batch changed")
            selected_ids = pd.read_csv(selection_path).sample_id.astype(str).tolist()
            if stable_hash(selected_ids) != record["outgoing_selected_ids_hash"]:
                raise RuntimeError("source outgoing selection hash mismatch")
            index = dict(zip(partition.sample_id.astype(str), partition.canonical_index.astype(int)))
            labeled, unlabeled = advance(labeled, unlabeled, [index[i] for i in selected_ids], outer)
            paths.append(selection_path)
    return {"seed": seed, "source_round": r_switch, "source_method": "gradient_lcmd",
            "switch_active_labels": len(labeled), "continuation_strategy": POLICY.after,
            "labeled_indices": labeled, "unlabeled_indices": unlabeled, "outer_indices": outer,
            "labeled_ids": ids(labeled), "unlabeled_ids": ids(unlabeled),
            "L_ids_hash": ids_hash(ids(labeled)), "U_ids_hash": ids_hash(ids(unlabeled)),
            "ordered_L_ids_hash": stable_hash(ids(labeled)), "ordered_U_ids_hash": stable_hash(ids(unlabeled)),
            "split_hash": stable_hash(partition.to_dict("list")), "files": hashes(paths)}


def compatible_gradient_extension(old_text, new_text):
    """Allow only the reviewed additive extractor extension, not changed old code."""
    old, new = ast.parse(old_text).body, ast.parse(new_text).body
    if len(new) < len(old):
        return False
    for a, b in zip(old, new):
        if isinstance(a, ast.ImportFrom) and a.module == "typing":
            if not isinstance(b, ast.ImportFrom) or b.module != "typing":
                return False
            a_names = {(v.name, v.asname) for v in a.names}
            b_names = {(v.name, v.asname) for v in b.names}
            if b_names not in (a_names, a_names | {("Mapping", None)}):
                return False
        elif ast.dump(a) != ast.dump(b):
            return False
    allowed = {"extract_linear_output_gradient_sketches", "extract_linear_output_gradient_sketches_many"}
    extra = new[len(old):]
    return all(isinstance(n, ast.FunctionDef) and n.name in allowed and not n.decorator_list for n in extra) and len({n.name for n in extra}) == len(extra)


def compatibility_audit(old_context):
    historical = read_json(IVR / "seal.json")
    files = {**{k: v for k, v in historical["files"].items() if k.endswith(".py")},
             **old_context["code_hashes"]}
    changes = []
    for name, digest in files.items():
        current = sha256_file(ROOT / name)
        if current == digest:
            continue
        if name != GRADIENT_MODULE:
            raise RuntimeError(f"incompatible historical code: {name}; a new predictor protocol is required")
        old = subprocess.check_output(["git", "show", f'{old_context["preregistration_commit"]}:{name}'], cwd=ROOT)
        if hashlib.sha256(old).hexdigest() != digest or not compatible_gradient_extension(old.decode(), (ROOT / name).read_text()):
            raise RuntimeError("historical gradient extractor changed incompatibly")
        changes.append({"path": name, "historical_sha256": digest, "current_sha256": current,
                        "compatibility": "original_module_AST_prefix_unchanged_except_typing_Mapping"})
    if package_versions() != read_json(IVR / "environment.json")["packages"]:
        raise RuntimeError("historical numerical environment differs")
    return {"checked_code_files": files, "compatible_additions": changes, "packages": package_versions()}


def make_context(seed, runtime):
    old = read_json(BASELINE / f"runtime/seed_{seed}/context.json")["contract"]
    compatibility_audit(old)
    context = SequentialSeedContext(seed, _partition(seed), runtime, old["preregistration_commit"])
    expected = dict(context.context_contract, code_hashes=old["code_hashes"])
    if expected != old:
        raise RuntimeError("historical preprocessing/source/split context mismatch")
    if sha256_file(runtime / "scaler.json") != sha256_file(BASELINE / f"runtime/seed_{seed}/scaler.json"):
        raise RuntimeError("historical scaler bytes differ")
    return context


def restore_truth(context, state):
    store = context.new_method_store()
    acquired = state["labeled_ids"][len(context.roles["l0"]):]
    store.freeze_acquisitions(acquired)
    truth = np.vstack([context.l0_truth, store.reveal(acquired, "after_acquisition_fit")])
    return store, truth, acquired


def check_checkpoint(context, state, truth):
    old_base = BASELINE / f"runtime/seed_{context.seed}"
    directory = old_base / f'lcmd/round_{state["source_round"]:02d}'
    record = read_json(directory / "contract.json")
    _assert_protected(directory, {k: v for k, v in record["files"].items() if k.startswith("model/")})
    model_dir = directory / "model"
    fit = read_json(model_dir / "fit_audit.json")
    old_context = read_json(old_base / "context.json")["contract"]
    seed_everything(initialization_seed(context.seed, 0))
    init_hash = state_dict_hash(build_predictor(context.normalization))
    expected = {
        "study": BASELINE.name, "context_hash": stable_hash(old_context), "method": "lcmd",
        "round": state["source_round"], "member": 0,
        "train_sample_ids": state["labeled_ids"], "validation_sample_ids": context.ids(context.roles["validation"]),
        "L_t_ids_hash": state["L_ids_hash"], "prediction_role": "test",
        "initialization_seed": initialization_seed(context.seed, 0), "initialization_hash": init_hash,
        "train_ids_hash": state["L_ids_hash"], "validation_ids_hash": ids_hash(context.ids(context.roles["validation"])),
        "training_config": context.training_config(0), "preprocessing": context.preprocessing,
        "normalization": asdict(context.normalization), "train_truth_hash": array_hash(truth.astype(np.float32)),
        "validation_truth_hash": array_hash(context.validation_truth.astype(np.float32)),
        "ordered_train_ids": state["labeled_ids"], "ordered_validation_ids": context.ids(context.roles["validation"]),
        "ordered_train_indices": state["labeled_indices"], "ordered_validation_indices": context.roles["validation"].tolist(),
        "prediction_sample_ids": context.ids(context.roles["test"]), "prediction_indices": context.roles["test"].tolist(),
    }
    if (fit["fit_contract_hash"] != hashlib_sha(expected) or fit["initialization_hash"] != init_hash
            or fit["test_labels_used_for_fit_or_checkpoint_selection"] != 0):
        raise RuntimeError("checkpoint exact training/initialization/truth contract mismatch")
    payload = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    validate_predictor_checkpoint(payload)
    if (payload["preprocessing"] != context.preprocessing or payload["training_config"] != context.training_config(0)
            or payload["normalization"] != asdict(context.normalization)
            or state_dict_hash(load_predictor_checkpoint(model_dir / "best.pt")) != fit["checkpoint_state_hash"]):
        raise RuntimeError("checkpoint model/preprocessing/state mismatch")
    return {"source_checkpoint_path": str((model_dir / "best.pt").relative_to(ROOT)),
            "source_checkpoint_sha256": sha256_file(model_dir / "best.pt"),
            "initialization_hash": init_hash, "fit_contract_hash": fit["fit_contract_hash"],
            "train_truth_hash": expected["train_truth_hash"],
            "checkpoint_state_hash": fit["checkpoint_state_hash"]}


def source_features(state):
    directory = BASELINE / f'runtime/seed_{state["seed"]}/lcmd/round_{state["source_round"]:02d}'
    path = directory / "acquisition_artifacts/current_gradient_features.npz"
    receipt_path = path.with_suffix(".npz.contract.json")
    config = read_json(receipt_path)["contract"]
    verify_cache(path, config)
    acquisition = read_json(directory / "acquisition_artifacts/contract.json")
    _assert_protected(directory, {"acquisition_artifacts/contract.json": read_json(directory / "contract.json")["files"]["acquisition_artifacts/contract.json"]})
    _assert_protected(directory, acquisition["files"])
    inputs = read_json(directory / "input_contract.json")
    expected_order = state["labeled_indices"] + state["unlabeled_indices"]
    expected = {"outer_seed": state["seed"], "method": "lcmd", "source_round": state["source_round"],
                "acquisition_round": state["source_round"] + 1, "L_t_ids_hash": state["L_ids_hash"],
                "U_t_ids_hash": state["U_ids_hash"], "evaluation_checkpoint_sha256": state["source_checkpoint_sha256"],
                "evaluation_checkpoint_state_hash": state["checkpoint_state_hash"],
                "endpoint_scales": inputs["endpoint_scales"], "dimension": 512,
                "sketch_seed": sketch_seed(state["seed"]), "ordered_indices": expected_order,
                "kind": "current_full_network_q50_gradient_CountSketch", "test_truth_access_count": 0,
                "scales": [inputs["endpoint_scales"][k] for k in ("V1", "V2")]}
    if any(config.get(k) != v for k, v in expected.items()):
        raise RuntimeError("source gradient provenance mismatch")
    with np.load(path) as bank:
        features, indices = bank["features"], bank["canonical_indices"]
    if indices.tolist() != expected_order or features.shape != (len(expected_order), 512) or not np.isfinite(features).all():
        raise RuntimeError("source gradient order/shape mismatch")
    position = {int(i): j for j, i in enumerate(indices)}
    features = features[[position[i] for i in state["outer_indices"]]]
    audit_path = directory / "acquisition_artifacts/gradient_audit.json"
    return features, read_json(audit_path), [path, receipt_path, audit_path, directory / "acquisition_artifacts/contract.json"]


def inspect_seed(seed, destination):
    state = reconstruct(seed)
    base = BASELINE / f"runtime/seed_{seed}"
    context = make_context(seed, destination)
    store, truth, _ = restore_truth(context, state)
    state["code_compatibility"] = compatibility_audit(read_json(base / "context.json")["contract"])
    state["checkpoint_reused"], state["gradient_reused"] = False, False
    state["fallback_reasons"] = []
    directory = base / f'lcmd/round_{state["source_round"]:02d}'
    record = read_json(directory / "contract.json")
    state["source_checkpoint_path"] = str((directory / "model/best.pt").relative_to(ROOT))
    state["expected_source_checkpoint_sha256"] = record["checkpoint_sha256"]
    paths = [base / "context.json", base / "scaler.json"]
    try:
        state.update(check_checkpoint(context, state, truth))
        state["checkpoint_reused"] = True
        model = ROOT / state["source_checkpoint_path"]
        paths += [model, model.parent / "fit_audit.json", model.parent / "predictions.csv.gz"]
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        state["fallback_reasons"].append(f"retrain_L653: {error}")
    if state["checkpoint_reused"]:
        try:
            bank, _, features = source_features(state)
            positions = [0, len(context.roles["l0"]), len(context.outer)-1]
            checked = context.outer[positions]
            probe = extract_q50_gradient_sketches(
                load_predictor_checkpoint(ROOT / state["source_checkpoint_path"]), context.atom, context.angle,
                checked, tuple(context.preprocessing["target_scales"][k] for k in ("V1", "V2")),
                dimension=512, sketch_seed=sketch_seed(seed))
            if not np.allclose(bank[positions], probe.features, rtol=1e-6, atol=1e-6):
                raise RuntimeError("recomputed source gradient spot check differs")
            state["gradient_spot_check"] = {"canonical_indices": checked.tolist(), "rows": len(positions),
                                            "max_absolute_difference": float(np.max(np.abs(bank[positions]-probe.features)))}
            state["gradient_reused"] = True
            state["source_gradient_path"] = str(features[0].relative_to(ROOT))
            state["source_gradient_sha256"] = sha256_file(features[0])
            paths += features
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            state["fallback_reasons"].append(f"reextract_gradient: {error}")
    state.update(source_sha256=sha256_file(SOURCE_DATA), graph_sha256=sha256_file(SOURCE_GRAPH_CACHE),
                 preprocessing=context.preprocessing, label_access_audit=store.audit,
                 target_scales=context.preprocessing["target_scales"], test_truth_access_count=0,
                 new_fits=0, planned_new_fits=11 + int(not state["checkpoint_reused"]))
    state["files"].update(hashes(paths))
    return state


def validate_seal():
    seal = read_json(STUDY / "seal.json")
    _assert_protected(ROOT, seal["files"])
    _assert_protected(ROOT, seal["source_files"])
    if (sha256_file(SOURCE_DATA) != seal["source_sha256"] or sha256_file(SOURCE_GRAPH_CACHE) != seal["graph_sha256"]
            or package_versions() != seal["packages"]):
        raise RuntimeError("dataset/graph/environment drift")
    return seal


def prepare(test_report):
    STUDY.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(STUDY / "runtime/prepare.lock"):
        if (STUDY / "seal.json").exists():
            return validate_seal()
        if list((STUDY / "runtime").glob("seed_*")):
            raise RuntimeError("cannot seal after execution")
        xml = ET.parse(test_report).getroot()
        cases = list(xml.iter("testcase"))
        if len(cases) < 11 or any(list(xml.iter(tag)) for tag in ("failure", "error", "skipped")):
            raise RuntimeError("passing non-skipped preflight suite required")
        sources = {}
        planned_fits = 0
        for seed in CONFIRMATION_SEEDS:
            with tempfile.TemporaryDirectory(prefix=f"switch_audit_{seed}_") as temporary:
                state = inspect_seed(seed, Path(temporary))
            _write_json_once(STUDY / f"lineage/seed_{seed}.json", state)
            sources.update(state["files"])
            planned_fits += state["planned_new_fits"]
            print(f"audited seed={seed} round={state['source_round']} checkpoint_reused={state['checkpoint_reused']} gradient_reused={state['gradient_reused']}", flush=True)
        for p in [BASELINE / "results/learning_curve_metrics.csv", BASELINE / "results/full_data_reference.csv",
                  IVR / "results/learning_curve_metrics.csv", IVR / "seal.json", IVR / "environment.json"]:
            sources.update(hashes([p]))
        test_copy = STUDY / "preflight_tests.xml"
        test_copy.write_bytes(Path(test_report).read_bytes())
        code = list((ROOT / "src/qgeognn_al").rglob("*.py"))
        code = [p for p in code if not p.name.startswith("short_sequential_")]
        code += [ENTRY, ROOT / "application/QGeoGNN.py", ROOT / "application/utils.py",
                 ROOT / "tests/active_learning_v2/test_lcmd_to_ivr.py", test_copy, STUDY / "PROTOCOL.md"]
        code += list((STUDY / "lineage").glob("*.json"))
        seal = {"status": "SEALED_PENDING_EXECUTION", "created_at": now(), "policy": asdict(POLICY),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "seeds": list(CONFIRMATION_SEEDS), "budgets": list(ACTIVE_LABEL_BUDGETS),
                "source_sha256": sha256_file(SOURCE_DATA), "graph_sha256": sha256_file(SOURCE_GRAPH_CACHE),
                "packages": package_versions(), "files": hashes(code), "source_files": sources,
                "evidence": "post_hoc_development_hypothesis_not_independent_confirmation",
                "new_fits": 0, "new_test_evaluations": 0, "planned_new_fits": planned_fits}
        _write_json_once(STUDY / "seal.json", seal)
        return seal
