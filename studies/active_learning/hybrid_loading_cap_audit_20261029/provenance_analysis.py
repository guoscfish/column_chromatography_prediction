"""Read-only provenance and validation audit of historical Scientist V2 runs.

No training, prediction, gradient extraction, transport, or label-store modules
are imported. Prediction/checkpoint files are streamed only for byte hashes.
Measured values are read solely from already frozen trajectory catalogs/feedback.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STUDIES = ROOT / "studies/active_learning"
SCIENTIST = STUDIES / "qgeognn_v2_row_llm_scientist_v2"
REVISIONS = SCIENTIST / "revisions"
ASTRA = REVISIONS / "astra_high_full_pool_20261001"
FREE_CONT = REVISIONS / "free_sol_xhigh_continuation_20261012"
INHERITED = REVISIONS / "sol_xhigh_hybrid_free_to1005_20261013"
RAW = STUDIES / "qgeognn_v2_row_sequential_b32"
HYBRID = "cw16_llm16_scientist_v2"
FREE = "free_llm32_scientist_v2"
CW = "center_width_lcmd"
METHODS = (HYBRID, FREE, CW, "raw_gradient_lcmd")
SEEDS = (157, 6101)
STAGES = ((333, 525), (525, 653), (653, 813), (333, 813))
HASHES: dict[str, str] = {}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def sha(path):
    path = Path(path)
    key = relative(path)
    if key not in HASHES:
        with path.open("rb") as handle:
            HASHES[key] = hashlib.file_digest(handle, "sha256").hexdigest()
    return HASHES[key]


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ids_hash(ids):
    return stable_hash(sorted(ids))


def selection_owner(seed, method, round_index):
    """Original execution owner, rather than a later inherited copy."""
    if method == HYBRID:
        return ASTRA
    if method == FREE:
        return ASTRA if round_index < {157: 6, 6101: 10}[seed] else FREE_CONT
    if method == "raw_gradient_lcmd":
        return RAW
    if method == CW:
        for end, name in ((2, "qgeognn_v2_row_short_sequential_b32"),
                          (5, "qgeognn_v2_row_cw_lcmd_to_525"),
                          (9, "qgeognn_v2_row_cw_lcmd_to_653"),
                          (20, "qgeognn_v2_row_cw_lcmd_to_1005")):
            if round_index <= end:
                return STUDIES / name
    raise ValueError((seed, method, round_index))


def fit_owner(seed, method, round_index):
    if method in (HYBRID, FREE):
        return ASTRA if round_index == 0 else selection_owner(seed, method, round_index - 1)
    if method == CW:
        return selection_owner(seed, method, max(0, round_index - 1))
    if method == "raw_gradient_lcmd":
        return RAW
    raise ValueError(method)


def csv_rows(path):
    sha(path)
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aulc(points, lower, upper):
    """Mean trapezoidal NRMSE on an exactly observed inclusive budget span."""
    rows = sorted((p for p in points if lower <= p["labels"] <= upper), key=lambda p: p["labels"])
    if len(rows) < 2 or rows[0]["labels"] != lower or rows[-1]["labels"] != upper:
        raise ValueError("stage endpoints must both be observed; no extrapolation")
    if any(b["labels"] - a["labels"] != 32 for a, b in zip(rows, rows[1:])):
        raise ValueError("incomplete or duplicated stage")
    return sum((b["labels"] - a["labels"]) * (a["validation_nrmse"] + b["validation_nrmse"]) / 2
               for a, b in zip(rows, rows[1:])) / (upper - lower)


def validate_boundary(packet, catalog, contract, selection, batch, feedback, receipt):
    """Deterministic identity and freeze-before-reveal checks; no truth lookup."""
    chosen = batch["batch_ids"]
    pending = batch["pending_ids"]
    assert len(chosen) == len(set(chosen)) == 32
    assert len(pending) == (16 if contract["method"] == HYBRID else 0)
    assert chosen == pending + [row["id"] for row in selection["choices"]]
    assert set(chosen) <= set(contract["U_t_ids"])
    assert not set(chosen) & set(contract["L_t_ids"])
    observed = {row["id"] for row in catalog["observed"]}
    candidates = {row["id"] for row in catalog["candidates"]}
    assert observed == set(contract["L_t_ids"])
    assert candidates.isdisjoint(pending)
    assert candidates | set(pending) == set(contract["U_t_ids"])
    assert {row["id"] for row in catalog["pending"]} == set(pending)
    assert set(batch["viewed_candidate_ids"]) == candidates
    assert packet["validation_test_records"] == 0
    assert packet["delivery"] == "all_candidates_one_request"
    assert packet["candidate_count"] == len(candidates)
    assert packet["observed_count"] == len(observed)
    assert packet["active_label_count"] == len(observed)
    assert selection["packet_sha256"] == packet["packet_sha256"]
    unsigned = {k: v for k, v in packet.items() if k != "packet_sha256"}
    assert stable_hash(unsigned) == packet["packet_sha256"]
    assert not batch["new_batch_labels_revealed"]
    lookup = {row["id"]: row for row in catalog["candidates"] + catalog["pending"]}
    for row in lookup.values():
        assert not {"V1_ml", "V2_ml", "true_V1_ml", "true_V2_ml"} & set(row)
    for name in ("candidates", "pending", "observed"):
        table = packet[name]
        assert len(table["rows"]) == len(catalog[name])
        assert all(len(row) == len(table["columns"]) for row in table["rows"])
        decoded = {dict(zip(table["columns"], row))["id"]: dict(zip(table["columns"], row)) for row in table["rows"]}
        assert set(decoded) == {row["id"] for row in catalog[name]}
        for raw in catalog[name]:
            enc = decoded[raw["id"]]
            assert packet["molecules"][enc["molecule_id"]]["smiles"] == raw["smiles"]
            assert dict(zip(packet["condition_columns"], enc["conditions"])) == raw["conditions"]
            for key, value in raw.items():
                if key not in ("id", "smiles", "conditions"):
                    assert enc[key] == value
    for hypothesis in selection["hypotheses"]:
        for field in ("supporting_observed_ids", "contradicting_observed_ids"):
            assert set(hypothesis[field]) <= observed
    assert receipt["audit"]["purpose"] == "after_acquisition_fit"
    assert receipt["audit"]["acquisitions_frozen"] is True
    assert receipt["audit"]["roles"] == "u0"
    assert receipt["audit"]["requested_rows"] == 32
    assert receipt["audit"]["requested_ids_hash"] == ids_hash(chosen)
    assert stable_hash(feedback) == receipt["feedback_sha256"]
    assert [row["candidate_id"] for row in feedback["records"]] == chosen
    assert [row["candidate_id"] for row in batch["premeasurement_predictions"]] == chosen
    for row, pred in zip(feedback["records"], batch["premeasurement_predictions"]):
        assert row["source"] == ("CW" if row["candidate_id"] in pending else "LLM")
        for target in ("V1", "V2"):
            assert row[f"pred_{target}_ml"] == pred[f"pred_{target}_ml"] == lookup[row["candidate_id"]][f"pred_{target}_ml"]
            assert row[f"error_{target}_ml"] == row[f"pred_{target}_ml"] - row[f"true_{target}_ml"]


def audit_fit(seed, method, r, initial):
    owner = fit_owner(seed, method, r)
    runtime_method = "lcmd" if method == "raw_gradient_lcmd" else method
    directory = owner / f"runtime/seed_{seed}/{runtime_method}/round_{r:02d}"
    frozen_path = directory / ("prediction_freeze.json" if method in (HYBRID, FREE) else "round_freeze.json")
    if method == "raw_gradient_lcmd":
        frozen_path = directory / "contract.json"
    frozen = read(frozen_path)
    assert frozen["test_truth_access_count"] == 0
    if method == "raw_gradient_lcmd":
        model = directory / "model/best.pt"
        predictions = directory / "model/predictions.csv.gz"
        expected_ids = frozen["input"]["L_t_ids_hash"]
    else:
        model, predictions = Path(frozen["checkpoint_path"]), Path(frozen["prediction_path"])
        expected_ids = frozen["L_t_ids_hash"] if method in (HYBRID, FREE) else frozen["input"]["L_t_ids_hash"]
    assert sha(model) == frozen["checkpoint_sha256"]
    assert sha(predictions) == frozen["prediction_sha256"]
    fit_path = model.parent / "fit_audit.json"
    fit = read(fit_path)
    assert fit["train_rows"] == 333 + 32 * r
    assert fit["train_ids_hash"] == expected_ids
    assert fit["initialization_hash"] == initial["initialization_hash"]
    assert fit["validation_ids_hash"] == initial["validation_ids_hash"]
    assert fit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
    assert fit["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    assert fit["prediction_sha256"] == frozen["prediction_sha256"]
    assert fit["test_graph_label_state"] == "zero_sentinel"
    history = csv_rows(model.parent / "history.csv")
    assert len(history) == fit["epochs_run"]
    best = min(history, key=lambda x: float(x["validation_selection_score"]))
    assert int(best["epoch"]) == fit["best_epoch"]
    assert float(best["validation_selection_score"]) == fit["best_validation_combined_normalized_rmse"]
    if method in (HYBRID, FREE):
        inherited = INHERITED / f"runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json"
        assert sha(inherited) == sha(frozen_path)
    return {"seed": seed, "method": method, "round": r, "labels": 333 + 32*r,
            "validation_nrmse": fit["best_validation_combined_normalized_rmse"],
            "owner_revision": owner.name, "fit_audit_path": relative(fit_path),
            "prediction_freeze_path": relative(frozen_path), "initialization_hash": fit["initialization_hash"],
            "validation_ids_hash": fit["validation_ids_hash"], "train_ids_hash": fit["train_ids_hash"],
            "checkpoint_sha256": fit["checkpoint_sha256"], "prediction_sha256": fit["prediction_sha256"],
            "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"]}


def audit_selection(seed, method, r, previous):
    owner = selection_owner(seed, method, r)
    directory = owner / f"selections/seed_{seed}/{method}/round_{r:02d}"
    packet, catalog, contract, selection, batch, feedback, receipt = [read(directory / (name + ".json")) for name in
        ("input", "catalog", "contract", "selection", "batch_freeze", "feedback", "label_access_receipt")]
    for name, digest in batch["artifact_hashes"].items():
        assert sha(directory / name) == digest, directory / name
    assert sha(directory / "input.json") == contract["input_sha256"]
    assert sha(directory / "catalog.json") == contract["catalog_sha256"]
    assert feedback["batch_freeze_sha256"] == receipt["batch_freeze_sha256"] == sha(directory / "batch_freeze.json")
    assert (contract["seed"], contract["method"], contract["round"]) == (seed, method, r)
    assert (feedback["seed"], feedback["method"], feedback["round"]) == (seed, method, r)
    assert len(contract["L_t_ids"]) == 333 + 32*r
    if previous is not None:
        assert contract["L_t_ids"] == previous["L_t_ids"] + previous["selected_ids"]
        assert contract["U_t_ids"] == [i for i in previous["U_t_ids"] if i not in set(previous["selected_ids"])]
    validate_boundary(packet, catalog, contract, selection, batch, feedback, receipt)
    accepted, all_models = [], set()
    turn_count = 0
    for path in sorted(directory.glob("turn_[0-9][0-9].json")):
        turn = read(path)
        turn_count += 1
        model = turn.get("provenance", {}).get("served_model", "unknown")
        all_models.add(model)
        assert turn.get("provenance", {}).get("native_tool_calls", 0) == 0
        try:
            answer = json.loads(turn["answer"])
        except (KeyError, ValueError):
            continue
        if answer == selection:
            accepted.append(model)
    # A documented ID transcription amendment is not an exact model response.
    accepted_model = ";".join(sorted(set(accepted))) if accepted else "unknown_authorized_transcription"
    amendments = [p.name for p in directory.glob("*correction*.json")]
    amendments += [p.name for p in directory.glob("*transcription*.json") if p.name not in amendments]
    for name in amendments:
        sha(directory / name)
    copy_checks = []
    for name in batch["artifact_hashes"]:
        path = INHERITED / f"selections/seed_{seed}/{method}/round_{r:02d}" / name
        assert sha(path) == sha(directory / name)
        copy_checks.append(name)
    for name in ("batch_freeze.json", "feedback.json", "label_access_receipt.json"):
        assert sha(INHERITED / f"selections/seed_{seed}/{method}/round_{r:02d}" / name) == sha(directory / name)
    return {"seed": seed, "method": method, "round": r, "labels_before": 333 + 32*r,
            "labels_after": 365 + 32*r, "owner_revision": owner.name, "directory": relative(directory),
            "selected_ids": batch["batch_ids"], "pending_ids": batch["pending_ids"],
            "L_t_ids": contract["L_t_ids"], "U_t_ids": contract["U_t_ids"],
            "checkpoint_sha256": contract["checkpoint_sha256"], "input_sha256": sha(directory / "input.json"),
            "catalog_sha256": sha(directory / "catalog.json"), "batch_freeze_sha256": sha(directory / "batch_freeze.json"),
            "served_model_exact_selection": accepted_model, "models_in_all_frozen_turns": sorted(all_models),
            "completed_plan_response_count": turn_count, "historical_amendments": sorted(amendments),
            "candidate_rows": len(catalog["candidates"]), "pending_rows": len(catalog["pending"]),
            "observed_rows": len(catalog["observed"]), "all_candidates_delivered": True,
            "freeze_before_reveal_receipt_verified": True, "inherited_copy_verified": True}


def audit_traditional_batch(seed, method, r, expected_labeled):
    owner = selection_owner(seed, method, r)
    runtime_method = "lcmd" if method == "raw_gradient_lcmd" else method
    directory = owner / f"runtime/seed_{seed}/{runtime_method}/round_{r:02d}"
    path = directory / "acquisition_artifacts/contract.json"
    contract = read(path)
    chosen_path = path.parent / "selected_next_batch.csv"
    chosen = [row["sample_id"] for row in csv_rows(chosen_path)]
    assert len(chosen) == len(set(chosen)) == 32
    assert stable_hash(chosen) == contract["selected_ids_ordered_hash"]
    if "selected_table_sha256" in contract:
        assert sha(chosen_path) == contract["selected_table_sha256"]
    for name, digest in contract.get("files", {}).items():
        assert sha(directory / name) == digest
    assert contract.get("test_truth_access_count", 0) == 0
    if method == CW:
        assert contract["input"]["L_t_ids_hash"] == ids_hash(expected_labeled)
        assert contract["input"]["current_model_round"] == r
        assert contract["input"]["batch_size"] == 32
        assert contract["input"]["centers_are_all_current_L_t"] is True
    else:
        state = read(directory / "input_contract.json")
        assert state["L_t_ids_hash"] == ids_hash(expected_labeled)
        assert state["round"] == r
    audit_path = owner / f"runtime/seed_{seed}/{runtime_method}/label_access_audit.csv"
    access = csv_rows(audit_path)
    matching = [row for row in access if row["requested_ids_hash"] == ids_hash(chosen)]
    assert len(matching) == 1
    assert matching[0]["purpose"] == "after_acquisition_fit"
    assert matching[0]["acquisitions_frozen"] == "True"
    assert matching[0]["roles"] == "u0"
    return {"seed": seed, "method": method, "round": r,
            "owner_revision": owner.name, "directory": relative(directory),
            "selected_ids": chosen, "acquisition_contract_sha256": sha(path),
            "freeze_before_reveal_receipt_verified": True}


def source_lineage():
    records = []
    original_protocol = read(ASTRA / "protocol.json")
    paths = ("scientist_study.py", "scientist_selector.py", "scientist_full_delivery.py", "scientist_transport.py",
             "llm_screen.py", "gradient_features.py", "gradient_transforms.py", "lcmd.py", "runner.py")
    for name in paths:
        path = ROOT / "src/qgeognn_al/active_learning_v2" / name
        current = path.read_bytes()
        assert sha(path) == original_protocol["code_hashes"][relative(path)]
        history = []
        for rev in ("8081e865", "30da9ae8", "1ec395f8", "93609bd9"):
            result = subprocess.run(["git", "show", f"{rev}:src/qgeognn_al/active_learning_v2/{name}"], cwd=ROOT,
                                    capture_output=True)
            if result.returncode:
                continue
            frozen = result.stdout
            current_ast = ast.parse(current)
            old_ast = ast.parse(frozen)
            funcs = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            a, b = funcs(current_ast), funcs(old_ast)
            history.append({"git_revision": rev, "sha256": hashlib.sha256(frozen).hexdigest(),
                            "bytes_equal_current": frozen == current,
                            "common_top_level_functions_classes": len(a.keys() & b.keys()),
                            "common_ast_equal": all(a[k] == b[k] for k in a.keys() & b.keys()),
                            "changed_common_symbols": sorted(k for k in a.keys() & b.keys() if a[k] != b[k])})
        records.append({"path": relative(path), "current_sha256": sha(path),
                        "matches_original_astra_protocol": True, "history": history})
    for path in (ASTRA / "frozen_scientist_study_original.py", ASTRA / "source_recovery_20261010/recovery.json",
                 ASTRA / "source_recovery_20261010/scientist_study_modified_2b0ebffbfd6cfcf5eac8d238618efdd2daefd9a99cabcefc102da73d50dbe76d.py",
                 ROOT / "reports/free_lcmd_five_seeds_to1485_20261026/frozen_sources/gradient_features.py"):
        sha(path)
    return records


def main():
    fits, batches, initial_states = [], [], {}
    for seed in SEEDS:
        initial_path = RAW / f"runtime/seed_{seed}/lcmd/round_00/model/fit_audit.json"
        initial = read(initial_path)
        initial_input = read(initial_path.parents[1] / "input_contract.json")
        config = initial_input["training_config"]
        split = RAW / f"splits/row_seed_{seed}.csv"
        split_manifest = read(RAW / "splits/split_manifest.json")
        split_record = next(x for x in split_manifest["splits"] if x["outer_seed"] == seed)
        assert sha(split) == split_record["sha256"]
        split_rows = csv_rows(split)
        split_columns = {key: [int(row[key]) if key in ("canonical_index", "outer_seed") else row[key]
                               for row in split_rows] for key in split_rows[0]}
        assert stable_hash(split_columns) == config["split_sha256"]
        assert config["checkpoint_selection"] == "validation_combined_normalized_rmse"
        assert not config["test_during_training"]
        assert initial_input["endpoint_scales_fit_ids_hash"] == initial_input["L_t_ids_hash"]
        for owner in (ASTRA, FREE_CONT, INHERITED):
            context = read(owner / f"runtime/seed_{seed}/context.json")
            assert context["split_hash"] == config["split_sha256"]
            assert context["preprocessing"]["target_scales"] == initial_input["endpoint_scales"]
            assert context["preprocessing"]["target_scale_fit_rows"] == 333
            assert context["preprocessing"]["target_scale_fit_ids_hash"] == initial_input["L_t_ids_hash"]
            assert context["preprocessing"]["test_rows_used"] == 0
        l0_catalog = read(ASTRA / f"selections/seed_{seed}/{HYBRID}/round_00/contract.json")
        l0 = l0_catalog["L_t_ids"]
        assert ids_hash(l0) == initial_input["L_t_ids_hash"]
        initial_states[str(seed)] = {"L333_ids_hash": ids_hash(l0), "validation_ids_hash": initial["validation_ids_hash"],
                                    "initialization_hash": initial["initialization_hash"],
                                    "checkpoint_sha256": initial["checkpoint_sha256"], "split_sha256": sha(split),
                                    "split_logical_partition_hash": config["split_sha256"],
                                    "endpoint_scales": initial_input["endpoint_scales"], "training_config": config}
        for method in METHODS:
            for r in range(16):
                fits.append(audit_fit(seed, method, r, initial))
            previous = None
            labeled = list(l0)
            acquired = []
            for r in range(15):
                if method in (HYBRID, FREE):
                    row = audit_selection(seed, method, r, previous)
                    fit = next(x for x in fits if x["seed"] == seed and x["method"] == method and x["round"] == r)
                    assert row["checkpoint_sha256"] == fit["checkpoint_sha256"]
                    previous = row
                else:
                    row = audit_traditional_batch(seed, method, r, labeled)
                acquired += row["selected_ids"]
                labeled += row["selected_ids"]
                next_fit = next(x for x in fits if x["seed"] == seed and x["method"] == method and x["round"] == r+1)
                assert next_fit["train_ids_hash"] == ids_hash(labeled)
                batches.append(row)
            assert len(acquired) == len(set(acquired)) == 480
    stages = []
    for seed in SEEDS:
        for method in METHODS:
            points = [x for x in fits if x["seed"] == seed and x["method"] == method]
            for lower, upper in STAGES:
                first = next(x for x in points if x["labels"] == lower)["validation_nrmse"]
                last = next(x for x in points if x["labels"] == upper)["validation_nrmse"]
                stages.append({"seed": seed, "method": method, "stage": f"{lower}-{upper}",
                               "labels_start": lower, "labels_end": upper, "fit_points": (upper-lower)//32+1,
                               "nrmse_start": first, "nrmse_end": last, "nrmse_improvement": first-last,
                               "validation_aulc": aulc(points, lower, upper),
                               "normalization": "original seed L333 population target std; fixed across all budgets"})
    frozen_snapshot = read(ROOT / "reports/llm_hybrid_complete_20261010/validation_curves.json")
    snapshot_checks = []
    for seed in SEEDS:
        for method in (HYBRID, FREE, CW):
            snap_method = CW if method == CW else method
            if snap_method not in frozen_snapshot[str(seed)]:
                continue
            observed = {x["labels"]: x["validation_nrmse"] for x in fits if x["seed"] == seed and x["method"] == method}
            snap = frozen_snapshot[str(seed)][snap_method]
            assert all(observed[x["labels"]] == x["nrmse"] for x in snap if x["labels"] <= 813)
            snapshot_checks.append({"seed": seed, "method": method, "historical_snapshot_last_budget": snap[-1]["labels"], "matches_reaudit": True})
    five_seed_snapshot = read(ROOT / "reports/free_llm_five_seeds_to1485_20261024/figures/validation_curves.json")
    for seed in SEEDS:
        for method, snap_method in ((FREE, FREE), ("raw_gradient_lcmd", "lcmd")):
            observed = {x["labels"]: x["validation_nrmse"] for x in fits if x["seed"] == seed and x["method"] == method}
            assert all(observed[x["labels"]] == x["nrmse"] for x in five_seed_snapshot[str(seed)][snap_method] if x["labels"] <= 813)
    dialog = STUDIES / "qgeognn_v2_row_llm_dialog_feedback_v1"
    dialog_protocol = read(dialog / "protocol.json")
    dialog_audit = read(dialog / "execution_boundary_audit.json")
    assert len(dialog_audit) == 12
    assert all(x["legal"] and not x["forbidden_records_exposed"] and not x["unobserved_responses_exposed"]
               and not x["native_tool_calls"] and x["premeasurement_errors_verified"] == 32 for x in dialog_audit)
    for path in (dialog / "FINAL_REPORT.md", dialog / "execution_boundary_audit.json", dialog / "results/aulc.csv",
                 ROOT / "studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/REPORT.md",
                 ROOT / "studies/active_learning/free_lcmd_mechanism_diagnostics_20261029/selection_behavior_analysis.py",
                 ASTRA / "protocol.json", ASTRA / "selector_prompt.txt", FREE_CONT / "protocol.json", FREE_CONT / "selector_prompt.txt",
                 FREE_CONT / "transition_manifest.json", SCIENTIST / "historical_runtime_manifest.json"):
        sha(path)
    prompts = []
    for owner in (ASTRA, FREE_CONT, INHERITED):
        protocol = read(owner / "protocol.json")
        path = owner / "selector_prompt.txt"
        assert stable_hash(path.read_text()) == protocol["prompt_sha256"]
        prompts.append({"revision": owner.name, "file_sha256": sha(path),
                        "protocol_prompt_stable_string_hash": protocol["prompt_sha256"]})
    assert len({x["file_sha256"] for x in prompts}) == 1
    lineage = source_lineage()
    archive_manifest = read(SCIENTIST / "historical_runtime_manifest.json")
    archive_path = SCIENTIST / archive_manifest["archive"]
    assert sha(archive_path) == archive_manifest["archive_sha256"]
    served = {}
    for seed in SEEDS:
        for method in (HYBRID, FREE):
            counts = {}
            for b in batches:
                if (b["seed"], b["method"]) == (seed, method):
                    model = b["served_model_exact_selection"]
                    counts[model] = counts.get(model, 0) + 1
            served[f"{seed}/{method}"] = counts
    sha(Path(__file__))
    summary = {"status": "PASS", "scope": "two paired seeds, 15 acquisitions, 16 validation fits per method; 333-813 only",
               "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "seeds": list(SEEDS), "methods": list(METHODS), "initial_states": initial_states,
               "verified_fit_count": len(fits), "verified_batch_count": len(batches),
               "unique_fit_audit_files": len({x["fit_audit_path"] for x in fits}),
               "verified_scientist_batch_count": 60, "selected_rows": len(batches)*32,
               "scientist_inherited_copies_equal_original": True, "source_file_count": len(HASHES),
               "prediction_csv_values_parsed": False, "checkpoint_deserialized": False,
               "new_training": 0, "new_llm_calls": 0, "new_label_requests": 0, "new_test_evaluations": 0,
               "served_models_exact_accepted_selection": served, "historical_curve_snapshot_checks": snapshot_checks,
               "system_prompt_hashes": prompts, "scientist_system_prompts_bytes_equal": True,
               "scientist_batches": [b for b in batches if b["method"] in (HYBRID, FREE)],
               "traditional_batches": [b for b in batches if b["method"] not in (HYBRID, FREE)],
               "source_lineage": lineage,
               "historical_metadata_archive": {"sha256_verified": True, "path": relative(archive_path),
                                                "recovery_commit": archive_manifest["recovery_commit"],
                                                "restored_or_modified": False},
               "dialog_v1": {"independent_auxiliary_only": True, "no_selection_or_curve_splicing": True,
                             "model": dialog_protocol["llm_model"], "reasoning_effort": dialog_protocol["reasoning_effort"],
                             "candidate_views_max": dialog_protocol["selector_settings"]["distinct_candidate_views_max"],
                             "boundary_audit": relative(dialog / "execution_boundary_audit.json")},
               "extension_policy": "Historical >813 continuations remain excluded from all two-seed common-interval means; no endpoints extrapolated.",
               "future_two_arm_cost_formula": {"budget_start": 333, "budget_end": 1005,
                    "acquisitions_per_arm_per_seed": 21, "new_fits_per_seed_two_arms": 42,
                    "shared_L333_fit_per_seed_if_not_reused": 1, "nominal_llm_plan_calls_per_seed_two_arms": 42,
                    "five_seed_new_fits": 210, "five_seed_shared_L333_fits_if_needed": 5,
                    "five_seed_nominal_plan_calls": 210,
                    "five_seed_max_completed_plan_responses_if_three_per_selection": 630,
                    "network_retry_note": "network retries add HTTP attempts; bound and log separately from completed plan responses"},
               "limitations": ["freeze receipts are recorded protocol evidence, not an external timestamp service",
                               "historically exposed validation/test splits make this developmental evidence",
                               "post-r0 methods have different labeled sets and current model states",
                               "Astra/high and Sol/xhigh periods are confounded with seed, budget, prompt and path history",
                               "raw LCMD has verified label-request ledgers but no audited measured-response feedback payload for residual-neighborhood analysis"]}
    write_csv(OUT / "validation_rounds.csv", fits)
    write_csv(OUT / "validation_stages.csv", stages)
    (OUT / "source_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT / "source_hashes.json").write_text(json.dumps(HASHES, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "verified_fit_count", "verified_batch_count", "served_models_exact_accepted_selection")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
