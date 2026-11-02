"""Isolated paired NoCW workflow. Defaults are offline; no legacy globals mutated."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from .benchmark_protocol import TRAINING_CONFIG, initialization_seed
from .dialog_bridge import once, read
from .maxdet_study import historical_round0_paths, package_versions
from .protocol import ids_hash, stable_hash, validate_row_protocol
from . import scientist_lifecycle as lifecycle
from .scientist_study import prediction_freeze, exclusive_lock
from .scientist_selector import ReadOnlyCatalog
from .scientist_full_delivery import encode_catalog
from .nocw_inputs import (VERSION, ARMS, input_view, prompt, pair_diff, audit_no_cw,
                          validate_plan, validate_history)

STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_llm_nocw_ablation"
SEEDS = (157, 6101)
METHODS = tuple(ARMS)
BUDGETS = tuple(range(333, 1006, 32))
LEGACY = ROOT / "studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001"
SPLITS = ROOT / "studies/active_learning/qgeognn_v2_row_short_sequential_b32/splits"


def initial_paths(seed):
    d = LEGACY / f"selections/seed_{seed}/free_llm32_scientist_v2/round_00"
    return {"catalog": d / "catalog.json", "contract": d / "contract.json",
            "split": SPLITS / f"row_seed_{seed}.csv", **historical_round0_paths(seed)}


def initial_state(seed):
    """Read ONLY frozen L333 catalog/contract, never legacy selections or feedback."""
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    paths = initial_paths(seed)
    raw, contract = read(paths["catalog"]), read(paths["contract"])
    partition = pd.read_csv(paths["split"])
    validate_row_protocol(partition)
    roles = {role: partition.loc[partition.role.eq(role), "sample_id"].astype(str).tolist()
             for role in ("l0", "u0", "validation", "test")}
    fit = read(paths["member0_audit"])
    old = read(paths["context"])["contract"]
    if (raw["pending"] or len(raw["observed"]) != 333 or len(raw["candidates"]) != 2997
            or [x["id"] for x in raw["observed"]] != roles["l0"]
            or [x["id"] for x in raw["candidates"]] != roles["u0"]
            or contract["L_t_ids"] != roles["l0"] or contract["U_t_ids"] != roles["u0"]
            or contract["catalog_sha256"] != sha256_file(paths["catalog"])
            or contract["checkpoint_sha256"] != sha256_file(paths["member0_model"])
            or fit["checkpoint_sha256"] != contract["checkpoint_sha256"]
            or fit["train_ids_hash"] != ids_hash(roles["l0"])
            or fit["initialization_seed"] != initialization_seed(seed, 0)
            or old["source_sha256"] != sha256_file(SOURCE_DATA)
            or old["source_graph_cache_sha256"] != sha256_file(SOURCE_GRAPH_CACHE)
            or old["l0_ids_hash"] != ids_hash(roles["l0"])
            or old["u0_ids_hash"] != ids_hash(roles["u0"])):
        raise RuntimeError("L333 reuse lineage mismatch")
    if any(row.get("record_kind") != "initial_L333" or any("error" in k for k in row)
           for row in raw["observed"]):
        raise RuntimeError("post-L333 observation contamination")
    manifest = {str(paths[k].relative_to(ROOT)): sha256_file(paths[k]) for k in (
        "catalog", "contract", "split", "member0_model", "member0_audit", "member0_predictions",
        "context", "scrubbed_graphs")}
    return raw, roles, {"seed": seed, "sources": manifest,
        "checkpoint_path": str(paths["member0_model"]), "checkpoint_sha256": fit["checkpoint_sha256"],
        "initialization_hash": fit["initialization_hash"], "L333_ids_hash": ids_hash(roles["l0"]),
        "U333_ids_hash": ids_hash(roles["u0"]), "normalization": old["normalization"],
        "preprocessing": old["preprocessing"], "legacy_feedback_read": False}


def code_hashes():
    # Freeze transitive local Python dependencies as well as the new adapter.
    paths = sorted((ROOT / "src/qgeognn_al").rglob("*.py"))
    paths += [ROOT / "scripts/studies/run_qgeognn_v2_row_llm_nocw.py"]
    paths += [ROOT / "tests/active_learning_v2/test_nocw_ablation.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def prepare():
    config = read(STUDY / "config.json")
    if (config["training"] != TRAINING_CONFIG or tuple(config["seeds"]) != SEEDS
            or config["arms"] != ARMS or config["initial_budget"] != BUDGETS[0]
            or config["primary_endpoint"] != BUDGETS[-1]
            or (config["batch_size"], config["selection_count"], config["pending_count"]) != (32, 32, 0)):
        raise ValueError("configuration disagrees with implemented protocol")
    sources = [initial_state(seed)[2] for seed in SEEDS]
    record = {"version": VERSION, "config": config, "initial_states": sources,
              "code_hashes": code_hashes(),
              "environment": package_versions(),
              "document_hashes": {name: sha256_file(STUDY / name) for name in ("PROTOCOL.md", "SCHEMA_DIFF.md", "CODE_AUDIT.md")},
              "prompt_hashes": {v: stable_hash(prompt(v)) for v in ARMS.values()}}
    once(STUDY / "protocol.lock.json", record)
    once(STUDY / "protocol.json", record)
    return {"status": "OFFLINE_PROTOCOL_FROZEN", "protocol_sha256": stable_hash(record)}


def validate():
    frozen = read(STUDY / "protocol.lock.json")
    if (read(STUDY / "protocol.json") != frozen
            or frozen["config"] != read(STUDY / "config.json") or frozen["code_hashes"] != code_hashes()
            or frozen["prompt_hashes"] != {v: stable_hash(prompt(v)) for v in ARMS.values()}):
        raise RuntimeError("frozen protocol/code/prompt drift")
    for source in frozen["initial_states"]:
        for path, digest in source["sources"].items():
            if sha256_file(ROOT / path) != digest:
                raise RuntimeError(f"initial source drift: {path}")
    if any(sha256_file(STUDY / name) != digest for name, digest in frozen["document_hashes"].items()):
        raise RuntimeError("frozen protocol document drift")
    if frozen["environment"] != package_versions():
        raise RuntimeError("frozen environment drift")
    return {"status": "VALID", "protocol_sha256": stable_hash(frozen)}


def preregister(*, authorize=False, provider_supported=False):
    if not authorize or not provider_supported:
        raise PermissionError("explicit formal-run authorization and provider-support attestation required")
    record = {**validate(), "formal_run_authorized": True,
              "provider_support_attested_by_operator": True,
              "transport": read(STUDY / "config.json")["transport"]}
    once(STUDY / "formal_authorization.json", record)
    return record


def require_authorization():
    frozen = validate()
    approval = read(STUDY / "formal_authorization.json")
    if not approval.get("formal_run_authorized") or approval["protocol_sha256"] != frozen["protocol_sha256"]:
        raise PermissionError("formal authorization does not bind this frozen protocol")


def selection_directory(seed, method, round_index):
    if seed not in SEEDS or method not in METHODS or round_index not in range(len(BUDGETS)-1):
        raise ValueError("unregistered seed/arm/round")
    return STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"


class Context(lifecycle.WorkflowContext):
    def __init__(self, seed):
        super().__init__(seed, STUDY)


def state(context, method, r):
    if method not in METHODS:
        raise ValueError("foreign method")
    result = lifecycle.state(context, method, r, sys.modules[__name__])
    validate_history(result[3], context.seed, method, r)
    return result


def write_stage(directory, raw, roles, source, seed, arm, r, history):
    catalog, view, packet = input_view(raw, seed, arm, r, history)
    if ([x["id"] for x in view["observed"]] != roles["l0"]
            or [x["id"] for x in view["candidates"]] != roles["u0"]):
        raise RuntimeError("catalog/trajectory IDs disagree")
    directory.mkdir(parents=True, exist_ok=True)
    once(directory / "catalog.json", view)
    once(directory / "input.json", packet)
    once(directory / "contract.json", {"seed": seed, "method": arm, "round": r,
        "variant": ARMS[arm], "protocol_hash": validate()["protocol_sha256"],
        "checkpoint_sha256": source["checkpoint_sha256"], "L_t_ids": roles["l0"], "U_t_ids": roles["u0"],
        "input_sha256": sha256_file(directory / "input.json"),
        "catalog_sha256": sha256_file(directory / "catalog.json"), "dry_run": "dry_run" in directory.parts})
    return catalog, packet


def stage(seed, method, r=0, *, allow_feature_compute=False):
    validate()
    directory = selection_directory(seed, method, r)
    if r == 0:
        raw, roles, source = initial_state(seed)
        # Freeze the shared historical predictor without opening any label store.
        runtime = STUDY / f"runtime/seed_{seed}/{method}/round_00"
        paths = initial_paths(seed)
        once(runtime / "prediction_freeze.json", {
            "checkpoint_path": str(paths["member0_model"]), "checkpoint_sha256": source["checkpoint_sha256"],
            "prediction_path": str(paths["member0_predictions"]),
            "prediction_sha256": sha256_file(paths["member0_predictions"]),
            "L_t_ids_hash": source["L333_ids_hash"], "initialization_hash": source["initialization_hash"],
            "test_truth_access_count": 0})
        history = []
    else:
        if not allow_feature_compute:
            raise PermissionError("later-round gradient extraction requires --allow-feature-compute")
        require_authorization()
        torch.set_num_threads(2)
        context = Context(seed)
        labeled, unlabeled, truth, history = state(context, method, r)
        checkpoint, runtime = prediction_freeze(context, method, r, labeled)
        _, raw = lifecycle.make_catalog(context, method, r, labeled, unlabeled, truth, history,
                                        checkpoint, runtime, pending_count=0, version=VERSION)
        roles = {"l0": context.ids(labeled), "u0": context.ids(unlabeled)}
        source = {"checkpoint_sha256": sha256_file(checkpoint)}
    write_stage(directory, raw, roles, source, seed, method, r, history)
    return {"status": "STAGED_WITHOUT_NEW_LABELS", "seed": seed, "arm": method, "round": r}


def live_call(messages, config):
    """Only paid entry. Exact SDK kwargs are frozen and tested without networking."""
    require_authorization()
    from openai import OpenAI
    from .scientist_transport import provider_key
    with OpenAI(api_key=provider_key(config), base_url=config["base_url"],
                max_retries=0, timeout=config["timeout_seconds"]) as client:
        with client.responses.stream(model=config["model"], input=messages,
                reasoning={"effort": config["reasoning_effort"]}, tools=[], store=False,
                max_output_tokens=config["max_output_tokens"], **config["sampling_parameters"]) as stream:
            response = stream.get_final_response()
    if (response.status != "completed" or
            not (response.model == config["model"] or response.model.startswith(config["model"] + "-")) or
            any(item.type not in ("message", "reasoning") for item in response.output)):
        raise RuntimeError("incomplete/wrong-model/tool response; fail closed")
    once(STUDY / "served_model_lock.json", {"served_model": response.model})
    return response.output_text, {"response_id": response.id, "served_model": response.model,
        "usage": response.usage.model_dump(mode="json") if response.usage else None, "native_tool_calls": 0}


def request_with_retries(messages, config, directory, turn, call):
    """Total attempt cap persists across restarts; ambiguous interruptions stop."""
    request_hash = stable_hash({"messages": messages, "config": config})
    for attempt in range(config["transport_attempts"]):
        path = directory / f"attempt_{turn:02d}_{attempt:02d}.json"
        started = path.with_suffix(".started.json")
        if path.exists():
            saved = read(path)
            if saved["request_sha256"] != request_hash:
                raise RuntimeError("request drift")
            if saved.get("success"):
                return saved
            continue
        if started.exists():
            raise RuntimeError("ambiguous interrupted provider request; manual accounting required")
        once(started, {"request_sha256": request_hash})
        t = time.monotonic()
        try:
            answer, provenance = call(copy.deepcopy(messages), copy.deepcopy(config))
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty answer")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            once(path, {"request_sha256": request_hash, "success": False,
                        "error_type": type(error).__name__, "elapsed_seconds": time.monotonic()-t})
            if attempt + 1 < config["transport_attempts"]:
                time.sleep(config["retry_backoff_seconds"] * (attempt+1))
            continue
        saved = {"request_sha256": request_hash, "success": True, "answer": answer,
                 "provenance": provenance, "elapsed_seconds": time.monotonic()-t}
        once(path, saved)
        return saved
    raise RuntimeError("transport attempt budget exhausted; no automatic fallback")


def run_selection(packet, catalog, directory, variant, config, call):
    directory = Path(directory)
    if any(packet.get(k) != v for k, v in encode_catalog(catalog).items()):
        raise ValueError("packet differs from complete catalog")
    if (packet.get("selection_count") != 32 or catalog.pending
            or packet.get("candidate_count") != len(catalog.candidates)
            or packet.get("observed_count") != len(catalog.observed)
            or packet.get("active_label_count") != len(catalog.observed)):
        raise ValueError("full Free32 packet count mismatch")
    if packet["packet_sha256"] != stable_hash({k: v for k, v in packet.items() if k != "packet_sha256"}):
        raise ValueError("packet binding mismatch")
    messages = [{"role": "system", "content": prompt(variant)},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}]
    for turn in range(config["selection_attempts"]):
        if variant == "no_cw":
            audit_no_cw(messages)
        # Store the actual outgoing serialization, not a reconstructed summary.
        once(directory / f"request_{turn:02d}.json", {"messages": messages, "config": config})
        saved = request_with_retries(messages, config, directory, turn, call)
        once(directory / f"turn_{turn:02d}.json", saved)
        try:
            result = validate_plan(json.loads(saved["answer"]), catalog, packet, variant)
        except (ValueError, TypeError, KeyError) as error:
            correction = {"error": str(error)[:1000], "instruction":
                "Return corrected final JSON from the SAME complete pool. Exactly 32 distinct candidate IDs. "
                "Measured evidence belongs only in typed observed-ID lists. No queries or tools."}
            once(directory / f"rejected_{turn:02d}.json", correction)
            messages += [{"role": "assistant", "content": saved["answer"]},
                         {"role": "user", "content": json.dumps(correction)}]
            continue
        once(directory / "selection.json", result)
        return result
    raise RuntimeError("selection remained invalid after bounded corrections; fail closed")


def freeze_batch(directory, result, catalog, packet):
    contract = read(directory / "contract.json")
    validate_plan(result, catalog, packet, contract["variant"])
    if not contract["dry_run"]:
        # Answers may use different whitespace, so compare their parsed JSON.
        accepted = []
        for p in directory.glob("turn_*.json"):
            turn = read(p)
            try:
                if json.loads(turn["answer"]) == result:
                    accepted.append(turn)
            except (ValueError, KeyError):
                continue
        served = read(STUDY / "served_model_lock.json")["served_model"]
        if not accepted or any(x["provenance"].get("mock") or
                not isinstance(x["provenance"].get("response_id"), str) or
                x["provenance"].get("served_model") != served for x in accepted):
            raise PermissionError("mock/unproven selection cannot freeze a formal batch")
    ids = [x["id"] for x in result["choices"]]
    if not set(ids) <= set(contract["U_t_ids"]) or set(ids) & set(contract["L_t_ids"]):
        raise ValueError("batch outside this trajectory U_t")
    names = ["catalog.json", "input.json", "contract.json", "selection.json"]
    names += [p.name for pattern in ("turn_*.json", "request_*.json", "attempt_*.json", "rejected_*.json")
              for p in sorted(directory.glob(pattern))]
    frozen = {"batch_ids": ids, "pending_ids": [], "premeasurement_predictions": [
        {"candidate_id": i, **{f"pred_{t}_ml": catalog.candidates[i][f"pred_{t}_ml"] for t in ("V1", "V2")}} for i in ids],
        "seed": contract["seed"], "method": contract["method"], "round": contract["round"],
        "protocol_hash": contract["protocol_hash"], "checkpoint_sha256": contract["checkpoint_sha256"],
        "new_batch_labels_revealed": False, "dry_run": contract["dry_run"],
        "artifact_hashes": {name: sha256_file(directory / name) for name in names}}
    once(directory / "batch_freeze.json", frozen)
    return frozen


def audit_batch(directory):
    directory = Path(directory)
    frozen = read(directory / "batch_freeze.json")
    for name, digest in frozen["artifact_hashes"].items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError("frozen batch artifact drift")
    contract, packet = read(directory / "contract.json"), read(directory / "input.json")
    expected = selection_directory(contract["seed"], contract["method"], contract["round"])
    if contract["dry_run"]:
        expected = STUDY / f"dry_run/seed_{contract['seed']}/{contract['method']}/round_{contract['round']:02d}"
    if (directory.resolve() != expected.resolve()
            or contract["variant"] != ARMS[contract["method"]]
            or contract["protocol_hash"] != validate()["protocol_sha256"]
            or contract["input_sha256"] != sha256_file(directory / "input.json")
            or contract["catalog_sha256"] != sha256_file(directory / "catalog.json")):
        raise RuntimeError("foreign/stale batch contract")
    catalog = ReadOnlyCatalog(**read(directory / "catalog.json"))
    def offline(*args):
        raise RuntimeError("audit cannot make provider requests")
    result = run_selection(packet, catalog, directory, contract["variant"],
                           read(STUDY / "config.json")["transport"], offline)
    return freeze_batch(directory, result, catalog, packet)


def select(seed, method, r=0, *, allow_paid_llm=False, allow_feature_compute=False):
    if not allow_paid_llm:
        raise PermissionError("selection requires --allow-paid-llm")
    require_authorization()
    with exclusive_lock(STUDY / ".operation.lock"):
        stage(seed, method, r, allow_feature_compute=allow_feature_compute)
        directory = selection_directory(seed, method, r)
        if (directory / "batch_freeze.json").exists():
            return audit_batch(directory)
        packet = read(directory / "input.json")
        catalog = ReadOnlyCatalog(**read(directory / "catalog.json"))
        result = run_selection(packet, catalog, directory, ARMS[method],
                               read(STUDY / "config.json")["transport"], live_call)
        return freeze_batch(directory, result, catalog, packet)


def advance(seed, method, r=0, *, allow_reveal=False, allow_training=False):
    if not allow_reveal or not allow_training:
        raise PermissionError("advance requires --allow-label-reveal AND --allow-training")
    require_authorization()
    with exclusive_lock(STUDY / ".operation.lock"):
        frozen = audit_batch(selection_directory(seed, method, r))
        if frozen["dry_run"]:
            raise PermissionError("mock batch can never reveal labels or train")
        context = Context(seed)
        labeled, _, _, _ = state(context, method, r)
        checkpoint, _ = prediction_freeze(context, method, r, labeled)
        if frozen["checkpoint_sha256"] != sha256_file(checkpoint):
            raise RuntimeError("batch bound to a different predictor")
        return lifecycle.advance(seed, method, r, sys.modules[__name__])


def mock_plan(packet, offset=0):
    ids = [r[0] for r in packet["candidates"]["rows"]][offset:offset+32]
    return {"type": "selection", "packet_sha256": packet["packet_sha256"],
        "batch_strategy": "OFFLINE MOCK; no scientific acquisition", "hypotheses": [],
        "choices": [{"id": i, "reason": "Mock parser and freeze verification.", "hypothesis_id": None} for i in ids],
        "feedback_interpretation": "" if packet["round"] == 0 else "Synthetic feedback only.",
        "batch_rationale": "OFFLINE MOCK; never eligible for label reveal."}


def dry_run():
    validate()
    config = read(STUDY / "config.json")["transport"]
    audits = []
    for seed in SEEDS:
        raw, roles, source = initial_state(seed)
        packets, batches = {}, {}
        for arm, variant in ARMS.items():
            directory = STUDY / f"dry_run/seed_{seed}/{arm}/round_00"
            catalog, packet = write_stage(directory, raw, roles, source, seed, arm, 0, [])
            response = mock_plan(packet, 0 if variant == "cw_aware" else 32)
            result = run_selection(packet, catalog, directory, variant, config,
                                   lambda *args: (json.dumps(response), {"mock": True, "usage": None}))
            frozen = freeze_batch(directory, result, catalog, packet)
            audit_batch(directory)
            # Verify a mock subsequent trajectory using synthetic labels only.
            chosen = set(frozen["batch_ids"])
            history = [{"seed": seed, "method": arm, "round": 0, "batch_freeze_sha256": sha256_file(directory / "batch_freeze.json"),
                "records": [{**p, "source": "LLM", "true_V1_ml": -123.0, "true_V2_ml": -122.0,
                    "error_V1_ml": p["pred_V1_ml"]+123., "error_V2_ml": p["pred_V2_ml"]+122.,
                    "reason": "Synthetic measurement; not real truth.", "hypothesis_id": None}
                    for p in frozen["premeasurement_predictions"]],
                "hypotheses": [], "feedback_interpretation": "Synthetic only.", "batch_rationale": "Mock."}]
            validate_history(history, seed, arm, 1)
            try:
                validate_history(history, seed, METHODS[1] if arm == METHODS[0] else METHODS[0], 1)
            except ValueError:
                pass
            else:
                raise RuntimeError("cross-arm history accepted")
            batches[arm] = frozen["batch_ids"]
            packets[arm] = packet
            once(directory / "synthetic_next_state.json", {"synthetic": True,
                "L_t_ids": roles["l0"] + frozen["batch_ids"],
                "U_t_ids": [i for i in roles["u0"] if i not in chosen],
                "feedback": history, "real_labels_revealed": 0})
            # A second real-format request tests nested feedback/memory as well.
            next_raw = copy.deepcopy(raw)
            next_raw["candidates"] = [x for x in next_raw["candidates"] if x["id"] not in chosen]
            lookup = {x["id"]: x for x in raw["candidates"]}
            for record in history[0]["records"]:
                card = lookup[record["candidate_id"]]
                next_raw["observed"].append({**{k: card[k] for k in ("id", "smiles", "conditions")},
                    "record_kind": "acquired_with_premeasurement_prediction", "V1_ml": -123., "V2_ml": -122.,
                    **{f"premeasurement_prediction_{t}_ml": record[f"pred_{t}_ml"] for t in ("V1", "V2")},
                    **{f"premeasurement_error_{t}_ml": record[f"error_{t}_ml"] for t in ("V1", "V2")}})
            next_roles = {"l0": roles["l0"] + frozen["batch_ids"],
                          "u0": [i for i in roles["u0"] if i not in chosen]}
            next_dir = directory.parent / "round_01"
            next_catalog, next_packet = write_stage(next_dir, next_raw, next_roles, source, seed, arm, 1, history)
            next_response = mock_plan(next_packet)
            next_result = run_selection(next_packet, next_catalog, next_dir, variant, config,
                lambda *args: (json.dumps(next_response), {"mock": True, "usage": None}))
            freeze_batch(next_dir, next_result, next_catalog, next_packet)
            audit_batch(next_dir)
        diff = pair_diff(packets[METHODS[0]], packets[METHODS[1]])
        once(STUDY / f"dry_run/seed_{seed}/input_diff.json", diff)
        audits.append({"seed": seed, "input_diff": {k: v for k, v in diff.items() if k != "removed"},
            "removed_CW_values": len(diff["removed"]), "initial_checkpoint": source,
            "mock_batches_independent": not set(batches[METHODS[0]]) & set(batches[METHODS[1]]),
            "no_cw_leak_audit": audit_no_cw({"prompt": prompt("no_cw"), "packet": packets[METHODS[1]]})})
    result = {"status": "PASS", "seeds": audits, "real_llm_calls": 0, "model_fits": 0,
              "new_real_labels_revealed": 0, "validation_test_truth_reads": 0,
              "synthetic_feedback_only": True, "formal_runtime_created": False}
    once(STUDY / "dry_run_audit.json", result)
    return result


def report(budget=1005):
    """Offline validation report; never exposed to the selection packet or memory."""
    validate()
    if budget not in (525, 1005):
        raise ValueError("report milestone must be 525 or 1005")
    rounds = (budget-333)//32
    # Whole paired matrix barrier BEFORE any validation labels or model loading.
    for seed in SEEDS:
        for arm in METHODS:
            for r in range(rounds):
                frozen = audit_batch(selection_directory(seed, arm, r))
                if frozen["dry_run"]:
                    raise PermissionError("cannot report mock trajectories")
                receipt = read(selection_directory(seed, arm, r) / "label_access_receipt.json")
                if receipt["batch_freeze_sha256"] != sha256_file(selection_directory(seed, arm, r)/"batch_freeze.json"):
                    raise RuntimeError("report lineage mismatch")
            read(STUDY / f"runtime/seed_{seed}/{arm}/round_{rounds:02d}/prediction_freeze.json")
    from ..models import load_predictor_checkpoint
    from .runner import predict_outputs
    from .benchmark_reporting import metric_row
    torch.set_num_threads(2)
    metrics, summaries, distributions, costs = [], [], [], []
    for seed in SEEDS:
        context = Context(seed)
        truth = context.new_store().reveal(context.ids(context.roles["validation"]), "initial_fit")
        for arm in METHODS:
            curve = []
            for r in range(rounds+1):
                labeled, _, _, _ = state(context, arm, r)
                checkpoint, _ = prediction_freeze(context, arm, r, labeled)
                prediction, order = predict_outputs(load_predictor_checkpoint(checkpoint), context.atom, context.angle, context.roles["validation"])
                if not np.array_equal(order, context.roles["validation"]):
                    raise RuntimeError("validation prediction ID drift")
                row = {"seed": seed, "arm": arm, "budget": BUDGETS[r],
                       **metric_row(truth, prediction, context.preprocessing["target_scales"])}
                metrics.append(row)
                curve.append(row["combined_normalized_RMSE"])
                if r == rounds:
                    continue
                directory = selection_directory(seed, arm, r)
                frozen, raw = audit_batch(directory), read(directory / "catalog.json")
                catalog = ReadOnlyCatalog(**raw)
                for i in frozen["batch_ids"]:
                    card = catalog.candidates[i]
                    distributions.append({"seed": seed, "arm": arm, "round": r, "id": i,
                        **{k: card[k] for k in ("smiles", "conditions", "descriptors", "molecule_novelty",
                            "condition_novelty", "max_similarity_to_observed", "loading_amount_mg")}})
                for p in sorted(directory.glob("attempt_??_??.json")):
                    attempt = read(p)
                    costs.append({"seed": seed, "arm": arm, "round": r,
                        "success": attempt["success"], "elapsed_seconds": attempt["elapsed_seconds"],
                        "usage": attempt.get("provenance", {}).get("usage")})
            area = sum((curve[j]+curve[j+1])*.5*32 for j in range(rounds))/(budget-333)
            summaries.append({"seed": seed, "arm": arm, "validation_AULC": area})
    result = {"budget": budget, "metrics": metrics, "AULC": summaries,
        "paired_delta_NoCW_minus_CWAware": {str(s): next(x["validation_AULC"] for x in summaries if x["seed"] == s and x["arm"] == METHODS[1]) -
            next(x["validation_AULC"] for x in summaries if x["seed"] == s and x["arm"] == METHODS[0]) for s in SEEDS},
        "sampling_distribution": distributions, "cost_records": costs,
        "currency_cost": None, "currency_cost_note": "No frozen provider tariff; report usage without inventing prices.",
        "selector_feedback": False, "test_truth_reads": 0}
    once(STUDY / f"reports/validation_to{budget}.json", result)
    return result
