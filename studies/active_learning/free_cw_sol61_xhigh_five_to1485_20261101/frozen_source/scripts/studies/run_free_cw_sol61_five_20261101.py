#!/usr/bin/env python3
"""Isolated CW-visible Free32 rerun; no arguments is a read-only status call."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

ENTRY_ROOT = Path(__file__).resolve().parents[2]
ROOT = ENTRY_ROOT.parents[3] if ENTRY_ROOT.name == "frozen_source" else ENTRY_ROOT
STUDY = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
FROZEN = STUDY / "frozen_source"
ENTRY = FROZEN / "scripts/studies" / Path(__file__).name
ARM = "free_cw_sol61_xhigh_rerun32"
VERSION = "free_cw_sol61_xhigh_five_to1485_20261101_v1"
SEEDS = (157, 887, 2357, 6101, 12203)
BUDGETS = tuple(range(333, 1486, 32))
MILESTONES = (525, 813, 1005, 1389, 1485)
NOCW = ROOT / "studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze_sources():
    """Only copies source; original NoCW and its transitive dependencies stay intact."""
    if (STUDY / "source_snapshot.json").exists():
        return
    if FROZEN.exists():
        raise RuntimeError("partial source snapshot requires inspection")
    files = sorted((ROOT / "src").rglob("*.py"))
    files += sorted((ROOT / "application").rglob("*.py"))
    files += sorted((ROOT / "scripts").rglob("*.py"))
    files += [ROOT / "tests/active_learning_v2/test_free_cw_sol61_five.py"]
    sources, copies = {}, {}
    for source in files:
        name = str(source.relative_to(ROOT))
        target = FROZEN / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = sha(source)
        if sha(target) != sources[name]:
            raise RuntimeError("source changed during snapshot")
        if name == "src/qgeognn_al/resources.py":
            original = target.read_text()
            needle = "ROOT = Path(__file__).resolve().parents[2]"
            if original.count(needle) != 1:
                raise RuntimeError("resource-root patch no longer matches")
            target.write_text(original.replace(needle, f"ROOT = Path({str(ROOT)!r})"))
        copies[name] = sha(target)
    # Standard-library atomic write, before importing any frozen study module.
    p = STUDY / "source_snapshot.json"
    p.write_text(json.dumps({"created_utc": now(), "source_hashes": sources,
        "frozen_hashes": copies, "only_copy_edit": "resources.ROOT binds the original read-only data repository",
        "shared_source_files_modified": 0}, indent=2) + "\n")


def import_frozen():
    global base, s, views, torch, pd, fitrunner, shortseq, stream
    if not FROZEN.is_dir():
        raise RuntimeError("run prepare before execution")
    sys.path.insert(0, str(FROZEN))
    import pandas as pd
    import torch
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_five as base
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    from src.qgeognn_al.active_learning_v2 import nocw_study as s, nocw_inputs as views
    from src.qgeognn_al.active_learning_v2 import runner as fitrunner
    from src.qgeognn_al.active_learning_v2 import short_sequential_runner as shortseq
    if not Path(s.__file__).resolve().is_relative_to(FROZEN):
        raise RuntimeError("shared module accidentally imported instead of frozen source")


def prompt(variant):
    if variant != "cw_aware":
        raise ValueError("only the CW-visible Free trajectory is registered")
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import DIRECT_PROMPT
    return DIRECT_PROMPT.replace("Pending-CW outcomes are identified separately.\n", "")


def input_view(raw, seed, arm, r, history):
    # Same positive schema and history checks as five-seed NoCW. Only its
    # no-CW negative scans are inapplicable to this registered CW-visible arm.
    _, view, packet = base.BASE_INPUT_VIEW(raw, seed, arm, r, history)
    view["salt"] = f"llm_scientist_v2:{seed}:free_llm32_scientist_v2:{r}"
    catalog = s.ReadOnlyCatalog(**view)
    packet.update(study_version="llm_scientist_v2", method="free_llm32_scientist_v2",
                  **s.encode_catalog(catalog))
    packet["memory"] = views.scientific_memory(history, seed, arm)
    packet.pop("packet_sha256", None)
    packet["packet_sha256"] = s.stable_hash(packet)
    return catalog, view, packet


def validate_plan(result, catalog, packet, variant):
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import validate_direct
    if variant != "cw_aware":
        raise ValueError("foreign variant")
    views._only(result, views.PLAN_FIELDS, "plan")
    validate_direct(result, catalog, packet)
    for h in result["hypotheses"]:
        views._only(h, views.HYPOTHESIS_FIELDS, "hypothesis")
    for choice in result["choices"]:
        views._only(choice, {"id", "reason", "hypothesis_id"}, "choice")
    return result


def code_hashes():
    snapshot = read(STUDY / "source_snapshot.json")
    return {str((FROZEN / name).relative_to(ROOT)): sha(FROZEN / name)
            for name in snapshot["frozen_hashes"]}


def once(path, value):
    path = Path(path)
    if path == STUDY / "served_model_lock.json":
        with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/served_model.lock"):
            return base.BASE_ONCE(path, value)
    return base.BASE_ONCE(path, value)


def credential_audit():
    result = base.credential_audit()
    # File aliases are sufficient provenance; never emit key values/fingerprints.
    return {"distinct_keys": result["distinct_keys"], "key_values_logged": False,
            "key_aliases": {str(seed): f"existing-independent-seed-{seed}" for seed in SEEDS}}


def activate(seed=None):
    base.STUDY, base.VERSION = STUDY, VERSION
    base.activate(seed)
    s.BUDGETS, s.ARMS, s.METHODS = BUDGETS, {ARM: "cw_aware"}, (ARM,)
    views.ARMS = {**views.ARMS, ARM: "cw_aware"}
    s.input_view, s.prompt, s.validate_plan = input_view, prompt, validate_plan
    s.code_hashes, s.once = code_hashes, once
    s.request_with_retries, s.live_call = request_with_retries, live_call
    # Add durable completion of the unchanged scratch-training call, not a new
    # optimizer/checkpoint rule. This patch only exists in this worker process.
    shortseq.fit_from_same_initialization = guarded_fit
    fitrunner.train_source = durable_train_source
    stream.STUDY = STUDY


class FatalProtocolError(RuntimeError):
    pass


def live_call(messages, config):
    raise RuntimeError("live requests require the durable request ledger")


def response_result(response, config, request_id=None):
    model = response.get("model")
    provenance = {"response_id": response.get("id"), "served_model": model,
        "request_id": request_id, "usage": response.get("usage"), "native_tool_calls": 0,
        "requested_model": config["model"], "reasoning_effort": config["reasoning_effort"],
        "received_utc": now(), "provider_source": "frozen-provider; per-seed credential alias"}
    answer = "".join(part.get("text", "") for item in response.get("output", [])
                     if item.get("type") == "message" for part in item.get("content", [])
                     if part.get("type") == "output_text")
    if (response.get("status") != "completed" or not isinstance(model, str) or
        not (model == config["model"] or model.startswith(config["model"] + "-")) or
        any(item.get("type") not in ("message", "reasoning") for item in response.get("output", [])) or
        not isinstance(response.get("id"), str) or not answer.strip()):
        raise FatalProtocolError("incomplete/wrong-model/tool/missing-response-id response")
    try:
        once(STUDY / "served_model_lock.json", {"served_model": model})
    except RuntimeError:
        raise FatalProtocolError("served model differs from the first locked returned identity") from None
    return answer, provenance


def live_stream(messages, config, directory, turn, attempt):
    import httpx
    from src.qgeognn_al.active_learning_v2.scientist_transport import provider_key
    s.require_authorization()
    prefix = directory / f"stream_{turn:02d}_{attempt:02d}"
    request_hash = s.stable_hash({"messages": messages, "config": config})
    state = {"seed": WORKER_SEED, "pid": os.getpid(), "turn": turn, "attempt": attempt,
        "request_sha256": request_hash, "phase": "CONNECTING", "started_utc": now(),
        "requested_model": config["model"], "reasoning_effort": config["reasoning_effort"],
        "response_id": None, "request_id": None, "event_count": 0}
    last_write = 0

    def persist(force=False):
        nonlocal last_write
        if force or time.monotonic() - last_write >= 15:
            state["updated_utc"] = now()
            base.atomic_json(prefix.with_suffix(".json"), state)
            base.atomic_json(STUDY / f"progress/stream_seed_{WORKER_SEED}.json", state)
            last_write = time.monotonic()

    def event(data):
        kind = data.get("type", "unknown")
        state.update(event_count=state["event_count"] + 1, last_event_type=kind,
                     last_event_utc=now(), last_sequence_number=data.get("sequence_number"))
        response = data.get("response", {})
        if response.get("id"):
            state["response_id"] = response["id"]
        if kind in ("response.created", "response.queued", "response.in_progress"):
            state["phase"] = kind.split(".")[1].upper()
            persist(True)
        if kind == "response.completed":
            # Save final text/usage before validation and socket close. Reasoning
            # content is never retained in the metadata stream journal.
            safe_response = {k: response.get(k) for k in ("id", "model", "status", "usage")}
            safe_response["output"] = [item if item.get("type") == "message" else {"type": item.get("type")}
                                       for item in response.get("output", [])]
            once(prefix.with_suffix(".response.json"), {"request_sha256": request_hash,
                "response": safe_response, "request_id": state["request_id"], "received_utc": now()})
            answer, provenance = response_result(safe_response, config, state["request_id"])
            result = {"request_sha256": request_hash, "answer": answer, "provenance": provenance,
                      "validated_completed": True}
            once(prefix.with_suffix(".completed.json"), result)
            state.update(phase="COMPLETED", served_model=provenance["served_model"], usage=provenance["usage"])
            persist(True)
            return answer, provenance
        if kind in ("response.failed", "response.incomplete", "error"):
            code = response.get("error", {}).get("code", data.get("code", "unknown"))
            if code in ("server_error", "rate_limit_exceeded"):
                raise httpx.RemoteProtocolError("provider transient terminal code: " + str(code))
            raise FatalProtocolError("provider terminal code: " + str(code)[:80])
        persist()
        return None

    persist(True)
    durable = None
    try:
        with stream.client_for(config) as client:
            with client.stream("POST", config["base_url"].rstrip("/") + "/responses",
                headers={"Authorization": "Bearer " + provider_key(config), "Accept": "text/event-stream"},
                json=stream.body_for(messages, config)) as response:
                state.update(http_status=response.status_code, phase="CONNECTED",
                    request_id=response.headers.get("x-request-id"), headers_received_utc=now())
                persist(True)
                if response.status_code not in (200,):
                    if response.status_code not in (408, 409, 429, 500, 502, 503, 504):
                        raise FatalProtocolError("provider HTTP " + str(response.status_code))
                    response.raise_for_status()
                lines = []
                for line in response.iter_lines():
                    state["last_line_utc"] = now()
                    persist()
                    if line.startswith("data:"):
                        lines.append(line[5:].lstrip(" "))
                    elif line == "" and lines:
                        payload, lines = "\n".join(lines), []
                        if payload.strip() == "[DONE]":
                            continue
                        durable = event(json.loads(payload))
                        if durable is not None:
                            return durable
                if lines and "\n".join(lines).strip() != "[DONE]":
                    durable = event(json.loads("\n".join(lines)))
                    if durable is not None:
                        return durable
                raise httpx.RemoteProtocolError("stream ended before response.completed")
    except Exception as error:
        if durable is not None:
            return durable
        state.update(phase="FAILED", error_type=type(error).__name__, ended_utc=now())
        persist(True)
        raise


def request_with_retries(messages, config, directory, turn, call):
    import httpx
    directory = Path(directory)
    request_hash = s.stable_hash({"messages": messages, "config": config})
    for attempt in range(config["transport_attempts"]):
        path = directory / f"attempt_{turn:02d}_{attempt:02d}.json"
        started = path.with_suffix(".started.json")
        prefix = directory / f"stream_{turn:02d}_{attempt:02d}"
        if path.exists():
            saved = read(path)
            if saved["request_sha256"] != request_hash:
                raise RuntimeError("request drift")
            if saved.get("fatal_protocol_error"):
                raise FatalProtocolError(saved["reason"])
            if saved.get("success"):
                return saved
            continue
        completed = prefix.with_suffix(".completed.json")
        raw_response = prefix.with_suffix(".response.json")
        if completed.exists() or raw_response.exists():
            cached = read(completed if completed.exists() else raw_response)
            if not started.exists() or read(started)["request_sha256"] != request_hash or cached["request_sha256"] != request_hash:
                raise RuntimeError("local completed-response binding mismatch")
            if completed.exists():
                answer, provenance = cached["answer"], cached["provenance"]
                model = provenance["served_model"]
                once(STUDY / "served_model_lock.json", {"served_model": model})
                if not (model == config["model"] or model.startswith(config["model"] + "-")):
                    raise FatalProtocolError("cached response has wrong model")
            else:
                answer, provenance = response_result(cached["response"], config, cached["request_id"])
            saved = {"request_sha256": request_hash, "success": True, "answer": answer,
                "provenance": provenance, "elapsed_seconds": max(0, (completed if completed.exists() else raw_response).stat().st_mtime-started.stat().st_mtime),
                "recovered_local_completion": True}
            once(path, saved)
            return saved
        if started.exists():
            raise RuntimeError("ambiguous interrupted provider request; accounting required; never auto-resend")
        once(started, {"request_sha256": request_hash, "started_utc": now(),
            "requested_model": config["model"], "reasoning_effort": config["reasoning_effort"]})
        begin = time.monotonic()
        try:
            answer, provenance = (live_stream(messages, config, directory, turn, attempt)
                                  if call is live_call else call(copy.deepcopy(messages), copy.deepcopy(config)))
            if not isinstance(answer, str) or not answer.strip():
                raise FatalProtocolError("empty answer")
        except FatalProtocolError as error:
            once(path, {"request_sha256": request_hash, "success": False, "fatal_protocol_error": True,
                "error_type": type(error).__name__, "reason": str(error), "ended_utc": now(),
                "elapsed_seconds": time.monotonic()-begin})
            raise
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            once(path, {"request_sha256": request_hash, "success": False, "error_type": type(error).__name__,
                "http_status": error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None,
                "ended_utc": now(), "elapsed_seconds": time.monotonic()-begin})
            if attempt + 1 < config["transport_attempts"]:
                delay = config["bounded_backoff_seconds"][attempt]
                base.status_record(WORKER_SEED, "TRANSPORT_BACKOFF", turn=turn, attempt=attempt, seconds=delay)
                time.sleep(delay)
            continue
        saved = {"request_sha256": request_hash, "success": True, "answer": answer, "provenance": provenance,
            "ended_utc": now(), "elapsed_seconds": time.monotonic()-begin}
        once(path, saved)
        return saved
    raise RuntimeError("bounded transport attempt budget exhausted; no fallback")


def guarded_fit(**kwargs):
    from src.qgeognn_al.active_learning_v2.cache import array_hash
    intent = {k: kwargs[k] for k in ("initialization_seed", "training_config", "contract", "preprocessing")}
    intent.update(train_truth_hash=array_hash(kwargs["train_truth"]),
                  validation_truth_hash=array_hash(kwargs["validation_truth"]))
    once(Path(kwargs["runtime"]) / "fit_intent.json", intent)
    return ORIGINAL_FIT(**kwargs)


def durable_train_source(model, atom, angle, train_indices, validation_indices, preprocessing, config, checkpoint_path, progress=None):
    from src.qgeognn_al.active_learning_v2.gradient_features import state_dict_hash
    checkpoint_path = Path(checkpoint_path)
    receipt = checkpoint_path.parent / "training_complete.json"
    intent = {"initialization_state_hash": state_dict_hash(model),
        "train_indices": list(map(int, train_indices)), "validation_indices": list(map(int, validation_indices)),
        "preprocessing": preprocessing, "training_config": config,
        "fit_intent_sha256": sha(checkpoint_path.parent / "fit_intent.json")}
    if receipt.exists():
        saved = read(receipt)
        if saved["intent"] != intent or saved["checkpoint_sha256"] != sha(checkpoint_path):
            raise RuntimeError("completed training receipt drift")
        return saved["history"], saved["best_epoch"]
    def heartbeat(event):
        base.atomic_json(STUDY / f"progress/training_seed_{WORKER_SEED}.json",
            {"seed": WORKER_SEED, "pid": os.getpid(), "updated_utc": now(), "phase": "TRAINING", **event})
        if progress:
            progress(event)
    history, best = ORIGINAL_TRAIN(model, atom, angle, train_indices, validation_indices, preprocessing,
                                   config, checkpoint_path, progress=heartbeat)
    once(receipt, {"intent": intent, "history": history, "best_epoch": best,
                   "checkpoint_sha256": sha(checkpoint_path), "completed_utc": now()})
    return history, best


def verify_fit(seed, r):
    d = STUDY / f"runtime/seed_{seed}/{ARM}/round_{r:02d}"
    frozen = read(d / "prediction_freeze.json")
    model = Path(frozen["checkpoint_path"]).parent
    audit = read(model / "fit_audit.json")
    for kind, filename in (("checkpoint", "best.pt"), ("prediction", "predictions.csv.gz")):
        if audit[kind+"_sha256"] != frozen[kind+"_sha256"] or sha(model / filename) != frozen[kind+"_sha256"]:
            raise RuntimeError("completed predictor artifact drift")
    initial = read(base.initial_paths(seed)["member0_audit"])
    if (audit["train_rows"] != BUDGETS[r] or audit["initialization_hash"] != initial["initialization_hash"] or
        audit["initialization_seed"] != seed+2000003 or audit["train_ids_hash"] != frozen["L_t_ids_hash"] or
        audit["test_labels_used_for_fit_or_checkpoint_selection"] != 0 or frozen["test_truth_access_count"] != 0):
        raise RuntimeError("fit budget/initialization/label barrier mismatch")
    return audit


def offline_check():
    s.validate()
    checks = []
    for seed in SEEDS:
        raw, roles, source = s.initial_state(seed)
        partition = pd.read_csv(base.initial_paths(seed)["split"])
        nocw_context = read(NOCW / f"runtime/seed_{seed}/context.json")
        if (nocw_context["split_hash"] != s.stable_hash(partition.to_dict("list")) or
            nocw_context["preprocessing"] != source["preprocessing"] or
            nocw_context["normalization"] != source["normalization"]):
            raise RuntimeError("actual NoCW split/preprocessing/normalization differs")
        # Independently reconstruct the true initial CW percentiles from the
        # hash-verified, checkpoint/IDs-bound L333 feature cache (no labels).
        import numpy as np
        from src.qgeognn_al.active_learning_v2.llm_screen import CW_STUDY
        from src.qgeognn_al.active_learning_v2.lcmd import _squared_distances
        cw_path = CW_STUDY / f"runtime/seed_{seed}/center_width_lcmd/round_00/acquisition_artifacts/current_transformed_gradient_features.npz"
        current = np.r_[partition.loc[partition.role.eq("l0"),"canonical_index"].to_numpy(int),
                        partition.loc[partition.role.eq("u0"),"canonical_index"].to_numpy(int)]
        if seed in (157,6101):
            cw_receipt_path = cw_path.with_suffix(".npz.contract.json")
            cache_receipt = read(cw_receipt_path)
            binding = cache_receipt["contract"]
            if (cache_receipt["sha256"] != sha(cw_path) or binding["checkpoint_sha256"] != source["checkpoint_sha256"] or
                binding["L_t_ids_hash"] != source["L333_ids_hash"] or binding["U_t_ids_hash"] != source["U333_ids_hash"]):
                raise RuntimeError("initial CW feature cache lineage mismatch")
            with np.load(cw_path) as cached:
                if not np.array_equal(cached["canonical_indices"], current):
                    raise RuntimeError("initial CW feature identity/order differs")
                features = cached["features"]
        else:
            # The three later historical seeds have the same L333 sketches in
            # their own initial banks, not in the earlier two-seed CW study.
            owner = ROOT / "studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol61_xhigh_free_five_seeds_to1005_20261023"
            cw_path = owner / f"runtime/seed_{seed}/free_llm32_scientist_v2/round_00/banks.npz"
            cw_receipt_path = cw_path.with_suffix(".json")
            cache_receipt = read(cw_receipt_path)
            binding = cache_receipt["contract"]
            if (cache_receipt["sha256"] != sha(cw_path) or binding["checkpoint"] != source["checkpoint_sha256"] or
                binding["indices"] != current.tolist() or binding["protocol"] != s.stable_hash(read(owner / "protocol.json"))):
                raise RuntimeError("initial bank hash/checkpoint/order/protocol mismatch")
            with np.load(cw_path) as cached:
                features = cached["cw"]
        ranks = pd.Series(np.sqrt(_squared_distances(features[333:],features[:333]).min(axis=1))).rank(pct=True).to_numpy()
        # dialog_study._card, used by the actual NoCW lifecycle, serializes
        # this column with Python round(value, 5). Preserve that exact rule.
        if [round(float(value),5) for value in ranks] != [row[views.CW_FIELD] for row in raw["candidates"]]:
            raise RuntimeError("catalog CW percentile differs from actual distance rank")
        # Reconstruct the actual NoCW view using its exact frozen wrapper.
        old_variant = views.ARMS[ARM]
        views.ARMS[ARM] = "no_cw"
        try:
            _, _, reconstructed = base.input_view(raw, seed, ARM, 0, [])
        finally:
            views.ARMS[ARM] = old_variant
        historical = read(NOCW / f"selections/seed_{seed}/llm32_nocw/round_00/input.json")
        if reconstructed != historical:
            raise RuntimeError("reconstructed NoCW packet differs from its actual initial input")
        directory = STUDY / f"dry_run/seed_{seed}/{ARM}/round_00"
        catalog, packet = s.write_stage(directory, raw, roles, source, seed, ARM, 0, [])
        expected = copy.deepcopy(packet)
        col = expected["candidates"]["columns"].index(views.CW_FIELD)
        expected["candidates"]["columns"].pop(col)
        removed = [{"table": "candidates", "row": j, "id": row[0], "field": views.CW_FIELD, "value": row.pop(col)}
                   for j, row in enumerate(expected["candidates"]["rows"])]
        expected.pop("packet_sha256")
        if expected != {k: v for k, v in historical.items() if k != "packet_sha256"}:
            raise RuntimeError("unregistered non-CW packet difference")
        if prompt("cw_aware").replace(views.CW_DESCRIPTION, "") != base.prompt("no_cw"):
            raise RuntimeError("unregistered prompt difference")
        if any(packet["memory"][k] for k in packet["memory"] if k != "memory_policy"):
            raise RuntimeError("initial memory is not empty")
        actual_freeze = read(NOCW / f"runtime/seed_{seed}/llm32_nocw/round_00/prediction_freeze.json")
        if (source["checkpoint_sha256"] != actual_freeze["checkpoint_sha256"] or
            sha(base.initial_paths(seed)["member0_predictions"]) != actual_freeze["prediction_sha256"] or
            source["L333_ids_hash"] != actual_freeze["L_t_ids_hash"]):
            raise RuntimeError("NoCW initial checkpoint/prediction/state differs")
        result = s.run_selection(packet, catalog, directory, "cw_aware", read(STUDY / "config.json")["transport"],
            lambda *args: (json.dumps(s.mock_plan(packet)), {"mock": True}))
        freeze = s.freeze_batch(directory, result, catalog, packet)
        s.audit_batch(directory)
        rejected = []
        for name, mutate in (("duplicate_candidate", lambda x: x["choices"][1].update(id=x["choices"][0]["id"])),
            ("observed_as_candidate", lambda x: x["choices"][0].update(id=next(iter(catalog.observed)))),
            ("candidate_as_evidence", lambda x: x.update(hypotheses=[{"id":"h", "content":"prior", "basis":"model_behavior",
                "supporting_observed_ids":[x["choices"][0]["id"]], "contradicting_observed_ids":[], "confidence":"low",
                "status":"proposed", "why_it_matters_for_model_learning":"test", "next_discriminating_question":"test?"}]))):
            bad = copy.deepcopy(result)
            mutate(bad)
            try:
                validate_plan(bad, catalog, packet, "cw_aware")
            except ValueError:
                rejected.append(name)
            else:
                raise RuntimeError("negative identity/evidence test failed: " + name)
        legal_cw = copy.deepcopy(result)
        legal_cw["batch_strategy"] = "Use CW distance percentile as optional model sensitivity evidence."
        validate_plan(legal_cw, catalog, packet, "cw_aware")
        diff = {"seed": seed, "passed": True, "changes": removed,
            "derived_hash_difference": {"new": packet["packet_sha256"], "nocw": historical["packet_sha256"]},
            "prompt_addition": views.CW_DESCRIPTION, "non_CW_content_and_order_identical": True,
            "NoCW_input_sha256": sha(NOCW / f"selections/seed_{seed}/llm32_nocw/round_00/input.json")}
        once(STUDY / f"offline_checks/packet_diff_seed_{seed}.json", diff)
        prediction = pd.read_csv(base.initial_paths(seed)["member0_predictions"])
        if prediction.sample_id.astype(str).tolist() != roles["test"]:
            raise RuntimeError("initial prediction identity/order mismatch")
        once(STUDY / f"offline_checks/initial_state_seed_{seed}.json", {"roles": roles, **source,
            "prediction_sha256": actual_freeze["prediction_sha256"], "NoCW_prediction_freeze": actual_freeze,
            "true_CW_rank_verified":True,"CW_cache_sha256":sha(cw_path),"CW_cache_receipt_sha256":sha(cw_receipt_path)})
        checks.append({"seed": seed, "L0": len(roles["l0"]), "U0": len(roles["u0"]),
            "checkpoint_sha256": source["checkpoint_sha256"], "prediction_sha256": actual_freeze["prediction_sha256"],
            "batch_size": len(freeze["batch_ids"]), "negative_tests_rejected": rejected,
            "initial_memory_empty": True, "true_CW_rank_verified":True,"actual_NoCW_split_and_preprocessing_identical":True,
            "full_input_diff_passed": True})
    snapshot = read(STUDY / "source_snapshot.json")
    for name, digest in snapshot["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("shared source changed during setup: " + name)
    result = {"status": "PASS", "seeds": checks, "real_llm_calls": 0, "new_labels_revealed": 0,
        "model_fits": 0, "shared_source_unchanged": True, "report_and_learning_curve_inputs": False,
        "initial_artifacts_reused_read_only": True, "test_truth_reads": 0}
    once(STUDY / "offline_checks/audit.json", result)
    return result


def prepare():
    credentials = credential_audit()
    config = read(NOCW / "config.json")
    config.update(version=VERSION, status="USER_AUTHORIZED_FORMAL_CW_FREE_RERUN", arms={ARM:"cw_aware"},
        primary_endpoint=1485, secondary_endpoints=list(MILESTONES),
        primary_metric="validation normalized trapezoidal raw label-AULC333..1485; fixed L333 target scales")
    config["transport"].update(selection_attempts=8, transport_attempts=4, bounded_backoff_seconds=[2,10,60],
        served_model_policy="exact returned identity locked on first formal completion; affected trajectory stops on drift")
    config["execution"] = {"seed_workers": 1, "training_workers": 1, "queue": "least completed rounds, fixed seed tie order",
        "target_budget":1485, "accepted_batches":180, "rounds_per_seed":36,
        "key_assignment":credentials["key_aliases"], "source_workers_interrupted":False,
        "provider_support":"first formal round only; no paid probes; no fallback",
        "maximum_local_restarts_per_seed":2, "maximum_answers_per_round":8,
        "maximum_network_attempts_per_answer":4, "maximum_network_attempts_per_round":32}
    once(STUDY / "config.json", config)
    once(STUDY / "credential_audit.json", credentials)
    docs = {
        "PROTOCOL.md": """# CW-visible Free32 rerun\n\nFive seeds 157,887,2357,6101,12203. L333 to L1485 in 36 batches of 32 per seed. Request gpt-6.1-sol / xhigh throughout. Scratch fit after every frozen batch, using the unchanged NoCW initialization, RNG, preprocessing, split, optimizer, epochs, patience and validation checkpoint rule. Initial L333 only is reused after byte/state hashes. New selections/history/models never import old Free/NoCW feedback. Reports never enter the selector. No test labels are accessed.\n\nComplete full catalogs; lossless precision; historical Free salt, ID/order/encoding; same five-seed NoCW memory and direct validator. Candidate IDs and measured evidence IDs have separate validators. No quotas or numerical acquisition formula.\n\nEight answer attempts (one initial plus seven corrections), four transport attempts each; backoff 2/10/60 seconds. Counts persist across restarts. No retries for model/config/credential/terminal protocol violations. Ambiguous in-flight requests without durable completion stop for accounting. Durable completed responses and scratch fits are replayed locally. At most two local recoverable worker restarts per seed.\n\nOne seed-round worker and one new training fit at a time; queue by completed round count then fixed seed order. Existing NoCW retains resources and its original budgets/config. NoCW ADVANCING phases get training priority.\n\nMilestones L525,L813,L1005,L1389,L1485 require all five seeds. Raw nonmonotone normalized trapezoidal AULC; never best-so-far. Paired comparisons only over five-seed common completed budgets. Historical Free keeps its mixed-model label. API randomness, dates and invisible backend changes prevent strict CW causal attribution.\n""",
        "SCHEMA_DIFF.md": """# Necessary interface differences\n\nVersus actual five-seed NoCW: restore cw_distance_percentile for each candidate and its original two-line definition; recompute packet binding hash. Skip NoCW-only negative scans on inputs, history, outputs and correction transcript; retain positive schemas and common direct checks. The private arm/version/output paths change. The visible historical study_version/method/salt remain identical. NoCW's inherited hybrid wording is retained to avoid unrelated prompt edits. Initial scientific memory is empty. The original memory-policy wording remains identical.\n\nNoCW's currently authorized unbounded retry extensions are replaced here with a preregistered finite 8-answer x 4-transport bound. Frozen SSE uses the same request body, direct route, timeout and model parameters; adds durable response/usage/request-ID journals. Training completion is journaled before prediction export, so export failures reuse finished scratch training. Neither addition changes training or checkpoint selection.\n\nNew source snapshot changes only resources.ROOT to address read-only original data. Runtime adapters are process-local and never edit NoCW dependencies. Initial packet diffs are enumerated per candidate in offline_checks/packet_diff_seed_*.json.\n""",
        "CODE_AUDIT.md": """# Boundaries\n\nRead applicable ancestor/repository AGENTS.md search: none found. Inspected actual five-seed NoCW, its L1389 extension, lifecycle, full delivery, positive schemas, negative scans, scratch fitter and prior diagnostics. Existing worktree was dirty and retained. No pull/checkout/reset. Frozen modules run in a separate process with independent study, locks, progress, logs and formal authorization.\n\nCatalog construction loads feature-only CSV columns. RestrictedLabelStore reveals L0 and selected frozen training batches; validation is used only for fitting/checkpoint selection. Test X and frozen test predictions are permitted; test truth remains unread. Initial-state manifest pins splits/catalog/checkpoint/prediction/context/scrubbed cache and data hashes. Reports run in separate processes, read saved metrics only, and cannot mutate selector memory. No paid offline checks.\n"""}
    for name, value in docs.items():
        path = STUDY / name
        if path.exists() and path.read_text() != value:
            raise RuntimeError("protocol document drift")
        path.write_text(value)
    prepared = s.prepare()
    checked = offline_check()
    once(STUDY / "formal_authorization.json", {**s.validate(), "formal_run_authorized":True,
        "user_authorization":"2026-11-01: 完整重跑带CW Free32五seed；离线检查后直接正式启动；API、冻结后标签揭示、逐轮重训及可恢复修复均已授权",
        "target_budget":1485, "accepted_batches":180, "paid_probes":False,
        "transport":config["transport"]})
    for seed in SEEDS:
        p = STUDY / f"progress/seed_{seed}.json"
        if not p.exists():
            base.status_record(seed, "QUEUED", current_budget=333, completed_rounds=0, target_budget=1485)
    return {**prepared, "offline_status":checked["status"], "authorized":True}


def completed_rounds(seed):
    count = 0
    for r in range(36):
        p = STUDY / f"progress/completed_seed_{seed}_round_{r:02d}.json"
        if not p.exists():
            break
        record = read(p)
        for name, digest in record["files"].items():
            if sha(STUDY / name) != digest:
                raise RuntimeError("completed round drift")
        count += 1
    return count


def training_priority_wait():
    # Never signal, edit, or acquire a lock owned by the existing NoCW jobs.
    roots = [NOCW, NOCW.parent / "five_seed_nocw_to1389_20261101"]
    while True:
        busy = []
        for root in roots:
            for p in (root / "progress").glob("seed_*.json"):
                state = read(p)
                if state.get("phase") == "ADVANCING" and alive(state.get("pid")):
                    busy.append({"seed":state["seed"], "pid":state["pid"]})
        if not busy:
            return
        base.status_record(WORKER_SEED, "WAITING_NOCW_TRAINING", source_workers=busy)
        time.sleep(15)


def worker(seed):
    global WORKER_SEED
    WORKER_SEED = seed
    activate(seed)
    s.require_authorization()
    credential_audit()
    os.environ.pop("SCIENTIST_API_KEY", None)
    os.environ["SCIENTIST_API_KEY_FILE"] = str(base.KEY_FILES[seed])
    torch.set_num_threads(2)
    with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/new_free_resource.lock"):
        with base.BASE_EXCLUSIVE_LOCK(STUDY / f"locks/worker_seed_{seed}.lock"):
            r = completed_rounds(seed)
            if r == 36:
                return
            try:
                for j in range(r):
                    s.audit_batch(s.selection_directory(seed, ARM, j))
                    verify_fit(seed, j+1)
                base.status_record(seed, "SELECTING", round=r, current_budget=BUDGETS[r], target_budget=1485)
                s.select(seed, ARM, r, allow_paid_llm=True, allow_feature_compute=r>0)
                training_priority_wait()
                base.status_record(seed, "ADVANCING", round=r, next_budget=BUDGETS[r+1], target_budget=1485)
                s.advance(seed, ARM, r, allow_reveal=True, allow_training=True)
                verify_fit(seed, r+1)
                d = s.selection_directory(seed, ARM, r)
                runtime = STUDY / f"runtime/seed_{seed}/{ARM}/round_{r+1:02d}"
                files = [d / n for n in ("batch_freeze.json", "feedback.json", "label_access_receipt.json")]
                files += [runtime / "prediction_freeze.json", runtime / "model/fit_audit.json", runtime / "model/history.csv"]
                once(STUDY / f"progress/completed_seed_{seed}_round_{r:02d}.json", {"seed":seed,"round":r,
                    "budget":BUDGETS[r+1], "completed_utc":now(), "files":{str(p.relative_to(STUDY)):sha(p) for p in files}})
                base.status_record(seed, "COMPLETE" if r==35 else "QUEUED", completed_rounds=r+1,
                    current_budget=BUDGETS[r+1], final_budget=1485 if r==35 else None, target_budget=1485)
            except BaseException as error:
                recoverable = isinstance(error, (OSError, MemoryError))
                base.status_record(seed, "RECOVERABLE_STOP" if recoverable else "FAILED_CLOSED", round=r,
                    current_budget=BUDGETS[r], error_type=type(error).__name__,
                    reason=str(error)[:700] if isinstance(error,(RuntimeError,ValueError,FatalProtocolError)) else type(error).__name__,
                    recoverable=recoverable, target_budget=1485)
                raise


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def resource_busy():
    path = STUDY / "locks/new_free_resource.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def status():
    launch = read(STUDY / "progress/launch.json") if (STUDY / "progress/launch.json").exists() else {}
    return {"study":str(STUDY), "target_budget":1485, "planned_accepted_batches":180,
        "maximum_new_concurrency":1, "launch":launch,
        "supervisor_alive":alive(launch.get("pid")),
        "seeds":[{**read(STUDY / f"progress/seed_{seed}.json"), "worker_alive":alive(read(STUDY / f"progress/seed_{seed}.json").get("pid"))}
                 for seed in SEEDS if (STUDY / f"progress/seed_{seed}.json").exists()]}


def update_reports():
    result = subprocess.run([sys.executable, str(FROZEN / "scripts/studies/report_free_cw_sol61_five_20261101.py")],
        cwd=ROOT, env=dict(os.environ), capture_output=True, text=True)
    if result.returncode:
        base.atomic_json(STUDY / "reports/report_error.json", {"time":now(),"returncode":result.returncode,"reason":result.stderr[-2000:]})
        return False
    return True


def supervise():
    activate()
    s.require_authorization()
    child, child_seed = None, None
    with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/supervisor.lock"):
        update_reports()
        while True:
            if child is not None and child.poll() is not None:
                progress = read(STUDY / f"progress/seed_{child_seed}.json")
                if child.returncode and progress.get("phase") not in ("FAILED_CLOSED", "RECOVERABLE_STOP"):
                    # Unrecorded crash can recover only after ledger inspection
                    # by the worker; ambiguous started requests fail closed there.
                    base.status_record(child_seed,"RECOVERABLE_STOP",error_type="UnexpectedWorkerExit",
                        reason="worker exited before phase receipt",recoverable=True)
                update_reports()
                child, child_seed = None, None
            rows = [read(STUDY / f"progress/seed_{seed}.json") for seed in SEEDS]
            if all(row["phase"] == "COMPLETE" for row in rows):
                reports_ok = update_reports()
                base.atomic_json(STUDY / "progress/supervisor.json", {"pid":os.getpid(),"phase":"ALL_FIVE_COMPLETE",
                    "updated_utc":now(),"target_budget":1485,"reports_updated":reports_ok,"seeds":rows})
                return
            if child is None:
                if resource_busy():
                    # A worker can survive its supervisor. Adopt its lease by
                    # waiting for release; never dispatch a duplicate request.
                    base.atomic_json(STUDY / "progress/supervisor.json", {"pid":os.getpid(),
                        "phase":"WAITING_EXISTING_WORKER_LEASE","updated_utc":now(),"seeds":rows})
                    time.sleep(10)
                    continue
                eligible = []
                for row in rows:
                    seed = row["seed"]
                    if row["phase"] in ("COMPLETE","FAILED_CLOSED"):
                        continue
                    ledger = STUDY / f"progress/local_restarts_seed_{seed}.json"
                    count = read(ledger)["count"] if ledger.exists() else 0
                    if row["phase"] == "RECOVERABLE_STOP":
                        if count >= 2:
                            base.status_record(seed,"FAILED_CLOSED",reason="bounded local restart budget exhausted")
                            continue
                    eligible.append((completed_rounds(seed),SEEDS.index(seed),seed))
                if eligible:
                    _, _, child_seed = min(eligible)
                    chosen = read(STUDY / f"progress/seed_{child_seed}.json")
                    if chosen["phase"] == "RECOVERABLE_STOP":
                        ledger = STUDY / f"progress/local_restarts_seed_{child_seed}.json"
                        count = read(ledger)["count"] if ledger.exists() else 0
                        base.atomic_json(ledger,{"count":count+1,"updated_utc":now()})
                    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
                    env.pop("SCIENTIST_API_KEY", None)
                    with (STUDY / f"progress/seed_{child_seed}.log").open("ab") as out:
                        child = subprocess.Popen([sys.executable,str(ENTRY),"worker","--seed",str(child_seed),
                            "--allow-paid-llm","--allow-training","--allow-label-reveal"],cwd=ROOT,env=env,
                            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                    base.atomic_json(STUDY / f"progress/dispatch_seed_{child_seed}.json",
                        {"seed":child_seed,"pid":child.pid,"started_utc":now()})
                elif all(row["phase"] in ("COMPLETE","FAILED_CLOSED") for row in rows):
                    update_reports()
                    base.atomic_json(STUDY / "progress/supervisor.json", {"pid":os.getpid(),"phase":"NEEDS_ATTENTION",
                        "updated_utc":now(),"seeds":rows})
                    return
            base.atomic_json(STUDY / "progress/supervisor.json", {"pid":os.getpid(),"phase":"RUNNING",
                "updated_utc":now(),"target_budget":1485,"active_seed":child_seed,
                "active_worker_pid":child.pid if child else None,"maximum_new_workers":1,"seeds":rows})
            time.sleep(10)


def launch():
    s.require_authorization()
    if read(STUDY / "offline_checks/audit.json")["status"] != "PASS":
        raise RuntimeError("offline checks did not pass")
    credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/launch.lock"):
        if (STUDY / "progress/launch.json").exists():
            raise RuntimeError("launch already recorded; inspect status instead of duplicate launch")
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        with (STUDY / "progress/supervisor.log").open("ab") as out:
            process = subprocess.Popen([sys.executable,str(ENTRY),"supervise","--allow-paid-llm","--allow-training",
                "--allow-label-reveal"],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        record = {"pid":process.pid,"started_utc":now(),"command":[sys.executable,str(ENTRY),"supervise",
            "--allow-paid-llm","--allow-training","--allow-label-reveal"],"maximum_new_workers":1,
            "maximum_new_training_workers":1,"target_budget":1485,"accepted_batches":180,"NoCW_modified":False}
        once(STUDY / "progress/launch.json", record)
        return record


WORKER_SEED = None


def main():
    global ORIGINAL_FIT, ORIGINAL_TRAIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",nargs="?",default="status",choices=("prepare","validate","status","launch","supervise","worker","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    for flag in ("allow-paid-llm","allow-training","allow-label-reveal"):
        parser.add_argument("--"+flag,action="store_true")
    args = parser.parse_args()
    if args.action in ("launch","supervise","worker") and not (args.allow_paid_llm and args.allow_training and args.allow_label_reveal):
        parser.error("all three explicit authorized cost/training/reveal flags required")
    if args.action == "prepare":
        freeze_sources()
    import_frozen()
    ORIGINAL_FIT, ORIGINAL_TRAIN = fitrunner.fit_from_same_initialization, fitrunner.train_source
    activate(args.seed if args.action=="worker" else None)
    if args.action == "worker":
        if args.seed is None:
            parser.error("worker requires --seed")
        result = worker(args.seed)
    elif args.action == "validate":
        result = s.validate()
    elif args.action == "report":
        result = {"report_updated":update_reports()}
    else:
        result = globals()[args.action]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
