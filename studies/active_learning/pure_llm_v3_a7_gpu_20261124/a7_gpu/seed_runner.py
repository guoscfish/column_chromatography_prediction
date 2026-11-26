"""Authorized A7/M2 selection-to-fit loop with independent seed receipts."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.context_builder import TARGETS
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.frozen_state import FrozenState
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import (
    _validation, select_stage, stage_state, verify_observed_alias_normalization,
)
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.trajectory import _build_feedback, _next_snapshot
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    SEALED_RELAY_RETRY_SECONDS, UNKNOWN_REPLAY_DECISION, TransportStop, check_request, is_sealed_relay_failure,
    make_request, reconcile, round_lock, terminal_record,
)

from .compact_payload import AUTHORITATIVE_PACKET_FILE, formal_wire_messages
from .formal_contract import OFFLINE, verify_authorization
from .metrics import record_batch_diagnostics

FIT_ARTIFACTS = (
    "best.pt", "predictions.csv.gz", "fit_audit.json", "history.csv", "validation_metrics.json", "fit_device.json",
    "training_complete.json",
)
PROMPT_PATH = OFFLINE / "PROMPT_PURE_LLM_V3.md"


@dataclass(frozen=True)
class SeedProgress:
    seed: int
    status: str
    completed_rounds: int
    active_labels: int


def seed_directory(directory, seed):
    return Path(directory) / f"seed_{seed}"


def round_directory(directory, round_index):
    return Path(directory) / f"round_{round_index:02d}"


def _scope(contract, authorization, seed, mock):
    return {"contract_sha256": contract.digest, "authorization_sha256": digest(authorization),
            "seed": seed, "mock": mock}


def _paused(run, seed):
    return (run / "pause.request").exists() or (seed / "pause.request").exists()


def _event(directory, scope, phase, status, round_index, error=None):
    record = {"scope": scope, "phase": phase, "status": status, "round": round_index,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    if error is not None:
        record["error_type"] = type(error).__name__
        if isinstance(error, TransportStop):
            record["transport_stop"] = str(error)
    once(directory / "events" / f"{uuid4().hex}.json", record)


def _state_roles(state, backend, scope, round_index, batch_size):
    l0, u0 = set(backend.l0_ids), set(backend.u0_ids)
    if (l0 & u0 or state.seed != scope["seed"] or state.round != round_index
            or state.packet.get("validation_test_records") != 0
            or len(state.history) != round_index):
        raise ValueError("state seed, round, or split roles differ from the formal trajectory")
    acquired = []
    for number, entry in enumerate(state.history):
        if ((entry["seed"], entry["round"], entry["method"]) != (state.seed, number, "pure_llm_v3")
                or len(entry["records"]) != batch_size
                or any(record["source"] != "LLM" for record in entry["records"])):
            raise ValueError("selection history contains a foreign acquisition")
        acquired.extend(record["candidate_id"] for record in entry["records"])
    if (len(set(acquired)) != round_index * batch_size or not set(acquired) <= u0
            or set(state.provenance["expected_observed"]) != l0 | set(acquired)
            or set(state.provenance["expected_candidates"]) != u0 - set(acquired)
            or state.provenance["split_sha256"] != backend.split_sha256):
        raise ValueError("L/U sets disagree with the split and the trajectory's exact acquisition lineage")
    return acquired


def _approved_assets(root, contract, scope):
    if scope["mock"]:
        return None
    for relative, expected in contract.asset_sha256.items():
        if Path(relative).suffix != ".pt" and file_digest(root / relative) != expected:
            raise ValueError("a formal data, split or preprocessing asset changed on disk")
    prompt = (root / PROMPT_PATH).read_bytes()
    if hashlib.sha256(prompt).hexdigest() != contract.source_sha256[PROMPT_PATH.as_posix()]:
        raise ValueError("the authorized A7 system prompt changed on disk")
    return prompt.decode().replace("\r\n", "\n").replace("\r", "\n").strip()


def _stage(root, state, snapshot, scope, contract):
    prompt = _approved_assets(root, contract, scope)
    stage = stage_state(root, state, arm="A7", memory_mode="M2", snapshot=snapshot)
    if prompt is not None and stage.messages[0] != {"role": "system", "content": prompt}:
        raise ValueError("system message differs from the approved A7 prompt")
    if (stage.packet["memory"]["mode"], stage.packet["layout"], stage.packet["compression"]) != (
        "M2", "B", "lossless"
    ):
        raise ValueError("staged packet is outside the formal A7 convention")
    if state.round == 0 and any(stage.packet["memory"].get(field) for field in (
        "latest_feedback", "hypotheses", "archived_hypotheses", "previous_feedback_interpretation"
    )):
        raise ValueError("L333 memory must start empty")
    binding = {**stage.binding, "a7_formal": scope}
    if contract.profile.get("normalize_observed_reference_aliases") is True:
        binding["normalize_observed_reference_aliases"] = True
    fixed_reference_repair = contract.profile.get("fixed_reference_repair")
    if fixed_reference_repair is not None:
        binding["fixed_reference_repair"] = fixed_reference_repair
    representation = contract.profile.get("wire_representation")
    if representation is None:
        return replace(stage, binding=binding)
    outgoing = formal_wire_messages(stage.messages, stage.packet, representation)
    binding.update(wire_representation=dict(representation),
                   authoritative_base_messages_sha256=binding["base_messages_sha256"],
                   wire_packet_sha256=digest(json.loads(outgoing[1]["content"])),
                   base_messages_sha256=digest(outgoing))
    return replace(stage, messages=outgoing, binding=binding)


def _initial_state(directory, backend, contract, scope, anchor_run, fit_slot):
    initial = directory / "initial"
    source, receipt = initial / "source.json", initial / "initial.json"
    anchor = str(anchor_run.resolve()) if anchor_run is not None else None
    origin = {"scope": scope, "anchor_run": anchor}
    saved = receipt.exists()
    if saved:
        recorded = read(receipt)
        fit_name = recorded.get("fit_path", "")
        if not isinstance(fit_name, str) or not (
            (anchor is not None and fit_name == "anchor") or (anchor is None and re.fullmatch(r"fit_[0-9]{3}", fit_name))
        ):
            raise ValueError("invalid initial artifact path")
        fit_dir = initial / fit_name
        state = FrozenState(**read(source))
    else:
        # The backend validates shared anchors. A fresh fit is permitted only in this authorized path.
        fresh = anchor_run is None
        if fresh:
            fit_dir = _fit_directory(initial)
            once(fit_dir / "fit_scope.json", {**origin, "phase": "initial"})
        else:
            fit_dir = initial / "anchor"
        # Shared checkpoints also perform qualified CUDA inference to reconstruct U predictions.
        with nullcontext() if (fit_dir / "initial_state_receipt.json").exists() else fit_slot():
            state = backend.initial_state(anchor_run, fit_dir, allow_fresh_fit=fresh)
    _state_roles(state, backend, scope, 0, contract.batch_size)
    predecessor = contract.predecessor or {}
    if not scope["mock"]:
        anchor_digests = predecessor.get("initial_checkpoint_sha256_by_seed")
        if anchor_digests is not None:
            expected = anchor_digests[str(state.seed)]
        else:
            expected = predecessor.get("seed_157_initial_checkpoint_sha256") if state.seed == 157 else None
        if expected is not None and state.checkpoint != expected:
            raise ValueError(f"bridge seed {state.seed} L333 checkpoint differs from the authenticated predecessor")
    if state.provenance["resource_receipt_sha256"] != backend.cache_receipt_sha256:
        raise ValueError("current GPU training resources differ from approved L333")
    if saved:
        if recorded != _initial_record(origin, fit_dir, source, state):
            raise ValueError("L333 receipt or selected GPU anchor changed")
        return state
    once(source, asdict(state))
    once(receipt, _initial_record(origin, fit_dir, source, state))
    return FrozenState(**read(source))


def _initial_record(origin, fit_dir, source, state):
    backend_receipt = fit_dir / "initial_state_receipt.json"
    receipt = read(backend_receipt)
    predictions = file_digest(fit_dir / "predictions.csv.gz")
    metrics = file_digest(fit_dir / "validation_metrics.json")
    if (receipt["checkpoint_sha256"] != state.checkpoint
            or file_digest(fit_dir / "best.pt") != state.checkpoint
            or receipt["prediction_sha256"] != predictions
            or receipt["validation_metrics_sha256"] != metrics
            or state.provenance["initial_state_receipt_sha256"] != digest(receipt)):
        raise ValueError("L333 source disagrees with the qualified initialization artifacts")
    return {**origin, "source_sha256": file_digest(source), "fit_path": fit_dir.name,
            "backend_receipt_sha256": file_digest(backend_receipt),
            "prediction_sha256": predictions, "validation_metrics_sha256": metrics}


def _selection_grant(scope, request_id):
    return {"scope": "paid_selection_only", "formal_run": scope, "request_id": request_id}


_AUDITED_TRANSPORT_ERRORS = frozenset({
    "timeout", "connection_interrupted", "sse_missing_completion", "sse_frame_error",
    "http_502", "http_503", "http_504", "http_524",
})


def _future_replay_authorization(run, scope, approval, request, terminal, config, *, sealed_relay_retry_seconds=None):
    """Approve only the unused original slot after an archived ambiguous network stop."""
    from .bridge_protocol import ATTEMPT_ID, read_bridge_json, validate_result, verify_job

    delayed = sealed_relay_retry_seconds is not None
    eligible = (is_sealed_relay_failure(terminal) if delayed else terminal.get("error") in _AUDITED_TRANSPORT_ERRORS)
    if (not eligible or terminal.get("status") != "stopped"
            or terminal.get("ambiguous") is not True or terminal.get("provider_dispatched") is not True
            or terminal.get("circuit_open") is not False or terminal.get("answer_sha256") is not None):
        return None
    attempt_id = terminal.get("attempt_id")
    if not isinstance(attempt_id, str) or not ATTEMPT_ID.fullmatch(attempt_id):
        raise PermissionError("the audited transport retry requires the exact attempt identity")
    job = read_bridge_json(run / "bridge/outbox" / f"{attempt_id}.json")
    result = validate_result(read_bridge_json(run / "bridge/results" / f"{attempt_id}.json"), job)
    original, original_config, started = verify_job(
        job, contract_sha256=scope["contract_sha256"], authorization_sha256=scope["authorization_sha256"],
    )
    failure = result["outcome"].get("failure") or {}
    if ((original, original_config) != (request, config)
            or started["attempt"] != 0 or any(terminal.get(k) != v for k, v in started.items())
            or terminal.get("bridge_job_sha256") != job["job_sha256"]
            or terminal.get("bridge_result_sha256") != result["result_sha256"]
            or result["outcome"]["status"] != "failed" or result["provider_dispatched"] is not True
            or result["raw_response"] is not None or failure.get("ambiguous") is not True
            or failure.get("code") != terminal["error"]):
        raise PermissionError("the original ambiguous result does not match the audited transport retry")
    authorization = {
        "decision": UNKNOWN_REPLAY_DECISION, "authorized_by": approval["authorized_by"],
        "authorization_reference": f"Audited one-slot future transport replay under {scope['contract_sha256']}",
        "accept_duplicate_billing_risk": True, "contract_sha256": scope["contract_sha256"],
        "request_id": request["request_id"], "attempt_id": attempt_id,
        "bridge_result_sha256": result["result_sha256"], "original_terminal_sha256": digest(terminal),
        "next_attempt": 1,
    }
    if delayed:
        identity = failure.get("reply_identity")
        if (type(sealed_relay_retry_seconds) is not int or sealed_relay_retry_seconds != SEALED_RELAY_RETRY_SECONDS
                or type(identity) is not dict or failure.get("retryable") is not False
                or not is_sealed_relay_failure({**identity, "error": failure["code"]})
                or identity["served_model"] not in (None, config.served_model) or identity["usage"] != {}
                or any(terminal.get(key) != value for key, value in identity.items())):
            raise PermissionError("the archived result does not prove the exact failed relay identity")
        authorization["sealed_relay_retry_seconds"] = sealed_relay_retry_seconds
    return authorization


def _verify_selection(directory, state, scope, config, batch_size, *, stage=None):
    freeze = read(directory / "batch_freeze.json")
    binding, batch = freeze["binding"], freeze["batch_ids"]
    if (freeze["mock"] is not scope["mock"] or freeze["label_access"] is not False
            or binding.get("a7_formal") != scope or binding["arm"] != "A7"
            or (binding["seed"], binding["round"], binding["checkpoint_sha256"])
            != (state.seed, state.round, state.checkpoint)
            or binding["state_sha256"] != digest(state.provenance)
            or len(batch) != batch_size or len(set(batch)) != batch_size
            or not set(batch) <= set(state.provenance["expected_candidates"])
            or freeze["selection_file_sha256"] != file_digest(directory / "selection.json")
            or freeze["memory_file_sha256"] != file_digest(directory / "memory.json")):
        raise ValueError("selection freeze is not bound to this authorized state")
    first = read(directory / "turn_00" / "request.json")
    check_request(first, binding, config)
    if read(directory / "authorization.json") != _selection_grant(scope, first["request_id"]):
        raise ValueError("the first logical request has no matching formal grant")
    turn = directory / f"turn_{freeze['accepted_logical_index']:02d}"
    request = read(turn / "request.json")
    check_request(request, binding, config)
    if "wire_representation" in binding and (
        digest(read(directory / AUTHORITATIVE_PACKET_FILE)) != binding["packet_sha256"]
        or any(digest(item["payload"]["input"][:2]) != binding["base_messages_sha256"] for item in (first, request))
    ):
        raise ValueError("authoritative A7 packet or accepted compact wire input drift")
    attempt = freeze["accepted_attempt"]
    answer = terminal_record(turn, attempt)
    started = read(turn / f"attempt_{attempt:02d}.started.json")
    if (request["request_id"] != freeze["request_id"] or answer["request_id"] != request["request_id"]
            or answer["status"] != "completed" or answer["served_model"] != config.served_model
            or answer["response_id"] != freeze["provider_response_id"]
            or answer["answer_sha256"] != digest(answer["text"])
            or any(answer.get(key) != value for key, value in started.items())
            or read(turn / "formal_request_authorization.json") != _selection_grant(scope, request["request_id"])
            or read(turn / "decision.json")["state"] != "legal"):
        raise ValueError("freeze is missing its accepted Responses receipt or legal plan")
    plan = read(directory / "selection.json")
    alias_evidence = turn / "observed_alias_normalization.json"
    if binding.get("normalize_observed_reference_aliases") is True:
        verify_observed_alias_normalization(turn, answer, plan, freeze)
    elif alias_evidence.exists() or "observed_alias_normalization_file_sha256" in freeze:
        raise ValueError("observed alias normalization is outside the approved scientific scope")
    fixed_reference_repair = binding.get("fixed_reference_repair")
    if fixed_reference_repair is not None:
        expected_projection = {"raw_reply_sha256": answer["answer_sha256"],
            "projected_plan_sha256": digest(plan), "reference_repair_sha256": digest(fixed_reference_repair)}
        if read(turn / "response_projection.json") != expected_projection:
            raise ValueError("the accepted plan does not match the archived scoped response projection")
    if stage is not None:
        if freeze["binding"] != stage.binding or not _validation(stage, plan).valid:
            raise ValueError("frozen plan failed validation against the reconstructed A7 packet")
        ordered_ids = [stage.context.all_row_ids[choice["candidate_id"]] for choice in plan["choices"]]
        if ordered_ids != batch:
            raise ValueError("frozen batch IDs disagree with the accepted plan")
    return {**freeze, "plan": plan, "freeze_sha256": file_digest(directory / "batch_freeze.json")}


def _feedback(stage, freeze, receipt, path, backend, prior_ids):
    batch = freeze["batch_ids"]
    if not path.exists():
        measured, audit = backend.reveal_batch(prior_ids, batch)
        record = _build_feedback(stage, freeze, dict(zip(batch, measured, strict=True)), audit)
        once(path, {**record, "selection_freeze_sha256": freeze["freeze_sha256"],
                    "formal_grant_sha256": digest(receipt)})
    saved = read(path)
    audit = saved["label_store_audit"]
    if ([row["candidate_id"] for row in saved["records"]] != batch
            or audit["purpose"] != "after_acquisition_fit" or audit["requested_rows"] != len(batch)
            or audit["requested_ids_hash"] != digest(sorted(batch)) or not audit["acquisitions_frozen"]
            or audit["roles"] != "u0"):
        raise ValueError("feedback ledger differs from the exact frozen batch")
    truth = {row["candidate_id"]: [row[f"true_{t}_ml"] for t in TARGETS] for row in saved["records"]}
    if any(not math.isfinite(value) for values in truth.values() for value in values):
        raise ValueError("non-finite feedback measurement")
    expected = _build_feedback(stage, freeze, truth, audit)
    if saved != {**expected, "selection_freeze_sha256": freeze["freeze_sha256"],
                 "formal_grant_sha256": digest(receipt)}:
        raise ValueError("feedback or its formal grant has changed")
    return saved


def _fit_directory(directory):
    resumable, partial = [], []
    for attempt in sorted(directory.glob("fit_[0-9][0-9][0-9]")):
        if ((attempt / "training_complete.json").is_file()
                or all((attempt / name).is_file() for name in ("best.pt", "predictions.csv.gz", "fit_audit.json"))):
            resumable.append(attempt)
        else:
            partial.append(attempt)
    if len(resumable) > 1:
        raise ValueError("multiple optimizer or fit receipts for one training step")
    if resumable:
        return resumable[0]
    number = max((int(path.name[4:]) for path in partial), default=-1) + 1
    if number >= 1000:
        raise ValueError("fit recovery path exhausted")
    return directory / f"fit_{number:03d}"


def _fit_audit(fit_dir, labeled, unlabeled):
    fit = read(fit_dir / "fit_audit.json")
    metrics = read(fit_dir / "validation_metrics.json")
    artifacts = {name: file_digest(fit_dir / name) for name in FIT_ARTIFACTS}
    if (read(fit_dir / "fit_artifact_receipt.json") != artifacts
            or fit["checkpoint_sha256"] != artifacts["best.pt"]
            or fit["prediction_sha256"] != artifacts["predictions.csv.gz"]
            or fit["train_ids_hash"] != digest(sorted(labeled))
            or fit["train_rows"] != len(labeled) or fit["prediction_rows"] != len(unlabeled)
            or fit["test_labels_used_for_fit_or_checkpoint_selection"] != 0
            or metrics["checkpoint_sha256"] != fit["checkpoint_sha256"]
            or metrics["best_epoch"] != fit["best_epoch"]
            or not math.isclose(metrics["combined_normalized_rmse"],
                                fit["best_validation_combined_normalized_rmse"], abs_tol=1e-6, rel_tol=0.0)):
        raise ValueError("completed fit audit does not match the formal L/U artifacts")
    return fit


def _fit_grant(scope, freeze, device):
    return {"scope": "reveal_and_fit_one_frozen_batch", "formal_run": scope,
            "device": device, "selection_freeze_sha256": freeze["freeze_sha256"]}


def _successor(directory, state, next_dir, scope, config, contract, backend):
    freeze = _verify_selection(directory, state, scope, config, contract.batch_size)
    source = next_dir / "source.json"
    receipt = read(next_dir / "transition.json")
    feedback_file = next_dir / "authorized_feedback.json"
    feedback = read(feedback_file)
    fit_base = _fit_directory(next_dir)
    grant = _fit_grant(scope, freeze, contract.device)
    prior = _state_roles(state, backend, scope, state.round, contract.batch_size)
    acquired = [*prior, *freeze["batch_ids"]]
    excluded = set(acquired)
    labeled = [*backend.l0_ids, *acquired]
    unlabeled = [key for key in backend.u0_ids if key not in excluded]
    fit = _fit_audit(fit_base, labeled, unlabeled)
    next_state = FrozenState(**read(source))
    expected = _transition_record(scope, freeze, source, feedback_file, fit_base, fit)
    if (receipt != expected or read(next_dir / "fit_authorization.json") != grant
            or read(fit_base / "fit_scope.json") != _fit_scope(grant, state.round + 1, feedback_file, labeled, unlabeled)
            or feedback["formal_grant_sha256"] != digest(grant)
            or feedback["selection_freeze_sha256"] != freeze["freeze_sha256"]
            or [row["candidate_id"] for row in feedback["records"]] != freeze["batch_ids"]
            or next_state.history != [*state.history, feedback]
            or next_state.checkpoint != fit["checkpoint_sha256"]
            or next_state.provenance["fit_audit_sha256"] != expected["fit_audit_sha256"]):
        raise ValueError("next state, feedback, and fit artifacts do not share the same causal lineage")
    _state_roles(next_state, backend, scope, state.round + 1, contract.batch_size)
    return next_state, read(directory / "memory.json")


def _transition_record(scope, freeze, source, feedback_file, fit_dir, fit):
    return {"formal_run": scope, "selection_freeze_sha256": freeze["freeze_sha256"],
            "memory_file_sha256": freeze["memory_file_sha256"], "fit_path": fit_dir.name,
            "fit_audit_sha256": file_digest(fit_dir / "fit_audit.json"),
            "fit_artifact_receipt_sha256": file_digest(fit_dir / "fit_artifact_receipt.json"),
            "validation_metrics_sha256": file_digest(fit_dir / "validation_metrics.json"),
            "checkpoint_sha256": fit["checkpoint_sha256"], "prediction_sha256": fit["prediction_sha256"],
            "feedback_sha256": file_digest(feedback_file), "source_sha256": file_digest(source)}


def _fit_scope(grant, round_index, feedback_file, labeled, unlabeled):
    return {"formal_grant_sha256": digest(grant), "fit_round": round_index,
            "feedback_sha256": file_digest(feedback_file),
            "labeled_order_sha256": digest(labeled), "unlabeled_order_sha256": digest(unlabeled)}


def qualification_status(run_directory, contract, authorization, *, mock=False):
    """Return the intact first-turn gate receipt when this contract requires one."""
    if not contract.requires_initial_qualification:
        return None
    run_directory = Path(run_directory)
    path = run_directory / "qualification.json"
    if not path.exists():
        return None
    authorization = verify_authorization(contract, authorization)
    scope = _scope(contract, authorization, contract.qualification_seed, mock)
    seed = seed_directory(run_directory, contract.qualification_seed)
    state = round_directory(seed, 1)
    record = read(path)
    expected = {"formal_run": scope, "status": "qualified", "round": 0,
                "selection_freeze_sha256": file_digest(round_directory(seed, 0) / "batch_freeze.json"),
                "transition_sha256": file_digest(state / "transition.json"),
                "next_source_sha256": file_digest(state / "source.json"),
                "next_request_id": record["next_request_id"]}
    if record != expected:
        raise ValueError("qualification receipt or formal qualification evidence has changed")
    return record


def _qualify(root, run, scope, contract, round_dir, state, snapshot):
    stage = _stage(root, state, snapshot, scope, contract)
    if (state.round != 1 or len(stage.context.observed) != contract.initial_labels + contract.batch_size
            or len(stage.packet["memory"]["latest_feedback"]) != contract.batch_size):
        raise ValueError("first formal round did not yield the expected next A7 state")
    next_round = round_directory(round_dir.parent, 1)
    once(run / "qualification.json", {
        "formal_run": scope, "status": "qualified", "round": 0,
        "selection_freeze_sha256": file_digest(round_dir / "batch_freeze.json"),
        "transition_sha256": file_digest(next_round / "transition.json"),
        "next_source_sha256": file_digest(next_round / "source.json"),
        "next_request_id": make_request(stage.binding, 0, stage.messages, contract.transport)["request_id"],
    })


def pin_boundary_user_message(stage, message, expected_sha):
    """Keep the approved original user message bytes after verifying the compact packet."""
    if (type(message) is not dict or set(message) != {"role", "content"}
            or message["role"] != "user" or not isinstance(message["content"], str)
            or digest(message) != expected_sha):
        raise PermissionError("the boundary user message differs from the formal contract")
    if digest(parse_response(message["content"])) != stage.binding.get("wire_packet_sha256"):
        raise ValueError("the boundary user message differs from the reconstructed scientific packet")
    messages = [stage.messages[0], message, *stage.messages[2:]]
    return replace(stage, messages=messages, binding={**stage.binding, "base_messages_sha256": digest(messages)})


def _import_boundary(directory, scope, boundary, expected, backend, batch_size, user_message):
    state, snapshot, start = boundary
    if not isinstance(state, FrozenState) or type(snapshot) is not dict:
        raise ValueError("resuming a boundary requires its exact frozen state and memory snapshot")
    encoded = asdict(state)
    identity = {"round": start, "state_sha256": digest(encoded), "memory_sha256": digest(snapshot)}
    if user_message is not None:
        identity["user_message_sha256"] = digest(user_message)
    if identity != expected:
        raise PermissionError("the supplied start boundary differs from the formal contract")
    _state_roles(state, backend, scope, start, batch_size)
    MemoryManager.from_snapshot(snapshot, seed=scope["seed"])
    if set(snapshot["rounds"]) != {str(index) for index in range(start)}:
        raise ValueError("boundary memory must contain exactly the completed prior rounds")
    round_dir = round_directory(directory, start)
    source = round_dir / "source.json"
    memory = round_dir / "boundary_memory.json"
    once(source, encoded)
    once(memory, snapshot)
    if user_message is not None:
        once(round_dir / "boundary_user_message.json", user_message)
    once(directory / "boundary.json", {"formal_run": scope, **identity})
    return FrozenState(**read(source)), read(memory)


def run_seed(*, root, run_directory, contract, authorization, seed, backend, call, anchor_run=None,
             max_rounds=None, api_slot=None, fit_slot=None, transport_options=None, mock=False,
             resume_boundary=None, boundary_user_message=None, response_projection=None):
    """Resume a contiguous prefix from L333 or from an externally verified, contract-bound boundary."""
    authorization = verify_authorization(contract, authorization)
    if seed not in contract.seed_ids or getattr(backend, "seed", None) != seed:
        raise ValueError("backend seed is outside the authorized run")
    if (getattr(backend, "is_mock", False) is not mock or not callable(call)
            or len(backend.l0_ids) != contract.initial_labels):
        raise ValueError("backend mode, authorized selector or initial label budget differs")
    stop = contract.rounds if max_rounds is None else max_rounds
    if type(stop) is not int or not 0 <= stop <= contract.rounds:
        raise ValueError("requested stop is outside the formal round schedule")
    expected_boundary = contract.profile.get("resume_boundaries", {}).get(str(seed))
    if (resume_boundary is None) != (expected_boundary is None):
        raise PermissionError("the supplied boundary and formal authorization must agree")
    pinned_message_sha = None if expected_boundary is None else expected_boundary.get("user_message_sha256")
    if (boundary_user_message is None) != (pinned_message_sha is None):
        raise PermissionError("the supplied boundary user message and formal authorization must agree")
    start = 0
    if resume_boundary is not None:
        if contract.requires_initial_qualification:
            raise PermissionError("an imported boundary requires an explicitly ungated contract")
        if type(resume_boundary) is not tuple or len(resume_boundary) != 3:
            raise ValueError("a resume boundary must supply the state, memory snapshot and round")
        start = resume_boundary[2]
        if type(start) is not int or not 0 <= start <= stop or start >= contract.rounds:
            raise ValueError("start boundary is outside the authorized round schedule")
    root, run = Path(root), Path(run_directory)
    directory = seed_directory(run, seed)
    anchor_run = None if anchor_run is None else Path(anchor_run).resolve()
    expected_anchor = Path(contract.anchor_root) / f"random-{seed}" if contract.anchor_root else None
    if anchor_run != expected_anchor and (resume_boundary is None or anchor_run is not None):
        raise ValueError("initial GPU anchor differs from the formal authorization")
    api_slot, fit_slot = api_slot or nullcontext, fit_slot or nullcontext
    options = dict(transport_options or {})
    if {"replay_authorization_factory", "attempt_limit", "sealed_relay_retry_seconds"} & options.keys():
        raise PermissionError("formal retries must use the internally verified archive and bound recovery budget")
    attempt_slot = options.get("attempt_slot")
    managed_attempt = options.get("managed_attempt")
    alias_normalization = contract.profile.get("normalize_observed_reference_aliases") is True
    audit_replay = contract.profile.get("audited_future_transport_replay") is True
    relay_delay = contract.profile.get("sealed_relay_retry_seconds")
    replay_start = contract.profile.get("replay_eligible_from_round")
    if relay_delay is not None and (not audit_replay or type(relay_delay) is not int
            or relay_delay != SEALED_RELAY_RETRY_SECONDS or contract.transport.transport_attempts != 2):
        raise PermissionError("sealed relay retry requires a signed managed two-attempt recovery")
    expected_replay_start = start if relay_delay is not None else start + 1
    if audit_replay and (mock or managed_attempt is None or type(replay_start) is not int
                         or replay_start != expected_replay_start):
        raise PermissionError("audited relay replay differs from the formal managed resume boundary")
    predecessor = contract.predecessor or {}
    cached_import = predecessor.get("cached_import")
    cached_dispatcher = predecessor.get("dispatcher_module") == "dispatch_v13_887_cached_r7"
    if cached_dispatcher or cached_import is not None:
        recovery_budget = predecessor.get("recovery_budget")
        if (not cached_dispatcher or not audit_replay or relay_delay is not None or (seed, start) != (887, 7)
                or type(cached_import) is not dict or cached_import.get("provider_post") is not False
                or recovery_budget != {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
                or any(type(value) is not int for value in recovery_budget.values())):
            raise PermissionError("the cached recovery authorizes one local decision and zero provider submissions")
    elif audit_replay:
        recovery_budget = predecessor.get("recovery_budget")
        if (recovery_budget != {"maximum_logical_requests": 1,
                                "maximum_physical_attempts": 2 if relay_delay is not None else 1}
                or any(type(value) is not int for value in recovery_budget.values())):
            raise PermissionError("the formal recovery boundary differs from its finite logical and physical budget")
    scope = _scope(contract, authorization, seed, mock)
    gate_seed = contract.qualification_seed if contract.requires_initial_qualification else None
    if gate_seed is not None and seed != gate_seed and qualification_status(run, contract, authorization, mock=mock) is None:
        return SeedProgress(seed, "awaiting_qualification", 0, contract.initial_labels)
    phase, index, completed = "initial", start, start
    with round_lock(directory):
        once(directory / "seed_contract.json", {"formal_run": scope, "device": contract.device})
        try:
            if (resume_boundary is None and _paused(run, directory)
                    and not (directory / "initial" / "initial.json").exists()):
                _event(directory, scope, phase, "paused", index)
                return SeedProgress(seed, "paused", 0, contract.initial_labels)
            _approved_assets(root, contract, scope)
            if resume_boundary is None:
                state = _initial_state(directory, backend, contract, scope, anchor_run, fit_slot)
                snapshot = None
            else:
                phase = "import_boundary"
                state, snapshot = _import_boundary(directory, scope, resume_boundary, expected_boundary,
                                                   backend, contract.batch_size, boundary_user_message)
            # Verify the completed prefix before applying a pause or an earlier stop boundary.
            for index in range(start, contract.rounds):
                round_dir, next_dir = (round_directory(directory, n) for n in (index, index + 1))
                phase = "verify_prefix"
                if (next_dir / "transition.json").exists():
                    state, snapshot = _successor(round_dir, state, next_dir, scope, contract.transport, contract, backend)
                    completed = index + 1
                    if index == 0 and seed == gate_seed:
                        _qualify(root, run, scope, contract, round_dir, state, snapshot)
                    continue
                if _paused(run, directory) or index >= stop:
                    break
                phase = "stage"
                prior = _state_roles(state, backend, scope, index, contract.batch_size)
                stage = _stage(root, state, snapshot, scope, contract)
                if index == start and boundary_user_message is not None:
                    stage = pin_boundary_user_message(stage, boundary_user_message, pinned_message_sha)
                if _paused(run, directory):
                    break
                if "wire_representation" in stage.binding:
                    once(round_dir / AUTHORITATIVE_PACKET_FILE, stage.packet)
                first = make_request(stage.binding, 0, stage.messages, contract.transport)
                once(round_dir / "authorization.json", _selection_grant(scope, first["request_id"]))
                once(round_dir / "turn_00" / "formal_request_authorization.json",
                     _selection_grant(scope, first["request_id"]))

                def authorize_request(request, config):
                    check_request(request, stage.binding, config)
                    turn = round_dir / f"turn_{request['logical_index']:02d}"
                    once(turn / "formal_request_authorization.json", _selection_grant(scope, request["request_id"]))

                def authorized_call(request, config):
                    authorize_request(request, config)
                    return call(request, config)

                @contextmanager
                def authorized_attempt_slot(request, config):
                    authorize_request(request, config)
                    with attempt_slot(request, config) as credential_id:
                        yield credential_id

                def authorized_managed_attempt(request, config, started):
                    authorize_request(request, config)
                    return managed_attempt(request, config, started)

                retry_options = dict(options)
                if attempt_slot is not None:
                    retry_options["attempt_slot"] = authorized_attempt_slot
                if managed_attempt is not None:
                    retry_options["managed_attempt"] = authorized_managed_attempt
                if audit_replay and index >= replay_start:
                    def audited_replay(request, terminal, config):
                        return _future_replay_authorization(run, scope, authorization, request, terminal, config,
                                                           sealed_relay_retry_seconds=relay_delay)

                    retry_options["replay_authorization_factory"] = audited_replay
                    if relay_delay is not None:
                        retry_options["sealed_relay_retry_seconds"] = relay_delay
                phase = "select"
                existing = (round_dir / "batch_freeze.json").exists()
                with nullcontext() if existing else api_slot():
                    if _paused(run, directory):
                        break
                    selection_options = {"normalize_observed_reference_aliases": True} if alias_normalization else {}
                    if audit_replay and index == start:
                        retry_options["attempt_limit"] = 2 if relay_delay is not None else 1
                        selection_options["logical_reply_limit"] = 1
                    select_stage(stage, round_dir, contract.transport, authorized_call, mock=mock,
                                 retry_options=retry_options, response_projection=response_projection,
                                 **selection_options)
                freeze = _verify_selection(round_dir, state, scope, contract.transport, contract.batch_size, stage=stage)
                if "wire_representation" in stage.binding:
                    record_batch_diagnostics(round_dir, freeze, stage.context.candidates, stage.context.observed)
                if _paused(run, directory):
                    break
                grant = _fit_grant(scope, freeze, contract.device)
                once(next_dir / "fit_authorization.json", grant)
                phase = "reveal"
                _approved_assets(root, contract, scope)
                feedback_file = next_dir / "authorized_feedback.json"
                feedback = _feedback(stage, freeze, grant, feedback_file, backend, prior)
                if _paused(run, directory):
                    break
                phase = "fit"
                acquired = [*prior, *freeze["batch_ids"]]
                excluded = set(acquired)
                new_l = [*backend.l0_ids, *acquired]
                new_u = [key for key in backend.u0_ids if key not in excluded]
                labels = [*backend.l0_truth, *([r[f"true_{t}_ml"] for t in TARGETS]
                          for h in (*state.history, feedback) for r in h["records"])]
                fit_dir = _fit_directory(next_dir)
                fit_scope = _fit_scope(grant, index + 1, feedback_file, new_l, new_u)
                sealed_fit = (fit_dir / "fit_audit.json").exists()
                if sealed_fit and read(fit_dir / "fit_scope.json") != fit_scope:
                    raise ValueError("completed fit does not belong to this frozen ordered L/U and feedback")
                if not sealed_fit:
                    with fit_slot():
                        if _paused(run, directory):
                            break
                        once(fit_dir / "fit_scope.json", fit_scope)
                        backend.fit_next(labeled_ids=new_l, labeled_truth=labels, unlabeled_ids=new_u,
                                         round_index=index + 1, freeze_sha256=freeze["freeze_sha256"], fit_dir=fit_dir)
                elif not (fit_dir / "fit_artifact_receipt.json").exists():
                    # The backend verifies the sealed fit contract and extracts metrics without fitting again.
                    backend.fit_next(labeled_ids=new_l, labeled_truth=labels, unlabeled_ids=new_u,
                                     round_index=index + 1, freeze_sha256=freeze["freeze_sha256"], fit_dir=fit_dir)
                fit = _fit_audit(fit_dir, new_l, new_u)
                phase = "commit"
                new_state = _next_snapshot(state, feedback, fit, fit_dir, new_l, new_u)
                next_source = next_dir / "source.json"
                once(next_source, asdict(new_state))
                # Stage the persisted representation on the first execution and on replay.
                new_state = FrozenState(**read(next_source))
                _state_roles(new_state, backend, scope, index + 1, contract.batch_size)
                next_memory = read(round_dir / "memory.json")
                _stage(root, new_state, next_memory, scope, contract)
                once(next_dir / "transition.json", _transition_record(
                    scope, freeze, next_source, feedback_file, fit_dir, fit))
                state, snapshot, completed = new_state, next_memory, index + 1
                if index == 0 and seed == gate_seed:
                    _qualify(root, run, scope, contract, round_dir, state, snapshot)
        except Exception as error:
            _event(directory, scope, phase, "failed", index, error)
            raise
        status = "complete" if completed == contract.rounds else "paused" if _paused(run, directory) else "yielded"
        if status == "paused":
            _event(directory, scope, phase, status, completed)
        return SeedProgress(seed, status, completed, contract.initial_labels + contract.batch_size * completed)


def reconcile_attempt(*, run_directory, contract, authorization, seed, round_index, logical_index,
                      attempt, reconciliation, reply=None, mock=False):
    """Use the existing provider receipt reconciliation under the exact formal run."""
    authorization = verify_authorization(contract, authorization)
    if (seed not in contract.seed_ids or not 0 <= round_index < contract.rounds
            or not 0 <= logical_index < contract.transport.logical_replies):
        raise ValueError("reconciliation is outside the formal scope")
    scope = _scope(contract, authorization, seed, mock)
    seed_dir = seed_directory(run_directory, seed)
    with round_lock(seed_dir):
        directory = round_directory(seed_dir, round_index)
        if (directory / "batch_freeze.json").exists():
            raise ValueError("a frozen batch cannot be reopened for provider recovery")
        turn = directory / f"turn_{logical_index:02d}"
        request = read(turn / "request.json")
        if (request["binding"].get("a7_formal") != scope
                or (request["binding"]["seed"], request["binding"]["round"]) != (seed, round_index)
                or read(turn / "formal_request_authorization.json") != _selection_grant(scope, request["request_id"])):
            raise PermissionError("request lacks this formal authorization")
        reconcile(directory, request, attempt, authorization=reconciliation, config=contract.transport, reply=reply)
        _event(seed_dir, scope, "reconcile", reconciliation["decision"], round_index)
