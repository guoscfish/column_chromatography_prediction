"""Read-only evidence for beginning a separately authorized A7 transport version."""
from contextlib import ExitStack, contextmanager
import fcntl
from pathlib import Path
import re

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig, check_request

PREDECESSOR_SHA256 = "3adec146134e47b237c40ea3b97e51ef4b9486e3007997be67e11bbadf16d5a2"
V2_SHA256 = "cd1fbd9e7c3ac46c6ef8f0c3aa275b91ecfb921e8f308bbe98891e4b8014d522"
V2_RESULT_SHA256 = "c7a67011cd47994d26076df87bc347839a40c850813f1bfe60f64f1557158507"
V3_SHA256 = "8c6e505da7e8ac957dfcb616e6814e72fa51183a872bc5f4cde83ce5f6737570"
V3_REQUEST_ID = "e5ce1c13338813b54dac1f919f7a67cd1930e9754aec04350dbf28cabb2160bc"
V3_ATTEMPT_ID = "8262eddaac1f4893b013bf5528960bd6"
V3_RESULT_SHA256 = "2211e652555712f28570080020ec2ab92a6e6e6cd828dd3925b03a6a27fadb10"
L333_ANCHOR_SHA256 = {
    "157": "affd5fc2c046beb35a666ba2d81bce6e5293244d8e0e0e58deaf5b7879ca8e46",
    "887": "78e6a2fc8edcdd6fa11361e6b2ca28342ce17640babd4a85a95098c5678ed588",
    "2357": "81b3d29462746e26f7634d5e911375067708c6d0d739cab565be6aea7c4591ca",
    "6101": "6959c9417166245403dad173ca82347336de53a37d643591ca449b0cf2a929c4",
    "12203": "e056a9924b408e8885e7fa619f0ce63a44b8b3b47a2126d57c1fdc80e3df428d",
}


def verify_v3_anchors(anchor_root):
    root = Path(anchor_root)
    for seed, expected in L333_ANCHOR_SHA256.items():
        checkpoint = root / f"random-{seed}/rounds/random/round_00/model/best.pt"
        if not checkpoint.is_file() or file_digest(checkpoint) != expected:
            raise ValueError(f"shared L333 CUDA checkpoint differs for seed {seed}")
    return dict(L333_ANCHOR_SHA256)


def inspect_predecessor(directory):
    """Admit only the part of the original run before any scientific decision."""
    directory = Path(directory).resolve(strict=True)
    contract_file = directory / "formal_contract.json"
    contract = read(contract_file)
    contract_sha256 = digest(contract)
    if contract_sha256 not in (PREDECESSOR_SHA256, V2_SHA256, V3_SHA256) or contract["run_directory"] != str(directory):
        raise ValueError("predecessor does not match the approved original A7 run")
    if (contract_sha256 != V3_SHA256 and not (directory / "pause.request").is_file()
            or (directory / "qualification.json").exists()
            or (directory / "provider_model_lock.json").exists()):
        raise ValueError("predecessor must remain paused before qualification")
    for name in ("batch_freeze.json", "selection.json", "authorized_feedback.json", "transition.json",
                 "fit_authorization.json", "memory.json"):
        if any(directory.glob(f"seed_*/round_*/{name}")):
            raise ValueError("predecessor contains a scientific decision or advancement")
    turn = directory / "seed_157/round_00/turn_00"
    request_file = turn / "request.json"
    request = read(request_file)
    binding = request["binding"]
    check_request(request, binding, TransportConfig(**contract["profile"]["transport"]))
    approval_file = directory / "authorization.json"
    approval = read(approval_file)
    scope = {"contract_sha256": contract_sha256, "authorization_sha256": digest(approval),
             "seed": 157, "mock": False}
    if (approval.get("contract_sha256") != contract_sha256 or binding.get("a7_formal") != scope
            or not re.fullmatch(r"[a-f0-9]{64}", binding["checkpoint_sha256"])):
        raise ValueError("predecessor request does not match its independently approved scope")
    if contract_sha256 == V2_SHA256:
        return _inspect_v2(directory, contract, request, scope)
    if contract_sha256 == V3_SHA256:
        with _quiescent_v3(directory) as stopped_evidence:
            return _inspect_v3(directory, contract, request, scope, stopped_evidence)
    attempts = sorted(directory.glob("seed_*/round_*/turn_*/attempt_*.started.json"))
    if attempts != [turn / f"attempt_{n:02d}.started.json" for n in range(2)] or (
        binding["seed"], binding["round"], request["logical_index"]
    ) != (157, 0, 0):
        raise ValueError("predecessor does not have exactly two initial connection attempts")
    evidence = {}
    for n, started_file in enumerate(attempts):
        terminal_file = turn / f"attempt_{n:02d}.terminal.json"
        terminal, started = read(terminal_file), read(started_file)
        if (
            started["request_id"] != request["request_id"] or started["attempt"] != n
            or any(terminal.get(k) != v for k, v in started.items())
            or terminal.get("error") != "connect_error" or terminal.get("ambiguous") is not False
            or terminal.get("status") != ("retryable" if n == 0 else "stopped")
            or any(terminal.get(k) for k in ("response_id", "provider_request_id", "usage", "text"))
        ):
            raise ValueError("predecessor transport has execution or response evidence requiring review")
        for path in (started_file, terminal_file):
            evidence[path.relative_to(directory).as_posix()] = file_digest(path)
    if any(turn.glob("*.reconciliation.json")) or any(turn.glob("*.recovery.terminal.json")):
        raise ValueError("predecessor changed through provider reconciliation")
    for path in (request_file, contract_file, approval_file, directory / "pause.request"):
        evidence[path.relative_to(directory).as_posix()] = file_digest(path)
    return {
        "run_directory": str(directory), "contract_sha256": PREDECESSOR_SHA256,
        "asset_sha256": contract["asset_sha256"], "anchor_root": contract["anchor_root"],
        "gpu_slots": contract["gpu_slots"], "lease_root": contract["lease_root"],
        "seed_157_initial_checkpoint_sha256": binding["checkpoint_sha256"],
        "profile": contract["profile"], "evidence_sha256": evidence,
        "prior_connection_attempts": 2, "complete_provider_responses": 0,
        "provider_billing_independently_verified": False,
        "scientific_decisions": 0, "reveal_batches": 0, "new_fits": 0,
    }


def _inspect_v2(directory, contract, request, scope):
    old = contract["predecessor"]
    original = Path(old["run_directory"])
    binding = request["binding"]
    if (old["contract_sha256"] != PREDECESSOR_SHA256 or original == directory
            or original.parent != directory.parent or inspect_predecessor(original) != old
            or binding["checkpoint_sha256"] != old["seed_157_initial_checkpoint_sha256"]):
        raise ValueError("v2 predecessor is detached from its authenticated v1 evidence")
    files = _inspect_ssh_failure(directory, request, scope, label="v2", result_sha256=V2_RESULT_SHA256,
                                 failure_code="provider_reasoning_rejected", ambiguous=False)
    return {
        **_ssh_history(directory, contract, request, (*files, directory / "pause.request"), V2_SHA256),
        "original_contract_sha256": PREDECESSOR_SHA256,
        "prior_reasoning_rejections": 1,
    }


@contextmanager
def _quiescent_v3(directory):
    locks = (directory / "coordinator.lock",
             *(directory / "workers" / f"seed_{seed}.lock" for seed in L333_ANCHOR_SHA256))
    with ExitStack() as stack:
        for path in locks:
            # Open the same flock files as the scheduler, with no writes or creation.
            lock = stack.enter_context(path.open("r+"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("v3 coordinator or a seed worker is still active") from None
        pause = directory / "pause.request"
        if pause.is_file():
            yield pause
        else:
            worker_file = directory / "workers/seed_157.json"
            worker = read(worker_file)
            if any(worker.get(key) != value for key, value in {
                "seed": 157, "status": "failed", "error_type": "TransportStop", "reason": "timeout",
            }.items()):
                raise ValueError("v3 needs the stopped seed 157 worker receipt when no pause marker exists")
            yield worker_file


def _inspect_v3(directory, contract, request, scope, stopped_evidence):
    old = contract["predecessor"]
    original = Path(old["run_directory"])
    checkpoints = verify_v3_anchors(contract["anchor_root"])
    if (old["contract_sha256"] != V2_SHA256 or original == directory or original.parent != directory.parent
            or {**inspect_predecessor(original), "initial_checkpoint_sha256_by_seed": checkpoints} != old):
        raise ValueError("v3 predecessor is detached from its authenticated v2 evidence")
    if request["request_id"] != V3_REQUEST_ID:
        raise ValueError("v3 predecessor differs from the documented request identity")
    if (any(directory.glob("seed_*/initial/fit_*/fit_*.json"))
            or any(directory.glob("seed_*/round_*/turn_*/*.reconciliation.json"))
            or any(directory.glob("seed_*/round_*/turn_*/*.recovery.terminal.json"))):
        raise ValueError("v3 predecessor contains a new fit or reconciliation evidence requiring review")
    files = _inspect_ssh_failure(directory, request, scope, label="v3", result_sha256=V3_RESULT_SHA256,
                                 failure_code="timeout", ambiguous=True, pinned_attempt_id=V3_ATTEMPT_ID)
    return {
        **_ssh_history(directory, contract, request, (*files, stopped_evidence), V3_SHA256),
        "original_contract_sha256": PREDECESSOR_SHA256,
        "v2_contract_sha256": V2_SHA256,
        "initial_checkpoint_sha256_by_seed": checkpoints,
        "prior_ambiguous_timeouts": 1,
        "request_id": V3_REQUEST_ID,
        "attempt_id": V3_ATTEMPT_ID,
        "bridge_result_sha256": V3_RESULT_SHA256,
        "provider_execution_and_billing": "unresolved",
    }


def _inspect_ssh_failure(directory, request, scope, *, label, result_sha256, failure_code, ambiguous,
                         pinned_attempt_id=None):
    from .bridge_protocol import validate_result, verify_job

    binding = request["binding"]
    if ((binding["seed"], binding["round"], request["logical_index"]) != (157, 0, 0)
            or binding["checkpoint_sha256"] != L333_ANCHOR_SHA256["157"]):
        raise ValueError(f"{label} predecessor request differs from seed 157 L333")
    turn = directory / "seed_157/round_00/turn_00"
    started_file, terminal_file = turn / "attempt_00.started.json", turn / "attempt_00.terminal.json"
    if (list(directory.glob("seed_*/round_*/turn_*/attempt_*.started.json")) != [started_file]
            or list(directory.glob("seed_*/round_*/turn_*/attempt_*.terminal.json")) != [terminal_file]
            or any(directory.glob("seed_*/round_*/turn_*/decision.json"))
            or any(turn.glob("*.reconciliation.json")) or any(turn.glob("*.recovery.terminal.json"))
            or any(directory.glob("seed_*/round_*/fit_*/fit_*.json"))):
        raise ValueError(f"{label} predecessor must have exactly one stopped attempt before any scientific decision")
    started, terminal = read(started_file), read(terminal_file)
    attempt_id = started["attempt_id"]
    if (not isinstance(attempt_id, str) or not re.fullmatch(r"[a-f0-9]{32}", attempt_id)
            or pinned_attempt_id not in (None, attempt_id)):
        raise ValueError(f"{label} predecessor bridge attempt identity is invalid")
    bridge = directory / "bridge"
    job_file = bridge / "outbox" / f"{attempt_id}.json"
    result_file = bridge / "results" / f"{attempt_id}.json"
    claim_file = bridge / "claims" / f"{attempt_id}.json"
    dispatcher_file = bridge / "dispatcher.json"
    grant_file = turn / "formal_request_authorization.json"
    if any(list(path.parent.glob("*.json")) != [path] for path in (job_file, result_file, claim_file)):
        raise ValueError(f"{label} predecessor has unexpected bridge attempts or receipts")
    job = read(job_file)
    checked_request, _, checked_started = verify_job(job, authorization_sha256=scope["authorization_sha256"],
                                                     contract_sha256=scope["contract_sha256"])
    result = validate_result(read(result_file), job)
    claim, dispatcher, grant = read(claim_file), read(dispatcher_file), read(grant_file)
    if (started != checked_started or request != checked_request
            or grant != {"scope": "paid_selection_only", "formal_run": scope, "request_id": request["request_id"]}
            or dispatcher.get("contract_sha256") != scope["contract_sha256"]
            or not re.fullmatch(r"[a-f0-9]{32}", dispatcher.get("dispatcher_id", ""))
            or claim != {"dispatcher_id": dispatcher["dispatcher_id"], "attempt_id": attempt_id,
                         "job_sha256": job["job_sha256"]}
            or result["result_sha256"] != result_sha256
            or result["provider_dispatched"] is not True or result["raw_response"] is not None
            or result["outcome"] != {"status": "failed", "failure": {
                "code": failure_code, "retryable": False, "retry_after": None,
                "ambiguous": ambiguous, "reply_identity": None,
            }}
            or any(terminal.get(key) != value for key, value in started.items())
            or terminal.get("bridge_result_sha256") != result["result_sha256"]
            or terminal.get("bridge_job_sha256") != job["job_sha256"]
            or terminal.get("provider_dispatched") is not True
            or terminal.get("error") != failure_code
            or terminal.get("ambiguous") is not ambiguous or terminal.get("status") != "stopped"
            or any(terminal.get(key) for key in ("response_id", "provider_request_id", "usage", "text"))):
        raise ValueError(f"{label} predecessor {failure_code} is detached from its immutable bridge evidence")
    if ambiguous and (terminal.get("provider_elapsed_seconds") != result["elapsed_seconds"]
                      or terminal.get("credential_id") != result["credential_id"]):
        raise ValueError(f"{label} predecessor terminal disagrees with the exact attempted transport evidence")
    return (directory / "formal_contract.json", directory / "authorization.json", turn / "request.json",
            grant_file, started_file, terminal_file, dispatcher_file,
            job_file, claim_file, result_file)


def _ssh_history(directory, contract, request, files, contract_sha256):
    return {
        **{key: contract[key] for key in ("run_directory", "profile", "asset_sha256", "anchor_root", "gpu_slots",
                                          "lease_root")},
        "contract_sha256": contract_sha256,
        "seed_157_initial_checkpoint_sha256": request["binding"]["checkpoint_sha256"],
        "evidence_sha256": {path.relative_to(directory).as_posix(): file_digest(path) for path in files},
        "complete_provider_responses": 0,
        "provider_billing_independently_verified": False,
        "scientific_decisions": 0, "reveal_batches": 0, "new_fits": 0,
    }
