"""Read-only evidence for beginning a separately authorized A7 transport version."""
from pathlib import Path
import re

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig, check_request

PREDECESSOR_SHA256 = "3adec146134e47b237c40ea3b97e51ef4b9486e3007997be67e11bbadf16d5a2"


def inspect_predecessor(directory):
    """Admit only the part of the original run before any scientific decision."""
    directory = Path(directory).resolve(strict=True)
    contract_file = directory / "formal_contract.json"
    contract = read(contract_file)
    if digest(contract) != PREDECESSOR_SHA256 or contract["run_directory"] != str(directory):
        raise ValueError("predecessor does not match the approved original A7 run")
    if (not (directory / "pause.request").is_file() or (directory / "qualification.json").exists()
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
    scope = {"contract_sha256": PREDECESSOR_SHA256, "authorization_sha256": digest(approval),
             "seed": 157, "mock": False}
    if (approval.get("contract_sha256") != PREDECESSOR_SHA256 or binding.get("a7_formal") != scope
            or not re.fullmatch(r"[a-f0-9]{64}", binding["checkpoint_sha256"])):
        raise ValueError("predecessor request does not match its independently approved scope")
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
