"""Draft or run a separately approved five-seed continuation from authenticated boundaries."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_BACKGROUND, TransportConfig, wire_payload,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending, read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    FormalContract, LEASE_ROOT, OFFLINE, ROOT, SEEDS, SSH_API, STUDY, V4_PILOT_RUN_NAME, request_timeouts,
    verify_authorization,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.supersession import L333_ANCHOR_SHA256
from . import prepare_five_seed_boundaries as prepared


SHA256 = re.compile(r"[a-f0-9]{64}\Z")
RUN_SLUG = re.compile(r"a7-[a-z0-9][a-z0-9-]{4,95}\Z")
TRANSPORT_ONLY = {(STUDY / "a7_gpu" / name).as_posix() for name in (
    "bridge_endpoint.py", "bridge_protocol.py", "dispatcher.py", "formal_contract.py",
    "key_pool.py", "provider.py", "ssh_courier.py",
)} | {(OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()}


class PreparedCohortContract(FormalContract):
    def record(self):
        return {**super().record(), "budget": self.predecessor["execution_budget"]}


def _qualification_model(qualification, transport):
    model = qualification["model_identity"]
    expected = {"provider": transport.provider, "requested_model": transport.requested_model,
                "served_model": transport.served_model, "requested_reasoning_effort": transport.reasoning_effort}
    if (type(model) is not dict or set(model) != {*expected, "reported_model_version", "reported_reasoning_effort"}
            or any(model[key] != value for key, value in expected.items())
            or model["reported_reasoning_effort"] not in (None, transport.reasoning_effort)
            or model["reported_model_version"] is not None
            and not isinstance(model["reported_model_version"], str)):
        raise PermissionError("the qualified route reported a different provider, model or reasoning effort")
    return model


def _verify_qualification_receipt(qualification, transport, *, background):
    path = Path(qualification["receipt_path"])
    if not path.is_absolute() or not path.is_file():
        raise PermissionError("the exact original route qualification receipt is missing or changed")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != qualification["receipt_sha256"]:
        raise PermissionError("the exact original route qualification receipt is missing or changed")
    receipt = json.loads(payload)
    expected = {key: getattr(transport, key) for key in (
        "provider", "base_url", "wire_mode", "requested_model", "reasoning_effort",
    )}
    receipt_fields = {"submission", "retrieval"} if background else {"completion", "long_context_completed"}
    if (type(receipt) is not dict or set(receipt) != {*expected, *receipt_fields}
            or any(receipt[key] != value for key, value in expected.items())):
        raise PermissionError("the qualification receipt does not bind this exact Responses route")
    model = _qualification_model(qualification, transport)
    completion = receipt["retrieval"] if background else receipt["completion"]
    if (type(completion) is not dict or set(completion) != {"response_id", "status", "model", "reasoning_effort"}
            or not isinstance(completion["response_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", completion["response_id"])
            or completion["status"] != "completed" or completion["model"] != transport.served_model
            or completion["reasoning_effort"] != model["reported_reasoning_effort"]):
        raise PermissionError("the qualification receipt does not show an exact completed model response")
    if background:
        if receipt["submission"] != {"response_id": completion["response_id"], "background": True} or (
            receipt["submission"].get("background") is not True
        ):
            raise PermissionError("the provider must accept background execution and complete the same retrieved response")
    elif receipt["long_context_completed"] is not True:
        raise PermissionError("the qualification receipt lacks a completed full-context stream")


def _provider_policy(policy):
    required = {"schema", "transport", "request_timeouts", "qualification", "dispatcher_module"}
    if type(policy) is not dict or set(policy) != required or policy["schema"] != "a7_route_policy_v1":
        raise ValueError("provide one exact reviewed Responses route policy")
    transport = TransportConfig(**policy["transport"])
    timeout = request_timeouts(policy)
    model_ids = {"gpt-6.2-astra"}
    if transport.provider == "aws-bedrock-mantle":
        model_ids.add("openai.gpt-6.2-astra")
    if (transport.requested_model not in model_ids or transport.served_model not in model_ids
            or (transport.reasoning_effort, transport.max_output_tokens, transport.logical_replies,
                transport.transport_attempts) != ("xhigh", 32768, 3, 2)
            or transport.timeout_seconds != timeout.total_seconds):
        raise PermissionError("the reviewed transport must preserve model, reasoning and finite A7 budgets")
    wire = wire_payload([], transport)
    if not isinstance(wire.get("store"), bool):
        raise ValueError("the bound wire protocol must explicitly specify its provider storage behavior")
    qualification = policy["qualification"]
    if qualification is not None:
        if (type(qualification) is not dict or set(qualification) != {
            "receipt_path", "receipt_sha256", "long_context_completed", "same_id_retrieval_completed", "model_identity",
        } or not isinstance(qualification["receipt_sha256"], str)
                or not SHA256.fullmatch(qualification["receipt_sha256"])
                or type(qualification["long_context_completed"]) is not bool
                or type(qualification["same_id_retrieval_completed"]) is not bool
                or not qualification["same_id_retrieval_completed" if wire.get("background") else "long_context_completed"]):
            raise PermissionError("the proposed route does not have its required completion or retrieval evidence")
        _verify_qualification_receipt(qualification, transport, background=wire.get("background") is True)
    dispatcher = policy["dispatcher_module"]
    if not isinstance(dispatcher, str) or not re.fullmatch(r"dispatch_[a-z0-9_]+", dispatcher):
        raise ValueError("the route policy must bind one installed Mac dispatcher module")
    return transport, timeout, wire["store"], qualification, dispatcher


def read_boundary(entry, seed, preparation):
    directory = Path(entry["boundary_directory"]).resolve(strict=True)
    if directory != preparation / "boundaries" / f"seed_{seed}":
        raise PermissionError("prepared boundary is outside its own seed custody")
    source, snapshot = read(directory / "state.json"), read(directory / "memory.json")
    descriptor = {"round": source["round"], "state_sha256": digest(source), "memory_sha256": digest(snapshot)}
    if (entry["descriptor"] != descriptor or source["seed"] != seed
            or entry["checkpoint_sha256"] != source["checkpoint"]
            or not SHA256.fullmatch(source["checkpoint"]) or type(source["round"]) is not int
            or not 0 <= source["round"] < 36
            or entry["active_labels"] != 333 + 32 * source["round"]
            or entry["active_labels"] != len(source["provenance"]["expected_observed"])):
        raise PermissionError("prepared state does not match its checkpoint, seed, scope and label count")
    MemoryManager.from_snapshot(snapshot, seed=seed)
    if set(snapshot["rounds"]) != {str(index) for index in range(source["round"])}:
        raise PermissionError("prepared state lacks the exact scientific memory through its start round")
    return seed_runner.FrozenState(**source), snapshot, source["round"]


def _archive_sources(draft, run):
    rows = draft["boundaries"]
    seed = SEEDS[0]
    historical = Path(rows[str(seed)]["evidence"]["run_directory"])
    if (historical.parent != run.parent or historical.name != f"{prepared.V11_RUN_PREFIX}-{seed}"):
        raise ValueError("the successor must be a sibling of the exact reviewed V11 archives")
    record = read(historical / "formal_contract.json")
    if digest(record) != prepared.V11_CONTRACTS[seed] or record["asset_sha256"] != draft["asset_sha256"]:
        raise PermissionError("the prepared draft does not match the archived scientific manifests")
    return record


def construct(root, draft_path, draft_sha, policy_path, run):
    """Bind exact immutable scientific boundaries and route configuration without launching."""
    draft, policy = read(draft_path), read(policy_path)
    if (digest(draft) != draft_sha or draft.get("schema") != "a7_five_seed_boundary_draft_v1"
            or draft.get("paid_requests_authorized") is not False or draft.get("gpu_slots") != [0]
            or draft.get("gpu_lease_root") != str(LEASE_ROOT)
            or draft.get("initial_anchor_sha256_by_seed") != L333_ANCHOR_SHA256
            or draft.get("improved_prompt_sha256") != prepared.PROMPT_SHA256
            or set(draft.get("boundaries", {})) != set(map(str, SEEDS))
            or not RUN_SLUG.fullmatch(run.name)):
        raise PermissionError("the candidate run needs the exact non-paying five-seed preparation and a separate run name")
    archive = _archive_sources(draft, run)
    archived_profile = FormalContract(
        archive["profile"], TransportConfig(**archive["profile"]["transport"]), archive["source_sha256"],
        archive["asset_sha256"]
    )
    scientific = prepared._scientific_profile(archived_profile)
    if draft["scientific_profile"] != scientific:
        raise PermissionError("scientific settings differ from the immutable reviewed experiment")
    sources = {name: file_digest(root / name) for name in archive["source_sha256"]}
    if any(sha != archive["source_sha256"][name] for name, sha in sources.items() if name not in TRANSPORT_ONLY):
        raise PermissionError("the candidate release changes approved science, training or prompt sources")
    transport, timing, store, qualification, dispatcher = _provider_policy(policy)
    route_paths = [STUDY / "scripts" / f"{name}.py"
                   for name in ("prepare_five_seed_boundaries", "run_prepared_five_seed", dispatcher)]
    if transport.wire_mode == RESPONSES_BACKGROUND:
        route_paths.extend(STUDY / "a7_gpu" / name for name in ("background_transport.py", "background_dispatcher.py"))
    sources.update({path.as_posix(): file_digest(root / path) for path in route_paths})
    descriptors = {}
    lanes = {}
    forbidden_runs = set()
    for seed in SEEDS:
        row = draft["boundaries"][str(seed)]
        state, _, start = read_boundary(row, seed, draft_path.parent)
        evidence = row["evidence"]
        cache = Path(row["cache_directory"]).resolve(strict=True)
        if seed in prepared.V11_CONTRACTS:
            original = run.parent / f"{prepared.V11_RUN_PREFIX}-{seed}"
            forbidden_runs.add(original)
            expected_cache = run.parent / V4_PILOT_RUN_NAME / "public_graphs" / f"seed_{seed}"
            if (evidence.get("source") != "sealed_v11" or evidence.get("run_directory") != str(original)
                    or evidence.get("contract_sha256") != prepared.V11_CONTRACTS[seed]
                    or start < prepared.V11_START[seed]
                    or file_digest(cache / "resource_receipt.json") != evidence.get("public_resource_receipt_sha256")):
                raise PermissionError("a prepared fitted boundary has different historical custody or training resources")
        else:
            expected_cache = draft_path.parent / "public_graphs" / f"seed_{seed}"
            if (evidence.get("source") != "shared_cuda_l333_anchor" or start != 0
                    or state.checkpoint != L333_ANCHOR_SHA256[str(seed)]
                    or file_digest(cache / "resource_receipt.json") != state.provenance["resource_receipt_sha256"]):
                raise PermissionError("a newly enabled seed does not own the exact approved L333 anchor")
        if cache != expected_cache:
            raise PermissionError("training cache does not belong to this seed's approved preparation")
        descriptors[str(seed)] = row["descriptor"]
        lanes[str(seed)] = {"checkpoint_sha256": state.checkpoint, "cache_directory": str(cache),
                            "boundary_directory": row["boundary_directory"], "start_round": start}
    if run in forbidden_runs or run == run.parent / V4_PILOT_RUN_NAME or (
        run.exists() and not (run / "formal_contract.json").is_file()
        and any(path.name != "coordinator.lock" for path in run.iterdir())
    ):
        raise PermissionError("the candidate run directory must not overwrite any existing run or artifacts")
    remaining = sum(36 - row["start_round"] for row in lanes.values())
    if remaining != draft["remaining_first_logical_requests"]:
        raise PermissionError("the prepared budget does not match the exact remaining trajectory lengths")
    profile = {**scientific, "store": store, "provider_capability": "verified_route" if qualification else "draft",
               "verified_served_model_reference": qualification["receipt_sha256"] if qualification else None,
               "transport": asdict(transport), "request_timeouts": asdict(timing), "resume_boundaries": descriptors}
    predecessor = {"prepared_draft_sha256": draft_sha, "transport_policy_sha256": digest(policy),
                   "initial_checkpoint_sha256_by_seed": L333_ANCHOR_SHA256,
                   "prior_v11_contract_sha256_by_seed": {str(seed): sha for seed, sha in prepared.V11_CONTRACTS.items()},
                   "provider_qualification": qualification, "dispatcher_module": dispatcher, "lanes": lanes,
                   "execution_budget": {"first_logical_requests": remaining,
                       "maximum_logical_requests": remaining * transport.logical_replies,
                       "maximum_physical_attempts": remaining * transport.logical_replies * transport.transport_attempts}}
    contract = PreparedCohortContract(profile, transport, sources, draft["asset_sha256"], str(LEASE_ROOT / "trajectories"),
                                      (0,), str(LEASE_ROOT), str(run), SSH_API, predecessor)
    if (run / "formal_contract.json").is_file() and read(run / "formal_contract.json") != contract.record():
        raise PermissionError("the existing target directory is bound to a different contract")
    return contract, draft


def _worker(root, run, contract, approval, preparation, row, seed, generation):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(contract.gpu_slots[0])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    status_path = run / "workers" / f"seed_{seed}.json"
    status = {"seed": seed, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    with scheduler._lock(run / "workers" / f"seed_{seed}.lock", busy="seed already has an active worker"):
        if read(run / "coordinator_generation.json")["id"] != generation:
            raise PermissionError("this worker belongs to an older coordinator")
        try:
            scheduler._atomic_status(status_path, {**status, "status": "initializing"})
            backend = TrainingBackend(root, row["cache_directory"], seed)
            boundary = read_boundary(row, seed, preparation)
            call, options = scheduler._provider_wiring(run, contract, approval, None, seed)
            result = seed_runner.run_seed(
                root=root, run_directory=run, contract=contract, authorization=approval, seed=seed, backend=backend,
                call=call, resume_boundary=boundary, transport_options=options,
                fit_slot=partial(scheduler.gpu_slot, run, seed, status["gpu"], contract.lease_root),
            )
            scheduler._atomic_status(status_path, {**status, **asdict(result)})
        except Exception as error:
            phase = "paused" if scheduler._pause_requested(run, seed) else (
                "bridge_pending" if isinstance(error, BridgePending) else "failed")
            scheduler._atomic_status(status_path, {**status, "status": phase, "error_type": type(error).__name__,
                                                   "reason": str(error)})
            raise


def launch(root, run, contract, draft, draft_path, approval, *, resume):
    qualification = contract.predecessor["provider_qualification"]
    if qualification is None:
        raise PermissionError("launch needs the verified qualification receipt for this exact route")
    verify_authorization(contract, approval)
    ctx = multiprocessing.get_context("spawn")
    with scheduler._lock(run / "coordinator.lock", busy="the five-seed coordinator is already active"):
        scheduler._assert_no_old_workers(run, contract.seed_ids)
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("the experiment is paused; use resume with its existing authorization")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", qualification["model_identity"])
        generation = uuid4().hex
        once(run / "events" / f"{generation}.json", {"action": "resume" if resume else "run",
             "authorization_sha256": digest(approval), "timestamp_unix": time.time()})
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": generation, "pid": os.getpid()})
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
        workers = {}
        for seed in contract.seed_ids:
            if scheduler._pause_requested(run):
                break
            row = draft["boundaries"][str(seed)]
            process = ctx.Process(target=_worker, name=f"a7-seed-{seed}",
                args=(root, run, contract, approval, draft_path.parent, row, seed, generation))
            process.start()
            workers[str(seed)] = process
        for process in workers.values():
            process.join()
        states = {seed: process.exitcode for seed, process in workers.items()}
        complete = len(states) == len(contract.seed_ids) and all(
            code == 0 and read(run / "workers" / f"seed_{seed}.json")["status"] == "complete"
            for seed, code in states.items())
        print(dumps({"status": "complete" if complete else "incomplete", "worker_exit": states}))
        return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="contract", choices=("contract", "run", "resume"))
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--draft-sha256", required=True)
    parser.add_argument("--transport-policy", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    draft_path = args.draft.resolve(strict=True)
    run = args.run_dir.resolve()
    contract, draft = construct(ROOT, draft_path, args.draft_sha256, args.transport_policy.resolve(strict=True), run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return 0
    if args.authorization is None or args.contract_sha256 != contract.digest:
        raise PermissionError("launch requires the exact approved five-seed contract SHA and authorization file")
    approval = read_bridge_json(args.authorization.resolve(strict=True))
    return launch(ROOT, run, contract, draft, draft_path, approval, resume=args.action == "resume")


if __name__ == "__main__":
    raise SystemExit(main())
