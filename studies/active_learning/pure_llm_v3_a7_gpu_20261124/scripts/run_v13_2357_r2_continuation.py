"""Resume the remaining V13 rounds only from a verified 2357 R2 CUDA fit."""

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_2357_r2_sse_extension as extension
from . import run_v13_lane_continuation as shared


RUN_NAME = "a7-gpt62-astra-xhigh-ipv4-v13-2357-r2-continuation-01"
DISPATCHER = "dispatch_v13_2357_r2_continuation"
LANE = shared.extension.Lane(extension.SEED, extension.ROUND, "timeout", extension.ATTEMPTS, extension.PAYLOAD)


def _sources(root, one_shot):
    sources = dict(one_shot.source_sha256)
    if any(file_digest(root / name) != expected for name, expected in sources.items()):
        raise PermissionError("source bytes differ from the exact one-shot SSE release")
    for module in ("run_v13_2357_r2_continuation", DISPATCHER, "run_v13_lane_continuation",
                   "run_v13_lane_extension", "dispatch_v13_lane_continuation", "dispatch_v13_lane_extension"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        actual = file_digest(root / name)
        if name in sources and actual != sources[name]:
            raise PermissionError("a continuation dependency differs from the one-shot source manifest")
        sources[name] = actual
    return sources


def construct(root, parent_release, parent_run, extension_release, extension_run, run, *,
              expected_extension_contract, expected_extension_approval):
    if run.name != RUN_NAME or run.parent != parent_run.parent or root in (parent_release, extension_release):
        raise PermissionError("use the registered 2357 R2 continuation sibling and its immutable release")
    retirement = shared._retirement(parent_run, LANE)
    one_shot, start, backend, original = extension.construct(extension_release, parent_release, parent_run, extension_run)
    approval = shared._extension_approval(one_shot, extension_run, expected_extension_contract, expected_extension_approval)
    parent, parent_approval, _, _ = extension._original(parent_release, parent_run)
    sources = _sources(root, one_shot)

    first = seed_runner.round_directory(seed_runner.seed_directory(extension_run, LANE.seed), LANE.round)
    next_round = seed_runner.round_directory(first.parent, LANE.round + 1)
    scope = seed_runner._scope(one_shot, approval, LANE.seed, False)
    if (read(first / "source.json") != asdict(start[0]) or read(first / "boundary_memory.json") != start[1]
            or read(first.parent / "boundary.json") != {
                "formal_run": scope, **one_shot.profile["resume_boundaries"][str(LANE.seed)],
            }):
        raise PermissionError("the completed R2 extension does not start at the verified L397 boundary")
    request = extension._stage_request(extension_release, one_shot, approval, start, original)
    attempt = shared._completed_attempt(extension_run, one_shot, approval, first, request)
    state, memory = seed_runner._successor(first, start[0], next_round, scope, one_shot.transport, one_shot, backend)
    budget = shared._budget(parent, LANE, state.round)
    marker = shared._completion(extension_run / "extension_completion.json", one_shot, LANE, state, memory,
                                next_round / "transition.json", budget)
    identity = read(extension_run / "provider_model_lock.json")
    if (identity != one_shot.predecessor["model_identity"] or identity != parent.predecessor["model_identity"]
            or one_shot.predecessor["parent_seed_pause_sha256"] != retirement):
        raise PermissionError("the R2 extension used different model or parent retirement evidence")

    transport = replace(one_shot.transport, logical_replies=parent.transport.logical_replies,
                        transport_attempts=parent.transport.transport_attempts)
    profile = {
        **one_shot.profile, "target_labels": parent.target_labels, "transport": asdict(transport),
        "resume_boundaries": {str(LANE.seed): {"round": state.round, "state_sha256": marker["state_sha256"],
                                              "memory_sha256": marker["memory_sha256"]}},
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent_run),
        "parent_release_directory": str(parent_release), "parent_contract_sha256": parent.digest,
        "parent_authorization_sha256": digest(parent_approval), "parent_seed_pause_sha256": retirement,
        "extension_run_directory": str(extension_run), "extension_release_directory": str(extension_release),
        "extension_contract_sha256": one_shot.digest, "extension_authorization_sha256": digest(approval),
        "extension_completion_sha256": digest(marker), "extension_transition_sha256": marker["transition_sha256"],
        "extension_fit_audit_sha256": state.provenance["fit_audit_sha256"], **attempt,
        "seed": LANE.seed, "round": state.round, "checkpoint_sha256": state.checkpoint,
        "cache_directory": str(backend.cache_dir), "public_resource_receipt_sha256": backend.cache_receipt_sha256,
        "model_identity": identity, "execution_budget": budget,
        "bridge_wait_seconds": one_shot.predecessor["bridge_wait_seconds"],
    }
    contract = replace(parent, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=predecessor)
    return contract, (state, memory, state.round), backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("continuation-release", "continuation-run", "extension-release", "extension-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--extension-contract-sha256", required=True)
    parser.add_argument("--extension-authorization-sha256", required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parent_release, parent, extension_release, extension_run, run = (
        path.resolve() for path in (args.continuation_release, args.continuation_run, args.extension_release,
                                    args.extension_run, args.run_dir)
    )
    contract, boundary, backend = construct(
        ROOT, parent_release, parent, extension_release, extension_run, run,
        expected_extension_contract=args.extension_contract_sha256,
        expected_extension_approval=args.extension_authorization_sha256,
    )
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact 33-round continuation authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = shared.execute(ROOT, parent, run, LANE, contract, approval, boundary, backend,
                            resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
