"""Run the first A7 pilot round under the revised response-format prompt."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    PROFILE_V4_PILOT, ROOT, SSH_API, STUDY, load_contract, verify_authorization,
)


PILOT_SHA = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
PROMPT_UPDATE = (
    'Response format check. Use exactly one key, "PE/EA", in each condition_anchor object. '
    'Copy the selected candidate\'s decoded raw "PE/EA" value character for character as a JSON string. '
    'Preserve punctuation and spacing. Return one JSON object using the registered keys. '
    'Include exactly 32 unique current candidate IDs with their matching molecule IDs. '
    'Follow the existing requirements for reasons, typed references, and hypothesis updates.'
)
SEED = 887
INDEX = 3
RUN_NAME = "a7-gpt62-astra-xhigh-ssh-v5-prompt-pilot-887"


def _load_boundary(original_root, original_run, run):
    record = read(original_run / "formal_contract.json")
    base = load_contract(original_root, original_root / PROFILE_V4_PILOT,
                         anchor_root=record["anchor_root"], gpu_slots=tuple(record["gpu_slots"]),
                         lease_root=record["lease_root"], run_directory=original_run, api_execution=SSH_API,
                         predecessor_run=record["predecessor"]["run_directory"])
    if (base.digest != PILOT_SHA or base.record() != record or run.parent != original_run.parent
            or run.name != RUN_NAME or not (original_run / "pause.request").is_file()):
        raise PermissionError("prompt successor requires the verified quiesced predecessor and its own run")
    base_approval = verify_authorization(base, read(original_run / "authorization.json"))
    old_seed = seed_runner.seed_directory(original_run, SEED)
    scope = seed_runner._scope(base, base_approval, SEED, False)
    backend = TrainingBackend(original_root, original_run / "public_graphs" / f"seed_{SEED}", SEED)
    anchor = Path(base.anchor_root) / f"random-{SEED}"
    state = seed_runner._initial_state(old_seed, backend, base, scope, anchor, None)
    memory = None
    for index in range(INDEX):
        before, after = (seed_runner.round_directory(old_seed, n) for n in (index, index + 1))
        state, memory = seed_runner._successor(before, state, after, scope, base.transport, base, backend)
    boundary = seed_runner.round_directory(old_seed, INDEX)
    if (state.round != INDEX or len(state.provenance["expected_observed"]) != 429
            or len(state.provenance["expected_candidates"]) != 2901 or (boundary / "batch_freeze.json").exists()):
        raise ValueError("expected the original sealed L429 state and an unfrozen fourth decision")
    return base, base_approval, state, memory, boundary


def _contract(root, original_root, original_run, run):
    base, base_approval, state, memory, boundary = _load_boundary(original_root, original_run, run)
    prompt = seed_runner.PROMPT_PATH
    if (root / prompt).read_text().strip() != (original_root / prompt).read_text().strip() + "\n\n" + PROMPT_UPDATE:
        raise ValueError("the source may only append the registered response-format guidance")
    modified = {prompt.as_posix(), (STUDY / "a7_gpu" / "seed_runner.py").as_posix()}
    all_sources = {name: file_digest(root / name) for name in base.source_sha256}
    if any(all_sources[name] != expected for name, expected in base.source_sha256.items() if name not in modified):
        raise PermissionError("successor changed code outside the registered prompt and boundary runner")
    for name in ("run_prompt_successor.py", "dispatch_prompt_successor.py"):
        path = (STUDY / "scripts" / name).as_posix()
        all_sources[path] = file_digest(root / path)
    old_request = read(boundary / "turn_00" / "request.json")
    descriptor = {"round": INDEX, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
                  "user_message_sha256": digest(old_request["payload"]["input"][1])}
    profile = {**base.profile, "resume_boundaries": {str(SEED): descriptor}}
    predecessor = {"contract_sha256": base.digest, "authorization_sha256": digest(base_approval),
                   "run_directory": str(original_run), "transition_sha256": file_digest(boundary / "transition.json"),
                   "checkpoint_sha256": state.checkpoint, "initial_new_request_limit": 1,
                   "uses_previous_incomplete_round_responses": False}
    contract = replace(base, profile=profile, source_sha256=all_sources, run_directory=str(run), predecessor=predecessor)
    return contract, state, memory, old_request


def _verify_fresh_input(root, contract, approval, state, memory, old_request):
    scope = seed_runner._scope(contract, approval, SEED, False)
    stage = seed_runner._stage(root, state, memory, scope, contract)
    old_messages = old_request["payload"]["input"]
    stage = seed_runner.pin_boundary_user_message(
        stage, old_messages[1], contract.profile["resume_boundaries"][str(SEED)]["user_message_sha256"])
    request = make_request(stage.binding, 0, stage.messages, contract.transport)
    messages = request["payload"]["input"]
    if ([message["role"] for message in messages] != ["system", "user"]
            or len(old_messages) != 2 or messages[1] != old_messages[1]
            or messages[0]["content"].replace("\n\n" + PROMPT_UPDATE, "", 1) != old_messages[0]["content"]
            or stage.binding["checkpoint_sha256"] != old_request["binding"]["checkpoint_sha256"]
            or stage.binding["packet_sha256"] != old_request["binding"]["packet_sha256"]
            or stage.binding["wire_packet_sha256"] != old_request["binding"]["wire_packet_sha256"]):
        raise PermissionError("fresh decision must use the original L429 candidate and observation packet")
    if request["request_id"] == old_request["request_id"]:
        raise ValueError("revised system prompt did not generate a new request identity")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run"))
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    original_root, original_run, run = (path.resolve() for path in (args.base_release, args.original_run, args.run_dir))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    contract, state, memory, old_request = _contract(ROOT, original_root, original_run, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        raise PermissionError("a fresh paid decision requires the exact successor contract authorization")
    approval = verify_authorization(contract, read(args.authorization))
    request = _verify_fresh_input(ROOT, contract, approval, state, memory, old_request)
    with scheduler._lock(run / "coordinator.lock", busy="prompt successor already running"):
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "fresh_request_scope.json", {"request_id": request["request_id"], "seed": SEED,
             "round": INDEX, "logical_index": 0, "attempt": 0, "prior_complete_response_in_payload": False})
        once(run / "provider_model_lock.json", read(original_run / "provider_model_lock.json"))
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": contract.digest, "pid": os.getpid()})
        backend = TrainingBackend(ROOT, original_run / "public_graphs" / f"seed_{SEED}", SEED)
        call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
        managed = options["managed_attempt"]

        def one_request(candidate, config, started):
            if candidate != request or started["attempt"] != 0:
                raise TransportStop("the prompt successor first run authorizes one new physical request")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(root=ROOT, run_directory=run, contract=contract, authorization=approval,
            seed=SEED, backend=backend, call=call, anchor_run=Path(contract.anchor_root) / f"random-{SEED}",
            max_rounds=INDEX + 1, fit_slot=partial(scheduler.gpu_slot, run, SEED, 0, contract.lease_root),
            transport_options={"managed_attempt": one_request}, resume_boundary=(state, memory, INDEX),
            boundary_user_message=old_request["payload"]["input"][1])
        print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
