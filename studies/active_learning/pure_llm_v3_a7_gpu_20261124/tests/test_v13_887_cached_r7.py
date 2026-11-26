"""Check the stopped original correction can be proven without a GPU terminal."""

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportConfig, TransportFailure, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_887_cached_r7 as mac, run_v13_887_cached_r7 as gpu,
)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    old = tmp_path / gpu.OLD.name
    spools = tmp_path / "spools"
    spool = spools / gpu.OLD.contract
    spool.mkdir(mode=0o700, parents=True)
    once(spool / "run_binding.json", {"contract_sha256": gpu.OLD.contract,
                                      "authorization_sha256": gpu.OLD.approval, "mock": False})
    seed = old / f"seed_{gpu.SEED}"
    round_dir = seed / f"round_{gpu.ROUND:02d}"
    config = TransportConfig("test-provider", "test-model", "test-model", "xhigh",
                             "https://example.invalid/v1", 600, 32768)
    packet = {"seed": gpu.SEED, "round": gpu.ROUND, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Immutable A7 prompt."}, {"role": "user", "content": dumps(packet)}]
    binding = {
        "seed": gpu.SEED, "round": gpu.ROUND, "arm": "A7", "prompt_version": "v3", "schema_version": SCHEMA_VERSION,
        **{field: digest(field) for field in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
        "prompt_sha256": digest(messages[0]["content"]), "packet_sha256": digest(packet),
        "base_messages_sha256": digest(messages),
        "a7_formal": {"contract_sha256": gpu.OLD.contract, "authorization_sha256": gpu.OLD.approval,
                      "seed": gpu.SEED, "mock": False},
    }
    config_kwargs = {"provider_dispatched": True, "credential_id": "key_01", "elapsed_seconds": 60.0}
    reply = ProviderReply("Commentary followed by an answer", "response-1", config.served_model,
                          {"input_tokens": 10, "output_tokens": 5}, provider_request_id="provider-request-1")
    raw = {
        "id": reply.response_id, "model": reply.served_model, "status": reply.status,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": reply.text}]}],
    }
    model = verify_reported_model(config, reply, {"reported_reasoning_effort": config.reasoning_effort,
                                                   "model_version": "test-version"})
    identity = "d" * 32
    once(spool / "dispatcher_identity.json", {"dispatcher_id": identity})
    once(spool / "provider_model_lock.json", model)
    attempts = []
    jobs = []
    result_sha = None
    for index in (0, 1):
        context = [*messages, {"role": "user", "content": "Correction context."}] if index else messages
        request = make_request(binding, index, context, config)
        attempt = f"{index + 1:032x}"
        started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": attempt,
                   "started_unix": 100.0, "transport": "mac_bridge"}
        job = make_job(request, config, started)
        if index == 0:
            result = make_result(job, reply=reply, model_identity=model, raw_response=raw, **config_kwargs)
        else:
            result = make_result(job, failure=TransportFailure("timeout", ambiguous=True), **config_kwargs)
        turn = round_dir / f"turn_{index:02d}"
        once(turn / "request.json", request)
        once(turn / "attempt_00.started.json", started)
        once(turn / "formal_request_authorization.json", gpu.seed_runner._selection_grant(
            binding["a7_formal"], request["request_id"]))
        name = f"{attempt}.json"
        once(old / "bridge/outbox" / name, job)
        once(old / "bridge/results" / name, result)
        once(spool / "jobs" / name, job)
        once(spool / "results" / name, result)
        once(spool / "delivered" / name, {"attempt_id": attempt, "result_sha256": result["result_sha256"]})
        once(old / "bridge/claims" / name, {"attempt_id": attempt, "job_sha256": job["job_sha256"],
                                             "dispatcher_id": identity})
        if index == 0:
            terminal = turn / "attempt_00.terminal.json"
            once(terminal, {**started, "status": "completed", "text": reply.text, "answer_sha256": digest(reply.text),
                            "response_id": reply.response_id, "served_model": reply.served_model, "usage": reply.usage,
                            "provider_request_id": reply.provider_request_id, "bridge_job_sha256": job["job_sha256"],
                            "bridge_result_sha256": result["result_sha256"], "provider_dispatched": True})
            once(turn / "decision.json", {"state": "malformed_complete_response"})
            monkeypatch.setattr(gpu, "OLD_TERMINAL_SHA", file_digest(terminal))
            result_sha = result["result_sha256"]
        attempts.append((attempt, request["request_id"], job["job_sha256"]))
        jobs.append(job)
    for module in (gpu, mac):
        monkeypatch.setattr(module, "ATTEMPTS", tuple(attempts))
        monkeypatch.setattr(module, "RESULT_SHA", result_sha)
        monkeypatch.setattr(module, "RAW_SHA", digest(raw))
    once(old / "pause.request", {"mode": "stop_before_next_operation"})
    monkeypatch.setattr(gpu, "PAUSE_MARKER_SHA", file_digest(old / "pause.request"))
    once(old / f"workers/seed_{gpu.SEED}.json", {"seed": gpu.SEED, "status": "paused", "error_type": "BridgePending",
                                                "reason": gpu.PAUSED_WAIT_REASON})
    return old, seed, spools, spool, config, model, jobs, reply, raw


def test_paused_archive_requires_matching_delivered_receipt_and_keeps_gpu_terminal_absent(archive):
    old, seed, spools, spool, config, model, jobs, _, _ = archive
    active, paused = gpu._verify_old_retired(old, seed)
    _, _, first = gpu._old_receipt(old, active, 0, config, None)
    _, _, second = gpu._old_receipt(old, active, 1, config, None, paused_waiter=True)
    proof = {"model_identity": model, "cached_import": {"prior": [first, second], "paused_waiter": paused}}
    assert second["terminal_sha256"] is None
    assert mac._archive(spools, proof) == config
    assert not (active / "turn_01/attempt_00.terminal.json").exists()

    bad_claim = {**second["claim"], "dispatcher_id": "b" * 32}
    proof["cached_import"]["prior"][1] = {**second, "claim": bad_claim}
    with pytest.raises(PermissionError, match="delivered Mac correction"):
        mac._archive(spools, proof)
    proof["cached_import"]["prior"][1] = second
    (spool / "delivered" / f"{jobs[1]['attempt_id']}.json").unlink()
    with pytest.raises(FileNotFoundError):
        mac._archive(spools, proof)


@pytest.mark.parametrize("changed", ["pause", "second_attempt", "third_turn", "freeze", "following_fit"])
def test_paused_archive_rejects_changed_boundary(archive, changed):
    old, seed, *_ = archive
    active, _ = gpu._verify_old_retired(old, seed)
    paths = {
        "pause": old / "pause.request",
        "second_attempt": active / "turn_01/attempt_01.started.json",
        "third_turn": active / "turn_02/request.json",
        "freeze": active / "batch_freeze.json",
        "following_fit": seed / f"round_{gpu.ROUND + 1:02d}/fit_authorization.json",
    }
    path = paths[changed]
    path.unlink(missing_ok=True)
    once(path, {"changed": changed})
    with pytest.raises(PermissionError):
        current, _ = gpu._verify_old_retired(old, seed)
        gpu._old_receipt(old, current, 1, archive[4], None, paused_waiter=True)


@pytest.mark.parametrize("variant", ["completed", "failed_with_raw", "failed_with_completed_identity"])
def test_paused_archive_never_supersedes_completed_correction(archive, variant):
    old, seed, _, _, _, model, jobs, reply, raw = archive
    active, _ = gpu._verify_old_retired(old, seed)
    job = jobs[1]
    if variant == "completed":
        result = make_result(job, reply=reply, model_identity=model, raw_response=raw,
                             provider_dispatched=True, credential_id="key_01")
    else:
        failure = TransportFailure("timeout", ambiguous=True)
        if variant == "failed_with_completed_identity":
            failure.reply_identity = {"response_id": reply.response_id, "served_model": reply.served_model,
                                      "usage": reply.usage, "provider_request_id": reply.provider_request_id,
                                      "provider_status": "completed"}
        result = make_result(job, failure=failure, raw_response=raw if variant == "failed_with_raw" else None,
                             provider_dispatched=True, credential_id="key_01")
    target = old / "bridge/results" / f"{job['attempt_id']}.json"
    target.unlink()
    once(target, result)
    with pytest.raises(PermissionError, match="successfully returned correction"):
        gpu._old_receipt(old, active, 1, archive[4], None, paused_waiter=True)
