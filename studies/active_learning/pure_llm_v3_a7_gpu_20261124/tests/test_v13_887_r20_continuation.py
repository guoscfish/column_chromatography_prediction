from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_887_r20_continuation as mac,
    run_v13_887_r20_continuation as gpu,
)


def test_r20_binds_actual_checkpoint_and_uses_the_finite_normal_continuation_budget():
    remaining = 36 - gpu.START
    assert gpu.START == 20 and gpu.HANDOFF["provider_post"] is False
    assert gpu.R20_REQUEST["checkpoint_sha256"] == gpu.HANDOFF["checkpoint_sha256"]
    assert gpu.BUDGET == {"first_logical_requests": remaining,
                          "maximum_logical_requests": 1 + (remaining - 1) * 3,
                          "maximum_physical_attempts": 2 + (remaining - 1) * 6}


def test_new_parser_is_the_only_permitted_change_to_inherited_science(monkeypatch, tmp_path):
    prompt = gpu.seed_runner.PROMPT_PATH
    old_parser, new_parser = b"original", b"reviewed"
    focus = {gpu.PARSER: new_parser, prompt.as_posix(): b"regular Prompt", gpu.CACHED_MODULE: b"archival gate",
             **{path.as_posix(): b"new entry" for path in gpu.SOURCES}}
    for name, content in focus.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    template = tmp_path / "separate_legacy_parser"
    template.write_bytes(old_parser)
    monkeypatch.setattr(gpu, "OLD_PARSER", file_digest(template))
    monkeypatch.setattr(gpu, "NEW_PARSER", file_digest(tmp_path / gpu.PARSER))
    monkeypatch.setattr(gpu, "NORMAL_PROMPT", file_digest(tmp_path / prompt))
    monkeypatch.setattr(gpu, "CACHED_HELPER", file_digest(tmp_path / gpu.CACHED_MODULE))
    parent = SimpleNamespace(source_sha256={gpu.PARSER: gpu.OLD_PARSER, prompt.as_posix(): gpu.NORMAL_PROMPT})
    imported = SimpleNamespace(source_sha256={gpu.CACHED_MODULE: gpu.CACHED_HELPER})
    manifest = gpu._normal_sources(tmp_path, parent, imported)
    assert manifest[gpu.PARSER] == gpu.NEW_PARSER
    assert manifest[prompt.as_posix()] == gpu.NORMAL_PROMPT
    assert manifest[gpu.CACHED_MODULE] == gpu.CACHED_HELPER
    (tmp_path / prompt).write_bytes(b"changed")
    with pytest.raises(PermissionError, match="only the exact reviewed parser"):
        gpu._normal_sources(tmp_path, parent, imported)


def test_old_handoff_is_verified_in_separate_process_with_archived_parser(monkeypatch, tmp_path):
    archive = tmp_path / "cached_release"
    for name, content in ((gpu.PARSER, b"unchanged legacy parser"), (gpu.CACHED_MODULE, b"legacy archive verifier")):
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(gpu, "OLD_PARSER", file_digest(archive / gpu.PARSER))
    monkeypatch.setattr(gpu, "CACHED_HELPER", file_digest(archive / gpu.CACHED_MODULE))
    monkeypatch.setenv("PYTHONPATH", "/wrong-source")
    imported = SimpleNamespace(source_sha256={gpu.PARSER: gpu.OLD_PARSER, gpu.CACHED_MODULE: gpu.CACHED_HELPER})
    old, old_release, run = [tmp_path / item for item in ("old_run", "old_release", "cache")]
    seen = []
    def invoke(command, **options):
        seen.append((command, options))
        return SimpleNamespace(returncode=0, stdout=json.dumps(gpu.HANDOFF))
    monkeypatch.setattr(gpu.subprocess, "run", invoke)
    monkeypatch.setattr(gpu.scheduler, "_assert_no_old_workers", lambda run, seeds: None)
    proof = gpu._verify_legacy_handoff(archive, old_release, old, run, imported)
    command, options = seen[0]
    assert proof == gpu.HANDOFF and options["cwd"] == archive
    assert "PYTHONPATH" not in options["env"] and options["env"]["PYTHONHASHSEED"] == "0"
    assert "verify-handoff" in command and gpu.CACHED.contract in command and gpu.CACHED.approval in command
    (archive / gpu.PARSER).write_bytes(b"unexpected parser change")
    with pytest.raises(PermissionError, match="immutable old parser"):
        gpu._verify_legacy_handoff(archive, old_release, old, run, imported)
    assert len(seen) == 1


def test_r20_preserves_exact_initial_packet_and_guards_the_second_physical_attempt(monkeypatch):
    dispatcher = mac.ContinuationDispatcher.__new__(mac.ContinuationDispatcher)
    dispatcher.scope = {"r20_request": gpu.R20_REQUEST}
    request = {"logical_index": 0, "payload_sha256": gpu.R20_REQUEST["payload_sha256"],
               "binding": {key: value for key, value in gpu.R20_REQUEST.items() if key != "payload_sha256"}}
    p0 = {"attempt": 0, "attempt_id": "first"}
    p1 = {"attempt": 1, "attempt_id": "second"}
    called = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda scoped, slot: called.append((scoped, slot)))
    dispatcher._check_boundary(request, p0)
    assert not called
    dispatcher._check_boundary(request, p1)
    assert called == [(request, p1)]
    for field in request["binding"]:
        changed = deepcopy(request)
        changed["binding"][field] = "f" * 64
        with pytest.raises(PermissionError, match="exact signed L973 request"):
            dispatcher._check_boundary(changed, p0)
    with pytest.raises(PermissionError, match="exact signed L973 request"):
        dispatcher._check_boundary({**request, "logical_index": 1}, p0)
