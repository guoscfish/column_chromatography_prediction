"""Verify the overlay leaves original scope, source, and attempt custody intact."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_with_sse_inactivity as overlay


@pytest.fixture
def sources(tmp_path, monkeypatch):
    base = tmp_path / "base"
    edited = tmp_path / "overlay"
    monkeypatch.setattr(overlay, "ROOT", edited)
    study = overlay.STUDY / "a7_gpu"
    files = [study / f"{name}.py" for name in (
        "bridge_protocol", "credentials", "dispatcher", "formal_contract", "key_pool", "provider", "ssh_courier",
    )]
    files.append(overlay.dispatcher.OFFLINE / "pure_llm_v3" / "transport_manager.py")
    manifest, patched = {}, {}
    for relative in files:
        name = relative.as_posix()
        for root in (base, edited):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("reviewed replacement" if root == edited and name in overlay.REPLACEMENTS else "original source")
        manifest[name] = file_digest(base / relative)
        if name in overlay.REPLACEMENTS:
            monkeypatch.setattr(overlay.REPLACEMENTS[name], "__file__", str(edited / relative))
            patched[name] = file_digest(edited / relative)
    return base, edited, {"contract": {"source_sha256": manifest}}, patched


def test_only_loaded_replacements_may_differ_from_the_exact_base(sources):
    base, edited, info, patched = sources
    assert overlay._verify_sources(info, base) == patched
    unaffected = overlay.STUDY / "a7_gpu" / "key_pool.py"
    (edited / unaffected).write_text("unapproved change")
    with pytest.raises(PermissionError, match="outside the approved transport"):
        overlay._verify_sources(info, base)
    (edited / unaffected).write_bytes((base / unaffected).read_bytes())
    (base / unaffected).write_text("old source drift")
    with pytest.raises(PermissionError):
        overlay._verify_sources(info, base)


def test_only_scope_validator_receives_the_replacement_manifest(sources, monkeypatch):
    _, _, info, patched = sources
    pilot, identity, transport = object(), object(), object()

    def approved(view, run, config):
        assert config is pilot and run == "/remote/base"
        assert view is not info and view["contract"] is not info["contract"]
        assert view["contract"]["source_sha256"] == {**info["contract"]["source_sha256"], **patched}
        return transport, identity, "prompt"

    monkeypatch.setattr(overlay, "_approved", approved)
    expected = overlay.PromptDispatcher, transport, identity, "prompt", 3
    assert overlay._authorized_scope(info, patched, overlay.MAIN, "/remote/base", Path("/spools"), pilot) == expected
    assert all(info["contract"]["source_sha256"][key] != value for key, value in patched.items())


@pytest.fixture(params=(overlay.MAIN, overlay.CONTINUATION))
def staged(tmp_path, request, monkeypatch):
    run = request.param
    contract, auth = overlay.RUN_IDENTITIES[run]
    spool = tmp_path / ".config/purellm/spool" / contract
    spool.mkdir(parents=True, mode=0o700)
    for folder in overlay.SPOOL_FOLDERS:
        (spool / folder).mkdir(mode=0o700)
    identity = "1" * 32
    once(spool / "run_binding.json", {"contract_sha256": contract, "authorization_sha256": auth, "mock": False})
    once(spool / "dispatcher_identity.json", {"dispatcher_id": identity})
    model = {"fixture": "locked provider"}
    once(spool / "provider_model_lock.json", model)
    remote_run = f"/gpfs/{run}"
    base = tmp_path / "unmodified"
    base.mkdir()
    authors = {"authorized_by": "test fixture", "authorization_reference": "explicit fixture approval"}
    schedule_path, schedule_sha, schedule_launcher = None, None, None
    if run == overlay.MAIN:
        schedule_launcher = tmp_path / "previous_schedule_launcher.py"
        schedule_launcher.write_text("prior verified launcher")
        schedule = {"schema": overlay.SCHEDULE_SCHEMA, "contract_sha256": contract,
                    "authorization_sha256": auth, "dispatcher_id": identity, "launcher_sha256": file_digest(schedule_launcher),
                    "previous_api_parallelism": 1, "api_parallelism": 3, "authorize_paid_api_scheduling": True, **authors}
        schedule_path = tmp_path / "schedule.json"
        once(schedule_path, schedule)
        schedule_sha = digest(schedule)
    replacement_hashes = {name: f"{index:064x}" for index, name in enumerate(overlay.REPLACEMENTS, start=1)}
    approval = {"schema": overlay.SCHEMA, "contract_sha256": contract, "authorization_sha256": auth,
                "dispatcher_id": identity, "launcher_sha256": file_digest(Path(overlay.__file__)),
                "replacement_source_sha256": replacement_hashes, "sse_inactivity_seconds": 900,
                "schedule_authorization_sha256": schedule_sha, "authorize_paid_api_scheduling": True, **authors}
    approval_path = tmp_path / "overlay.json"
    once(approval_path, approval)
    args = SimpleNamespace(contract_sha256=contract, authorization_sha256=auth, base_release=base,
                           overlay_authorization=approval_path, overlay_authorization_sha256=digest(approval),
                           schedule_authorization=schedule_path, schedule_authorization_sha256=schedule_sha,
                           schedule_launcher=schedule_launcher, verify_only=True, ssh_host="offline",
                           remote_root="/gpfs/current-release", remote_run_dir=remote_run, remote_python="/python")
    info, transport = object(), object()
    endpoint = SimpleNamespace(info=lambda: info)
    pool = SimpleNamespace(directory=tmp_path / "shared-pool", config=SimpleNamespace(parallelism=3))
    launch = SimpleNamespace(dispatcher_id=identity)
    used = {}

    def dispatch(path, **kwargs):
        assert path == spool and kwargs["info"] is info and kwargs["transport"] is transport
        assert kwargs["contract_sha256"] == contract and kwargs["sse_inactivity_seconds"] == 900
        assert not kwargs["mock"] and kwargs["pool"].directory == pool.directory
        used["pool"] = kwargs["pool"]
        return launch

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(overlay, "SSHCourier", lambda *_: endpoint)
    monkeypatch.setattr(overlay, "verify_info", lambda current, **_: current)
    monkeypatch.setattr(overlay, "_verify_sources", lambda *_: replacement_hashes)
    monkeypatch.setattr(overlay, "_pool", lambda _: (pool, None, object()))
    monkeypatch.setattr(overlay, "_authorized_scope", lambda *_: (
        dispatch, transport, model, "prompt", 3 if run == overlay.MAIN else 1,
    ))
    return SimpleNamespace(args=args, spool=spool, launch=launch, endpoint=endpoint, used=used, approval=approval)


def test_verify_only_reports_unsealed_attempt_without_dispatch(staged, monkeypatch):
    started = staged.spool / "attempts" / ("2" * 32) / "started.json"
    once(started, {"fixture": "in-flight"})
    monkeypatch.setattr(overlay.dispatcher, "run_dispatcher", lambda *_, **__: pytest.fail("verification dispatched"))
    view = overlay.run(staged.args)
    assert view["verified"] and view["unsealed_attempts"] == 1 and view["sse_inactivity_seconds"] == 900
    assert view["api_workers"] == (3 if staged.args.schedule_authorization else 1)
    assert staged.used == {}


def test_existing_scheduler_cannot_start_with_an_unsealed_attempt(staged, monkeypatch):
    identity = "2" * 32
    once(staged.spool / "attempts" / identity / "started.json", {"fixture": "in-flight"})
    staged.args.verify_only = False

    def execute(mac, endpoint, *, watch, workers):
        assert mac is staged.launch and endpoint is staged.endpoint and watch
        assert mac.prompt_sha == "prompt"
        assert workers == (3 if staged.args.schedule_authorization else 1)
        staged.used["pool"].prepare()
        return {"started": True}

    monkeypatch.setattr(overlay.dispatcher, "run_dispatcher", execute)
    monkeypatch.setattr(overlay.SharedKeyPool, "prepare", lambda _: None)
    with pytest.raises(PermissionError, match="drain and seal"):
        overlay.run(staged.args)
    once(staged.spool / "results" / f"{identity}.json", {"fixture": "sealed"})
    assert overlay.run(staged.args) == {"started": True}


def test_missing_parallel_evidence_or_mismatched_overlay_fails_before_dispatch(staged, monkeypatch):
    monkeypatch.setattr(overlay, "_pool", lambda _: pytest.fail("bad authorization accessed the pool"))
    if staged.args.schedule_authorization:
        staged.args.schedule_launcher.write_text("changed schedule launcher")
        with pytest.raises(PermissionError, match="parallel schedule"):
            overlay.run(staged.args)
    else:
        staged.args.overlay_authorization_sha256 = "0" * 64
        with pytest.raises(PermissionError, match="SSE inactivity overlay"):
            overlay.run(staged.args)
