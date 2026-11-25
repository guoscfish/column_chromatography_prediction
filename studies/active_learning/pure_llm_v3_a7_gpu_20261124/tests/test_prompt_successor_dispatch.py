from copy import deepcopy
from types import SimpleNamespace
import time

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import SCHEMA
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prompt_successor as successor


def test_one_shot_checks_unchanged_l429_data_and_disallows_failed_answers():
    original = {"binding": {**dict.fromkeys(successor.DATA_BINDINGS, "original"), "prompt_sha256": "old"},
                "payload": {"input": [{"role": "system", "content": "old"}, {"role": "user", "content": "L429"}]}}
    fresh = deepcopy(original)
    fresh.update(logical_index=0)
    fresh["binding"].update(seed=887, round=3, prompt_sha256="approved")
    fresh["payload"]["input"][0]["content"] = "fresh"
    dispatcher = SimpleNamespace(validate=lambda job: (job, None, {"attempt": 0}))
    assert successor._validate(dispatcher, fresh, original, "approved") is fresh
    changes = (
        lambda job: job["binding"].update(round=4),
        lambda job: job["binding"].update(checkpoint_sha256="other"),
        lambda job: job["payload"]["input"][1].update(content="changed"),
        lambda job: job["payload"]["input"].append({"role": "user", "content": "failed answer"}),
        lambda job: job["binding"].update(prompt_sha256="old"),
    )
    for alter in changes:
        changed = deepcopy(fresh)
        alter(changed)
        with pytest.raises(PermissionError, match="updated prompt and original"):
            successor._validate(dispatcher, changed, original, "approved")


def test_attempt_selection_is_immutable_across_retry_of_delivery(tmp_path):
    for name in successor.JOURNAL_DIRS:
        (tmp_path / name).mkdir()
    contract, auth, identity, job_sha = "a" * 64, "b" * 64, "c" * 32, "d" * 64
    info = {"contract_sha256": contract, "authorization_sha256": auth, "mock": False}
    dispatcher = SimpleNamespace(path=tmp_path, info=info)
    job = {"attempt_id": identity, "job_sha256": job_sha}
    inventory = {"schema": SCHEMA, **info, "paused": False, "attempt_ids": [identity]}
    courier = SimpleNamespace(list=lambda: inventory, get=lambda _: job)
    deadline = time.monotonic() + 1
    assert successor._selection(dispatcher, courier, lambda value: value, None, deadline, .01) == job
    assert read(tmp_path / successor.SELECTION) == job
    with pytest.raises(PermissionError, match="immutable one-shot"):
        successor._selection(dispatcher, courier, lambda value: value, "e" * 32, deadline, .01)
    inventory["attempt_ids"] = []
    assert successor._selection(dispatcher, courier, lambda value: value, None, deadline, .01) == job
    successor._check_journal_scope(tmp_path, identity)
    (tmp_path / "attempts" / ("e" * 32)).mkdir()
    with pytest.raises(PermissionError, match="foreign attempt"):
        successor._check_journal_scope(tmp_path, identity)
