"""Offline workflow with exclusive, durable freeze before label callbacks.

    fit/validate/feedback belong to future separately qualified workflow
    extensions. The current CLI stops after freeze and never reveals labels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping

from ..acquisitions.base import Selection, select
from ..agents.plans import validate_plan
from ..infra.contracts import canonical_bytes, content_hash
from ..representations.base import FeatureSnapshot


class SelectionWorkflow:
    def __init__(self, snapshot: FeatureSnapshot):
        snapshot.verify()
        self.snapshot = snapshot
        self.state = "ready"
        self.selection: Selection | None = None
        self.freeze_path: Path | None = None
        self.freeze_sha256: str | None = None

    def plan(self, *, acquisition: str, budget: int, seed: int) -> Selection:
        if self.state != "ready":
            raise ValueError("selection can only be planned from ready state")
        result = select(self.snapshot, acquisition=acquisition, budget=budget, seed=seed)
        self.selection = result
        self.state = "selected"
        return result

    def _payload(self) -> dict[str, object]:
        if self.selection is None:
            raise ValueError("no selection has been planned")
        self.snapshot.verify()
        return {"schema_version": 1, "event": "batch_frozen",
                **self.snapshot.predictor.to_dict(), "representation": self.snapshot.config.kind,
                "representation_provenance": self.snapshot.provenance(), **self.selection.to_dict(),
                "test_accessed": False, "labels_revealed": False,
                "checkpoint_loaded": False, "training_performed": False}

    def freeze(self, path: Path) -> dict[str, object]:
        if self.state != "selected":
            raise ValueError("freeze requires selected state")
        payload = self._payload()
        envelope = {"sha256": content_hash(payload), "payload": payload}
        path = Path(path)
        # Exclusive creation refuses to overwrite historical or active files.
        with path.open("xb") as handle:
            handle.write(canonical_bytes(envelope) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.freeze_path = path
        self.freeze_sha256 = envelope["sha256"]
        self.state = "frozen"
        return envelope

    def _verify_freeze(self) -> dict[str, object]:
        if self.freeze_path is None or self.freeze_sha256 is None:
            raise ValueError("durable batch freeze is required before reveal")
        envelope = json.loads(self.freeze_path.read_bytes())
        payload = envelope["payload"]
        if envelope["sha256"] != self.freeze_sha256 or content_hash(payload) != self.freeze_sha256:
            raise ValueError("batch freeze changed")
        if payload != self._payload():
            raise ValueError("frozen batch differs from current authorized snapshot or plan")
        return payload

    def reveal(self, callback: Callable[[tuple[str, ...]], Mapping[str, object]]) -> dict[str, object]:
        if self.state != "frozen":
            raise ValueError("reveal requires a verified durable freeze")
        self._verify_freeze()
        assert self.selection is not None
        assert self.freeze_path is not None
        # Durable exclusive marker is written before callback. Resume cannot
        # silently repeat a reveal after a crash or an ambiguous callback failure.
        marker = self.freeze_path.with_name(self.freeze_path.name + ".reveal_started.json")
        with marker.open("xb") as handle:
            handle.write(canonical_bytes({"freeze_sha256": self.freeze_sha256,
                                          "event": "reveal_started"}) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.state = "reveal_started"
        revealed = dict(callback(self.selection.selected_ids))
        if set(revealed) != set(self.selection.selected_ids):
            raise ValueError("label callback must return exactly the frozen selected IDs")
        self.state = "revealed"
        return revealed

    @classmethod
    def resume_frozen(cls, snapshot: FeatureSnapshot, path: Path) -> "SelectionWorkflow":
        envelope = json.loads(Path(path).read_bytes())
        payload = envelope["payload"]
        if content_hash(payload) != envelope["sha256"]:
            raise ValueError("batch freeze hash mismatch")
        chosen = validate_plan({"selected_ids": payload["selected_ids"]},
                               candidate_ids=snapshot.candidate_ids, budget=payload["budget"])
        workflow = cls(snapshot)
        workflow.selection = Selection(chosen, payload["acquisition"], payload["seed"], payload["budget"],
                                       payload["feature_hash"], canonical_bytes(payload["trace"]).decode())
        workflow.freeze_path, workflow.freeze_sha256 = Path(path), envelope["sha256"]
        workflow._verify_freeze()
        marker = Path(path).with_name(Path(path).name + ".reveal_started.json")
        workflow.state = "reveal_started" if marker.exists() else "frozen"
        return workflow
