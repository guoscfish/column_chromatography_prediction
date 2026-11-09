#!/usr/bin/env python3
"""Run a synthetic config -> frozen representation -> acquisition -> trace demo.

Use `python -m scripts.run_offline_selection --config ... --output ...` from an
isolated checkout. This entrypoint accepts only synthetic vectors, has no data,
checkpoint, provider or training loader and stops at durable batch freezing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.qgeognn_al.infra.contracts import BaselineProvenance
from src.qgeognn_al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.workflows import SelectionWorkflow


def run(config: dict[str, object], output: Path) -> dict[str, object]:
    expected = {"schema_version", "representation", "acquisition", "seed", "budget",
                "train_ids", "candidate_ids", "train_features", "candidate_features", "provenance"}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("offline config schema mismatch")
    provenance = BaselineProvenance(**config["provenance"])
    if provenance.source_kind != "synthetic":
        raise ValueError("this demonstration CLI accepts synthetic inputs only")
    authorization = Authorization(tuple(config["train_ids"]), tuple(config["candidate_ids"]))
    snapshot = FeatureSnapshot(authorization=authorization, train_ids=config["train_ids"],
                               candidate_ids=config["candidate_ids"],
                               train_features=config["train_features"], candidate_features=config["candidate_features"],
                               config=RepresentationConfig(**config["representation"]), predictor=provenance)
    workflow = SelectionWorkflow(snapshot)
    workflow.plan(acquisition=config["acquisition"], budget=config["budget"], seed=config["seed"])
    return workflow.freeze(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": result["sha256"],
                      "selected_ids": result["payload"]["selected_ids"]}, sort_keys=True))


if __name__ == "__main__":
    main()
