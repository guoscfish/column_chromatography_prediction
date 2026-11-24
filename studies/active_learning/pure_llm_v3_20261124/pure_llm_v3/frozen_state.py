"""Read-only bridge for SHA-bound production NoCW snapshots and past feedback."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .records import digest, file_digest, read

HISTORICAL = Path("studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions")
FIRST = HISTORICAL / "five_seed_nocw_to1005_20261029"
SPLITS = Path("studies/active_learning/qgeognn_v2_row_sequential_b32/splits")
SEEDS = (157, 887, 2357, 6101, 12203)
METHOD = "llm32_nocw"


@dataclass(frozen=True)
class FrozenState:
    seed: int
    round: int
    checkpoint: str
    packet: dict
    messages: list
    history: list
    provenance: dict


def _matches(path, expected):
    if file_digest(path) != expected:
        raise ValueError(f"frozen hash mismatch: {path}")


def verified_split(root, seed):
    manifest = read(root / SPLITS / "split_manifest.json")
    split = next(item for item in manifest["splits"] if item["outer_seed"] == seed)
    file = root / SPLITS / f"row_seed_{seed}.csv"
    _matches(file, split["sha256"])
    with file.open(newline="") as stream:
        roles = {key: [] for key in ("l0", "u0", "validation", "test")}
        for row in csv.DictReader(stream):
            roles[row["role"]].append(row["sample_id"])
    if [len(ids) for ids in roles.values()] != [333, 2997, 416, 417]:
        raise ValueError("frozen partition counts differ")
    if len(set().union(*map(set, roles.values()))) != 4163:
        raise ValueError("frozen partition is not disjoint")
    return set(roles["l0"]), set(roles["u0"]), split["sha256"]


def _receipt(directory, seed, round_index, observed, candidates):
    contract = read(directory / "contract.json")
    freeze_path = directory / "batch_freeze.json"
    freeze = read(freeze_path)
    expected = {"seed": seed, "round": round_index, "method": METHOD, "dry_run": False}
    if any(record.get(k) != v for record in (contract, freeze) for k, v in expected.items()):
        raise ValueError("foreign or mock historical freeze")
    if freeze["checkpoint_sha256"] != contract["checkpoint_sha256"]:
        raise ValueError("checkpoint drift")
    for name in ("contract.json", "catalog.json", "input.json", "request_00.json", "selection.json"):
        _matches(directory / name, freeze["artifact_hashes"][name])
    _matches(directory / "input.json", contract["input_sha256"])
    _matches(directory / "catalog.json", contract["catalog_sha256"])
    if (len(contract["L_t_ids"]) != len(observed) or set(contract["L_t_ids"]) != observed
            or len(contract["U_t_ids"]) != len(candidates) or set(contract["U_t_ids"]) != candidates):
        raise ValueError("historical state differs from frozen L/U lineage")
    return contract, freeze


def _feedback(directory, freeze, seed, round_index, candidates):
    feedback = read(directory / "feedback.json")
    receipt = read(directory / "label_access_receipt.json")
    freeze_sha = file_digest(directory / "batch_freeze.json")
    batch = freeze["batch_ids"]
    if (
        len(batch) != 32 or len(set(batch)) != 32 or not set(batch) <= candidates
        or freeze["pending_ids"] or freeze["new_batch_labels_revealed"] is not False
        or receipt["feedback_sha256"] != digest(feedback)
        or feedback["batch_freeze_sha256"] != freeze_sha
        or receipt["batch_freeze_sha256"] != freeze_sha
        or receipt["audit"]["requested_ids_hash"] != digest(sorted(batch))
        or receipt["audit"]["requested_rows"] != 32
        or receipt["audit"]["purpose"] != "after_acquisition_fit"
        or not receipt["audit"]["acquisitions_frozen"]
        or (feedback["seed"], feedback["round"], feedback["method"]) != (seed, round_index, METHOD)
        or [r["candidate_id"] for r in feedback["records"]] != batch
        or [r["candidate_id"] for r in freeze["premeasurement_predictions"]] != batch
    ):
        raise ValueError("unverified historical feedback")
    for row, prior in zip(feedback["records"], freeze["premeasurement_predictions"], strict=True):
        if row["source"] != "LLM":
            raise ValueError("foreign acquisition source")
        for target in ("V1", "V2"):
            prediction = f"pred_{target}_ml"
            if (row[prediction] != prior[prediction]
                    or row[f"error_{target}_ml"] != row[prediction] - row[f"true_{target}_ml"]):
                raise ValueError("acquisition prediction or signed error drift")
    return {**feedback, "checkpoint_sha256": freeze["checkpoint_sha256"]}


def load_historical(root, seed, round_index, revision=FIRST):
    """Fix roles from the split and move rows using prior freezes with label receipts."""
    if seed not in SEEDS or not 0 <= round_index < 36:
        raise ValueError("unregistered historical seed or round")
    observed, candidates, split_sha = verified_split(root, seed)
    base = root / revision / "selections" / f"seed_{seed}" / METHOD
    history = []
    for index in range(round_index + 1):
        directory = base / f"round_{index:02d}"
        contract, freeze = _receipt(directory, seed, index, observed, candidates)
        if index < round_index:
            feedback = _feedback(directory, freeze, seed, index, candidates)
            batch = set(freeze["batch_ids"])
            history.append(feedback)
            observed.update(batch)
            candidates.difference_update(batch)
    packet = read(directory / "input.json")
    if digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != packet["packet_sha256"]:
        raise ValueError("historical packet digest drift")
    if (packet["seed"], packet["round"], packet["validation_test_records"], packet["selection_count"]) != (
        seed, round_index, 0, 32
    ):
        raise ValueError("invalid historical packet identity")
    messages = read(directory / "request_00.json")["messages"]
    if (
        [m.get("role") for m in messages] != ["system", "user"]
        or any(set(m) != {"role", "content"} for m in messages)
        or digest(json.loads(messages[1]["content"])) != digest(packet)
    ):
        raise ValueError("historical first request differs from its audited packet")
    return FrozenState(seed, round_index, contract["checkpoint_sha256"], packet, messages, history,
        {"split_sha256": split_sha, "input_sha256": contract["input_sha256"],
         "expected_observed": sorted(observed), "expected_candidates": sorted(candidates),
         "source_directory": str(directory.relative_to(root)),
         "batch_freeze_sha256": file_digest(directory / "batch_freeze.json")})
