"""Rebuild the V3 metadata audit using only projected metadata and frozen identities."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT))

from src.qgeognn_al.active_learning_v2.protocol import ids_hash
from src.qgeognn_al.artifacts import sha256_file
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import (
    BENCHMARK_PATH, CANONICAL_COLUMNS, CONDITION_AXES, CONFIRMATION_SEEDS, E0_PATH, LEGACY_AXES,
    PUBLIC_CONDITIONS, RAW_COLUMNS, RAW_PATH, condition_changes, load_4g_metadata, load_frozen_split,
    projected_rows, purity_value,
)


AL = Path("studies/active_learning")
SOURCE_CASES = (
    ("adjudicated_NoCW_subtest", AL / "nocw_failure_attribution_20261112/tables/nocw_scientific_chain_adjudications.csv",
     ("seed", "round", "hypothesis_id", "subtest_axis", "id_a", "id_b"), "id_a", "id_b", "subtest_axis"),
    ("NoCW_diagnostic_pair", AL / "nocw_failure_attribution_20261112/tables/condition_nocw_pair_signed_events.csv",
     ("seed", "round", "axis", "row_a", "row_b"), "row_a", "row_b", "axis"),
    ("legacy_axis_pair", AL / "4g_single_factor_response_20261123/tables/legacy_165_pairs.csv",
     ("cohort", "seed", "round", "low_id", "high_id"), "low_id", "high_id", None),
)
CONTRAST_LANGUAGE = re.compile(r"\b(?:replicat\w*|match(?:es|ed|ing)?|single[- ]factor|controlled comparison|"
                               r"controlled pair|changes only|varying only|fixed loading|fixed ratio)\b", re.I)
ROW_ID = re.compile(r"\b[0-9a-f]{20}\b")
REVIEWED_LITERAL_PAIRS = {
    ("9205619d3836954c6591", "40fd02db79e028fe05a9"): (
        "nocw_vs_cw_lcmd_five_to1485_20261112", "PE/EA"),
    ("f402144b8672dd5ab2ab", "d448d8de2d8dacec62c3"): (
        "astra_high_full_pool_20261001", "sample_volume"),
    ("5e5b7c697bc0c268b288", "5f02dc3c13f7f2daed32"): (
        "sol61_xhigh_free_five_seeds_to1005_20261023", "loading_solvent_volume"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def put_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def put_csv(path, rows, columns=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(cohort, seed, role, state_id, flows, purities):
    flow = Counter(flows)
    purity = Counter(purities)
    kinds = Counter()
    for value, count in purity.items():
        kinds[purity_value(value)[0]] += count
    return {
        "cohort": cohort, "seed": seed, "role": role, "state_id": state_id, "rows": sum(flow.values()),
        **{f"flow_{value}_ml_min": flow[value] for value in ("4", "5", "6", "8", "10")},
        "flow_counts_ml_min": compact(flow), "purity_raw_counts": compact(purity),
        "purity_kind_counts": compact(kinds),
    }


def id_distribution(cohort, seed, role, state_id, ids, metadata):
    return distribution(cohort, seed, role, state_id, (metadata[i].flow for i in ids),
                        (metadata[i].purity_raw for i in ids))


def history_file_digest(path, expected):
    """Earlier workflows used either file bytes or canonical JSON digests."""
    if not expected or not path.is_file():
        return False
    payload = path.read_bytes()
    return sha256(payload).hexdigest() == expected or sha256(
        json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() == expected


def frozen_identity(root, path, freeze, known_roles):
    relative = path.relative_to(root)
    seed = int(path.parents[2].name.removeprefix("seed_"))
    method = path.parent.parent.name
    round_index = int(path.parent.name.removeprefix("round_"))
    contract_path = path.parent / "contract.json"
    artifacts = freeze.get("artifact_hashes", {})
    if freeze.get("dry_run") or any("dry_run" in part or "mock" in part for part in relative.parts):
        raise ValueError("dry_run_or_mock_path")
    if "contract" in freeze:
        contract = freeze["contract"]
        labeled, unlabeled = contract["ordered_l"], contract["ordered_u"]
        credential = "embedded_ordered_ids"
    elif contract_path.is_file():
        digest = artifacts.get("contract.json") or freeze.get("contract_sha256")
        if not history_file_digest(contract_path, digest):
            raise ValueError("contract_sha_unverified")
        contract = read_json(contract_path)
        labeled, unlabeled = contract["L_t_ids"], contract["U_t_ids"]
        credential = "frozen_contract_sha256"
    elif "L_t_ids_hash" in freeze:
        contract = freeze
        state_key = seed, freeze["L_t_ids_hash"], freeze["U_t_ids_hash"]
        if state_key not in known_roles:
            raise ValueError("no_matching_identity_for_state_hashes")
        labeled, unlabeled = known_roles[state_key]
        credential = "saved_role_sha256"
    else:
        raise ValueError("complete_l_u_identity_unavailable")

    if (int(contract.get("seed", seed)), contract.get("method", method), int(contract.get("round", round_index))) != (
            seed, method, round_index):
        raise ValueError("contract_seed_method_round_mismatch")
    if contract.get("dry_run"):
        raise ValueError("dry_run_or_mock_contract")
    batch = freeze.get("batch_ids", freeze.get("ids"))
    return seed, method, round_index, tuple(labeled), tuple(unlabeled), tuple(batch), credential


def post_reveal_receipt(path, batch):
    receipt = path.parent / "label_access_receipt.json"
    if not receipt.is_file():
        return False
    saved = read_json(receipt)
    audit = saved.get("audit", {})
    return ((saved.get("batch_freeze_sha256") == sha256_file(path)
             or saved.get("selection_freeze_sha256") == sha256_file(path))
            and audit.get("requested_ids_hash") == ids_hash(batch)
            and audit.get("purpose") == "after_acquisition_fit"
            and audit.get("acquisitions_frozen") is True)


def state_csv_credentials(root, metadata, expected, known, register, omissions):
    """Admit saved states whose IDs match a freeze or a verified preceding acquisition."""
    deferred = []
    ordered_identity = tuple(metadata)
    for path in sorted((root / AL).rglob("state.csv")):
        if not path.with_name("round_freeze.json").is_file() or "dry_run" in path.parts:
            continue
        try:
            seed = int(path.parents[2].name.removeprefix("seed_"))
            method = path.parent.parent.name
            round_id = int(path.parent.name.removeprefix("round_"))
            freeze_path = path.with_name("round_freeze.json")
            freeze = read_json(freeze_path)
            rows = projected_rows(path, ("sample_id", "canonical_index", "role"))
            if any(ordered_identity[int(row["canonical_index"])] != row["sample_id"] for row in rows):
                raise ValueError("canonical_identity_mismatch")
            l_ids = tuple(row["sample_id"] for row in rows if row["role"] == "labeled")
            u_ids = tuple(row["sample_id"] for row in rows if row["role"] == "unlabeled")
            l0, outer = expected[seed]
            if (len(rows) != len(outer) or set(l_ids) | set(u_ids) != outer
                    or not l0 <= set(l_ids)):
                raise ValueError("does_not_preserve_frozen_row_partition")
            contract = freeze.get("input", {})
            if not contract and "selected_next_batch" in freeze:
                deferred.append((path, seed, method, round_id, l_ids, u_ids, freeze))
                continue
            if (contract.get("L_t_ids_hash"), contract.get("U_t_ids_hash")) != (ids_hash(l_ids), ids_hash(u_ids)):
                raise ValueError("round_freeze_role_hash_unavailable_or_mismatch")
            register(seed, l_ids, u_ids, {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                                         "evidence": "round_freeze_role_hashes", "method": method, "round": round_id})
        except (OSError, KeyError, IndexError, ValueError, TypeError) as error:
            omissions.append({"path": path.relative_to(root).as_posix(), "reason": str(error)})

    for path, seed, method, round_id, l_ids, u_ids, freeze in deferred:
        key = seed, ids_hash(l_ids), ids_hash(u_ids)
        if key not in known:
            omissions.append({"path": path.relative_to(root).as_posix(), "reason": "no_anchored_predecessor_state"})
            continue
        register(seed, l_ids, u_ids, {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                                     "evidence": "anchored_round_state", "method": method, "round": round_id})
        batch = freeze["selected_next_batch"]
        if not batch:
            continue
        receipt_path = path.with_name("label_access_receipt.json")
        if not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        audit = receipt.get("audit", {})
        freeze_path = path.with_name("round_freeze.json")
        selected = set(batch)
        if (receipt.get("round_freeze_sha256") == sha256_file(freeze_path)
                and receipt.get("batch_ids") == batch and audit.get("requested_ids_hash") == ids_hash(batch)
                and audit.get("acquisitions_frozen") is True and audit.get("purpose") == "after_acquisition_fit"
                and len(selected) == len(batch) and selected <= set(u_ids)):
            register(seed, l_ids + tuple(batch), tuple(i for i in u_ids if i not in selected),
                     {"path": freeze_path.relative_to(root).as_posix(), "sha256": sha256_file(freeze_path),
                      "evidence": "bound_post_reveal_round_state", "method": method, "round": round_id})


def sequential_role_anchors(root, expected, register, omissions):
    runtime = root / AL / "qgeognn_v2_row_sequential_b32/runtime"
    for path in sorted(runtime.glob("seed_*/*/round_*/contract.json")):
        seed = int(path.parents[2].name.removeprefix("seed_"))
        if seed not in expected:
            continue
        try:
            freeze = read_json(path)
            contract = freeze["input"]
            paths = (path.with_name("labeled_ids.npy"), path.with_name("unlabeled_ids.npy"))
            role_ids = []
            for file in paths:
                receipt = read_json(file.with_name(file.name + ".contract.json"))
                if sha256_file(file) != receipt["sha256"]:
                    raise ValueError("sequential_id_file_sha_mismatch")
                role_ids.append(tuple(np.load(file, allow_pickle=False).tolist()))
            l_ids, u_ids = role_ids
            l0, outer = expected[seed]
            if (len(l_ids) + len(u_ids) != len(outer) or set(l_ids) | set(u_ids) != outer or not l0 <= set(l_ids)
                    or (ids_hash(l_ids), ids_hash(u_ids)) != (contract["L_t_ids_hash"], contract["U_t_ids_hash"])):
                raise ValueError("sequential_identity_mismatch")
            register(seed, l_ids, u_ids, {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                                         "evidence": "saved_sequential_role_hashes", "method": contract["method"],
                                         "round": contract["round"]})
        except (OSError, KeyError, ValueError, TypeError) as error:
            omissions.append({"path": path.relative_to(root).as_posix(), "reason": str(error)})


def frozen_distributions(root, metadata, splits):
    states, omitted, omitted_state_csvs, accepted = {}, [], [], []
    known = {}

    def register(seed, l_ids, u_ids, source):
        key = seed, ids_hash(l_ids), ids_hash(u_ids)
        if key not in states:
            states[key] = {"seed": seed, "L": len(l_ids), "U": len(u_ids),
                           "L_sha256": key[1], "U_sha256": key[2], "sources": []}
            known[key] = (l_ids, u_ids)
        states[key]["sources"].append(source)

    for seed, roles in splits.items():
        register(seed, roles["l0"], roles["u0"], {"evidence": "frozen_row_split"})
    paths = sorted(path for base in (root / AL, root / "reports")
                   for path in base.rglob("batch_freeze.json"))
    expected = {seed: (frozenset(roles["l0"]), frozenset(roles["l0"] + roles["u0"]))
                for seed, roles in splits.items()}
    sequential_role_anchors(root, expected, register, omitted_state_csvs)
    state_csv_credentials(root, metadata, expected, known, register, omitted_state_csvs)

    # Process saved identity lists before hash-only receipts so identical predecessor states resolve.
    pending = []
    for path in paths:
        try:
            freeze = read_json(path)
            if "L_t_ids_hash" in freeze:
                pending.append((path, freeze))
                continue
            resolved = frozen_identity(root, path, freeze, known)
            seed, method, round_id = record_freeze(root, path, freeze, resolved, expected, register)
            accepted.append((path, freeze, seed, method, round_id, resolved[3], resolved[4]))
        except (OSError, KeyError, ValueError, TypeError) as error:
            omitted.append({"path": path.relative_to(root).as_posix(), "reason": str(error)})

    while pending:
        remaining = []
        for path, freeze in pending:
            try:
                resolved = frozen_identity(root, path, freeze, known)
                seed, method, round_id = record_freeze(root, path, freeze, resolved, expected, register)
                accepted.append((path, freeze, seed, method, round_id, resolved[3], resolved[4]))
            except (OSError, KeyError, ValueError, TypeError) as error:
                if str(error) == "no_matching_identity_for_state_hashes":
                    remaining.append((path, freeze))
                else:
                    omitted.append({"path": path.relative_to(root).as_posix(), "reason": str(error)})
        if len(remaining) == len(pending):
            omitted.extend({"path": path.relative_to(root).as_posix(), "reason": "no_matching_identity_for_state_hashes"}
                           for path, _ in remaining)
            break
        pending = remaining
    distributions = [
        id_distribution("frozen_state", seed, role, l_sha, known[key][index], metadata)
        for key in sorted(states, key=lambda value: (value[0], states[value]["L"], value[1]))
        for seed, l_sha, _ in (key,)
        for index, role in enumerate(("L", "U"))
    ]
    return distributions, list(states.values()), omitted, omitted_state_csvs, accepted


def record_freeze(root, path, freeze, resolved, expected, register):
    seed, method, round_id, l_ids, u_ids, batch, credential = resolved
    if seed not in expected:
        raise ValueError("seed_not_in_frozen_row_manifest")
    l0_ids, outer_train = expected[seed]
    left, right, selected = frozenset(l_ids), frozenset(u_ids), frozenset(batch)
    if len(left) != len(l_ids) or len(right) != len(u_ids) or left & right or left | right != outer_train:
        raise ValueError("does_not_preserve_frozen_row_partition")
    if not l0_ids <= left or not selected or selected - right or len(selected) != len(batch):
        raise ValueError("invalid_l0_or_frozen_batch_membership")
    evidence = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                "method": method, "round": round_id}
    register(seed, l_ids, u_ids, {**evidence, "evidence": credential})
    if post_reveal_receipt(path, batch):
        register(seed, l_ids + batch, tuple(i for i in u_ids if i not in selected),
                 {**evidence, "evidence": "bound_post_reveal_receipt"})
    return seed, method, round_id


def metadata_identity(metadata, raw_rows):
    by_cas_purity, by_structure_purity = defaultdict(set), defaultdict(set)
    by_structure_cas, by_cas_structure = defaultdict(set), defaultdict(set)
    by_structure_name, by_cas_name, by_name_cas = defaultdict(set), defaultdict(set), defaultdict(set)
    old_groups, full_groups = defaultdict(list), defaultdict(list)
    for row in metadata.values():
        by_cas_purity[row.cas].add(row.purity_raw)
        by_structure_purity[row.canonical_smiles].add(row.purity_raw)
        by_structure_cas[row.canonical_smiles].add(row.cas)
        by_cas_structure[row.cas].add(row.canonical_smiles)
        by_structure_name[row.canonical_smiles].add(row.name)
        by_cas_name[row.cas].add(row.name)
        by_name_cas[row.name].add(row.cas)
        old_groups[row.identity_key(LEGACY_AXES)].append(row)
        full_groups[row.identity_key()].append(row)
    group_details = []
    for rows in old_groups.values():
        if len(rows) > 1 and len({row.identity_key() for row in rows}) > 1:
            group_details.append({"ids": [row.sample_id for row in rows],
                                  "flow": sorted({row.flow for row in rows}),
                                  "purity_raw": sorted({row.purity_raw for row in rows}),
                                  "CAS": sorted({row.cas for row in rows})})

    def multi(values):
        return {key: sorted(items) for key, items in values.items() if len(items) > 1}

    raw_cas_purity, raw_smiles_purity = defaultdict(set), defaultdict(set)
    for row in raw_rows:
        raw_cas_purity[row["CAS"]].add(row["Purity"])
        if row["smiles"]:
            raw_smiles_purity[row["smiles"]].add(row["Purity"])
    duplicates = [[row.sample_id for row in rows] for rows in full_groups.values() if len(rows) > 1]
    return {
        "unique_canonical_molecules": len(by_structure_name), "unique_cas": len(by_cas_name),
        "raw_cas_with_multiple_purity": multi(raw_cas_purity), "raw_smiles_with_multiple_purity": multi(raw_smiles_purity),
        "canonical_cas_with_multiple_purity": multi(by_cas_purity),
        "canonical_structure_with_multiple_purity": multi(by_structure_purity),
        "canonical_structure_with_multiple_cas": multi(by_structure_cas),
        "cas_with_multiple_canonical_structures": multi(by_cas_structure),
        "canonical_structure_with_multiple_names": multi(by_structure_name),
        "cas_with_multiple_names": multi(by_cas_name), "name_with_multiple_cas": multi(by_name_cas),
        "old_condition_groups": len(old_groups), "full_condition_groups": len(full_groups),
        "old_duplicate_groups": sum(len(rows) > 1 for rows in old_groups.values()),
        "full_duplicate_groups": len(duplicates), "full_duplicate_rows": sum(map(len, duplicates)),
        "potential_repeats_ids": duplicates,
        "old_groups_split_by_full_metadata": group_details,
        "old_groups_split_by_flow": sum(len(group["flow"]) > 1 for group in group_details),
        "old_groups_split_by_purity": sum(len(group["purity_raw"]) > 1 for group in group_details),
        "old_groups_split_by_cas": sum(len(group["CAS"]) > 1 for group in group_details),
    }


def contrast_record(metadata, source, source_row, case_type, method, seed, round_id, left, right,
                    declared_axis="", reference_role=""):
    left_row, right_row = metadata.get(left), metadata.get(right)
    old_changes, full_changes = (), ()
    same_structure = None
    verdict = "unverified_identity"
    if left_row and right_row:
        same_structure = left_row.canonical_smiles == right_row.canonical_smiles
        conditions = left_row.public_conditions(), right_row.public_conditions()
        old_changes = condition_changes(*conditions, LEGACY_AXES)
        full_changes = condition_changes(*conditions)
        if left == right:
            verdict = "same_row"
        elif not same_structure:
            verdict = "cross_structure"
        elif not full_changes:
            verdict = "same_full_condition_different_row"
        elif len(full_changes) == 1:
            verdict = "one_recorded_axis"
        else:
            verdict = "multiple_recorded_axes"
    return {
        "case_type": case_type, "method": method, "seed": seed, "round": round_id,
        "source": source, "source_row_or_slot": source_row, "row_a": left, "row_b": right,
        "reference_role": reference_role, "declared_axis": declared_axis, "same_structure": same_structure,
        "old_changed_axes": "|".join(old_changes), "full_changed_axes": "|".join(full_changes),
        "newly_visible_changes": "|".join(axis for axis in full_changes if axis not in LEGACY_AXES),
        "flow_a": left_row.flow if left_row else "", "flow_b": right_row.flow if right_row else "",
        "purity_a": left_row.purity_raw if left_row else "", "purity_b": right_row.purity_raw if right_row else "",
        "classification": verdict,
    }


def historical_contrasts(root, metadata, accepted):
    matches, seen = [], set()
    for case_type, source, columns, left_key, right_key, axis_key in SOURCE_CASES:
        for index, row in enumerate(projected_rows(root / source, columns), start=2):
            left, right = row[left_key], row[right_key]
            axis = row.get(axis_key, "")
            key = case_type, row["seed"], row["round"], *sorted((left, right)), axis
            if key in seen:
                continue
            seen.add(key)
            matches.append(contrast_record(metadata, source.as_posix(), index, case_type, row.get("cohort", "NoCW"),
                                           row["seed"], row["round"], left, right, axis))
    for path, freeze, seed, method, round_id, l_ids, u_ids in accepted:
        selection = path.parent / "selection.json"
        if not history_file_digest(selection, freeze.get("artifact_hashes", {}).get("selection.json")):
            continue
        selected = read_json(selection)
        roles = {i: "observed" for i in l_ids}
        roles.update({i: "candidate" for i in u_ids})
        for slot, choice in enumerate(selected.get("choices", []), start=1):
            left = choice.get("id")
            reason = choice.get("reason", "")
            if not isinstance(reason, str) or not CONTRAST_LANGUAGE.search(reason):
                continue
            for right in sorted(set(ROW_ID.findall(reason)) - {left}):
                # A mentioned literal may be another comparison in that reason. Keep this as a screen.
                reviewed = REVIEWED_LITERAL_PAIRS.get((left, right))
                axis = reviewed[1] if reviewed and reviewed[0] in selection.parts else ""
                case_type = "reviewed_literal_comparison" if axis else "literal_id_contrast_sentence_screen"
                matches.append(contrast_record(metadata, selection.relative_to(root).as_posix(), slot,
                                               case_type, method, seed, round_id, left, right, axis or "",
                                               reference_role=roles.get(right, "unverified")))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=STUDY)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = projected_rows(root / RAW_PATH, RAW_COLUMNS)
    metadata = load_4g_metadata(root)
    canonical = projected_rows(root / E0_PATH / "canonical_4g.csv", CANONICAL_COLUMNS)
    split_manifest = root / BENCHMARK_PATH / "splits/split_manifest.json"
    seeds = [item["outer_seed"] for item in read_json(split_manifest)["splits"]]
    splits = {seed: load_frozen_split(root, seed, metadata) for seed in seeds}
    distributions = [
        distribution("raw", "", "all", "", (row["Flow mL/min"] for row in raw), (row["Purity"] for row in raw)),
        id_distribution("canonical", "", "all", "", tuple(metadata), metadata),
    ]
    for seed, roles in splits.items():
        views = {"outer_train": roles["l0"] + roles["u0"], **roles}
        for role, values in views.items():
            distributions.append(id_distribution("frozen_row_split", seed, role, "", values, metadata))
    frozen, states, omitted, omitted_state_csvs, accepted = frozen_distributions(root, metadata, splits)
    put_csv(out / "flow_purity_distribution.csv", distributions + frozen)
    summaries = []
    for scope, rows in (("raw", raw), ("canonical", canonical)):
        for column in RAW_COLUMNS + (("canonical_smiles", "sample_id", "source_row_1based") if scope == "canonical" else ()):
            values = [row[column] for row in rows]
            summaries.append({"scope": scope, "field": column, "rows": len(values),
                              "blank_rows": values.count(""), "distinct_nonempty": len(set(values) - {""})})
    put_csv(out / "raw_canonical_metadata_summary.csv", summaries)
    identity = metadata_identity(metadata, raw)
    put_json(out / "metadata_identity_audit.json", identity)
    sidecar = out / "label_free_metadata.csv"
    put_csv(sidecar, [{**{key: value for key, value in asdict(row).items() if key in
                       ("sample_id", "source_row_1based", "canonical_smiles", "source_smiles", "ending_time")},
                       **row.public_conditions()} for row in metadata.values()])
    contrasts = historical_contrasts(root, metadata, accepted)
    put_csv(out / "historical_contrast_reclassification.csv", contrasts)
    dataset_headers = {}
    for path in sorted((root / "dataset").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            dataset_headers[path.name] = next(csv.reader(handle))
    source_paths = [root / RAW_PATH, root / E0_PATH / "canonical_4g.csv", root / E0_PATH / "data_metadata.json",
                    root / BENCHMARK_PATH / "protocol.json", split_manifest,
                    *(root / BENCHMARK_PATH / "splits" / f"row_seed_{seed}.csv" for seed in seeds),
                    *(root / item[1] for item in SOURCE_CASES)]
    put_json(out / "label_free_metadata_manifest.json", {
        "source_files": {path.relative_to(root).as_posix(): sha256_file(path) for path in source_paths},
        "dataset_csv_headers_only": dataset_headers, "raw_metadata_usecols": RAW_COLUMNS,
        "canonical_metadata_usecols": CANONICAL_COLUMNS, "public_conditions": PUBLIC_CONDITIONS,
        "condition_key_fields": ("canonical_smiles", *CONDITION_AXES), "confirmation_seeds": CONFIRMATION_SEEDS,
        "sidecar": {"path": sidecar.name, "sha256": sha256_file(sidecar), "rows": len(metadata)},
        "new_label_fields_projected": 0, "historical_frozen_state_count": len(states),
        "batch_freezes_admitted": len(accepted), "batch_freezes_omitted": omitted,
        "state_csvs_omitted": omitted_state_csvs,
        "contrast_rows": dict(Counter(item["case_type"] for item in contrasts)),
        "frozen_states": states,
    })
    print(compact({"canonical": len(metadata), "freezes": len(accepted), "states": len(states),
                   "omitted_freezes": len(omitted), "contrast_rows": len(contrasts)}))


if __name__ == "__main__":
    main()
