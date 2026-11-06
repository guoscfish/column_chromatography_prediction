"""Verify real frozen packets using offline A/B/C envelopes, without production imports."""
from __future__ import annotations

import copy
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from arm_adapter import (assert_frozen_input_invariants, build_arm_inputs,
                         freeze_shared_inputs)
from controller import stable_hash


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
FROZEN = BASELINE / "frozen_source/src/qgeognn_al/active_learning_v2"
FROZEN_RUNNER = BASELINE / "frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py"
SEEDS = (157, 6101)
METHOD = "free_cw_sol61_xhigh_rerun32"
SOURCE_MANIFEST = HERE / "real_frozen_input_sources.json"
LEGACY_SETUP_MANIFEST = HERE / "real_frozen_input_sources.v1_setup.json"
QUALIFICATION_SNAPSHOT = HERE / "results/gp_qualification_snapshot.json"
OUTPUT = HERE / "results/real_frozen_input_check.json"
TRANSPORT_FIELDS = {
    "backend", "model", "reasoning_effort", "base_url", "key_env", "key_file_env",
    "sampling_parameters", "sampling_note", "max_output_tokens",
    "maximum_plan_characters", "selection_attempts", "transport_attempts",
    "retry_backoff_seconds", "timeout_seconds", "sdk_max_retries", "tools", "store",
    "served_model_policy", "provider_support_verified", "pricing_verified",
    "bounded_backoff_seconds",
}
PACKET_FIELDS = {
    "study_version", "seed", "round", "active_label_count", "selection_count",
    "objective", "memory", "validation_test_records", "delivery", "candidate_count",
    "observed_count", "condition_columns", "molecules", "candidates", "observed",
    "pending", "method", "packet_sha256",
}
CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent",
              "Volume of loading solvent/ul")
PREDICTIONS = {f"pred_{t}_{q}_ml" for t in ("V1", "V2") for q in ("q10", "q50", "q90")}
CANDIDATE_FIELDS = {
    "id", "molecule_id", "conditions", "V1_quantile_width_ml", "V2_quantile_width_ml",
    "condition_novelty", "cw_distance_percentile", "loading_amount_mg",
    "max_similarity_to_observed", "max_similarity_to_pending", "molecule_novelty",
    "pred_V1_ml", "pred_V2_ml", *PREDICTIONS,
}
OBSERVED_FIELDS = {
    "id", "molecule_id", "conditions", "V1_ml", "V2_ml", "condition_novelty",
    "loading_amount_mg", "max_similarity_to_observed", "max_similarity_to_pending",
    "molecule_novelty", "record_kind", "premeasurement_prediction_V1_ml",
    "premeasurement_prediction_V2_ml", "premeasurement_error_V1_ml",
    "premeasurement_error_V2_ml", "historical_abs_error_ml",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(raw: bytes) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise ValueError("Nonfinite JSON constant is prohibited: " + value)

    return json.loads(raw.decode("utf-8"), parse_constant=reject_nonfinite)


def source_paths() -> list[Path]:
    paths = [BASELINE / "config.json", BASELINE / "served_model_lock.json", FROZEN_RUNNER,
             FROZEN / "scientist_full_delivery.py", FROZEN / "nocw_inputs.py",
             FROZEN / "scientist_selector.py", FROZEN / "protocol.py",
             QUALIFICATION_SNAPSHOT, HERE / "results/metrics.json"]
    for seed in SEEDS:
        directory = BASELINE / "selections" / f"seed_{seed}" / METHOD / "round_00"
        paths.extend(directory / name for name in ("input.json", "catalog.json", "contract.json"))
    return paths


def decode_table(packet: dict, name: str) -> list[dict]:
    table = packet[name]
    require(set(table) == {"columns", "rows"}, "Unexpected table structure")
    columns = table["columns"]
    require(len(set(columns)) == len(columns), "Duplicate table column")
    allowed = CANDIDATE_FIELDS if name == "candidates" else OBSERVED_FIELDS
    if name == "pending":
        allowed = {"id", "molecule_id", "conditions"}
    require(set(columns) <= allowed, "Non-allowlisted field in " + name)
    records = []
    for values in table["rows"]:
        require(len(values) == len(columns), "Table width mismatch")
        row = dict(zip(columns, values))
        molecule = packet["molecules"][row.pop("molecule_id")]
        require(set(molecule) == {"smiles", "descriptors"}, "Unexpected molecule fields")
        row.update(copy.deepcopy(molecule))
        require(len(row["conditions"]) == len(CONDITIONS), "Condition width mismatch")
        row["conditions"] = dict(zip(CONDITIONS, row["conditions"]))
        records.append(row)
    require(len({row["id"] for row in records}) == len(records), "Duplicate table ID")
    return records


def check_contract_packet(contract: dict, packet: dict, catalog: dict,
                          input_raw: bytes, catalog_raw: bytes, seed: int) -> dict:
    require(contract["seed"] == packet["seed"] == seed, "Seed mismatch")
    require(contract["round"] == packet["round"] == 0, "Only frozen round zero is permitted")
    require(contract["method"] == METHOD
            and packet["method"] == "free_llm32_scientist_v2", "Method mismatch")
    require(contract["variant"] == "cw_aware" and contract["dry_run"] is False,
            "Expected real CW-Free frozen input")
    require(digest(input_raw) == contract["input_sha256"], "Input bytes violate contract")
    require(digest(catalog_raw) == contract["catalog_sha256"], "Catalog bytes violate contract")
    require(set(packet) == PACKET_FIELDS, "Unregistered packet fields")
    require(packet["validation_test_records"] == 0, "Validation/test records are prohibited")
    require(packet["selection_count"] == 32, "Expected a full LLM32 proposal")
    require(tuple(packet["condition_columns"]) == CONDITIONS, "Condition definition drift")
    require(set(catalog) == {"candidates", "observed", "pending", "salt"},
            "Unregistered catalog fields")
    without_hash = {key: value for key, value in packet.items() if key != "packet_sha256"}
    # Historical production hashing uses default ensure_ascii=True. Do not change it.
    historical_hash = digest(json.dumps(without_hash, sort_keys=True,
                                        separators=(",", ":")).encode())
    require(historical_hash == packet["packet_sha256"], "Historical packet hash mismatch")
    labeled, candidates = contract["L_t_ids"], contract["U_t_ids"]
    require(len(labeled) == len(set(labeled)) == 333, "Expected unique L333")
    require(len(candidates) == len(set(candidates)) == 2997, "Expected unique U2997")
    require(not set(labeled) & set(candidates), "L/U overlap")
    require(packet["active_label_count"] == packet["observed_count"] == len(labeled),
            "Observed count mismatch")
    require(packet["candidate_count"] == len(candidates), "Candidate count mismatch")
    for role, ids in (("observed", labeled), ("candidates", candidates)):
        require([row["id"] for row in catalog[role]] == ids, "Catalog/contract ID order differs")
        decoded = decode_table(packet, role)
        decoded_by_id = {row["id"]: row for row in decoded}
        require(set(decoded_by_id) == set(ids), "Packet/contract ID membership differs")
        for raw_row in catalog[role]:
            for key, value in raw_row.items():
                require(key in decoded_by_id[raw_row["id"]]
                        and decoded_by_id[raw_row["id"]][key] == value,
                        "Raw catalog value changed in encoded packet")
    require(not packet["pending"]["rows"] and not catalog["pending"], "Pending rows are prohibited")
    return {"contract_input_bytes_match": True, "contract_catalog_bytes_match": True,
            "historical_packet_hash_match": True, "catalog_contract_order_match": True,
            "encoded_catalog_values_match": True, "validation_test_records": 0,
            "host_method": contract["method"], "frozen_model_packet_method": packet["method"],
            "labeled_count": len(labeled), "candidate_count": len(candidates),
            "checkpoint_identity_sha256": contract["checkpoint_sha256"],
            "checkpoint_identity_source": "existing contract only; weights never opened",
            "checkpoint_content_rehashed": False,
            "model_visible_candidate_order_sha256": stable_hash(
                [row[packet["candidates"]["columns"].index("id")]
                 for row in packet["candidates"]["rows"]])}


def rejected_mutation(inputs: dict, kind: str) -> dict:
    changed = copy.deepcopy(inputs)
    if kind == "memory":
        changed["B"]["shared_base_payload"]["baseline_components"]["scientific_memory"][
            "previous_feedback_interpretation"] = "in-memory deliberate mutation"
    elif kind == "candidate_order":
        rows = changed["C"]["shared_base_payload"]["baseline_components"][
            "candidate_input"]["rows"]
        rows[0], rows[1] = rows[1], rows[0]
    elif kind == "gp_sidecar":
        changed["C"]["gp_information"]["formal_guidance_eligible"] = True
    else:
        raise ValueError("Unknown mutation kind")
    try:
        assert_frozen_input_invariants(changed)
    except ValueError as error:
        return {"rejected": True, "diagnostic": str(error), "mutation_scope": "memory_only"}
    raise AssertionError("Deliberate mutation was not rejected: " + kind)


def write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def qualification_snapshot() -> dict:
    verification = load_json((HERE / "results/verification.json").read_bytes())
    fields = ("scope", "source_checks_passed", "historical_seeds", "initial_labels_per_seed",
              "response_forward_evaluation_per_seed", "risk_training_per_seed",
              "risk_evaluation_per_seed", "screen", "formal_GP_guidance_eligible",
              "risk_current_checkpoint_transport", "all_fits_successful", "boundary_fit_count",
              "fit_count", "maximum_jitter", "actions", "source_manifest_sha256", "offline_plan_sha256")
    snapshot = {
        "schema_version": "stable-gp-development-qualification-v1",
        "scope": "Retrospective qualification only; no candidate numerical values or metric values",
        "excluded_volatile_fields": ["elapsed_seconds"],
        "verification_fields": {field: verification[field] for field in fields},
        "metrics_sha256": digest((HERE / "results/metrics.json").read_bytes()),
        "offline_plan_sha256": verification["offline_plan_sha256"],
    }
    if QUALIFICATION_SNAPSHOT.exists():
        require(load_json(QUALIFICATION_SNAPSHOT.read_bytes()) == snapshot,
                "Stable GP qualification fields changed; snapshot will not be replaced")
    else:
        write_json(QUALIFICATION_SNAPSHOT, snapshot, exclusive=True)
    return snapshot


def main() -> None:
    # Read exactly the named artifacts; never discover files, credentials, or labels.
    qualification = qualification_snapshot()
    paths = source_paths()
    raw = {str(path.relative_to(ROOT)): path.read_bytes() for path in paths}
    hashes = {name: digest(value) for name, value in sorted(raw.items())}
    manifest = {"schema_version": "real-frozen-input-sources-v2",
                "scope": "Explicit frozen baseline artifacts and retrospective qualification receipts",
                "format_revision": "Replace volatile verification.json byte lock with stable qualification snapshot; original GP source locks unchanged",
                "source_sha256": hashes}
    manifest_exists = SOURCE_MANIFEST.exists()
    migrate_legacy_manifest = False
    if manifest_exists:
        previous = load_json(SOURCE_MANIFEST.read_bytes())
        if previous.get("schema_version") == "real-frozen-input-sources-v1":
            # The only permitted one-time revision removes the elapsed-time-sensitive file.
            old_sources = dict(previous["source_sha256"])
            new_sources = dict(hashes)
            old_sources.pop(str((HERE / "results/verification.json").relative_to(ROOT)))
            new_sources.pop(str(QUALIFICATION_SNAPSHOT.relative_to(ROOT)))
            require(old_sources == new_sources, "Source drift is not a manifest format revision")
            if LEGACY_SETUP_MANIFEST.exists():
                require(load_json(LEGACY_SETUP_MANIFEST.read_bytes()) == previous,
                        "Legacy setup manifest already exists with different contents")
            else:
                write_json(LEGACY_SETUP_MANIFEST, previous, exclusive=True)
            migrate_legacy_manifest = True
        else:
            require(previous == manifest,
                    "Frozen input source drift; existing manifest will not be replaced")

    def artifact(path: Path) -> Any:
        return load_json(raw[str(path.relative_to(ROOT))])

    config = artifact(BASELINE / "config.json")
    transport = config["transport"]
    require(set(transport) == TRANSPORT_FIELDS, "Transport fields differ from frozen allowlist")
    require(transport["tools"] == [], "Native tools must remain disabled")
    require(transport["model"] == "gpt-6.1-sol" and transport["reasoning_effort"] == "xhigh",
            "Frozen model configuration changed")
    require(isinstance(transport["key_env"], str) and isinstance(transport["key_file_env"], str),
            "Credential references must be environment variable names only")
    served_lock = artifact(BASELINE / "served_model_lock.json")
    require(set(served_lock) == {"served_model"}, "Unexpected served-model lock fields")
    require(served_lock["served_model"] == transport["model"], "Served model lock mismatch")
    verification = qualification["verification_fields"]
    require(verification["formal_GP_guidance_eligible"] is False,
            "This study envelope only accepts the unqualified development GP")
    source_hashes = {name: hashes[str((FROZEN / name).relative_to(ROOT))]
                     for name in ("scientist_full_delivery.py", "nocw_inputs.py",
                                  "scientist_selector.py", "protocol.py")}
    protocol_record = {
        "baseline": "CW-Free frozen source; full-catalog direct scientific protocol",
        "source_sha256": source_hashes,
        "frozen_runner_sha256": hashes[str(FROZEN_RUNNER.relative_to(ROOT))],
        "prompt_source_chain": ["scientist_selector.SYSTEM_PROMPT",
                                "scientist_full_delivery.DIRECT_PROMPT",
                                "frozen run_free_cw_sol61_five_20261101.prompt(cw_aware)"],
        "runner_prompt_definition": 'DIRECT_PROMPT.replace("Pending-CW outcomes are identified separately.\\n", "")',
        "nocw_inputs_COMMON_PROMPT_is_not_effective_runner_prompt": True,
        "shared_engineering_validator": "frozen runner validate_plan override; source identity bound, not invoked",
        "production_components_imported": False,
        "actual_system_prompt_reconstructed": False,
        "config_file_sha256": hashes[str((BASELINE / "config.json").relative_to(ROOT))],
        "config_transport_is_historical_record_not_effective_overlay_reconstruction": True,
    }
    # Inspect the AST without executing a production import or function.
    runner_ast = ast.parse(raw[str(FROZEN_RUNNER.relative_to(ROOT))].decode("utf-8"))
    prompt_function = next(node for node in runner_ast.body
                           if isinstance(node, ast.FunctionDef) and node.name == "prompt")
    return_values = [node.value for node in ast.walk(prompt_function) if isinstance(node, ast.Return)]
    require(len(return_values) == 1, "Unexpected frozen runner prompt return structure")
    expected_expression = ast.parse(
        'DIRECT_PROMPT.replace("Pending-CW outcomes are identified separately.\\n", "")', mode="eval").body
    require(ast.dump(return_values[0]) == ast.dump(expected_expression),
            "Frozen runner prompt override differs from recorded expression")
    protocol_record["runner_prompt_expression_AST_verified_without_execution"] = True
    llm_configuration = {"transport": copy.deepcopy(transport),
                         "served_model_lock": served_lock,
                         "credential_env_values_read": False,
                         "credential_files_read": False}
    states = []
    for seed in SEEDS:
        directory = BASELINE / "selections" / f"seed_{seed}" / METHOD / "round_00"
        packet, catalog, contract = [artifact(directory / name)
                                     for name in ("input.json", "catalog.json", "contract.json")]
        checks = check_contract_packet(contract, packet, catalog,
                                       raw[str((directory / "input.json").relative_to(ROOT))],
                                       raw[str((directory / "catalog.json").relative_to(ROOT))], seed)
        shared = freeze_shared_inputs(
            checkpoint_sha256=contract["checkpoint_sha256"],
            labeled_ids=contract["L_t_ids"], candidate_ids=contract["U_t_ids"],
            scientific_memory=packet["memory"], candidate_input=packet["candidates"],
            candidate_metadata={"condition_columns": packet["condition_columns"],
                                "molecules": packet["molecules"], "catalog_salt": catalog["salt"]},
            baseline_packet=packet, selector_protocol=protocol_record,
            llm_configuration=llm_configuration,
            tool_contract={"delivery": "all_candidates_one_request", "native_tools": [],
                           "retrieval_tools": [], "new_label_access": False,
                           "validation_test_records": 0})
        # Qualification postdates the anchor. It is mock metadata, never a historical request.
        gp_information = {
            "status": "unqualified_development_no_execution",
            "formal_guidance_eligible": False,
            "retrospective_only": True,
            "historical_replay_permitted": False,
            "evaluation_receipt_postdates_anchor": True,
            "numeric_candidate_signal_payload": None,
            "qualification": [item for item in verification["screen"] if item["seed"] == seed],
            "definitions": {
                "response_gp": "fixed-r00 L333 response model; uncertainty screen failed",
                "risk_gp": "fixed-r00 log-risk; later current-checkpoint transfer unvalidated",
                "controller": "numerical quality checks only; no proposal or label execution",
            },
            "retrospective_results_sha256": {
                name: hashes[str((HERE / "results" / name).relative_to(ROOT))]
                for name in ("gp_qualification_snapshot.json", "metrics.json")},
            "future_evaluation_metric_values_in_model_envelope": False,
        }
        require(len(gp_information["qualification"]) == 2,
                "Expected the raw and CW development qualification receipts")
        inputs = build_arm_inputs(shared, gp_information)
        invariants = assert_frozen_input_invariants(inputs)
        states.append({"seed": seed, "round": 0, "baseline_checks": checks,
                       "envelope_invariants": invariants,
                       "frozen_component_hashes": shared["frozen_state_manifest"],
                       "arm_envelope_sha256": {arm: stable_hash(value) for arm, value in inputs.items()},
                       "A_B_shared_baseline_identical": True,
                       "B_C_initial_envelope_identical": stable_hash(inputs["B"]) == stable_hash(inputs["C"]),
                       "deliberate_mutations": {kind: rejected_mutation(inputs, kind)
                                                for kind in ("memory", "candidate_order", "gp_sidecar")}})
    # Fail closed if a source changed while it was being inspected.
    require(all(digest(path.read_bytes()) == hashes[str(path.relative_to(ROOT))] for path in paths),
            "Source changed during frozen-input verification")
    require(qualification_snapshot() == qualification,
            "Stable GP qualification changed while verifying inputs")
    if not manifest_exists:
        write_json(SOURCE_MANIFEST, manifest, exclusive=True)
    elif migrate_legacy_manifest:
        write_json(SOURCE_MANIFEST, manifest)
    report = {
        "schema_version": "real-frozen-input-check-v1", "passed": True,
        "scope": "真实冻结输入上的离线 mock envelope 一致性；不是生产请求重建、历史重放或上线授权",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": digest(SOURCE_MANIFEST.read_bytes()),
        "source_file_count": len(paths), "source_hashes_captured_once_and_verified_on_rerun": True,
        "source_manifest_format_revision": {
            "version": 2,
            "reason": "verification.json 的 elapsed_seconds 随重跑改变；只冻结稳定资格字段、metrics 与 offline plan 哈希",
            "legacy_setup_manifest": str(LEGACY_SETUP_MANIFEST.relative_to(ROOT))
                if LEGACY_SETUP_MANIFEST.exists() else None,
            "original_GP_input_manifest_modified": False,
        },
        "baseline_protocol_record": protocol_record, "llm_configuration_record": llm_configuration,
        "states": states,
        "limitations": [
            "checkpoint 只采用已有 contract 中的身份哈希；未打开或重新哈希权重文件",
            "完整现有 memory/candidates/observed/pending packet 被绑定；不等于生产真实请求逐字节重建",
            "GP 侧车仅含事后开发资格拒绝与定义及结果哈希；未携带候选数值或未来评价指标值",
            "当前回顾性资格信息晚于历史 r00，禁止作为该历史状态的正式 selector 重放输入",
            "config 仅记录冻结 baseline 运输字段，不重建运行中的恢复 overlay 或实际最终有效运输配置",
            "一致性哈希不能独立认证上游数据来源；合法性/证据生产 validator 未被本脚本重新执行",
        ],
        "actions": {"paid_llm_calls": 0, "production_imports": 0,
                    "credential_values_read": 0, "checkpoint_loads": 0,
                    "new_labels_revealed": 0, "test_truth_reads": 0,
                    "QGeoGNN_training_runs": 0, "input_manifest_modified": False},
    }
    write_json(OUTPUT, report)
    print(json.dumps({"passed": True, "source_file_count": len(paths),
                      "shared_base_sha256": {str(state["seed"]): state["envelope_invariants"][
                          "shared_base_sha256"] for state in states},
                      "report": str(OUTPUT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
