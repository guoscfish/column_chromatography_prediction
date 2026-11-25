"""Stage and run a SHA-bound PureLLM-V3 selection with no label-store dependency."""
import argparse
import copy
from dataclasses import dataclass
import re
from pathlib import Path

from .context_builder import (CONDITIONS, LEGACY_CONDITIONS, LEGACY_HYPOTHESIS_FIELDS,
                              encode_context, make_context)
from .deterministic_validator import ValidationResult, validate_plan
from .frozen_state import FIRST, SEEDS, load_historical
from .memory_manager import MEASUREMENT_FIELDS, MemoryManager
from .metadata_view import load_4g_metadata
from .prompt_builder import REPAIR_INSTRUCTIONS, REPLAN_INSTRUCTIONS, STUDY_PATH, messages
from .records import digest, dumps, file_digest, once, read
from .repair_protocol import ReplanRequested, apply_repair, build_repair_request
from .response_schema import SCHEMA_VERSION, ResponseFormatError, parse_response
from .transport_manager import (
    TransportConfig, TransportStop, check_request, make_request, responses_call, round_lock, run_request,
)

ROOT = Path(__file__).resolve().parents[4]
PRESETS = {
    "A0": ("historical", "legacy", "A", False, False, "M0", True),
    "A1": ("v3_legacy", "legacy", "A", False, False, "M0", True),
    "A2": ("v3_legacy", "legacy", "A", False, True, "M0", True),
    "A3": ("v3_legacy", "short", "A", False, True, "M0", False),
    "A4": ("v3_legacy", "short", "A", True, True, "M0", False),
    "A5": ("v3_legacy", "short", "B", True, True, "M0", False),
    "A6": ("v3", "short", "B", True, True, "M0", False),
    "A7": ("v3", "short", "B", True, True, "M2", False),
}
SAFE_ID = re.compile(r"(?:[CO]-[0-9]{1,12}|[0-9a-f]{20}|H-[A-Za-z0-9_-]{1,48})\Z")
PROTOCOL_PATH = STUDY_PATH / "execution_config.json"


@dataclass(frozen=True)
class Stage:
    packet: dict
    messages: list
    binding: dict
    context: object
    memory: MemoryManager
    schema: str


def _legacy_metadata_packet(state, context):
    user_packet = copy.deepcopy(state.packet)
    user_packet["condition_columns"] = [*LEGACY_CONDITIONS, *[k for k in CONDITIONS if k not in ("PE/EA",)]]
    for name in ("candidates", "observed"):
        rows = getattr(context, name)
        for values in user_packet[name]["rows"]:
            conditions = rows[values[0]]["conditions"]
            values[2].extend(conditions[key] for key in user_packet["condition_columns"][len(LEGACY_CONDITIONS):])
    user_packet["packet_sha256"] = digest({k: v for k, v in user_packet.items() if k != "packet_sha256"})
    return user_packet


def _memory_view(memory, context, mode, *, use_legacy_history=False):
    view = memory.prompt_view(mode, observed=context.observed, all_row_ids=context.all_row_ids)
    if use_legacy_history and mode == "M0":
        real_to_visible = {context.all_row_ids[key]: key for key in context.observed}
        view["legacy_previous_hypotheses"] = [
            {**hypothesis, **{field: [real_to_visible[key] for key in hypothesis[field]]
                              for field in ("supporting_observed_ids", "contradicting_observed_ids")}}
            for hypothesis in context.legacy_memory["previous_hypotheses"]
        ]
        view["previous_feedback_interpretation"] = context.legacy_memory["previous_feedback_interpretation"]
    fields = {"mode", "latest_feedback", "historical_top_abs_errors", "previous_feedback_interpretation",
              "hypotheses", "archived_hypotheses", "hypothesis_updates", "legacy_previous_hypotheses"}
    if set(view) - fields:
        raise ValueError("memory has an unregistered outbound field")
    for name in ("latest_feedback", "historical_top_abs_errors"):
        seen = set()
        for outcome in view.get(name, []):
            if set(outcome) - ({"observed_id"} | set(MEASUREMENT_FIELDS)):
                raise ValueError("unauthorized measurement field in memory")
            row = context.observed.get(outcome.get("observed_id"))
            if (row is None or outcome["observed_id"] in seen
                    or any(row.get(k) != v for k, v in outcome.items() if k != "observed_id")):
                raise ValueError("memory measurement differs from current observed facts")
            seen.add(outcome["observed_id"])
    return view


def stage_state(root, state, *, arm="A7", memory_mode=None, layout_override=None,
                identity_override=None, snapshot=None):
    prompt, identity, layout, compress, metadata, default_mode, echo = PRESETS[arm]
    mode = memory_mode or default_mode
    if not isinstance(echo, bool):
        raise ValueError("invalid preset echo flag")
    if (prompt != "v3" and (memory_mode not in (None, "M0") or snapshot is not None)
            or echo and (layout_override or identity_override)):
        raise ValueError("legacy replay requires its registered layout, ID and memory")
    identity = identity_override or identity
    layout = layout_override or layout
    context = make_context(state, load_4g_metadata(root), identity=identity,
                           include_metadata=metadata, legacy_molecules=echo)
    memory = MemoryManager(state.seed) if snapshot is None else MemoryManager.from_snapshot(snapshot, seed=state.seed)
    memory.register_observed(observed=context.observed, all_row_ids=context.all_row_ids, facts=context.facts)
    if arm in ("A0", "A1"):
        packet = state.packet
    elif arm == "A2":
        packet = _legacy_metadata_packet(state, context)
    else:
        packet = encode_context(context, layout=layout, compress=compress,
                                memory=_memory_view(memory, context, mode, use_legacy_history=snapshot is None))
    outgoing = messages(root, prompt, packet, full_metadata=metadata,
                        legacy_reference=state.messages if arm in ("A0", "A1") else None)
    schema = SCHEMA_VERSION if prompt == "v3" else "historical-scientist-plan"
    binding = {"seed": state.seed, "round": state.round, "checkpoint_sha256": state.checkpoint,
               "packet_sha256": digest(packet), "mapping_sha256": context.mapping_digest,
               "prompt_version": prompt, "prompt_sha256": digest(outgoing[0]["content"]),
               "base_messages_sha256": digest(outgoing), "schema_version": schema, "arm": arm,
               "split_sha256": state.provenance["split_sha256"],
               "state_sha256": digest(state.provenance)}
    return Stage(packet, outgoing, binding, context, memory, schema)


def _validation(stage, proposal):
    ctx = stage.context
    if stage.schema == SCHEMA_VERSION:
        hypotheses = stage.memory.hypothesis_view(observed=ctx.observed, all_row_ids=ctx.all_row_ids)
        visible = None
        if stage.packet["memory"]["mode"] == "M0":
            visible = {update["hypothesis_id"] for update in stage.packet["memory"]["hypothesis_updates"]}
        return validate_plan(proposal, candidates=ctx.candidates, observed=ctx.observed,
                             all_row_ids=ctx.all_row_ids, hypotheses=hypotheses,
                             visible_hypothesis_ids=visible)
    return _validate_legacy(stage, proposal)


def _validate_legacy(stage, proposal):
    """Keep the historical direct-plan structural/evidence contract for the ablation."""
    from types import SimpleNamespace
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import validate_direct

    result = ValidationResult(expected_count=32)
    allowed = {"type", "choices", "hypotheses", "feedback_interpretation", "batch_rationale", "batch_strategy"}
    echo = stage.packet.get("packet_sha256")
    if echo is not None:
        allowed.add("packet_sha256")
    if not isinstance(proposal, dict) or set(proposal) != allowed:
        result.errors.append({"code": "legacy_response_fields", "path": ""})
        return result
    # Existing validator reads table identities and uses no predictor, labels or catalog scoring.
    ctx = stage.context
    empty = {"columns": [], "rows": []}
    packet = {"packet_sha256": echo, "selection_count": 32, "round": ctx.round, "molecules": {"m": {}},
              "condition_columns": [], "candidates": {"columns": ["id", "molecule_id", "conditions"],
              "rows": [[key, "m", []] for key in ctx.candidates]}, "observed": empty}
    try:
        validate_direct(proposal, SimpleNamespace(candidates=ctx.candidates, observed=ctx.observed), packet)
    except (KeyError, TypeError, ValueError):
        result.errors.append({"code": "legacy_direct_validator_rejected", "path": ""})
        return result
    for idx, choice in enumerate(proposal["choices"]):
        if set(choice) != {"id", "reason", "hypothesis_id"}:
            result.errors.append({"code": "legacy_choice_fields", "path": f"/choices/{idx}"})
        else:
            result.legal_choice_indices.append(idx)
            result.legal_candidate_indices.append(idx)
    for idx, hypothesis in enumerate(proposal["hypotheses"]):
        if set(hypothesis) != LEGACY_HYPOTHESIS_FIELDS:
            result.errors.append({"code": "legacy_hypothesis_fields", "path": f"/hypotheses/{idx}"})
    if stage.binding["arm"] in ("A0", "A1", "A2"):
        from src.qgeognn_al.active_learning_v2.nocw_inputs import audit_no_cw

        try:
            audit_no_cw(proposal)
        except ValueError:
            result.errors.append({"code": "legacy_keyword_validator", "path": ""})
    return result


def _safe_repair(contract, stage):
    """Only source-registered IDs and structural diagnostics cross the rejected-plan boundary."""
    secure = {"repair_id": contract["repair_id"], "mode": contract["mode"]}
    for field in ("replace_choice_slots", "replace_hypothesis_update_slots", "replace_fields",
                  "append_choice_count", "locked_choice_slots", "locked_candidate_id_slots"):
        secure[field] = contract[field]
    allowed = {**stage.context.candidates, **stage.context.observed}
    diagnostics = []
    for issue in contract["diagnosis"]["errors"]:
        item = {k: issue[k] for k in ("code", "path")}
        for k in ("candidate_id", "reference_id", "hypothesis_id"):
            if isinstance(issue.get(k), str) and SAFE_ID.fullmatch(issue[k]):
                item[k] = issue[k]
        for k in ("expected", "actual", "position", "retained_position"):
            if type(issue.get(k)) is int:
                item[k] = issue[k]
        for k in ("allowed", "actual_changed_axes", "missing_axes"):
            if isinstance(issue.get(k), list):
                item[k] = [axis for axis in issue[k] if axis in (*CONDITIONS, *LEGACY_CONDITIONS)]
        if isinstance(issue.get("expected_molecule_id"), str) and re.fullmatch(
            r"(?:M-\d{4}|m\d+)", issue["expected_molecule_id"]
        ):
            item["expected_molecule_id"] = issue["expected_molecule_id"]
        diagnostics.append(item)
    secure["errors"] = diagnostics
    proposal = contract.get("proposal", {})
    choices = proposal.get("choices", [])
    secure["available_choice_anchors"] = [
        {"slot": index, "candidate_id": key, "molecule_id": allowed[key]["molecule_id"],
         "references": [r for r in choice.get("references", [])
                        if r.get("reference_id") in allowed and isinstance(r.get("relationship"), str)
                        and r["relationship"] in {
                             "single_factor_contrast", "cross_molecule_analogy", "condition_response",
                             "replicate", "multifactor_contrast", "prediction_comparison", "observed_evidence"
                        }]}
        for index, choice in enumerate(choices)
        if (key := choice.get("candidate_id")) in allowed
    ]
    # Each displayed reference comes from the host membership view, not the rejected proposal's prose.
    for choice in secure["available_choice_anchors"]:
        choice["references"] = [
            {"reference_id": r["reference_id"],
             "reference_kind": "observed" if r["reference_id"] in stage.context.observed else "candidate",
             "relationship": r["relationship"],
             **({"changed_axis": r["changed_axis"]} if r.get("changed_axis") in CONDITIONS else {})}
            for r in choice["references"]
        ]
    secure["available_hypothesis_anchors"] = [
        {"slot": slot, "hypothesis_id": h["hypothesis_id"],
         **{field: [key for key in h.get(field, []) if key in stage.context.observed]
            for field in ("supporting_observed_ids", "contradicting_observed_ids")}}
        for slot, h in enumerate(proposal.get("hypothesis_updates", []))
        if isinstance(h.get("hypothesis_id"), str) and SAFE_ID.fullmatch(h["hypothesis_id"])
        and h["hypothesis_id"].startswith("H-")
    ]
    return secure


def _next_messages(stage, repair=None):
    if repair is None:
        return stage.messages
    projected = (_safe_repair(repair, stage) if "repair_id" in repair else repair)
    instruction = REPAIR_INSTRUCTIONS if repair["mode"] == "patch" else REPLAN_INSTRUCTIONS
    return [*stage.messages, {"role": "user", "content": instruction + "\n" + dumps(projected)}]


def select_stage(stage, directory, config, call, *, mock, retry_options=None, response_projection=None):
    """Persist each phase and resume deterministically from complete response receipts."""
    retry_options = retry_options or {}
    reference_repair = stage.binding.get("fixed_reference_repair")
    if (response_projection is None) != (reference_repair is None):
        raise PermissionError("the response projection must match an explicitly bound repair")
    if response_projection is not None and (mock or config.logical_replies != 1 or config.transport_attempts != 1):
        raise PermissionError("the scoped reference repair authorizes one real attempt")
    with round_lock(directory):
        if digest(stage.packet) != stage.binding["packet_sha256"] or (
            digest(stage.messages) != stage.binding["base_messages_sha256"]
            or stage.context.mapping_digest != stage.binding["mapping_sha256"]
        ):
            raise ValueError("frozen stage binding drift")
        # Re-check the decoded role data against the exact outbound packet.
        if stage.binding["arm"] not in ("A0", "A1", "A2"):
            from .context_builder import decode_context
            if decode_context(stage.packet) != {role: getattr(stage.context, role) for role in ("observed", "candidates")}:
                raise ValueError("staged role data differs from outbound packet")
        frozen = directory / "batch_freeze.json"
        once(directory / "binding.json", {"binding": stage.binding, "configuration": config.__dict__, "mock": mock})
        if frozen.exists():
            saved = read(frozen)
            if saved["binding"] != stage.binding or saved["mock"] != mock:
                raise ValueError("existing batch belongs to another contract")
            if file_digest(directory / "selection.json") != saved["selection_file_sha256"]:
                raise ValueError("frozen selection hash drift")
            if saved.get("memory_file_sha256") and file_digest(directory / "memory.json") != saved["memory_file_sha256"]:
                raise ValueError("frozen memory hash drift")
            return saved

        proposal = repair = None
        seen_response_ids = set()
        for index in range(config.logical_replies):
            request = make_request(stage.binding, index, _next_messages(stage, repair), config)
            check_request(request, stage.binding, config)
            answer = run_request(directory, request, config, call, **retry_options)
            if answer["response_id"] in seen_response_ids:
                raise TransportStop("provider reused an identity for a different logical request")
            seen_response_ids.add(answer["response_id"])
            turn = directory / f"turn_{index:02d}"
            try:
                response = parse_response(answer["text"])
            except ResponseFormatError:
                once(turn / "decision.json", {"state": "malformed_complete_response"})
                repair = repair or {"mode": "full_replan", "errors": [{"code": "parse_error", "path": ""}]}
                continue
            if response_projection is not None:
                try:
                    response = response_projection(response)
                except ResponseFormatError:
                    once(turn / "decision.json", {"state": "patch_rejected", "reason": "outside_scoped_reference_repair"})
                    continue
                once(turn / "response_projection.json", {"raw_reply_sha256": answer["answer_sha256"],
                     "projected_plan_sha256": digest(response), "reference_repair_sha256": digest(reference_repair)})
            if repair is not None and repair["mode"] == "patch":
                try:
                    response = apply_repair(proposal, repair, response, binding=stage.binding,
                                            validator=lambda plan: _validation(stage, plan))
                except ReplanRequested as error:
                    once(turn / "decision.json", {"state": "model_requested_replan", "reason": str(error)})
                    repair = {"mode": "full_replan", "errors": []}
                    continue
                except ValueError:
                    once(turn / "decision.json", {"state": "patch_rejected"})
                    continue
            proposal = response
            validation = _validation(stage, proposal)
            once(turn / "decision.json", {"state": "legal" if validation.valid else "rejected",
                                          "diagnosis": validation.as_dict()})
            if validation.valid:
                once(directory / "selection.json", proposal)
                if stage.schema == SCHEMA_VERSION:
                    memory = MemoryManager.from_snapshot(stage.memory.snapshot(), seed=stage.context.seed)
                    memory.record_round(stage.context.round, proposal, observed=stage.context.observed,
                        all_row_ids=stage.context.all_row_ids, checkpoint_sha256=stage.context.checkpoint,
                        facts=stage.context.facts, candidates=stage.context.candidates)
                    once(directory / "memory.json", memory.snapshot())
                id_key = "candidate_id" if stage.schema == SCHEMA_VERSION else "id"
                ids = [stage.context.all_row_ids[row[id_key]] for row in proposal["choices"]]
                memory_hash = (file_digest(directory / "memory.json") if stage.schema == SCHEMA_VERSION else None)
                record = {"binding": stage.binding, "request_id": request["request_id"],
                          "provider_response_id": answer["response_id"], "mock": mock,
                          "accepted_logical_index": index, "accepted_attempt": answer["attempt"],
                          "selection_file_sha256": file_digest(directory / "selection.json"),
                          "memory_file_sha256": memory_hash,
                          "batch_ids": ids, "label_access": False}
                once(frozen, record)
                return record
            repair = (build_repair_request(proposal, validation=validation, binding=stage.binding)
                      if stage.schema == SCHEMA_VERSION else {"mode": "full_replan",
                      "errors": [{"code": "legacy_validation_failure", "path": ""}]})
        raise TransportStop("logical response budget exhausted")


def _authorize(path, stage, config):
    if path is None:
        raise PermissionError("paid selection requires a reviewed request-bound authorization file")
    approval = read(path)
    # The approved hash includes dataset/split, provider/config, model and the complete first payload.
    expected = make_request(stage.binding, 0, stage.messages, config)["request_id"]
    if (
        approval.get("request_id") != expected or approval.get("scope") != "paid_selection_only"
        or not isinstance(approval.get("authorized_by"), str) or not approval["authorized_by"].strip()
        or not approval.get("authorization_reference")
    ):
        raise PermissionError("paid selection authorization does not match this request")
    return approval


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("stage", "select"))
    cli.add_argument("--seed", required=True, type=int, choices=SEEDS)
    cli.add_argument("--round", required=True, type=int)
    cli.add_argument("--arm", default="A7", choices=PRESETS)
    cli.add_argument("--memory-mode", choices=("M0", "M1", "M2"))
    cli.add_argument("--layout", choices=("A", "B"))
    cli.add_argument("--identity", choices=("short", "legacy"))
    cli.add_argument("--revision", type=Path, default=FIRST)
    cli.add_argument("--config", type=Path, default=ROOT / PROTOCOL_PATH)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--trajectory", type=Path, help="independent future V3 trajectory directory")
    cli.add_argument("--authorization", type=Path)
    args = cli.parse_args()
    if not 0 <= args.round < 36:
        cli.error("selection rounds are 0 through 35, ending at L1485")
    config = TransportConfig(**read(args.config)["transport"])
    snapshot = None
    if args.trajectory is not None:
        from .trajectory import load_transition, round_directory

        if (args.arm not in ("A6", "A7") or args.layout or args.identity or args.revision != FIRST):
            cli.error("trajectory mode requires a fixed A6 or A7 preset")
        if args.action == "select" and args.out.resolve() != round_directory(args.trajectory, args.round).resolve():
            cli.error("trajectory selection output must be its round directory")
        contract = {"seed": args.seed, "arm": args.arm,
                    "memory_mode": args.memory_mode or PRESETS[args.arm][5],
                    "transport_sha256": digest(config.__dict__)}
        if args.round and read(args.trajectory / "trajectory.json") != contract:
            cli.error("trajectory contract differs from the previous selection")
        source, snapshot = load_transition(ROOT, args.trajectory, args.seed, args.round)
    else:
        source = load_historical(ROOT, args.seed, args.round, args.revision)
    stage = stage_state(ROOT, source, arm=args.arm, memory_mode=args.memory_mode,
                        layout_override=args.layout, identity_override=args.identity, snapshot=snapshot)
    request = make_request(stage.binding, 0, stage.messages, config)
    if args.action == "stage":
        once(args.out / "staged_request.json", request)
        print(dumps({"status": "STAGED", "eligible": len(stage.context.candidates),
                     "observed": len(stage.context.observed), "request_id": request["request_id"]}))
        return
    approval = _authorize(args.authorization, stage, config)
    if args.trajectory is not None:
        once(args.trajectory / "trajectory.json", contract)
    once(args.out / "authorization.json", approval)
    frozen = select_stage(stage, args.out, config, responses_call, mock=False)
    print(dumps({"status": "FROZEN", "selection_count": len(frozen["batch_ids"])}))


if __name__ == "__main__":
    main()
