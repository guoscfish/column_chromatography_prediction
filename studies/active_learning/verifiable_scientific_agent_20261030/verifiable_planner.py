"""Offline, label-free planning diagnostics and mock-only localized repair.

Nothing here calls a model, trains a predictor, or opens a label store. Authoritative
acceptance remains the original validator. Added findings are diagnostics only.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Unanalyzable(ValueError):
    """No unique complete JSON interpretation, so no batch may be inferred."""


def parse_complete_json(text: str) -> Any:
    """One complete JSON value; reject duplicate keys and non-finite constants.

    This deliberately has no brace search, regex ID recovery, trailing-text removal,
    quote repair, or truncated-output completion.
    """
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Unanalyzable(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise Unanalyzable(f"non-finite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise Unanalyzable(f"overflowing/non-finite JSON number: {value}")
        return number

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant,
                          parse_float=finite_float)
    except (ValueError, TypeError) as error:
        raise Unanalyzable(str(error)) from error


def format_only_repair(text: str) -> tuple[str, str]:
    """Only unwrap exactly one whole JSON code fence, with no surrounding prose.

    The interior must itself be complete unambiguous JSON. No fields, choices,
    strings, evidence references or experiment identities are changed.
    """
    try:
        parse_complete_json(text)
        return text, "identity"
    except Unanalyzable:
        pass
    match = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*", text)
    if match is None or "```" in match.group(1):
        raise Unanalyzable("not a single complete JSON fence")
    inner = match.group(1)
    parse_complete_json(inner)
    return inner, "unwrap_single_json_fence"


def table_ids(packet: dict, table: str) -> tuple[str, ...]:
    data = packet[table]
    columns = data["columns"]
    if len(columns) != len(set(columns)) or "id" not in columns:
        raise ValueError(f"ambiguous {table} columns")
    position = columns.index("id")
    if any(not isinstance(row, list) or len(row) != len(columns) for row in data["rows"]):
        raise ValueError(f"malformed {table} table")
    ids = tuple(row[position] for row in data["rows"])
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"nonunique/nonstring {table} IDs")
    return ids


@dataclass(frozen=True)
class FrozenContext:
    """All identity sets must be explicitly supplied; no inference from labels."""
    candidate_ids: tuple[str, ...]
    labeled_ids: frozenset[str]
    observed_ids: frozenset[str]
    packet_sha256: str
    selection_count: int
    round: int
    variant: str = "direct"
    pending_ids: tuple[str, ...] = ()
    validator_kind: str = "auto"
    binding_sha256: str = field(init=False)

    def __post_init__(self):
        candidates = tuple(self.candidate_ids)
        labeled, observed = frozenset(self.labeled_ids), frozenset(self.observed_ids)
        if (any(not isinstance(i, str) or not i for i in (*candidates, *labeled, *observed))
                or len(candidates) != len(set(candidates))):
            raise ValueError("explicit identity sets must contain unique nonempty string IDs")
        pending = tuple(self.pending_ids)
        if (len(pending) != len(set(pending)) or
                any(not isinstance(i, str) or not i for i in pending) or
                set(pending) & (set(candidates) | labeled)):
            raise ValueError("invalid/overlapping pending identities")
        if set(candidates) & labeled or not observed <= labeled:
            raise ValueError("U_t/L_t overlap or observations outside explicit L_t")
        if (type(self.selection_count) is not int or not 0 < self.selection_count <= len(candidates)
                or type(self.round) is not int or self.round < 0
                or self.variant not in ("direct", "no_cw", "cw_aware")
                or not isinstance(self.packet_sha256, str) or not self.packet_sha256):
            raise ValueError("invalid frozen planning contract")
        kind = self.validator_kind
        if kind == "auto":
            kind = "direct" if self.variant == "direct" else "nocw_five" if self.variant == "no_cw" else "nocw_paired"
        if kind not in ("direct", "nocw_five", "nocw_paired"):
            raise ValueError("unknown original validator contract")
        object.__setattr__(self, "validator_kind", kind)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "labeled_ids", labeled)
        object.__setattr__(self, "observed_ids", observed)
        object.__setattr__(self, "pending_ids", pending)
        object.__setattr__(self, "binding_sha256", digest({
            "ordered_candidates": candidates, "pending": pending,
            "L_t": sorted(labeled), "observed": sorted(observed),
            "packet_sha256": self.packet_sha256, "selection_count": self.selection_count,
            "round": self.round, "variant": self.variant, "validator_kind": kind}))

    @classmethod
    def from_packet(cls, packet: dict, *, labeled_ids, unlabeled_ids, variant="direct", validator_kind="auto"):
        """Only read identity columns. Full packet labels are never dereferenced."""
        candidates, observed = table_ids(packet, "candidates"), table_ids(packet, "observed")
        pending = table_ids(packet, "pending") if "pending" in packet else ()
        unlabeled = frozenset(unlabeled_ids)
        if not set(candidates) <= unlabeled or not set(pending) <= unlabeled:
            raise ValueError("supplied candidates/pending outside explicit U_t")
        if set(candidates) & set(pending) or unlabeled & set(labeled_ids):
            raise ValueError("pending/candidate or U_t/L_t overlap")
        # Hybrid candidate table excludes its 16 pending choices; no silent union.
        if set(candidates) | set(pending) != unlabeled:
            raise ValueError("full U_t is not accounted for by candidates and pending")
        return cls(candidates, frozenset(labeled_ids), frozenset(observed),
                   packet["packet_sha256"], packet["selection_count"], packet["round"], variant, pending,
                   validator_kind)


def frozen_validator_bundle(repo_root: Path | None = None) -> dict:
    """Execute only original pure validator ASTs; avoid RDKit/training imports.

    ID-only tables are behaviorally equivalent for these functions: they inspect
    candidate/evidence identity, prose and schema, never candidate chemistry/labels.
    SHA256 values identify the exact production source files that were read.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    source_dir = root / "src/qgeognn_al/active_learning_v2"
    namespace = {"copy": copy, "json": json, "re": re, "hashlib": hashlib}
    sources = {}
    selections = {
        "protocol.py": {"stable_hash"},
        "scientist_selector.py": {"MAX_HYPOTHESES"},
        "scientist_full_delivery.py": {"decode_table", "validate_direct"},
        "nocw_inputs.py": {"CW_PATTERN", "PLAN_FIELDS", "HYPOTHESIS_FIELDS", "_only",
                           "audit_no_cw", "validate_plan"},
    }
    for filename, names in selections.items():
        path = source_dir / filename
        data = path.read_text()
        sources[str(path.relative_to(root))] = hashlib.sha256(data.encode()).hexdigest()
        selected = []
        for node in ast.parse(data).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                      for t in node.targets):
                selected.append(node)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    # The actual five-seed runner overrides nocw_inputs.validate_plan. Its wrapper
    # intentionally retains Free's prose-ID rules. Load only that function and bind
    # its local import to the very same extracted validate_direct function above.
    driver = root / "scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py"
    data = driver.read_text()
    sources[str(driver.relative_to(root))] = hashlib.sha256(data.encode()).hexdigest()
    wrapper = next(node for node in ast.parse(data).body
                   if isinstance(node, ast.FunctionDef) and node.name == "validate_plan")
    wrapper.name = "validate_plan_five"
    wrapper.body = [node for node in wrapper.body if not isinstance(node, ast.ImportFrom)]
    namespace["views"] = SimpleNamespace(**{key: namespace[key] for key in
        ("_only", "PLAN_FIELDS", "HYPOTHESIS_FIELDS", "audit_no_cw")})
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(driver), "exec"), namespace)
    namespace["source_sha256"] = sources
    return namespace


def baseline_validate(plan: Any, context: FrozenContext, bundle: dict) -> tuple[bool, str]:
    # Preserve actual packet binding; only fields irrelevant to validation are redacted.
    packet = {"packet_sha256": context.packet_sha256, "selection_count": context.selection_count,
              "round": context.round, "condition_columns": [], "molecules": {"m0": {}},
              "candidates": {"columns": ["id", "molecule_id", "conditions"],
                             "rows": [[i, "m0", []] for i in context.candidate_ids]}}
    catalog = SimpleNamespace(candidates=dict.fromkeys(context.candidate_ids),
                              observed=dict.fromkeys(context.observed_ids),
                              references=dict.fromkeys((*context.candidate_ids,
                                                        *context.observed_ids, *context.pending_ids)))
    try:
        if context.validator_kind == "direct":
            bundle["validate_direct"](plan, catalog, packet)
        elif context.validator_kind == "nocw_five":
            bundle["validate_plan_five"](plan, catalog, packet, context.variant)
        else:
            bundle["validate_plan"](plan, catalog, packet, context.variant)
        return True, ""
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return False, f"{type(error).__name__}: {error}"


@dataclass
class Diagnosis:
    authoritative_legal: bool
    baseline_error: str
    analyzable: bool
    findings: list[dict]
    legal_choice_indices: list[int]
    legal_candidate_indices: list[int]
    missing_count: int
    full_replanning_required: bool

    def as_dict(self):
        return copy.deepcopy(self.__dict__)


def diagnose(plan: Any, context: FrozenContext, bundle: dict) -> Diagnosis:
    """Collect all detectable hard issues; do not change original acceptance."""
    accepted, baseline_error = baseline_validate(plan, context, bundle)
    issues: list[dict] = []

    def issue(code, path, message, *, mutable=True, **details):
        issues.append({"code": code, "path": path, "message": message,
                       "mutable": mutable, **details})

    if not isinstance(plan, dict):
        issue("schema_root", "", "plan must be an object", mutable=False)
        return Diagnosis(accepted, baseline_error, False, issues, [], [],
                         context.selection_count, True)
    for key, expected in (("type", "selection"), ("packet_sha256", context.packet_sha256)):
        if plan.get(key) != expected:
            issue("binding_mismatch", f"/{key}", "frozen packet/type binding mismatch", mutable=False)
    allowed = {"type", "packet_sha256", "batch_strategy", "choices", "hypotheses",
               "feedback_interpretation", "batch_rationale"}
    if context.validator_kind != "direct":
        for key in sorted(set(plan) - allowed):
            issue("schema_unauthorized_field", f"/{key}", "unauthorized plan field", mutable=False)
    choices = plan.get("choices")
    hypotheses = plan.get("hypotheses")
    if not isinstance(choices, list):
        issue("schema_choices", "/choices", "choices must be an array", mutable=False)
        choices = []
    missing = max(context.selection_count - len(choices), 0)
    if len(choices) != context.selection_count:
        issue("selection_count", "/choices", "wrong number of experiments", mutable=missing > 0,
              expected=context.selection_count, actual=len(choices), missing=missing,
              excess=max(len(choices) - context.selection_count, 0))
    seen_hypotheses: set[str] = set()
    if not isinstance(hypotheses, list) or len(hypotheses) > 8:
        issue("schema_hypotheses", "/hypotheses", "expected zero to eight objects", mutable=False)
        hypotheses = []
    hypothesis_fields = {"id", "content", "basis", "supporting_observed_ids",
                         "contradicting_observed_ids", "confidence", "status",
                         "why_it_matters_for_model_learning", "next_discriminating_question"}
    for index, hypothesis in enumerate(hypotheses):
        prefix = f"/hypotheses/{index}"
        if not isinstance(hypothesis, dict):
            issue("schema_hypothesis", prefix, "hypothesis must be an object", mutable=False)
            continue
        for name in ("id", "content", "why_it_matters_for_model_learning", "next_discriminating_question"):
            value = hypothesis.get(name)
            if not isinstance(value, str) or not value.strip():
                issue("schema_hypothesis_field", f"{prefix}/{name}", "nonempty string required",
                      mutable=name != "id")
        identity = hypothesis.get("id")
        if isinstance(identity, str):
            if identity in seen_hypotheses:
                issue("duplicate_hypothesis_id", f"{prefix}/id", "ambiguous hypothesis identity", mutable=False)
            seen_hypotheses.add(identity)
        for name, values in (("basis", ("chemical_prior", "observed_data", "model_behavior", "mixed")),
                             ("confidence", ("low", "medium", "high")),
                             ("status", ("proposed", "supported", "weakened", "rejected", "unresolved"))):
            if hypothesis.get(name) not in values:
                issue("schema_hypothesis_enum", f"{prefix}/{name}", "invalid enum value")
        for name in ("supporting_observed_ids", "contradicting_observed_ids"):
            evidence = hypothesis.get(name)
            path = f"{prefix}/{name}"
            if not isinstance(evidence, list) or any(not isinstance(i, str) for i in evidence):
                issue("schema_evidence", path, "evidence must be a string ID array")
            else:
                invalid = [i for i in evidence if i not in context.observed_ids]
                if invalid:
                    issue("unobserved_evidence", path, "only supplied measured observations may be cited",
                          invalid_ids=invalid,
                          retained_valid_ids=[i for i in evidence if i in context.observed_ids])
        if context.validator_kind != "direct":
            for name in sorted(set(hypothesis) - hypothesis_fields):
                issue("schema_unauthorized_field", f"{prefix}/{name}", "unauthorized hypothesis field", mutable=False)
    seen_ids: dict[str, int] = {}
    legal_candidates = []
    legal_choices = []
    supplied = set(context.candidate_ids)
    for index, choice in enumerate(choices):
        prefix = f"/choices/{index}"
        first_issue = len(issues)
        if not isinstance(choice, dict):
            issue("schema_choice", prefix, "choice must be an object", mutable=False)
            continue
        identity = choice.get("id")
        valid_identity = isinstance(identity, str)
        if not valid_identity:
            issue("schema_choice_id", f"{prefix}/id", "candidate ID must be a string")
        else:
            if identity in seen_ids:
                issue("duplicate_candidate_id", f"{prefix}/id", "duplicate within batch", id=identity,
                      first_index=seen_ids[identity], duplicate_index=index)
                valid_identity = False
            else:
                seen_ids[identity] = index
            if identity not in supplied:
                issue("out_of_pool_id", f"{prefix}/id", "ID absent from current supplied candidate pool", id=identity)
                valid_identity = False
            if identity in context.labeled_ids:
                issue("already_measured_id", f"{prefix}/id", "ID belongs to explicit L_t", id=identity)
                valid_identity = False
        if valid_identity:
            legal_candidates.append(index)
        if not isinstance(choice.get("reason"), str) or not choice["reason"].strip():
            issue("schema_choice_reason", f"{prefix}/reason", "nonempty scientific reason required")
        reference = choice.get("hypothesis_id")
        if "hypothesis_id" not in choice or (reference is not None and
                (not isinstance(reference, str) or reference not in seen_hypotheses)):
            issue("invalid_hypothesis_reference", f"{prefix}/hypothesis_id", "unknown hypothesis reference")
        if context.validator_kind != "direct":
            for name in sorted(set(choice) - {"id", "reason", "hypothesis_id"}):
                issue("schema_unauthorized_field", f"{prefix}/{name}", "unauthorized choice field", mutable=False)
        if len(issues) == first_issue:
            legal_choices.append(index)
    for name in ("batch_strategy", "batch_rationale", "feedback_interpretation"):
        value = plan.get(name)
        if not isinstance(value, str) or ((name != "feedback_interpretation" or context.round > 0)
                                          and not value.strip()):
            issue("schema_plan_field", f"/{name}", "required scientific prose is missing")
    if len(str(plan)) > 30000:
        issue("size_bound", "", "plan exceeds frozen memory bound", mutable=False)
    if context.validator_kind == "nocw_paired":
        prose = []
        for name in ("batch_strategy", "batch_rationale", "feedback_interpretation"):
            if isinstance(plan.get(name), str):
                prose.append((f"/{name}", plan[name]))
        prose += [(f"/choices/{i}/reason", x["reason"]) for i, x in enumerate(choices)
                  if isinstance(x, dict) and isinstance(x.get("reason"), str)]
        for i, hypothesis in enumerate(hypotheses):
            if isinstance(hypothesis, dict):
                prose += [(f"/hypotheses/{i}/{key}", hypothesis[key]) for key in
                          ("content", "why_it_matters_for_model_learning", "next_discriminating_question")
                          if isinstance(hypothesis.get(key), str)]
        # Known references only; opaque unknown prose IDs cannot safely be inferred.
        references = (*context.candidate_ids, *context.observed_ids, *context.pending_ids)
        for path, text in prose:
            cited = [i for i in references if re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)]
            if cited:
                issue("untyped_evidence_prose", path, "row IDs must use typed evidence lists",
                      cited_ids=cited)
    if context.variant == "no_cw":
        try:
            bundle["audit_no_cw"](plan)
        except ValueError as error:
            issue("nocw_forbidden_content", "", str(error), mutable=False)
    # Every issue should be explained, but unforeseen original-validator failures
    # stay authoritative and fail closed instead of inviting speculative repairs.
    if not accepted and not issues:
        issue("unlocalized_baseline_failure", "", baseline_error, mutable=False)
    full = any(not x["mutable"] for x in issues)
    structure_reliable = isinstance(plan.get("choices"), list) and all(
        isinstance(choice, dict) and isinstance(choice.get("id"), str)
        and isinstance(choice.get("reason"), str) and "hypothesis_id" in choice
        and (choice["hypothesis_id"] is None or isinstance(choice["hypothesis_id"], str))
        for choice in plan["choices"])
    full = full or not structure_reliable
    return Diagnosis(accepted, baseline_error, structure_reliable, issues, legal_choices, legal_candidates, missing, full)


def build_repair_request(plan: dict, context: FrozenContext, bundle: dict, *, intent_is_clear: bool) -> dict:
    """Prepare an unsent localized-patch protocol, never a replacement selector.

    The caller must use parse_complete_json first and explicitly assess semantic intent.
    Prefer prepare_repair_from_response at the raw-response boundary. The program cannot verify
    that a new scientific explanation means the same thing as the old explanation.
    """
    diagnosis = diagnose(plan, context, bundle)
    mode = ("already_verified" if diagnosis.authoritative_legal else
            "full_replanning_required" if not intent_is_clear or diagnosis.full_replanning_required else
            "localized_patch_request")
    mutable = sorted({x["path"] for x in diagnosis.findings
                      if x["mutable"] and x["path"] != "/choices"})
    # A reason attached to an illegal identity is not a verified explanation of a
    # different experiment. Ask the planner to explicitly replace that whole slot.
    identity_codes = {"duplicate_candidate_id", "out_of_pool_id", "already_measured_id", "schema_choice_id"}
    expanded_scopes = {x["path"].rsplit("/", 1)[0] for x in diagnosis.findings
                       if x["code"] in identity_codes}
    # An invalid measured-evidence relation can change content/confidence/status.
    # Allow explicit revision of the affected hypothesis rather than mechanically
    # replacing citation strings under a fixed scientific claim.
    expanded_scopes.update(x["path"].rsplit("/", 1)[0] for x in diagnosis.findings
                           if x["code"] in {"unobserved_evidence", "schema_evidence"})
    mutable = sorted({path for path in mutable
                      if not any(path.startswith(scope+"/") for scope in expanded_scopes)} | expanded_scopes)
    return {"protocol": "localized_patch_v1_mock_only", "mode": mode,
            "base_proposal_sha256": digest(plan), "context_sha256": context.binding_sha256,
            "packet_sha256": context.packet_sha256, "diagnosis": diagnosis.as_dict(),
            "locked_proposal": copy.deepcopy(plan), "replace_paths": mutable,
            "append_choice_count": diagnosis.missing_count,
            "legal_candidate_indices_to_preserve": diagnosis.legal_candidate_indices,
            "fully_legal_choice_indices_to_preserve": diagnosis.legal_choice_indices,
            "intent_is_clear": intent_is_clear,
            "instructions": "Return explicit replace operations only at listed paths and append exactly the missing count. "
            "For each illegal identity replace the entire choice, explicitly supplying its new scientific reason/reference. "
            "For invalid evidence revise the whole affected hypothesis, preserve its ID and all valid observed citations "
            "across the union of support/contradict lists; roles/content/status may explicitly change. "
            "Preserve every other field. Never guess IDs or measured evidence. "
            "If the original scientific intent is unclear, return full_replanning_required. "
            "Identity repair does not by itself verify scientific intent."}


def prepare_repair_from_response(text: str, context: FrozenContext, bundle: dict, *, intent_is_clear: bool) -> dict:
    """Safe response boundary: ambiguous/truncated JSON never becomes a patch plan."""
    try:
        formatted, operation = format_only_repair(text)
        plan = parse_complete_json(formatted)
    except Unanalyzable as error:
        return {"protocol": "localized_patch_v1_mock_only", "mode": "full_replanning_required",
                "reason": f"no unique complete proposal: {error}",
                "source_response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "context_sha256": context.binding_sha256, "unambiguous_proposal": False}
    request = build_repair_request(plan, context, bundle, intent_is_clear=intent_is_clear)
    # Provenance is returned separately so apply_mock_patch receives the exact host
    # contract, which it independently reconstructs and checks for tampering.
    return {"unambiguous_proposal": True, "format_operation": operation,
            "source_response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "plan": plan, "request": request}


def apply_mock_patch(plan: dict, context: FrozenContext, request: dict, patch: dict, bundle: dict) -> dict:
    """Apply an explicitly supplied mock patch, checking freeze, locality and validity.

    No provider/network code exists here. Passing these structural checks does not
    establish preservation of scientific intent; a future planner must assess that.
    """
    if request["mode"] != "localized_patch_request":
        raise ValueError("localized repair not authorized by protocol; full replan required")
    expected_request = build_repair_request(plan, context, bundle,
                                            intent_is_clear=request.get("intent_is_clear") is True)
    if request != expected_request:
        raise ValueError("repair request differs from the host-generated locked contract")
    if (request["base_proposal_sha256"] != digest(plan) or
            patch.get("base_proposal_sha256") != digest(plan) or
            patch.get("context_sha256") != context.binding_sha256 or
            request["context_sha256"] != context.binding_sha256):
        raise ValueError("stale patch or frozen-state binding mismatch")
    if patch.get("mode") == "full_replanning_required":
        raise ValueError("planner declared ambiguous intent; full replan required")
    if not isinstance(patch.get("intent_preservation_statement"), str) or not patch["intent_preservation_statement"].strip():
        raise ValueError("explicit planner intent statement is required, not proof of semantic equivalence")
    result = copy.deepcopy(plan)
    used = set()
    operations = patch.get("operations")
    if not isinstance(operations, list):
        raise ValueError("patch operations must be an array")
    for operation in operations:
        if (not isinstance(operation, dict) or set(operation) != {"op", "path", "value"}
                or operation["op"] != "replace" or operation["path"] not in request["replace_paths"]
                or operation["path"] in used):
            raise ValueError("patch modifies locked data or repeats an operation")
        path = operation["path"]
        used.add(path)
        components = path.lstrip("/").split("/")
        parent = result
        for component in components[:-1]:
            parent = parent[int(component)] if isinstance(parent, list) else parent[component]
        key = components[-1]
        old = parent.get(key) if isinstance(parent, dict) else parent[int(key)]
        value = copy.deepcopy(operation["value"])
        if len(components) == 2 and components[0] == "hypotheses":
            if not isinstance(value, dict) or value.get("id") != old.get("id"):
                raise ValueError("affected hypothesis must retain its unambiguous identity")
            names = ("supporting_observed_ids", "contradicting_observed_ids")
            retained = Counter(i for name in names for i in
                               (old[name] if isinstance(old.get(name), list) else [])
                               if isinstance(i, str)
                               and i in context.observed_ids)
            if any(not isinstance(value.get(name), list) for name in names):
                raise ValueError("revised hypothesis requires typed evidence lists")
            revised = Counter(i for name in names for i in value[name] if isinstance(i, str))
            if any(revised[i] < n for i, n in retained.items()):
                raise ValueError("valid measured evidence cannot silently be dropped; roles may explicitly change")
        if key in ("supporting_observed_ids", "contradicting_observed_ids") and isinstance(old, list):
            retained = Counter(i for i in old if isinstance(i, str) and i in context.observed_ids)
            if not isinstance(value, list) or any(value.count(i) < n for i, n in retained.items()):
                raise ValueError("valid measured evidence cannot silently be dropped")
        if isinstance(parent, list):
            parent[int(key)] = value
        else:
            parent[key] = value
    appended = patch.get("append_choices", [])
    if not isinstance(appended, list) or len(appended) != request["append_choice_count"]:
        raise ValueError("patch must append exactly the missing number of choices")
    result["choices"].extend(copy.deepcopy(appended))
    for i in request["legal_candidate_indices_to_preserve"]:
        if result["choices"][i]["id"] != plan["choices"][i]["id"]:
            raise ValueError("legal candidate identity changed")
    for i in request["fully_legal_choice_indices_to_preserve"]:
        if result["choices"][i] != plan["choices"][i]:
            raise ValueError("fully legal choice and scientific reason changed")
    diagnosis = diagnose(result, context, bundle)
    if not diagnosis.authoritative_legal:
        raise ValueError(f"patched plan still invalid: {diagnosis.baseline_error}")
    return result


class StableCandidateIndex:
    """A one-based full-pool index bound to the exact packet and unchanged row order."""

    def __init__(self, packet: dict, context: FrozenContext):
        ordered = table_ids(packet, "candidates")
        if ordered != context.candidate_ids or packet["packet_sha256"] != context.packet_sha256:
            raise ValueError("candidate index differs from frozen packet/order")
        if "local_index" in packet["candidates"]["columns"]:
            raise ValueError("index column already present")
        self._packet = copy.deepcopy(packet)
        self._ids = ordered
        self._reverse = {identity: i + 1 for i, identity in enumerate(ordered)}
        self.context_sha256 = context.binding_sha256
        self.binding_sha256 = digest({"version": "full_pool_local_index_v1", "packet": packet,
                                      "context": context.binding_sha256, "ordered_ids": ordered})

    def to_id(self, index: int, *, binding_sha256: str) -> str:
        if binding_sha256 != self.binding_sha256 or type(index) is not int or not 1 <= index <= len(self._ids):
            raise ValueError("stale binding, noninteger or out-of-range local index")
        return self._ids[index - 1]

    def to_index(self, identity: str, *, binding_sha256: str) -> int:
        if binding_sha256 != self.binding_sha256 or identity not in self._reverse:
            raise ValueError("stale binding or unknown candidate ID")
        return self._reverse[identity]

    def presentation(self) -> dict:
        packet = copy.deepcopy(self._packet)
        packet["candidates"]["columns"].append("local_index")
        for index, row in enumerate(packet["candidates"]["rows"], 1):
            row.append(index)
        # Adding an index creates a new presentation; never reuse the source's hash
        # as if it authenticated the modified packet bytes.
        packet["source_packet_sha256"] = packet.pop("packet_sha256")
        packet["indexed_packet_sha256"] = digest(packet)
        return {"source_packet_sha256": self._packet["packet_sha256"],
                "index_binding_sha256": self.binding_sha256, "packet": packet,
                "candidate_count": len(self._ids), "index_base": 1,
                "ordering_policy": "unchanged complete source candidate-table order"}

    def decode_proposal(self, plan: dict, *, context: FrozenContext, bundle: dict) -> dict:
        if (context.binding_sha256 != self.context_sha256
                or plan.get("index_binding_sha256") != self.binding_sha256):
            raise ValueError("proposal index is stale or from another frozen state")
        result = copy.deepcopy(plan)
        result.pop("index_binding_sha256")
        if not isinstance(result.get("choices"), list):
            raise ValueError("choices must be an array")
        for choice in result["choices"]:
            if not isinstance(choice, dict) or "local_index" not in choice or "id" in choice:
                raise ValueError("one unambiguous local_index per choice required")
            choice["id"] = self.to_id(choice.pop("local_index"), binding_sha256=self.binding_sha256)
        diagnosis = diagnose(result, context, bundle)
        if not diagnosis.authoritative_legal:
            raise ValueError(f"decoded plan fails original validator: {diagnosis.baseline_error}")
        return result
