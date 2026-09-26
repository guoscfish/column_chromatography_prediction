"""JSON relay for audited, independent Codex selector tasks (no API key).

Native tools are available in the product. A response is accepted only after
the full task rollout is audited to contain zero model tool calls. This is
an audit boundary, explicitly not a hard operating-system sandbox.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ..training.predictor import atomic_json
from .full_pool_selector import (SYSTEM_PROMPT, ReadOnlyCatalog, MODEL_CALL_BUDGET,
                                validate_selection)
from .protocol import stable_hash

DIALOG_PROMPT = SYSTEM_PROMPT.replace(
    "No code, file, network, or external tool use is available.",
    "Do not invoke ANY native tool, including task-reading, shell, file, web or code tools. "
    "Every native tool call invalidates the task. Only use the JSON relay below.") + """
This is a separate selector task for one trajectory. Use only the supplied messages.
Do not look up the source task. Do not provide commentary. Every answer must be one
JSON object, without Markdown fences. To retrieve data, return:
{"type":"query","queries":[{"operation":"search_observed","args":{...}}, ...]}.
The program executes these read-only queries and returns exact results. Operations:
get_candidates / get_observed: args ids (list of IDs), limit (maximum 24).
search_candidates / search_observed: optional args smiles (substring),
same_molecule_as (candidate or observed ID), similar_to (ID), min_tanimoto,
conditions (exact field-value object), descriptor (MW_g_mol, LogP, TPSA_A2, HBD, HBA),
min_value, max_value, limit (maximum 24). Empty filters search all records.
No matching record means empty results, not permission to invent an experiment.
At most 12 queries, 240 distinct candidate views including 20 initial cards,
and 16 assistant answers per round. Batch queries when useful and retain room
for a supplementary retrieval. The final answer has type:"selection" plus
packet_sha256, choices, hypotheses, feedback_interpretation as specified above.
Give precise evidence IDs where available, distinguish chemical prior from observation,
and allow modest reasons when a detailed mechanism is not supported.
"""


def read(path):
    return json.loads(Path(path).read_text())


def once(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"immutable dialog artifact changed: {path.name}")
    else:
        atomic_json(path, value)


def export_catalog(directory, catalog):
    once(directory / "catalog.json", {
        "candidates": list(catalog.candidates.values()), "observed": list(catalog.observed.values()),
        "pending": list(catalog.pending.values()), "reference_ids": catalog.reference_ids})
    packet = read(directory / "input.json")
    prompt = DIALOG_PROMPT + "\nCURRENT ROUND PACKET:\n" + json.dumps(packet, ensure_ascii=False)
    path = directory / "dialog_prompt.txt"
    if path.exists() and path.read_text() != prompt:
        raise RuntimeError("dialog prompt changed")
    path.write_text(prompt)


def load_catalog(directory):
    catalog = ReadOnlyCatalog(**read(directory / "catalog.json"))
    catalog.initial_cards()
    return catalog


def rollout_path(thread_id):
    if not re.fullmatch(r"[0-9a-f-]{36}", thread_id):
        raise ValueError("invalid task ID")
    paths = list((Path.home() / ".codex/sessions").rglob(f"*{thread_id}.jsonl"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one local task rollout")
    return paths[0]


def audit_bytes(raw):
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    turns, finals, usage, items = [], [], {}, []
    for row in rows:
        payload = row.get("payload", {})
        if row["type"] == "turn_context":
            if payload.get("model") != "gpt-6-sol" or payload.get("effort") != "high":
                raise RuntimeError("selector model or reasoning setting changed")
            turns.append({"model": payload["model"], "effort": payload["effort"]})
        if row["type"] == "response_item":
            kind = payload.get("type", "")
            if "call" in kind and not kind.endswith("_output"):
                raise RuntimeError("native tool call invalidates selector task")
            items.append({"type": kind, "sha256": stable_hash(payload)})
            if kind == "message" and payload.get("role") == "assistant" and payload.get("phase") in ("final", "final_answer"):
                text = "".join(part.get("text", "") for part in payload.get("content", []))
                finals.append({"id": payload.get("id"), "text": text})
        if row["type"] == "event_msg" and payload.get("type") == "token_count":
            usage = payload.get("info", {}).get("total_token_usage", {})
    if not turns:
        raise RuntimeError("missing model provenance")
    return {"turns": turns, "finals": finals, "usage": usage, "item_digests": items,
            "native_tool_calls": 0, "prefix_bytes": len(raw), "prefix_sha256": hashlib.sha256(raw).hexdigest()}


def bind(directory, thread_id, first=False):
    directory = Path(directory)
    packet = read(directory / "input.json")
    for existing in directory.parents[2].glob("seed_*/*/round_*/dialog_binding.json"):
        previous = read(existing)
        previous_packet = read(existing.parent / "input.json")
        same_trajectory = (previous_packet["seed"], previous_packet["method"]) == (packet["seed"], packet["method"])
        if (previous["thread_id"] == thread_id) != same_trajectory:
            raise RuntimeError("task must be exclusive to one trajectory and remain fixed")
    if first != (packet["round"] == 0):
        raise ValueError("first binding must match round zero")
    audit = audit_bytes(rollout_path(thread_id).read_bytes())
    binding = {"thread_id": thread_id, "packet_sha256": read(directory / "input.json")["packet_sha256"],
               "start_final_count": 0 if first else len(audit["finals"]),
               "usage_before": {} if first else audit["usage"]}
    once(directory / "dialog_binding.json", binding)
    return binding


def replay(directory, catalog):
    for path in sorted(directory.glob("relay_*.json")):
        receipt = read(path)
        for query, expected in zip(receipt["queries"], receipt["results"]):
            actual = catalog.query(query["operation"], query["args"])
            if actual != expected:
                raise RuntimeError("query replay changed")


def process(directory):
    directory = Path(directory)
    if (directory / "dialog_acceptance.json").exists():
        catalog = load_catalog(directory)
        accepted_selection(directory, catalog, read(directory / "input.json"))
        return {"status": "ALREADY_ACCEPTED"}
    binding = read(directory / "dialog_binding.json")
    audit = audit_bytes(rollout_path(binding["thread_id"]).read_bytes())
    finals = audit["finals"][binding["start_final_count"]:]
    if not finals:
        return {"status": "WAITING_FOR_TASK"}
    if len(finals) > MODEL_CALL_BUDGET:
        raise RuntimeError("round answer budget exceeded")
    current = finals[-1]
    value = json.loads(current["text"])
    catalog = load_catalog(directory)
    replay(directory, catalog)
    target = directory / f"relay_{len(finals):02d}.json"
    if target.exists():
        return {"status": "QUERY_RESULT", "message_path": str(target.with_suffix('.txt'))}
    if value.get("type") == "query":
        queries = value.get("queries")
        if not isinstance(queries, list) or not queries or len(catalog.queries) + len(queries) > catalog.query_budget:
            raise ValueError("query count exceeds round budget")
        results = [catalog.query(item["operation"], item["args"]) for item in queries]
        once(target, {"assistant_message_id": current["id"], "queries": queries, "results": results})
        message = {"query_results": results, "remaining_queries": catalog.query_budget-len(catalog.queries),
                   "remaining_candidate_views": catalog.view_budget-len(catalog.viewed),
                   "remaining_assistant_answers": MODEL_CALL_BUDGET-len(finals),
                   "instruction": "Continue with JSON query or final selection only; no native tools."}
        target.with_suffix('.txt').write_text(json.dumps(message, ensure_ascii=False))
        return {"status": "QUERY_RESULT", "message_path": str(target.with_suffix('.txt'))}
    if value.get("type") != "selection":
        raise ValueError("expected query or selection JSON")
    packet = read(directory / "input.json")
    result = validate_selection(value, catalog, packet_sha256=packet["packet_sha256"], require_feedback=packet["round"]>0)
    # Do not archive developer/system instructions or private reasoning text.
    once(directory / "dialog_audit.json", audit)
    acceptance = {"thread_id": binding["thread_id"], "packet_sha256": packet["packet_sha256"],
                  "result": result, "viewed_candidate_ids": sorted(catalog.viewed),
                  "queries": catalog.queries, "assistant_answers_this_round": len(finals),
                  "audit_hash": stable_hash(audit), "usage_cumulative": audit["usage"],
                  "model": "gpt-6-sol", "effort": "high", "native_tool_calls": 0}
    once(directory / "dialog_acceptance.json", acceptance)
    return {"status": "SELECTION_ACCEPTED", "seed": packet["seed"], "round": packet["round"],
            "queries": len(catalog.queries), "viewed": len(catalog.viewed),
            "outside_128": sum(v["id"] not in catalog.reference_ids for v in result["choices"])}


def accepted_selection(directory, catalog, packet):
    path = directory / "dialog_acceptance.json"
    if not path.exists():
        return None
    accepted, audit = read(path), read(directory / "dialog_audit.json")
    if accepted["packet_sha256"] != packet["packet_sha256"] or accepted["audit_hash"] != stable_hash(audit):
        raise RuntimeError("dialog evidence binding changed")
    with rollout_path(accepted["thread_id"]).open('rb') as handle:
        raw = handle.read(audit["prefix_bytes"])
    if audit_bytes(raw) != audit:
        raise RuntimeError("original selector rollout changed")
    replay(directory, catalog)
    if sorted(catalog.viewed) != accepted["viewed_candidate_ids"]:
        raise RuntimeError("candidate-view audit mismatch")
    validate_selection({**accepted["result"], "packet_sha256": packet["packet_sha256"]}, catalog,
                       packet_sha256=packet["packet_sha256"], require_feedback=packet["round"]>0)
    return accepted["result"]
