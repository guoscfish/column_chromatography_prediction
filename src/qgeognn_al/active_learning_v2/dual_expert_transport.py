"""Independent role transcripts using the same transport as the free/CW study."""
from __future__ import annotations

import json
from pathlib import Path

from .dialog_bridge import once, read
from .protocol import stable_hash
from .scientist_full_delivery import encode_catalog
from .scientist_transport import (settings as scientist_settings, cli_call, responses_call,
                                  _call_with_retries, FULL_SELECTION_ATTEMPTS)
from .dual_expert_selector import (EXPERT_MODEL_CALL_BUDGET, PLANNER_MODEL_CALL_BUDGET,
    ROLE_PROMPTS, FULL_ROLE_PROMPTS, validate_expert_memo, validate_planner_selection)


def settings(backend="responses", model="gpt-6.1-sol", base_url="https://token4research.cn/v1",
             reasoning_effort="xhigh"):
    if reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("unsupported reasoning effort")
    config = scientist_settings(backend, model, base_url)
    config["reasoning_effort"] = reasoning_effort
    return config


def run_agent(role, packet, catalog, directory, config, call=None, extra_format_attempts=0):
    if packet.get("delivery") != "all_candidates_one_request":
        return _run_relay_agent(role, packet, catalog, directory, config, call)
    if role not in FULL_ROLE_PROMPTS or packet.get("role") != role:
        raise ValueError("unknown or mismatched role")
    encoded = encode_catalog(catalog)
    if any(packet.get(key) != value for key, value in encoded.items()):
        raise ValueError("role packet differs from complete authorized catalog")
    if packet.get("candidate_count") != len(catalog.candidates) or packet.get("observed_count") != len(catalog.observed):
        raise ValueError("role catalog counts mismatch")
    if packet.get("packet_sha256") != stable_hash({k: v for k, v in packet.items() if k != "packet_sha256"}):
        raise ValueError("role packet hash mismatch")
    catalog.viewed.update(catalog.candidates)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    call = call or (cli_call if config["backend"] == "codex_cli" else responses_call)
    messages = [{"role": "system", "content": FULL_ROLE_PROMPTS[role]},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}]
    if extra_format_attempts not in (0, 1):
        raise ValueError("at most one explicitly registered extra format attempt")
    attempt_budget = FULL_SELECTION_ATTEMPTS + extra_format_attempts
    contract = ("MEMO FORMAT: state_summary must be a string. Use only type, role, packet_sha256, "
                "state_summary, important_observations, candidate_scientific_questions, possible_model_blind_spots, "
                "important_model_behaviors, possible_learning_bottlenecks, coverage_or_redundancy_concerns, "
                "warnings, recommended_followup_queries, particularly_informative_candidates. "
                "Put hypothesis revisions inside these fields; no hypothesis_updates or revised_hypotheses keys. "
                "Keep the COMPLETE JSON below 9,500 characters to stay safely within the 12,000-character limit. "
                "Return the correct role and packet_sha256. No choices or selection field.")
    if role == "scientific_planner":
        contract = ("PLAN FORMAT: return type selection and the supplied packet_sha256, expert_synthesis "
                    "with chemistry_takeaways, ml_takeaways, agreements, disagreements, planner_resolution; "
                    "nonempty batch_strategy, batch_rationale, expected_learning_value and feedback_interpretation. "
                    "Exactly 32 distinct candidate-table IDs in choices, with id, reason, knowledge_basis "
                    "(chemistry/machine_learning/observed_feedback) and hypothesis_id or related_hypothesis_id. "
                    "hypotheses MUST be a LIST of 0..8 objects, each with id, content, why_it_matters_for_model_learning, "
                    "next_discriminating_question, basis (chemical_prior/observed_data/model_behavior/mixed), "
                    "confidence (low/medium/high), status (proposed/supported/weakened/rejected/unresolved), "
                    "supporting_observed_ids and contradicting_observed_ids. Choice hypothesis references must be null "
                    "or an ID in that list. Only observed-table sample IDs can be measured evidence; candidates and "
                    "molecule IDs cannot. For prediction-only hypotheses, use empty evidence lists. "
                    "Keep complete JSON below 28,000 characters.")
    for index in range(attempt_budget):
        path = directory / f"turn_{index:02d}.json"
        request_hash = stable_hash({"messages": messages, "config": config})
        if path.exists():
            receipt = read(path)
            if receipt["request_sha256"] != request_hash:
                raise RuntimeError("agent transcript/request drift")
        else:
            answer, provenance = _call_with_retries(messages, config, call, directory, index, request_hash)
            receipt = {"request_sha256": request_hash, "answer": answer, "provenance": provenance,
                       "role": role, "call_index": index}
            once(path, receipt)
        try:
            value = json.loads(receipt["answer"])
            if role == "scientific_planner":
                result = validate_planner_selection(value, catalog, packet)
            else:
                result = validate_expert_memo(value, role)
                if result.get("packet_sha256") != packet["packet_sha256"]:
                    raise ValueError("memo packet binding mismatch")
                if not isinstance(result.get("state_summary"), str) or not result["state_summary"].strip():
                    raise ValueError("memo needs state_summary")
                if len(str(result)) > 12000:
                    raise ValueError("memo exceeds size bound")
        except (ValueError, TypeError, KeyError) as error:
            rejected_path = directory / f"rejected_turn_{index:02d}.json"
            if rejected_path.exists():
                feedback = read(rejected_path)
            else:
                feedback = {"error": str(error)[:1000],
                            "instruction": "Return corrected final JSON from the SAME complete catalog. No queries. Only observed-table IDs are measured evidence."
                                + " " + contract}
                once(rejected_path, feedback)
            if index+1 == attempt_budget:
                raise RuntimeError("agent response remained invalid after bounded corrections") from error
            messages += [{"role": "assistant", "content": receipt["answer"]},
                         {"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}]
            if extra_format_attempts and index+1 == FULL_SELECTION_ATTEMPTS:
                detail = " Current validation detail: " + str(error)[:3000] if role == "scientific_planner" else ""
                messages.append({"role": "user", "content": contract + detail})
            continue
        once(directory / ("selection.json" if role == "scientific_planner" else "memo.json"), result)
        return result
    raise RuntimeError("agent did not return a valid response")


def _run_relay_agent(role, packet, catalog, directory, config, call=None):
    if role not in ROLE_PROMPTS:
        raise ValueError("unknown role")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    call = call or (cli_call if config["backend"] == "codex_cli" else responses_call)
    max_calls = PLANNER_MODEL_CALL_BUDGET if role == "scientific_planner" else EXPERT_MODEL_CALL_BUDGET
    messages = [{"role": "system", "content": ROLE_PROMPTS[role]},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
    for index in range(max_calls):
        path = directory / f"turn_{index:02d}.json"
        request_hash = stable_hash({"messages": messages, "config": config, "role": role})
        if path.exists():
            receipt = read(path)
            if receipt["request_sha256"] != request_hash:
                raise RuntimeError("agent transcript/request drift")
        else:
            started = path.with_suffix(".started.json")
            if started.exists():
                raise RuntimeError("interrupted agent call requires explicit protocol revision")
            once(started, {"request_sha256": request_hash, "role": role})
            answer, provenance = call(messages, config)
            receipt = {"request_sha256": request_hash, "answer": answer, "provenance": provenance,
                       "role": role, "call_index": index}
            once(path, receipt)
        value = json.loads(receipt["answer"])
        if value.get("type") == "query":
            queries = value.get("queries")
            if not isinstance(queries, list) or not queries:
                raise ValueError("query response must contain queries")
            if len(catalog.queries) + len(queries) > catalog.query_budget:
                raise ValueError("agent exceeded query budget")
            results = [catalog.query(q["operation"], q["args"]) for q in queries]
            messages.append({"role": "assistant", "content": receipt["answer"]})
            messages.append({"role": "user", "content": json.dumps({"query_results": results}, ensure_ascii=False)})
            continue
        if role == "scientific_planner":
            result = validate_planner_selection(value, catalog, packet)
        else:
            result = validate_expert_memo(value, role)
        once(directory / ("selection.json" if role == "scientific_planner" else "memo.json"), result)
        return result
    raise ValueError("agent exhausted model-call budget")
