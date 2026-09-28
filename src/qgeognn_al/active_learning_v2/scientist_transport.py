"""Stateless Responses or isolated Codex CLI calls for the V2 JSON relay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import tomllib

from .dialog_bridge import once, read
from .protocol import stable_hash
from .scientist_selector import SYSTEM_PROMPT, MODEL_CALL_BUDGET, validate_selection


def settings(backend="codex_cli", model="gpt-5.5", base_url="https://token4research.cn"):
    if backend not in ("codex_cli", "responses"):
        raise ValueError("unknown LLM backend")
    if not base_url.startswith("https://") or "@" in base_url or "?" in base_url:
        raise ValueError("provider URL must be HTTPS without embedded credentials")
    return {"backend": backend, "model": model, "base_url": base_url.rstrip("/"),
            "reasoning_effort": "high",
            "authentication": "codex_home" if backend == "codex_cli" else "environment_key",
            "key_env": None if backend == "codex_cli" else "SCIENTIST_API_KEY",
            "automatic_fallback": False, "responses_store": False,
            "cli_response_storage": "controlled by installed CLI/provider; not asserted by an undocumented flag",
            "cli_isolation": "fresh CODEX_HOME and cwd; no inherited config/history; native-call audit",
            "hard_filesystem_read_sandbox": False}


def audit_cli(events, rollouts, model, effort):
    allowed_items = {"user_message", "UserMessage", "agent_message", "AgentMessage",
                     "reasoning", "Reasoning"}
    messages, completed, configuration_warnings = [], False, []
    for event in events:
        if event.get("type") in ("error", "turn.failed", "turn_failed"):
            raise RuntimeError("Codex CLI failed")
        event_type = event.get("type", "")
        if event_type.startswith("item.") or event_type.startswith("item_"):
            item = event.get("item", {})
            # CLI 0.155 emits ignored-setting notices as error items before a turn.
            warning = item.get("message", "")
            if item.get("type") == "error" and re.fullmatch(
                    r"Codex is ignoring \d+ unrecognized configuration settings\. Check for typos or deprecated settings\."
                    r"(?:\n  user \([^\n]+\): `(?:disable_response_storage|mcp_servers\.computer-use\.type)` is ignored\.)+",
                    warning):
                configuration_warnings.append(warning)
                continue
            if item.get("type") not in allowed_items:
                raise RuntimeError("native tool or unknown CLI item invalidates selection")
            if event_type in ("item.completed", "item_completed") and item["type"] in ("agent_message", "AgentMessage"):
                messages.append(item.get("text") if "text" in item else
                                "".join(c.get("text", "") for c in item.get("content", [])))
        completed |= event_type in ("turn.completed", "turn_completed", "task_complete")
    turns, finals = [], []
    for row in rollouts:
        payload = row.get("payload", {})
        if row.get("type") == "turn_context":
            if payload.get("model") != model or payload.get("effort") != effort:
                raise RuntimeError("CLI model/effort provenance mismatch")
            turns.append({"model": model, "effort": effort})
        if row.get("type") == "response_item":
            if payload.get("type") not in ("message", "reasoning"):
                raise RuntimeError("native call or unknown rollout item invalidates selection")
            if payload.get("role") == "assistant" and payload.get("phase") in ("final", "final_answer"):
                finals.append("".join(c.get("text", "") for c in payload.get("content", [])))
    if not completed or not turns or len(finals) != 1 or not messages or messages[-1] != finals[0]:
        raise RuntimeError("missing or inconsistent CLI final/provenance")
    item_types = [row.get("payload", {}).get("type") for row in rollouts if row.get("type") == "response_item"]
    return finals[0], {"turns": turns, "native_tool_calls": 0, "response_item_types": item_types,
                       "configuration_warnings": configuration_warnings,
                       "event_sha256": stable_hash(events), "rollout_sha256": stable_hash(rollouts)}


def recover_cli_rollout(messages, config, *, started_at, codex_home=None):
    """Recover a completed CLI answer when only the CLI event-name parser failed."""
    codex_home = Path(codex_home or (Path.home() / ".codex"))
    expected_input = json.dumps(messages, ensure_ascii=False)
    recovered = []
    for path in (codex_home / "sessions").rglob("*.jsonl"):
        if path.stat().st_mtime < started_at:
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        contexts = [i for i, row in enumerate(rows) if row.get("type") == "turn_context"]
        for position, index in enumerate(contexts):
            context = rows[index]["payload"]
            if context.get("model") != config["model"] or context.get("effort") != config["reasoning_effort"]:
                continue
            end = contexts[position+1] if position+1 < len(contexts) else len(rows)
            turn = rows[index:end]
            matching_inputs = ["".join(part.get("text", "") for part in row["payload"].get("content", [])
                                      if isinstance(part, dict)) for row in turn
                               if row.get("type") == "response_item" and row.get("payload", {}).get("role") == "user"]
            if matching_inputs != [expected_input]:
                continue
            inputs = []
            finals = []
            response_types = []
            final_ids = []
            agent_event_ids = []
            task_complete = False
            for row in turn:
                payload = row.get("payload", {})
                if row.get("type") == "response_item":
                    kind = payload.get("type")
                    response_types.append(kind)
                    if kind not in ("message", "reasoning"):
                        raise RuntimeError("native tool call in completed CLI rollout")
                    if payload.get("role") == "user":
                        inputs.append("".join(part.get("text", "") for part in payload.get("content", [])
                                               if isinstance(part, dict)))
                    if payload.get("role") == "assistant" and payload.get("phase") in ("final", "final_answer"):
                        finals.append("".join(part.get("text", "") for part in payload.get("content", [])
                                               if isinstance(part, dict)))
                elif row.get("type") == "event_msg":
                    event = row.get("payload", {})
                    if event.get("type") in ("error", "turn_failed"):
                        raise RuntimeError("failed Codex CLI rollout cannot be recovered")
                    task_complete |= event.get("type") == "task_complete"
                    item = event.get("item", {})
                    item_type = item.get("type")
                    if event.get("type") in ("item_started", "item_completed") and item_type not in (
                            "UserMessage", "Reasoning", "AgentMessage"):
                        raise RuntimeError("native or unknown CLI item in completed rollout")
                    if event.get("type") == "item_completed" and item_type == "AgentMessage":
                        agent_event_ids.append(item.get("id"))
                if row.get("type") == "response_item":
                    payload = row.get("payload", {})
                    if payload.get("role") == "assistant" and payload.get("phase") in ("final", "final_answer"):
                        final_ids.append(payload.get("id"))
            if (inputs == [expected_input] and len(finals) == 1 and response_types
                    and set(response_types) <= {"message", "reasoning"} and task_complete
                    and len(agent_event_ids) == 1 and agent_event_ids == final_ids):
                try:
                    value = json.loads(finals[0])
                except json.JSONDecodeError as error:
                    raise RuntimeError("recovered CLI answer is not JSON") from error
                if value.get("type") not in ("query", "selection"):
                    raise RuntimeError("recovered CLI answer has an unknown type")
                recovered.append((finals[0], {"audit": {"model": config["model"],
                    "effort": config["reasoning_effort"], "native_tool_calls": 0,
                    "response_item_types": response_types,
                    "rollout_sha256": stable_hash(rows)}, "recovered_from_rollout": str(path),
                    "recovered_after_cli_event_parser_failure": True}))
    if len(recovered) != 1:
        raise RuntimeError("expected exactly one matching completed CLI rollout")
    return recovered[0]


def cli_call(messages, config):
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("codex CLI is not installed")
    codex_home = Path.home() / ".codex"
    user_config = tomllib.loads((codex_home / "config.toml").read_text())
    provider_id = user_config.get("model_provider", "openai")
    provider = user_config.get("model_providers", {}).get(provider_id, {})
    if (provider.get("base_url", "https://api.openai.com/v1").rstrip("/") != config["base_url"].rstrip("/")
            or provider.get("wire_api", "responses") != "responses"):
        raise RuntimeError("Codex provider endpoint differs from the frozen study transport")
    # Use the user's configured provider/auth, but keep the working directory and
    # system prompt isolated. read-only does not deny file reads; audit every item.
    with tempfile.TemporaryDirectory(prefix="scientist-cli-") as tmp:
        root = Path(tmp)
        work = root / "work"
        work.mkdir()
        instructions = root / "instructions.txt"
        instructions.write_text(SYSTEM_PROMPT)
        overrides = {
            "model_reasoning_effort": config["reasoning_effort"],
            "approval_policy": "never", "web_search": "disabled",
            "features.shell_tool": False, "features.multi_agent": False,
            "features.goals": False, "features.memories": False,
            "mcp_servers": {},
            "model_instructions_file": str(instructions),
        }
        command = [executable, "exec", "--ignore-rules", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--json",
                   "--color", "never", "--model", config["model"], "--cd", str(work)]
        for name, value in overrides.items():
            command += ["-c", f"{name}={json.dumps(value)}"]
        command += ["-"]
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR") if k in os.environ}
        env.update({"HOME": str(Path.home()), "CODEX_HOME": str(codex_home)})
        existing_rollouts = set((codex_home / "sessions").rglob("*.jsonl"))
        start_time = time.time()
        result = subprocess.run(command, input=json.dumps(messages, ensure_ascii=False),
                                capture_output=True, text=True, timeout=600, env=env, cwd=work)
        if result.returncode:
            # Do not persist provider error bodies or subprocess stderr containing credentials.
            raise RuntimeError(f"Codex CLI exited {result.returncode}; no selection accepted")
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        files = [path for path in (codex_home / "sessions").rglob("*.jsonl")
                 if path not in existing_rollouts and path.stat().st_mtime >= start_time]
        if len(files) != 1:
            raise RuntimeError("expected one isolated CLI rollout")
        rollouts = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        answer, audit = audit_cli(events, rollouts, config["model"], config["reasoning_effort"])
        usage = [e.get("usage", {}) for e in events if e.get("type") == "turn.completed"]
        return answer, {"audit": audit, "usage": usage,
                        "cli_version": subprocess.check_output([executable, "--version"], text=True).strip()}


def responses_call(messages, config):
    from openai import OpenAI
    key = os.environ.get(config["key_env"])
    if not key:
        raise RuntimeError(f"missing {config['key_env']}")
    client = OpenAI(api_key=key, base_url=config["base_url"], max_retries=0, timeout=600)
    response = client.responses.create(model=config["model"], input=messages,
        reasoning={"effort": config["reasoning_effort"]}, tools=[], store=False)
    if response.model != config["model"] and not response.model.startswith(config["model"]+"-"):
        raise RuntimeError("provider returned a different model; no fallback is allowed")
    if any(item.type not in ("message", "reasoning") for item in response.output):
        raise RuntimeError("native tool output invalidates selector response")
    return response.output_text, {"response_id": response.id, "served_model": response.model,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "native_tool_calls": 0}


def run_selector(packet, catalog, directory, config, call=None):
    directory = Path(directory)
    call = call or (cli_call if config["backend"] == "codex_cli" else responses_call)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
    for index in range(MODEL_CALL_BUDGET):
        path = directory / f"turn_{index:02d}.json"
        request_hash = stable_hash({"messages": messages, "config": config})
        if path.exists():
            receipt = read(path)
            if receipt["request_sha256"] != request_hash:
                raise RuntimeError("selector transcript/request drift")
        else:
            started = path.with_suffix(".started.json")
            if started.exists():
                if config["backend"] != "codex_cli":
                    raise RuntimeError("interrupted/failed selector call; explicit protocol revision required, no silent retry")
                marker = read(started)
                if marker["request_sha256"] != request_hash:
                    raise RuntimeError("started CLI request does not match current packet")
                answer, provenance = recover_cli_rollout(messages, config, started_at=started.stat().st_mtime)
            else:
                once(started, {"request_sha256": request_hash})
                answer, provenance = call(messages, config)
            receipt = {"request_sha256": request_hash, "answer": answer, "provenance": provenance}
            once(path, receipt)
        value = json.loads(receipt["answer"])
        if value.get("type") == "selection":
            result = validate_selection(value, catalog, packet)
            once(directory / "selection.json", result)
            return result
        queries = value.get("queries")
        if value.get("type") != "query" or not isinstance(queries, list) or not queries:
            raise ValueError("expected query or selection")
        if index == MODEL_CALL_BUDGET-1 or len(catalog.queries)+len(queries) > catalog.query_budget:
            raise ValueError("selector exceeded query/answer budget")
        results = []
        errors = []
        for query in queries:
            try:
                results.append(catalog.query(query["operation"], query["args"]))
                errors.append(None)
            except (ValueError, TypeError, KeyError) as error:
                # Feed malformed relay syntax back to the model without guessing
                # intent or counting the failed request as a catalog query.
                results.append({"error": type(error).__name__, "message": str(error)})
                errors.append(str(error))
        if any(errors):
            once(directory / f"query_error_{index:02d}.json", {
                "queries": queries, "results": results,
                "errors": errors, "remaining_answers": MODEL_CALL_BUDGET-index-1})
        else:
            once(directory / f"query_{index:02d}.json", {"queries": queries, "results": results})
        feedback = {"query_results": results, "remaining_answers": MODEL_CALL_BUDGET-index-1}
        if any(errors):
            feedback["query_errors"] = errors
        messages += [{"role": "assistant", "content": receipt["answer"]},
                     {"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}]
    raise RuntimeError("selector did not return a selection")
