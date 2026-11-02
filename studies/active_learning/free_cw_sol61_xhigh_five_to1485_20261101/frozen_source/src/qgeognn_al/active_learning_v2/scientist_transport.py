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

from .dialog_bridge import once, read
from .protocol import stable_hash
from .scientist_selector import SYSTEM_PROMPT, MODEL_CALL_BUDGET, validate_selection
from .scientist_full_delivery import DIRECT_PROMPT, encode_catalog, validate_direct


# A provider failure must not invalidate an entire trajectory, but an unhealthy
# provider must also not be polled forever.  Retries are for the same logical
# request and are recorded separately from accepted selector turns.
MAX_CALL_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
CLI_TIMEOUT_SECONDS = 600


def settings(backend="responses", model="gpt-6-astra", base_url="https://token4research.cn/v1"):
    if backend not in ("codex_cli", "responses"):
        raise ValueError("unknown LLM backend")
    if not base_url.startswith("https://") or "@" in base_url or "?" in base_url:
        raise ValueError("provider URL must be HTTPS without embedded credentials")
    return {"backend": backend, "model": model, "base_url": base_url.rstrip("/"),
            "reasoning_effort": "high", "candidate_delivery": "all_candidates",
            "authentication": "explicit_provider_key",
            "key_env": "SCIENTIST_API_KEY", "key_file_env": "SCIENTIST_API_KEY_FILE",
            "automatic_fallback": False, "responses_store": False,
            "cli_response_storage": "controlled by installed CLI/provider; not asserted by an undocumented flag",
            "cli_isolation": "fresh CODEX_HOME and cwd; no inherited config/history; native-call audit",
            "hard_filesystem_read_sandbox": False}


def provider_key(config):
    """Read only the experiment's explicit credential; never borrow ChatGPT tokens."""
    key = os.environ.get(config["key_env"], "").strip()
    if not key:
        filename = os.environ.get(config["key_file_env"])
        if not filename and config["base_url"] == "https://token4research.cn/v1":
            default = Path.home() / ".config/qgeognn-scientist/token4research.api-key"
            if default.is_file():
                filename = default
        if filename:
            try:
                key = Path(filename).expanduser().read_text().strip()
            except (OSError, UnicodeError):
                raise RuntimeError("cannot read SCIENTIST_API_KEY_FILE") from None
    if not key or any(c.isspace() for c in key):
        raise RuntimeError("set SCIENTIST_API_KEY or SCIENTIST_API_KEY_FILE to the frozen provider's API key")
    return key


def check_transport(config):
    """Local preflight only; does not send a request or expose credential values."""
    if config["backend"] == "codex_cli" and not shutil.which("codex"):
        raise RuntimeError("codex CLI is not installed")
    provider_key(config)
    return {"status": "LOCAL_TRANSPORT_READY", "backend": config["backend"],
            "base_url": config["base_url"], "model": config["model"],
            "credential_present": True, "live_provider_verified": False}


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
    key = provider_key(config)
    # Both provider config and session storage belong exclusively to this call.
    # Global Codex login/configuration never enters the third-party subprocess.
    with tempfile.TemporaryDirectory(prefix="scientist-cli-") as tmp:
        root = Path(tmp)
        work = root / "work"
        work.mkdir()
        isolated_home = root / "codex-home"
        isolated_home.mkdir()
        (isolated_home / "config.toml").write_text(
            'model_provider = "scientist"\n'
            '[model_providers.scientist]\n'
            'name = "Scientist study provider"\n'
            f'base_url = {json.dumps(config["base_url"])}\n'
            'wire_api = "responses"\n'
            f'env_key = {json.dumps(config["key_env"])}\n'
            'requires_openai_auth = false\n')
        instructions = root / "instructions.txt"
        instructions.write_text(next((m["content"] for m in messages if m.get("role") == "system"), SYSTEM_PROMPT))
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
        env.update({"HOME": str(root), "CODEX_HOME": str(isolated_home)})
        env[config["key_env"]] = key
        existing_rollouts = set((isolated_home / "sessions").rglob("*.jsonl"))
        start_time = time.time()
        try:
            result = subprocess.run(command, input=json.dumps(messages, ensure_ascii=False),
                                    capture_output=True, text=True, timeout=CLI_TIMEOUT_SECONDS,
                                    env=env, cwd=work)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Codex CLI timed out after {CLI_TIMEOUT_SECONDS}s") from error
        if result.returncode:
            # Do not persist provider error bodies or subprocess stderr containing credentials.
            raise RuntimeError(f"Codex CLI exited {result.returncode}; no selection accepted")
        try:
            events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise RuntimeError("Codex CLI emitted invalid JSONL") from error
        files = [path for path in (isolated_home / "sessions").rglob("*.jsonl")
                 if path not in existing_rollouts and path.stat().st_mtime >= start_time]
        if len(files) != 1:
            raise RuntimeError("expected one isolated CLI rollout")
        try:
            rollouts = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise RuntimeError("Codex CLI rollout contains invalid JSONL") from error
        answer, audit = audit_cli(events, rollouts, config["model"], config["reasoning_effort"])
        usage = [e.get("usage", {}) for e in events if e.get("type") == "turn.completed"]
        return answer, {"audit": audit, "usage": usage,
                        "cli_version": subprocess.check_output([executable, "--version"], text=True).strip()}


def responses_call(messages, config):
    from openai import OpenAI
    key = provider_key(config)
    client = OpenAI(api_key=key, base_url=config["base_url"], max_retries=0, timeout=600)
    # Consume the stream to avoid proxy timeouts while long reasoning completes.
    # Accept only the complete response, never a partial streamed selection.
    with client:
        with client.responses.stream(model=config["model"], input=messages,
                reasoning={"effort": config["reasoning_effort"]}, tools=[], store=False) as stream:
            response = stream.get_final_response()
    if response.status != "completed":
        raise RuntimeError("provider response did not complete; no selection accepted")
    if response.model != config["model"] and not response.model.startswith(config["model"]+"-"):
        raise RuntimeError("provider returned a different model; no fallback is allowed")
    if any(item.type not in ("message", "reasoning") for item in response.output):
        raise RuntimeError("native tool output invalidates selector response")
    return response.output_text, {"response_id": response.id, "served_model": response.model,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "native_tool_calls": 0}


def _record_call_failure(directory, index, attempt, request_hash, error):
    """Persist a bounded retry failure without persisting provider response bodies."""
    retry_dir = Path(directory) / "retries" / f"turn_{index:02d}"
    retry_dir.mkdir(parents=True, exist_ok=True)
    message = str(error)
    for name in ("SCIENTIST_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "<redacted>")
    message = message[:1000]
    payload = {
        "request_sha256": request_hash, "attempt": attempt,
        "error_type": type(error).__name__, "message": message,
    }
    target = retry_dir / f"attempt_{attempt:02d}.json"
    while target.exists() and read(target) != payload:
        attempt += 1
        payload["attempt"] = attempt
        target = retry_dir / f"attempt_{attempt:02d}.json"
    once(target, payload)


def _call_with_retries(messages, config, call, directory, index, request_hash):
    """Call one logical selector turn with bounded, auditable retries."""
    directory = Path(directory)
    started = directory / f"turn_{index:02d}.started.json"

    # A completed CLI process can outlive a parser error. Recovering it avoids a
    # duplicate provider request; an incomplete/invalid rollout falls through to
    # a fresh isolated attempt.
    if started.exists():
        marker = read(started)
        if marker.get("request_sha256") != request_hash:
            raise RuntimeError("interrupted request does not match current packet")
    if started.exists() and config["backend"] == "codex_cli":
        try:
            return recover_cli_rollout(messages, config, started_at=started.stat().st_mtime)
        except Exception as error:
            _record_call_failure(directory, index, 0, request_hash, error)

    retry_dir = directory / "retries" / f"turn_{index:02d}"
    existing = sorted(retry_dir.glob("attempt_*.json")) if retry_dir.exists() else []
    first_attempt = len(existing) + 1
    last_error = None
    for attempt in range(first_attempt, first_attempt + MAX_CALL_ATTEMPTS):
        if not started.exists():
            once(started, {"request_sha256": request_hash})
        try:
            answer, provenance = call(messages, config)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("selector returned an empty answer")
            return answer, provenance
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            last_error = error
            _record_call_failure(directory, index, attempt, request_hash, error)
            if attempt < first_attempt + MAX_CALL_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt - first_attempt + 1))
    raise RuntimeError(
        f"selector turn {index} failed after {MAX_CALL_ATTEMPTS} attempts; "
        f"see {retry_dir}"
    ) from last_error


def run_selector(packet, catalog, directory, config, call=None):
    if packet.get("delivery") == "all_candidates_one_request":
        return run_full_selector(packet, catalog, directory, config, call)
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
            try:
                value = json.loads(receipt["answer"])
            except json.JSONDecodeError:
                # Older runs could persist a provider's Markdown-wrapped answer
                # before parsing it. Keep that immutable evidence, but continue
                # the conversation with an explicit correction request.
                messages += [{"role": "assistant", "content": receipt["answer"]},
                             {"role": "user", "content": json.dumps(
                                 {"error": "answer must be one JSON object without Markdown"},
                                 ensure_ascii=False)}]
                continue
        else:
            answer, provenance = _call_with_retries(
                messages, config, call, directory, index, request_hash
            )
            try:
                value = json.loads(answer)
            except json.JSONDecodeError as error:
                once(directory / f"rejected_turn_{index:02d}.json", {
                    "request_sha256": request_hash, "error_type": type(error).__name__,
                    "message": "selector answer was not valid JSON",
                })
                messages += [{"role": "assistant", "content": answer},
                             {"role": "user", "content": json.dumps(
                                 {"error": "answer must be one JSON object without Markdown"},
                                 ensure_ascii=False)}]
                continue
            receipt = {"request_sha256": request_hash, "answer": answer, "provenance": provenance}
            once(path, receipt)
        if value.get("type") == "selection":
            try:
                result = validate_selection(value, catalog, packet)
            except ValueError as error:
                if index == MODEL_CALL_BUDGET - 1:
                    raise
                once(directory / f"rejected_turn_{index:02d}.json", {
                    "request_sha256": request_hash, "error_type": type(error).__name__,
                    "message": str(error)[:1000],
                })
                messages += [{"role": "assistant", "content": receipt["answer"]},
                             {"role": "user", "content": json.dumps(
                                 {"error": str(error), "instruction": "return a corrected JSON selection"},
                                 ensure_ascii=False)}]
                continue
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


FULL_SELECTION_ATTEMPTS = 3


def run_full_selector(packet, catalog, directory, config, call=None):
    """Send the whole catalog; bounded corrections can repair invalid plans only."""
    directory = Path(directory)
    encoded = encode_catalog(catalog)
    if any(packet.get(key) != value for key, value in encoded.items()):
        raise ValueError("full-pool packet differs from complete authorized catalog")
    if packet.get("candidate_count") != len(catalog.candidates) or packet.get("observed_count") != len(catalog.observed):
        raise ValueError("full-pool counts differ from authorized catalog")
    if packet.get("packet_sha256") != stable_hash({k: v for k, v in packet.items() if k != "packet_sha256"}):
        raise ValueError("full-pool packet hash mismatch")
    catalog.viewed.update(catalog.candidates)
    call = call or (cli_call if config["backend"] == "codex_cli" else responses_call)
    messages = [{"role": "system", "content": DIRECT_PROMPT},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}]
    for index in range(FULL_SELECTION_ATTEMPTS):
        path = directory / f"turn_{index:02d}.json"
        request_hash = stable_hash({"messages": messages, "config": config})
        if path.exists():
            receipt = read(path)
            if receipt["request_sha256"] != request_hash:
                raise RuntimeError("full-pool transcript/request drift")
        else:
            answer, provenance = _call_with_retries(messages, config, call, directory, index, request_hash)
            # Persist every answer, including malformed JSON, for exact replay.
            receipt = {"request_sha256": request_hash, "answer": answer, "provenance": provenance}
            once(path, receipt)
        try:
            result = validate_direct(json.loads(receipt["answer"]), catalog, packet)
        except (ValueError, TypeError, KeyError) as error:
            feedback = {"error": str(error)[:1000],
                        "instruction": "Return a corrected final JSON selection from the SAME complete catalog. No queries. Only observed-table IDs may be cited as measured evidence; use empty evidence lists for prediction-only hypotheses."}
            once(directory / f"rejected_turn_{index:02d}.json", feedback)
            if index + 1 == FULL_SELECTION_ATTEMPTS:
                raise RuntimeError("full-pool selection remained invalid after bounded corrections") from error
            messages += [{"role": "assistant", "content": receipt["answer"]},
                         {"role": "user", "content": json.dumps(feedback, ensure_ascii=False)}]
            continue
        once(directory / "selection.json", result)
        return result
    raise RuntimeError("full-pool selector did not return a valid selection")
