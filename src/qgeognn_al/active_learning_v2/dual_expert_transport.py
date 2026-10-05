"""Independent, auditable JSON relays for the two advisers and planner."""
from __future__ import annotations

import json
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import time
import tomllib

from .dialog_bridge import once, read
from .scientist_selector import MODEL_CALL_BUDGET as _unused
from .protocol import stable_hash
from .dual_expert_selector import (EXPERT_MODEL_CALL_BUDGET, PLANNER_MODEL_CALL_BUDGET,
                                   ROLE_PROMPTS, validate_expert_memo, validate_planner_selection)


def settings(backend="codex_cli", model="gpt-6-sol", base_url="https://token4research.cn"):
    if backend not in ("codex_cli", "responses"):
        raise ValueError("unknown backend")
    if not base_url.startswith("https://") or "@" in base_url or "?" in base_url:
        raise ValueError("provider URL must be HTTPS without embedded credentials")
    return {"backend": backend, "model": model, "base_url": base_url.rstrip("/"),
            "reasoning_effort": "high", "authentication": "codex_home" if backend == "codex_cli" else "environment_key",
            "key_env": None if backend == "codex_cli" else "SCIENTIST_API_KEY", "automatic_fallback": False,
            "responses_store": False, "cli_isolation": "fresh temporary cwd with native-call audit"}


def _responses_call(messages, config):
    from openai import OpenAI
    key = os.environ.get(config["key_env"])
    if not key:
        raise RuntimeError(f"missing {config['key_env']}")
    client = OpenAI(api_key=key, base_url=config["base_url"], max_retries=0, timeout=600)
    response = client.responses.create(model=config["model"], input=messages,
                                       reasoning={"effort": config["reasoning_effort"]}, tools=[], store=False)
    if response.model != config["model"] and not response.model.startswith(config["model"] + "-"):
        raise RuntimeError("provider returned a different model")
    if any(item.type not in ("message", "reasoning") for item in response.output):
        raise RuntimeError("native tool output invalidates response")
    return response.output_text, {"response_id": response.id, "served_model": response.model,
                                  "usage": response.usage.model_dump(mode="json") if response.usage else None,
                                  "native_tool_calls": 0}


def _cli_call(messages, config, system_prompt):
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("codex CLI is not installed")
    codex_home = Path.home() / ".codex"
    user_config = tomllib.loads((codex_home / "config.toml").read_text())
    provider_id = user_config.get("model_provider", "openai")
    provider = user_config.get("model_providers", {}).get(provider_id, {})
    if provider.get("base_url", "https://api.openai.com/v1").rstrip("/") != config["base_url"].rstrip("/"):
        raise RuntimeError("Codex provider endpoint differs from frozen transport")
    with tempfile.TemporaryDirectory(prefix="dual-expert-cli-") as tmp:
        root, work = Path(tmp), Path(tmp) / "work"
        work.mkdir()
        instructions = root / "instructions.txt"
        instructions.write_text(system_prompt)
        command = [executable, "exec", "--ignore-rules", "--skip-git-repo-check", "--sandbox", "read-only",
                   "--json", "--color", "never", "--model", config["model"], "--cd", str(work)]
        overrides = {"model_reasoning_effort": config["reasoning_effort"], "approval_policy": "never",
                     "web_search": "disabled", "features.shell_tool": False, "features.multi_agent": False,
                     "features.goals": False, "features.memories": False, "mcp_servers": {},
                     "model_instructions_file": str(instructions)}
        for name, val in overrides.items():
            command += ["-c", f"{name}={json.dumps(val)}"]
        command += ["-"]
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR") if k in os.environ}
        env.update({"HOME": str(Path.home()), "CODEX_HOME": str(codex_home)})
        started = time.time()
        result = subprocess.run(command, input=json.dumps(messages, ensure_ascii=False), capture_output=True,
                                text=True, timeout=600, env=env, cwd=work)
        if result.returncode:
            raise RuntimeError(f"Codex CLI exited {result.returncode}")
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        errors = [e for e in events if e.get("type") in ("error", "turn.failed", "turn_failed")]
        if errors:
            raise RuntimeError("Codex CLI failed")
        answers = []
        for event in events:
            item = event.get("item", {})
            if event.get("type") in ("item.completed", "item_completed") and item.get("type") in ("agent_message", "AgentMessage"):
                answers.append(item.get("text", "") or "".join(part.get("text", "") for part in item.get("content", [])))
            if event.get("type", "").startswith("item.") or event.get("type", "").startswith("item_"):
                if item.get("type") not in ("agent_message", "AgentMessage", "reasoning", "Reasoning", "error"):
                    raise RuntimeError("native tool output invalidates CLI call")
        if len(answers) != 1:
            raise RuntimeError("missing CLI final answer")
        return answers[0], {"cli_version": subprocess.check_output([executable, "--version"], text=True).strip(),
                            "model": config["model"], "effort": config["reasoning_effort"],
                            "native_tool_calls": 0, "event_sha256": stable_hash(events),
                            "started_at": started}


def run_agent(role, packet, catalog, directory, config, call=None):
    if role not in ROLE_PROMPTS:
        raise ValueError("unknown role")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    call = call or (lambda messages, cfg: _cli_call(messages, cfg, ROLE_PROMPTS[role])) if config["backend"] == "codex_cli" else call or _responses_call
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
