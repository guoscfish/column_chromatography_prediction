"""Restricted SSH file courier for one authorized GPFS A7 run."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, once, read

from .formal_contract import AUTH_KEYS, SSH_API

SCHEMA = "purellm_a7_ssh_bridge_v1"
ATTEMPT = re.compile(r"[a-f0-9]{32}\Z")
MAX_RESULT_BYTES = 16 * 1024 * 1024


def info(run):
    record = read(run / "formal_contract.json")
    approval = read(run / "authorization.json")
    sha = digest(record)
    if (
        record.get("api_execution") != SSH_API or record.get("run_directory") != str(run)
        or not isinstance(record.get("predecessor"), dict) or set(approval) != AUTH_KEYS
        or approval["scope"] != "pure_llm_v3_a7_formal" or approval["contract_sha256"] != sha
        or any(approval[k] is not True for k in (
            "authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training"))
        or any(not isinstance(approval[k], str) or not approval[k].strip()
               for k in ("authorized_by", "authorization_reference"))
    ):
        raise PermissionError("run is not authorized for the Mac SSH dispatcher")
    return {"schema": SCHEMA, "contract_sha256": sha, "authorization_sha256": digest(approval),
            "contract": record, "mock": False, "paused": (run / "pause.request").exists()}


def _job(run, attempt_id, authority):
    from .bridge_protocol import read_bridge_json, verify_job

    if not isinstance(attempt_id, str) or not ATTEMPT.fullmatch(attempt_id):
        raise ValueError("invalid physical attempt identity")
    job = read_bridge_json(run / "bridge/outbox" / f"{attempt_id}.json")
    verify_job(job, authorization_sha256=authority["authorization_sha256"],
               contract_sha256=authority["contract_sha256"])
    request, started = job["request"], job["started"]
    binding = request["binding"]
    scope = {"contract_sha256": authority["contract_sha256"],
             "authorization_sha256": authority["authorization_sha256"],
             "seed": binding["seed"], "mock": False}
    contract = authority["contract"]
    if (
        started["attempt_id"] != attempt_id or binding["a7_formal"] != scope
        or binding["seed"] not in contract["profile"]["confirmation_seeds"]
        or not 0 <= binding["round"] < 36 or request["configuration"] != contract["profile"]["transport"]
    ):
        raise ValueError("outbox job is outside the authorized run")
    turn = run / f"seed_{binding['seed']}/round_{binding['round']:02d}/turn_{request['logical_index']:02d}"
    expected_grant = {"scope": "paid_selection_only", "formal_run": scope, "request_id": request["request_id"]}
    if (
        request != read(turn / "request.json")
        or started != read(turn / f"attempt_{started['attempt']:02d}.started.json")
        or read(turn / "formal_request_authorization.json") != expected_grant
    ):
        raise ValueError("job differs from the coordinator's staged physical attempt")
    return job


def _terminal(run, job):
    request, started = job["request"], job["started"]
    binding = request["binding"]
    turn = run / f"seed_{binding['seed']}/round_{binding['round']:02d}/turn_{request['logical_index']:02d}"
    return (turn / f"attempt_{started['attempt']:02d}.terminal.json").exists()


def _dispatcher_claim(spool, job, attempt_id, identity):
    if not isinstance(identity, str) or not ATTEMPT.fullmatch(identity):
        raise ValueError("invalid dispatcher identity")
    return spool / "claims" / f"{attempt_id}.json", {
        "attempt_id": attempt_id, "dispatcher_id": identity, "job_sha256": job["job_sha256"],
    }


def exchange(run, action, attempt_id=None, result=None, dispatcher_id=None):
    run = Path(run).resolve(strict=True)
    authority = info(run)
    if action == "info":
        return authority
    spool = run / "bridge"
    if action == "list":
        attempts = []
        for path in sorted((spool / "outbox").glob("*.json")):
            if ATTEMPT.fullmatch(path.stem) and not (spool / "results" / path.name).exists():
                if not _terminal(run, _job(run, path.stem, authority)):
                    attempts.append(path.stem)
        return {k: v for k, v in authority.items() if k != "contract"} | {"attempt_ids": attempts}
    job = _job(run, attempt_id, authority)
    target = spool / "results" / f"{attempt_id}.json"
    if action in ("get", "claim"):
        if target.exists() or _terminal(run, job):
            raise ValueError("attempt already has a returned or terminal result")
        if action == "claim":
            if authority["paused"] or (run / f"seed_{job['request']['binding']['seed']}/pause.request").exists():
                raise PermissionError("run or seed is paused")
            from .bridge_protocol import private_directory

            private_directory(spool / "claims")
            claim_path, claim = _dispatcher_claim(spool, job, attempt_id, dispatcher_id)
            once(spool / "dispatcher.json", {"dispatcher_id": dispatcher_id,
                 "contract_sha256": authority["contract_sha256"]})
            once(claim_path, claim)
        return job
    if action == "put":
        from .bridge_protocol import private_directory, validate_result

        private_directory(target.parent)
        claim_path, claim = _dispatcher_claim(spool, job, attempt_id, dispatcher_id)
        if read(claim_path) != claim:
            raise PermissionError("result sender does not own this job's durable claim")
        validate_result(result, job)
        if _terminal(run, job) and not target.exists():
            raise ValueError("attempt reconciled before result upload; keep local receipt for audit")
        once(target, result)
        return {"attempt_id": attempt_id, "result_sha256": result["result_sha256"]}
    raise ValueError("unknown courier action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("info", "list", "get", "claim", "put"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--dispatcher-id")
    args = parser.parse_args()
    if not args.run_dir.is_absolute():
        parser.error("the remotely authorized run directory must be absolute")
    os.umask(0o077)
    result = None
    if args.action == "put":
        raw = sys.stdin.buffer.read(MAX_RESULT_BYTES + 1)
        if len(raw) > MAX_RESULT_BYTES:
            parser.error("bridge result exceeds conservative transport envelope limit")
        result = json.loads(raw)
    print(dumps(exchange(args.run_dir, args.action, args.attempt_id, result, args.dispatcher_id), sort_keys=True))


if __name__ == "__main__":
    main()
