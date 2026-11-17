#!/usr/bin/env python3
"""Reviewable five-seed extension of the unlaunched upstream-error recovery."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
HELPER = ROOT / "scripts/studies/resume_nocw_upstream_20261114.py"
spec = importlib.util.spec_from_file_location("upstream_recovery_five_base", HELPER)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
OLD_PLAN = r.PLAN
r.SEEDS = {157: 27, 887: 24, 2357: 28, 6101: 28, 12203: 30}
r.SCRIPT = SCRIPT
r.PLAN = r.m.STUDY / "upstream_recovery_all_five_plan_20261114.json"
r.APPROVAL = r.m.STUDY / "upstream_recovery_all_five_authorization_20261114.json"
original_require = r.require_approval


def require_approval():
    record = r.m.read(r.PLAN)
    if record.get("helper_script_sha256") != r.m.sha(HELPER):
        raise RuntimeError("Tested upstream-recovery helper changed")
    approval = r.m.read(r.APPROVAL) if r.APPROVAL.exists() else {}
    if approval.get("seeds") != list(r.SEEDS):
        raise PermissionError("Explicit recovery approval for all five seeds is pending")
    return original_require()


r.require_approval = require_approval


def prepare():
    if r.PLAN.exists():
        record = r.m.read(r.PLAN)
        if (record["recovery_script_sha256"] != r.m.sha(SCRIPT)
                or record["helper_script_sha256"] != r.m.sha(HELPER)):
            raise RuntimeError("Prepared all-five recovery adapter changed")
        return record
    if (r.m.STUDY / "upstream_recovery_launch_20261114.json").exists():
        raise RuntimeError("Earlier recovery was dispatched; reconcile before expanding its scope")
    record = r.prepare()
    record.update(scope="all five stopped NoCW workers; no CW worker restart",
                  helper_script_sha256=r.m.sha(HELPER),
                  supersedes_unapproved_two_seed_plan_sha256=r.m.sha(OLD_PLAN),
                  reviewed_seed_scope=list(r.SEEDS))
    r.m.atomic(r.PLAN, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--seed", type=int, choices=r.SEEDS)
    args = parser.parse_args()
    if args.action == "worker":
        if args.seed is None:
            parser.error("worker requires --seed")
        result = r.worker(args.seed)
    elif args.action == "prepare":
        result = prepare()
    else:
        result = r.launch()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
