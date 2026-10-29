#!/usr/bin/env python3
"""Five independent seed workers for the authorized NoCW-only experiment.

The frozen two-seed implementation is adapted in each process; its files and
protocol are not edited. No arguments performs an offline status inspection.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import torch

from src.qgeognn_al.active_learning_v2 import nocw_study as s
from src.qgeognn_al.active_learning_v2 import nocw_inputs as views
from src.qgeognn_al.active_learning_v2.benchmark_protocol import load_features
from src.qgeognn_al.active_learning_v2.gradient_transforms import center_width_transform
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, validate_row_protocol
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.training.predictor import atomic_json

SEEDS = (157, 887, 2357, 6101, 12203)
VERSION = "llm32_nocw_five_seed_historical_control_v1"
BASE_STUDY = s.STUDY
STUDY = BASE_STUDY / "revisions/five_seed_nocw_to1005_20261029"
SCRIPT = Path(__file__).resolve()
KEY_ROOT = Path.home() / ".config/qgeognn-scientist"
KEY_FILES = {157: KEY_ROOT / "seed157-continuation.api-key",
             887: KEY_ROOT / "dual-expert-813.api-key",
             2357: KEY_ROOT / "dual-expert-813-seed6101.api-key",
             6101: KEY_ROOT / "seed6101-token4research.api-key",
             12203: KEY_ROOT / "free-12203-20261023.api-key"}
BASE_INITIAL_PATHS = s.initial_paths
BASE_CODE_HASHES = s.code_hashes
BASE_EXCLUSIVE_LOCK = s.exclusive_lock
BASE_ONCE = s.once
BASE_INPUT_VIEW = views.input_view
BASE_VALIDATE_PLAN = views.validate_plan


def prompt(variant):
    if variant != "no_cw":
        raise ValueError("only NoCW is authorized")
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import DIRECT_PROMPT
    return DIRECT_PROMPT.replace(views.CW_DESCRIPTION, "").replace(
        "Pending-CW outcomes are identified separately.\n", "")


def input_view(raw, seed, arm, r, history):
    """Use the historical Free salt/encoding, with only the CW column removed."""
    _, view, packet = BASE_INPUT_VIEW(raw, seed, arm, r, history)
    view["salt"] = f"llm_scientist_v2:{seed}:free_llm32_scientist_v2:{r}"
    catalog = s.ReadOnlyCatalog(**view)
    packet.update(study_version="llm_scientist_v2", method="free_llm32_scientist_v2",
                  **s.encode_catalog(catalog))
    packet["memory"] = views.scientific_memory(history, seed, arm)
    packet.pop("packet_sha256", None)
    packet["packet_sha256"] = s.stable_hash(packet)
    views.audit_no_cw({"prompt": prompt("no_cw"), "catalog": view, "packet": packet})
    return catalog, view, packet


def validate_plan(result, catalog, packet, variant):
    # Preserve Free's evidence-ID and prose rules. No extra prose-ID restriction
    # from the unrun paired protocol is introduced into this historical comparison.
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import validate_direct
    views._only(result, views.PLAN_FIELDS, "plan")
    validate_direct(result, catalog, packet)
    for h in result["hypotheses"]:
        views._only(h, views.HYPOTHESIS_FIELDS, "hypothesis")
    for choice in result["choices"]:
        views._only(choice, {"id", "reason", "hypothesis_id"}, "choice")
    views.audit_no_cw(result)
    return result


def now():
    return datetime.now(timezone.utc).isoformat()


def initial_paths(seed):
    paths = BASE_INITIAL_PATHS(seed)
    paths["split"] = s.ROOT / f"studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv"
    if seed not in (157, 6101):
        root = s.ROOT / "studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol61_xhigh_free_five_seeds_to1005_20261023"
        d = root / f"selections/seed_{seed}/free_llm32_scientist_v2/round_00"
        paths.update(catalog=d / "catalog.json", contract=d / "contract.json")
    return paths


class Context(s.lifecycle.WorkflowContext):
    """Same lifecycle context, with the five baseline splits and seed whitelist."""
    def __init__(self, seed):
        if seed not in SEEDS:
            raise ValueError("unregistered seed")
        self.seed, self.study = seed, STUDY
        self.runtime = STUDY / f"runtime/seed_{seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        paths = initial_paths(seed)
        self.partition = pd.read_csv(paths["split"])
        validate_row_protocol(self.partition)
        self.data = load_features()
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError("source order mismatch")
        self.roles = {r: self.partition.loc[self.partition.role.eq(r), "canonical_index"].to_numpy(int)
                      for r in ("l0", "u0", "validation", "test")}
        old = s.read(paths["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(paths["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("scrubbed graph labels must be zero")
        store = RestrictedLabelStore(s.SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = None
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = s.stable_hash(s.read(STUDY / "protocol.json"))
        s.once(self.runtime / "context.json", {"seed": seed, "protocol_hash": self.protocol_hash,
            "split_hash": s.stable_hash(self.partition.to_dict("list")), "preprocessing": self.preprocessing,
            "normalization": asdict(self.normalization), "initial_label_access": store.audit})


def code_hashes():
    return {**BASE_CODE_HASHES(), str(SCRIPT.relative_to(s.ROOT)): s.sha256_file(SCRIPT),
            "tests/active_learning_v2/test_nocw_five_seed.py":
                s.sha256_file(s.ROOT / "tests/active_learning_v2/test_nocw_five_seed.py")}


@contextmanager
def seed_operation_lock(seed):
    if seed not in SEEDS:
        raise ValueError("registered worker seed required")
    with BASE_EXCLUSIVE_LOCK(STUDY / f"locks/operation_seed_{seed}.lock"):
        yield


def once(path, value):
    path = Path(path)
    if path == STUDY / "served_model_lock.json":
        # First responses from different keys can arrive together. This is the
        # only global lock, held solely for the model-identity comparison/write.
        lock = STUDY / "locks/served_model.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                BASE_ONCE(path, value)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    else:
        BASE_ONCE(path, value)


def activate(worker_seed=None):
    s.STUDY, s.SEEDS, s.VERSION = STUDY, SEEDS, VERSION
    s.ARMS = {"llm32_nocw": "no_cw"}
    s.METHODS = ("llm32_nocw",)
    views.VERSION = VERSION
    s.initial_paths, s.Context, s.code_hashes, s.once = initial_paths, Context, code_hashes, once
    s.input_view, s.prompt, s.validate_plan = input_view, prompt, validate_plan
    s.exclusive_lock = (lambda ignored: seed_operation_lock(worker_seed)) if worker_seed else BASE_EXCLUSIVE_LOCK


def credential_audit():
    values = [KEY_FILES[seed].read_bytes().strip() for seed in SEEDS]
    if any(not value or any(chr(c).isspace() for c in value) for value in values):
        raise ValueError("nonempty single-token credentials required")
    if len(set(values)) != len(SEEDS):
        raise ValueError("each seed must use a different key")
    return {"distinct_keys": 5, "key_values_logged": False,
            "key_files": {str(seed): str(path) for seed, path in KEY_FILES.items()}}


def prepare():
    activate()
    credentials = credential_audit()
    config = s.read(BASE_STUDY / "config.json")
    config.update(version=VERSION, seeds=list(SEEDS), arms=s.ARMS,
        status="USER_AUTHORIZED_FORMAL_FIVE_SEED_NOCW_ONLY_REVISION",
        execution={"seed_workers": 5, "arms_per_worker": 1, "target_budget": 1005,
                   "key_assignment": credentials["key_files"],
                   "operation_lock": "per seed; global model-identity lock only",
                   "cw_aware_execution": False,
                   "comparison": "historical CW-aware reference; model/call-configuration confounding remains",
                   "provider_support": "verify from first authorized full-catalog selection; fail closed"})
    config["ordering"] = "historical Free: sha256([llm_scientist_v2:seed:free_llm32_scientist_v2:round, opaque ID])"
    config["transport"].update(model="gpt-6.1-sol", reasoning_effort="xhigh")
    once(STUDY / "config.json", config)
    once(STUDY / "credential_audit.json", credentials)
    return s.prepare()


def authorize():
    activate()
    record = {**s.validate(), "formal_run_authorized": True,
        "user_authorization": ["开始运行吧，每个seed用一个单独的key，五个并行跑", "CW-aware是不是都跑完了，跑NoCW就行", "这些任务统一都用6.1 sol xhigh"],
        "scope": "five seed workers; NoCW ONLY; L333 to L1005; reveal and retrain after each frozen batch; no new CW-aware calls or fits",
        "provider_support_attested_by_operator": False,
        "provider_support_verification": "first actual authorized full request; never change model or truncate input on failure",
        "transport": s.read(STUDY / "config.json")["transport"]}
    once(STUDY / "formal_authorization.json", record)
    return record


def status_record(seed, phase, **details):
    record = {"seed": seed, "phase": phase, "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(STUDY / f"progress/seed_{seed}.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def arm_order(seed, r):
    return ("llm32_nocw",)


def dry_run():
    activate()
    s.validate()
    audits = []
    config = s.read(STUDY / "config.json")["transport"]
    for seed in SEEDS:
        raw, roles, source = s.initial_state(seed)
        directory = STUDY / f"dry_run/seed_{seed}/llm32_nocw/round_00"
        catalog, packet = s.write_stage(directory, raw, roles, source, seed, "llm32_nocw", 0, [])
        # Compare the actual request to the old Free catalog, including its salt.
        reference = s.ReadOnlyCatalog(**raw)
        expected = copy.deepcopy(s.encode_catalog(reference))
        column = expected["candidates"]["columns"].index(views.CW_FIELD)
        expected["candidates"]["columns"].pop(column)
        for row in expected["candidates"]["rows"]:
            row.pop(column)
        if expected != s.encode_catalog(catalog):
            raise RuntimeError("historical Free catalog/order differs beyond CW removal")
        result = s.run_selection(packet, catalog, directory, "no_cw", config,
            lambda *args: (json.dumps(s.mock_plan(packet)), {"mock": True}))
        frozen = s.freeze_batch(directory, result, catalog, packet)
        s.audit_batch(directory)
        audits.append({"seed": seed, "candidate_count": len(catalog.candidates), "observed_count": len(catalog.observed),
            "checkpoint_sha256": source["checkpoint_sha256"], "free_order_and_non_CW_values_identical": True,
            "cw_values_removed": 2997, "batch_count": len(frozen["batch_ids"]),
            "no_cw_audit": views.audit_no_cw({"prompt": prompt("no_cw"), "packet": packet})})
    result = {"status": "PASS", "seeds": audits, "real_llm_calls": 0, "model_fits": 0,
              "new_real_labels_revealed": 0, "new_cw_aware_executions": 0}
    once(STUDY / "dry_run_audit.json", result)
    return result


def report(budget):
    """Use saved validation metrics at each chosen checkpoint; no test reveal."""
    activate()
    s.validate()
    rounds = (budget-333)//32
    rows, summary, costs = [], [], []
    for seed in SEEDS:
        curve = []
        for r in range(rounds+1):
            if r:
                s.audit_batch(s.selection_directory(seed, "llm32_nocw", r-1))
                model = STUDY / f"runtime/seed_{seed}/llm32_nocw/round_{r:02d}/model"
            else:
                model = initial_paths(seed)["member0_model"].parent
            audit = s.read(model / "fit_audit.json")
            if audit["checkpoint_sha256"] != s.sha256_file(model / "best.pt"):
                raise RuntimeError("checkpoint drift before validation report")
            history = pd.read_csv(model / "history.csv")
            chosen = history.loc[history.epoch.eq(audit["best_epoch"])]
            if len(chosen) != 1:
                raise RuntimeError("chosen validation epoch missing")
            record = chosen.iloc[0].to_dict()
            metrics = {k: v for k, v in record.items() if k.startswith("validation_")}
            score = float(record["validation_selection_score"])
            curve.append(score)
            rows.append({"seed": seed, "budget": s.BUDGETS[r], **metrics})
        summary.append({"seed": seed, "validation_AULC":
            sum((curve[j]+curve[j+1])*.5*32 for j in range(rounds))/(budget-333)})
        for r in range(rounds):
            d = s.selection_directory(seed, "llm32_nocw", r)
            for p in sorted(d.glob("attempt_??_??.json")):
                a = s.read(p)
                costs.append({"seed": seed, "round": r, "success": a["success"],
                    "elapsed_seconds": a["elapsed_seconds"], "usage": a.get("provenance", {}).get("usage")})
    result = {"budget": budget, "metrics": rows, "AULC": summary, "costs": costs,
        "comparison": "historical CW-aware reference; not a newly paired strict causal ablation",
        "test_truth_reads": 0, "selector_reward": False}
    once(STUDY / f"reports/validation_to{budget}.json", result)
    return result


def worker(seed, *, allow_paid_llm=False, allow_training=False, allow_reveal=False):
    if not (allow_paid_llm and allow_training and allow_reveal):
        raise PermissionError("worker requires all three explicit cost/reveal flags")
    activate(seed)
    s.require_authorization()
    credential_audit()
    os.environ.pop("SCIENTIST_API_KEY", None)
    os.environ["SCIENTIST_API_KEY_FILE"] = str(KEY_FILES[seed])
    torch.set_num_threads(2)
    with BASE_EXCLUSIVE_LOCK(STUDY / f"locks/worker_seed_{seed}.lock"):
        try:
            for r in range(len(s.BUDGETS)-1):
                arms = arm_order(seed, r)
                # Only NoCW is authorized. Freeze before revealing this batch.
                for arm in arms:
                    status_record(seed, "SELECTING", round=r, arm=arm, current_budget=s.BUDGETS[r])
                    s.select(seed, arm, r, allow_paid_llm=True, allow_feature_compute=r > 0)
                for arm in arms:
                    status_record(seed, "ADVANCING", round=r, arm=arm, next_budget=s.BUDGETS[r+1])
                    s.advance(seed, arm, r, allow_reveal=True, allow_training=True)
            status_record(seed, "COMPLETE", final_budget=1005, arms=list(s.METHODS))
        except BaseException as error:
            status_record(seed, "FAILED_CLOSED", error_type=type(error).__name__,
                          reason=str(error)[:700])
            raise


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def launch(*, allow_paid_llm=False, allow_training=False, allow_reveal=False):
    if not (allow_paid_llm and allow_training and allow_reveal):
        raise PermissionError("launch requires all three explicit cost/reveal flags")
    activate()
    s.require_authorization()
    credential_audit()
    logdir = STUDY / "progress"
    logdir.mkdir(parents=True, exist_ok=True)
    with BASE_EXCLUSIVE_LOCK(STUDY / "locks/launch.lock"):
        launchpath = STUDY / "launch.json"
        if launchpath.exists():
            previous = s.read(launchpath)
            if any(process_alive(w["pid"]) for w in previous["workers"]):
                raise RuntimeError("existing seed workers are still running")
            raise RuntimeError("launch already recorded; inspect failures before explicit per-seed resume")
        workers = []
        for seed in SEEDS:
            env = os.environ.copy()
            env.pop("SCIENTIST_API_KEY", None)
            env.update(SCIENTIST_API_KEY_FILE=str(KEY_FILES[seed]), KMP_DUPLICATE_LIB_OK="TRUE",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
            log = logdir / f"seed_{seed}.log"
            with log.open("ab") as output:
                process = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed),
                    "--allow-paid-llm", "--allow-training", "--allow-label-reveal"],
                    cwd=s.ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append({"seed": seed, "pid": process.pid, "key_file": str(KEY_FILES[seed]), "log": str(log)})
        record = {"status": "LAUNCHED", "launched_utc": now(), "workers": workers,
                  "target_budget": 1005, "nocw_trajectories": 5, "cw_aware_trajectories": 0,
                  "planned_primary_requests": 105, "planned_new_fits": 105,
                  "automatic_restart": False}
        once(launchpath, record)
        return record


def status():
    record = s.read(STUDY / "launch.json") if (STUDY / "launch.json").exists() else {"status": "NOT_LAUNCHED", "workers": []}
    for w in record["workers"]:
        w["process_alive"] = process_alive(w["pid"])
        p = STUDY / f"progress/seed_{w['seed']}.json"
        w["progress"] = s.read(p) if p.exists() else None
        w.pop("key_file", None)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", default="status", choices=("prepare", "dry-run", "validate", "authorize", "launch", "worker", "status", "report"))
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--allow-paid-llm", action="store_true")
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--allow-label-reveal", action="store_true")
    p.add_argument("--budget", type=int, choices=(525, 1005), default=1005)
    a = p.parse_args(argv)
    if a.action == "worker" and a.seed is None:
        p.error("worker requires --seed")
    activate(a.seed if a.action == "worker" else None)
    if a.action == "prepare": result = prepare()
    elif a.action == "authorize": result = authorize()
    elif a.action in ("launch", "worker"):
        kwargs = dict(allow_paid_llm=a.allow_paid_llm, allow_training=a.allow_training, allow_reveal=a.allow_label_reveal)
        result = worker(a.seed, **kwargs) if a.action == "worker" else launch(**kwargs)
    elif a.action == "status": result = status()
    elif a.action == "report": result = report(a.budget)
    elif a.action == "dry-run": result = dry_run()
    else: result = getattr(s, a.action.replace("-", "_"))()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
