#!/usr/bin/env python3
"""Paired Free32 CW-information ablation. No arguments runs only offline dry-run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.qgeognn_al.active_learning_v2 import nocw_study as study


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", default="dry-run",
                   choices=("prepare", "validate", "dry-run", "preregister", "stage", "select", "advance", "report"))
    p.add_argument("--seed", type=int, choices=study.SEEDS)
    p.add_argument("--arm", choices=study.METHODS)
    p.add_argument("--round", type=int, default=0, choices=range(21))
    p.add_argument("--budget", type=int, default=1005, choices=(525, 1005))
    p.add_argument("--allow-paid-llm", action="store_true")
    p.add_argument("--allow-label-reveal", action="store_true")
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--allow-feature-compute", action="store_true")
    p.add_argument("--authorize-formal-run", action="store_true")
    p.add_argument("--confirm-provider-support", action="store_true")
    a = p.parse_args(argv)
    if a.action in ("stage", "select", "advance") and (a.seed is None or a.arm is None):
        p.error("--seed and --arm required; commands act on exactly one batch")
    if a.action == "preregister":
        result = study.preregister(authorize=a.authorize_formal_run, provider_supported=a.confirm_provider_support)
    elif a.action == "stage":
        result = study.stage(a.seed, a.arm, a.round, allow_feature_compute=a.allow_feature_compute)
    elif a.action == "select":
        result = study.select(a.seed, a.arm, a.round, allow_paid_llm=a.allow_paid_llm,
                              allow_feature_compute=a.allow_feature_compute)
    elif a.action == "advance":
        result = study.advance(a.seed, a.arm, a.round, allow_reveal=a.allow_label_reveal, allow_training=a.allow_training)
    elif a.action == "report":
        result = study.report(a.budget)
    else:
        result = getattr(study, a.action.replace("-", "_"))()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
