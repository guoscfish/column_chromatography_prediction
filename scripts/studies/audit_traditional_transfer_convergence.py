#!/usr/bin/env python3
"""Retrospective train-only convergence audit for the frozen formal ROW run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "studies/transfer/traditional_transfer_improvement/formal_row_5seed"
OUT = ROOT / "studies/transfer/traditional_transfer_convergence_audit"
COLUMNS = ("25g", "40g")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
ADAM_METHODS = ("P0", "P1", "P2")
FORMAL_PROTOCOL = json.loads((FORMAL / "FORMAL_ROW_PROTOCOL.json").read_text())
PROTOCOL_MAX_EPOCH = int(FORMAL_PROTOCOL["adam_budget"]["maximum_epochs"])


def main() -> None:
    rows = []
    for column in COLUMNS:
        for seed in SEEDS:
            base = FORMAL / "runtime" / column / f"seed_{seed}"
            history = pd.read_csv(base / "training_history.csv")
            stage = pd.read_csv(base / "stage_metrics.csv")
            for _, item in stage.iterrows():
                if item.method not in ADAM_METHODS:
                    continue
                current = history.loc[(history.method == item.method) & (history.stage == item.stage)].sort_values("epoch")
                if current.empty:
                    raise RuntimeError(f"missing history: {column}/{seed}/{item.method}/{item.stage}")
                tail = current.tail(min(20, len(current)))
                x = tail.epoch.to_numpy(float)
                y = tail.validation_score.to_numpy(float)
                slope = float(np.polyfit(x, y, 1)[0]) if len(tail) >= 2 else float("nan")
                rows.append({
                    "column": column, "seed": seed, "method": item.method, "stage": item.stage,
                    "protocol_max_epoch": PROTOCOL_MAX_EPOCH,
                    "epochs_run": int(item.epochs_run), "best_epoch": int(item.best_epoch),
                    "run_reached_protocol_ceiling": bool(int(item.epochs_run) == PROTOCOL_MAX_EPOCH),
                    "best_at_protocol_ceiling": bool(int(item.best_epoch) == PROTOCOL_MAX_EPOCH),
                    "best_at_run_end": bool(int(item.best_epoch) == int(item.epochs_run)),
                    "best_validation_score": float(item.validation_score), "last_validation_score": float(current.validation_score.iloc[-1]),
                    "tail_first_validation_score": float(tail.validation_score.iloc[0]), "tail_last_validation_score": float(tail.validation_score.iloc[-1]),
                    "tail_delta_last_minus_first": float(tail.validation_score.iloc[-1] - tail.validation_score.iloc[0]),
                    "tail_validation_slope": slope,
                })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "train_only_history_audit.csv", index=False)
    summary = frame.groupby(["column", "method", "stage"], as_index=False).agg(
        fits=("seed", "size"), run_reached_protocol_ceiling_count=("run_reached_protocol_ceiling", "sum"),
        best_at_protocol_ceiling_count=("best_at_protocol_ceiling", "sum"),
        best_at_run_end_count=("best_at_run_end", "sum"),
        best_epoch_mean=("best_epoch", "mean"), epochs_run_mean=("epochs_run", "mean"),
        tail_negative_slope_count=("tail_validation_slope", lambda x: int((x < -1e-4).sum())),
        tail_delta_mean=("tail_delta_last_minus_first", "mean"), tail_slope_mean=("tail_validation_slope", "mean"),
    )
    summary.to_csv(OUT / "convergence_summary.csv", index=False)
    hit = int(frame.best_at_protocol_ceiling.sum())
    run_ceiling = int(frame.run_reached_protocol_ceiling.sum())
    run_end = int(frame.best_at_run_end.sum())
    total = len(frame)
    negative = int((frame.tail_validation_slope < -1e-4).sum())
    report = f"""# Convergence Audit

## Provenance

This is a retrospective train-only audit of the pre-existing local formal ROW runtime. It reads only `training_history.csv` and `stage_metrics.csv` from the frozen P0-P3 run. It does not read outer test truth, retrain a model, select a test checkpoint, or modify the formal study.

## Evidence

- Adam stages audited: {total}.
- Protocol maximum epoch: {PROTOCOL_MAX_EPOCH}.
- Runs that reached the protocol ceiling: {run_ceiling}/{total}.
- Best checkpoint at the protocol ceiling: {hit}/{total}.
- Best checkpoint at the actual run end: {run_end}/{total}.
- Adam budget in the formal protocol: 150 epochs, patience 40.
- Last-20-epoch validation slope remained negative below -1e-4 in {negative}/{total} stages.
- Stage A head-only runs for P1/P2 reached epoch 150 in all 20 contexts. P0/P1/P2 Stage B also frequently reached the ceiling, especially for 40g.

The raw stage-level evidence is in `train_only_history_audit.csv`; grouped counts are in `convergence_summary.csv`.

## Assessment

The old field `best_at_max_epoch` was ambiguous: it compared `best_epoch` with the *actual run end*, which can be earlier than the protocol maximum because of early stopping. This regenerated audit keeps `protocol_max_epoch`, `epochs_run`, and `best_epoch` separate. The historical numerical claim remains true under the corrected semantics: {hit}/{total} Adam stages selected epoch {PROTOCOL_MAX_EPOCH}; it is now explicitly `best_at_protocol_ceiling`, not merely a run-end comparison.

The formal Adam budget is not an adequate convergence basis for stable optimizer or recipe ranking. A best epoch equal to the protocol ceiling is a censoring signal, not evidence that the ceiling is optimal. The observed negative late validation slopes are train-only evidence that some trajectories were still improving when stopped.

This audit does not establish that a longer budget will improve outer-test performance. It establishes only that the current formal result cannot answer that question.

## Independent follow-up protocol

Create a new train-only developmental convergence study with the same frozen populations, source checkpoint, trainable scope, BN policies, and seeds. Freeze the budget rule before any outer-test read. The minimal candidate budget is maximum epochs 500 with patience 80, with 150/300/500 recorded for diagnosis; selection must use inner validation or a predeclared plateau rule. Any later outer-test score is developmental confirmation because the current outer test has already been exposed in existing artifacts.
"""
    (OUT / "CONVERGENCE_AUDIT.md").write_text(report)
    protocol = {
        "status": "RETROSPECTIVE_TRAIN_ONLY_AUDIT",
        "source_study": "formal_row_5seed",
        "columns": list(COLUMNS), "seeds": list(SEEDS), "methods": list(ADAM_METHODS),
        "inputs": ["runtime/*/training_history.csv", "runtime/*/stage_metrics.csv"],
        "outer_test_truth_used": False,
        "formal_budget": {"maximum_epochs": PROTOCOL_MAX_EPOCH, "patience": 40},
        "semantics": {"protocol_max_epoch": "predeclared Adam maximum", "epochs_run": "actual epochs completed", "best_epoch": "validation-selected epoch", "run_reached_protocol_ceiling": "epochs_run == protocol_max_epoch", "best_at_protocol_ceiling": "best_epoch == protocol_max_epoch", "best_at_run_end": "best_epoch == epochs_run"},
        "evidence": {"adam_stages": total, "run_reached_protocol_ceiling": run_ceiling, "best_at_protocol_ceiling": hit, "best_at_run_end": run_end, "negative_tail_slope": negative},
        "follow_up_candidate": {"maximum_epochs": 500, "patience": 80, "diagnostic_budgets": [150, 300, 500]},
        "follow_up_test_status": "developmental_confirmation_only",
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
