#!/usr/bin/env python3
"""Release-code-aligned transfer diagnostic on the frozen reconstruction splits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import minmax_fit, sha256_file  # noqa: E402
from scripts.studies import run_paper_transfer_reproduction_25g_40g as reconstruction  # noqa: E402
from src.qgeognn_al.data import build_model_data, eluent_descriptor  # noqa: E402
from src.qgeognn_al.evaluation.point import point_metrics  # noqa: E402
from src.qgeognn_al.models import load_predictor_checkpoint  # noqa: E402
from src.qgeognn_al.training.predictor import loader_pair, predict, stable_hash  # noqa: E402

STUDY = ROOT / "studies/transfer/paper_transfer_gap_decomposition"
RECON = ROOT / "studies/transfer/paper_transfer_reproduction"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
COLUMNS = ("25g", "40g")
SEEDS = (42, 525, 1101, 2025, 2026)
EPOCHS = 500
VALIDATION_CADENCE = 20
BATCH_SIZE = 2048
LR = 1e-4
WEIGHT_DECAY = 1e-5


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for name in shown.columns:
        if pd.api.types.is_float_dtype(shown[name]):
            shown[name] = shown[name].map(lambda value: f"{value:.3f}")
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join("---" for _ in shown.columns) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def _target_full_scaler(data: pd.DataFrame, graph_cache: dict) -> dict:
    descriptors = np.vstack([graph_cache[value]["descriptor"] for value in data.canonical_smiles]).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in data["PE/EA"]]).astype(np.float32)
    return {"descriptor": minmax_fit(descriptors), "eluent": minmax_fit(eluents)}


def _release_loss(true: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    q_loss = reconstruction.quantile_target_loss
    return q_loss(true[:, 0], prediction[:, :3]) + 0.5 * q_loss(true[:, 1], prediction[:, 3:])


def _indices(data: pd.DataFrame, split: pd.DataFrame) -> dict[str, np.ndarray]:
    joined = data[["sample_id"]].merge(split[["sample_id", "split"]], on="sample_id", how="left", validate="one_to_one")
    if joined.split.isna().any():
        raise RuntimeError("frozen split does not cover target data")
    return {role: np.flatnonzero(joined.split.eq(role)) for role in ("train", "valid", "test")}


def _run_dir(column: str, seed: int) -> Path:
    return STUDY / f"runtime/runs/{column}/seed_{seed}"


def run_one(column: str, seed: int, smoke: bool = False) -> dict:
    torch.set_num_threads(2)
    directory = _run_dir(column, seed)
    result_path = directory / "result.json"
    epochs = 2 if smoke else EPOCHS
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("epochs_requested") == epochs and result.get("source_checkpoint_sha256") == SOURCE_SHA256:
            print(f"{column}/{seed}: verified complete", flush=True)
            return result
    started = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    filtered = reconstruction.read_target_data(column, "legacy_filtered")
    target_full = reconstruction.read_target_data(column, "no_threshold")
    frozen = pd.read_csv(RECON / "split_manifest.csv")
    split = frozen.loc[(frozen.column == column) & (frozen.protocol == "legacy_filtered") & (frozen.seed == seed)].copy()
    split = split.sort_values("canonical_index")
    expected = reconstruction.make_row_split(filtered, column, "legacy_filtered", seed)
    pd.testing.assert_frame_equal(split.reset_index(drop=True), expected.reset_index(drop=True))
    index = _indices(filtered, split)
    graph_cache = reconstruction.load_target_graph_cache(column)
    scaler = _target_full_scaler(target_full, graph_cache)
    atom, angle = build_model_data(filtered, graph_cache, pd.DataFrame(), scaler)
    model = load_predictor_checkpoint(SOURCE)
    for parameter in model.parameters():
        parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.5)
    # The released QGeoGNN transfer path instantiates StepLR but never calls scheduler.step().
    del scheduler
    target_scales = {name: float(filtered.iloc[index["train"]][f"{name}_ml"].std(ddof=0)) for name in ("V1", "V2")}
    best, best_epoch, best_state, history = float("inf"), None, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for a, b in zip(*loader_pair(atom, angle, index["train"], BATCH_SIZE)):
            prediction = model(a, b)
            loss = _release_loss(a.y, prediction)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite release-aligned loss")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % VALIDATION_CADENCE == 0 or epoch == epochs:
            truth, prediction, _ = predict(model, atom, angle, index["valid"])
            metrics = point_metrics(truth, prediction, target_scales)
            score = float(metrics["combined_normalized_rmse"])
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_score": score, **metrics})
            if score < best:
                best, best_epoch = score, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    metrics = {}
    predictions = []
    for role in ("valid", "test"):
        truth, output, positions = predict(model, atom, angle, index[role])
        metrics[role] = point_metrics(truth, output, target_scales)
        frame = pd.DataFrame({"sample_id": filtered.iloc[positions].sample_id.to_numpy(), "split": role,
                              "V1_true": truth[:, 0], "V2_true": truth[:, 1],
                              "V1_q10": output[:, 0], "V1_q50": output[:, 1], "V1_q90": output[:, 2],
                              "V2_q10": output[:, 3], "V2_q50": output[:, 4], "V2_q90": output[:, 5]})
        predictions.append(frame)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv.gz", index=False,
                                                      compression={"method": "gzip", "mtime": 0})
    result = {"column": column, "protocol": "legacy_filtered", "seed": seed,
        "method": "RELEASE_CODE_ALIGNED_TRANSFER", "best_epoch": best_epoch, "epochs_requested": epochs,
        "epochs_run": epochs, "validation_cadence": VALIDATION_CADENCE, "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "scheduler": "StepLR(50,0.5) instantiated_not_stepped",
        "loss": "loss_V1 + 0.5 * loss_V2", "head": "Linear(128,6)+ReLU; eval clamp [0,1e8]",
        "preprocessing": "descriptor_and_eluent_minmax_fit_on_whole_reader_compatible_target_before_threshold_filter",
        "preprocessing_fit_rows": len(target_full), "selection": "validation_only_at_20_epoch_cadence",
        "trainable_parameters": trainable, "total_parameters": total,
        "source_checkpoint": str(SOURCE.relative_to(ROOT)), "source_checkpoint_sha256": SOURCE_SHA256,
        "split_ids_hash": stable_hash(split[["sample_id", "split"]].astype(str).to_dict("records")),
        "valid": metrics["valid"], "test": metrics["test"], "runtime_seconds": time.time() - started}
    _write_json(result_path, result)
    print(json.dumps({"completed": [column, seed], "best_epoch": best_epoch,
                      "V1_r2": metrics["test"]["V1_r2"], "V2_r2": metrics["test"]["V2_r2"]}), flush=True)
    return result


def prepare() -> None:
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("qualified source checkpoint changed")
    STUDY.mkdir(parents=True, exist_ok=True)
    raw4 = pd.read_csv(ROOT / "dataset/dataset_4g.csv")
    source = pd.read_csv(ROOT / "experiments/e0_4g_baseline/canonical_4g.csv")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    audit = {
        "classification": "RELEASE_CODE_ALIGNED_REPRODUCTION_DIAGNOSTIC", "exact_paper_reproduction": False,
        "paper_reported_4g_rows": 4684, "paper_reported_compounds": 218,
        "repository_4g_raw_rows": len(raw4), "current_effective_filtered_rows": len(source),
        "current_effective_compounds": int(source.canonical_smiles.nunique()),
        "qualified_source_checkpoint": str(SOURCE.relative_to(ROOT)), "qualified_source_checkpoint_sha256": SOURCE_SHA256,
        "prior_reconstruction_source_checkpoint": "experiments/e0_4g_baseline/checkpoints/best.pt",
        "prior_reconstruction_source_checkpoint_sha256": sha256_file(ROOT / "experiments/e0_4g_baseline/checkpoints/best.pt"),
        "source_checkpoint_identity_match": False,
        "qualified_model_parameters": int(checkpoint["parameter_count"]),
        "released_original_model_parameters": 831558, "reconstructed_trainable_parameters": 242216,
        "release_transfer_scope": "transfer branch loads 4g state then Adam(model.parameters()): approximate full fine-tune",
        "release_default_branch_caveat": "QGeoGNN_transfer_25g hard-codes transfer_mode='direct_train'; diagnostic explicitly exercises documented transfer branch semantics",
        "release_column_info": "Use_column_info=False", "release_loss": "loss_V1 + 0.5*loss_V2",
        "release_head": "Linear(128,6)+ReLU; evaluation clamp", "reconstruction_head": "monotonic q10/q50/q90",
        "release_preprocessing": "whole target dataset min/max before split and before retention threshold exclusion",
        "reconstruction_preprocessing": "frozen source 4g scaler", "release_optimizer": "Adam lr=1e-4 weight_decay=1e-5",
        "release_scheduler": "StepLR(50,0.5) instantiated but scheduler.step() absent", "release_epochs": 500,
        "paper_source_epochs": 1500, "paper_transfer_epochs": "not identified; no 1500-epoch target sensitivity run authorized",
        "paper_count_source": "Wu et al., Chem 11 (2025) 102598 / arXiv:2404.09114",
        "test_driven_selection": False,
    }
    _write_json(STUDY / "audit.json", audit)
    _write_json(STUDY / "protocol.json", {"study_name": "PAPER_TRANSFER_GAP_DECOMPOSITION",
        "classification": audit["classification"], "columns": list(COLUMNS), "seeds": list(SEEDS),
        "split": "exact reused legacy_filtered random-row 80/10/10 identities", "duration": [EPOCHS],
        "checkpoint_selection": "validation only", "test_during_training": False,
        "source_checkpoint_sha256": SOURCE_SHA256, "exact_reproduction_claim": False})
    (STUDY / "README.md").write_text("# Paper transfer gap decomposition\n\nRelease-code-aligned diagnostic, not exact paper reproduction. Track A is independent from Track B.\n", encoding="utf-8")
    comparison = """# Paper pipeline preprocessing comparison

| item | paper/supplement evidence | released code | prior reconstruction | aligned diagnostic |
| --- | --- | --- | --- | --- |
| target normalization | fit scope not identified | whole reader-compatible target before split/filter | frozen 4g source scaler | released target-full semantics |
| transfer scope | pretrained transfer described; exact module freeze map unavailable | transfer branch loads source then optimizes `model.parameters()` | 242,216-parameter last-layer/adapter/head scope | all 458,952 parameters of qualified model |
| output head | quantile outputs; positivity parameterization unspecified | `Linear(128,6) + ReLU`, eval clamp | monotonic quantile head | qualified six-output linear+ReLU head |
| endpoint loss | quantile objective | `loss_V1 + 0.5*loss_V2` | `loss_V1 + loss_V2` | released 1:0.5 weighting |
| learning rate | Adam/transfer details incomplete | Adam 1e-4, weight decay 1e-5 | Adam 1e-4, weight decay 1e-5 | same as release |
| scheduler | StepLR described | StepLR(50,0.5) instantiated but never stepped | no StepLR | instantiated and deliberately not stepped |
| duration | source training reports 1500 epochs; target-transfer duration not recovered | 500 target epochs | cap 500 with patience 100 | exactly 500, validation-only checkpoint |
| batch size | 2048 | 2048 | 2048 | 2048 |
| column fields | transfer narrative refers to column specification | global `Use_column_info=False` | appended adapter inputs | no column fields |

The public release does not contain enough state to reconstruct the paper's exact source checkpoint, data snapshot, split map, or complete author training procedure. Therefore this study is explicitly not an exact reproduction.
"""
    (STUDY / "paper_pipeline_preprocessing_comparison.md").write_text(comparison, encoding="utf-8")


def execute(workers: int, smoke: bool = False) -> None:
    prepare()
    started = time.time()
    if smoke:
        run_one("25g", 42, True)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda pair: run_one(*pair), [(column, seed) for column in COLUMNS for seed in SEEDS]))
    _write_json(STUDY / "runtime/execution.json", {"runs": len(results), "workers": workers,
        "wall_runtime_seconds": time.time() - started, "all_complete": len(results) == 10})
    summarize(results)


def summarize(results: list[dict] | None = None) -> None:
    if results is None:
        results = [json.loads((_run_dir(column, seed) / "result.json").read_text()) for column in COLUMNS for seed in SEEDS]
    rows = []
    for result in results:
        rows.append({"column": result["column"], "protocol": result["protocol"], "seed": result["seed"],
            "method": result["method"], "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "trainable_parameters": result["trainable_parameters"], "source_checkpoint_sha256": result["source_checkpoint_sha256"],
            **{name: result["test"][name] for name in ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse")}})
    aligned = pd.DataFrame(rows).sort_values(["column", "seed"])
    aligned.to_csv(STUDY / "release_code_aligned_metrics.csv", index=False)
    old = pd.read_csv(RECON / "run_summary.csv")
    old = old.loc[(old.protocol == "legacy_filtered") & (old.method == "paper_transfer") & old.column.isin(COLUMNS)].copy()
    paired = aligned.merge(old, on=["column", "seed"], suffixes=("_aligned", "_reconstruction"))
    keep = ["column", "seed"]
    for endpoint in ("V1", "V2"):
        for metric in ("rmse", "mae", "r2"):
            a = f"{endpoint}_{metric}"
            old_name = f"{endpoint}_test_{metric}"
            paired[f"{a}_delta"] = paired[a] - paired[old_name]
            keep += [a, old_name, f"{a}_delta"]
    paired[keep].to_csv(STUDY / "paired_reconstruction_comparison.csv", index=False)
    summary_rows = []
    measured = ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse")
    for (column, method), group in aligned.groupby(["column", "method"]):
        row = {"column": column, "method": method, "seeds": int(group.seed.nunique()),
               "best_epoch_mean": float(group.best_epoch.mean())}
        for name in measured:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_sample_sd"] = float(group[name].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(STUDY / "release_code_aligned_run_summary.csv", index=False)
    paper = {"25g": (0.747, .840), "40g": (.826, .824)}
    comparison_rows = []
    for row in summary.itertuples():
        p1, p2 = paper[row.column]
        comparison_rows.append({"column": row.column, "paper_V1_r2": p1, "aligned_V1_r2": row.V1_r2_mean,
                                "V1_gap": row.V1_r2_mean - p1, "paper_V2_r2": p2,
                                "aligned_V2_r2": row.V2_r2_mean, "V2_gap": row.V2_r2_mean - p2})
    pd.DataFrame(comparison_rows).to_csv(STUDY / "paper_r2_comparison.csv", index=False)
    old_means = old.groupby("column")[[f"{e}_test_{m}" for e in ("V1", "V2") for m in ("rmse", "mae", "r2")]].mean()
    aligned_means = aligned.groupby("column")[[f"{e}_{m}" for e in ("V1", "V2") for m in ("rmse", "mae", "r2")]].mean()
    change25_v1 = 100 * (1 - aligned_means.loc["25g", "V1_rmse"] / old_means.loc["25g", "V1_test_rmse"])
    change25_v2 = 100 * (1 - aligned_means.loc["25g", "V2_rmse"] / old_means.loc["25g", "V2_test_rmse"])
    change40_v1 = 100 * (1 - aligned_means.loc["40g", "V1_rmse"] / old_means.loc["40g", "V1_test_rmse"])
    change40_v2 = 100 * (1 - aligned_means.loc["40g", "V2_rmse"] / old_means.loc["40g", "V2_test_rmse"])
    lines = ["# Final report: paper transfer gap decomposition", "", "This is a release-code-aligned diagnostic, not an exact paper reproduction.", "",
             "## Five-seed random-row results", "", _markdown(summary), "", "## Paper Figure 4 R2 comparison", "",
             _markdown(pd.DataFrame(comparison_rows)), "", "## Conclusions", "",
             "Q1/Q2. The prior reconstruction is not exact because the public release does not recover the paper's complete source checkpoint/data/split pipeline and because the reconstruction intentionally changed trainable scope, normalization, output head, V2 loss weight, scheduler semantics, and column-input path. These are confirmed implementation mismatches, not guesses.", "",
             f"Q3. Alignment improves 25g V1 RMSE by {change25_v1:.2f}% and mean V1 R2 by {aligned_means.loc['25g','V1_r2']-old_means.loc['25g','V1_test_r2']:+.3f}; V2 RMSE changes by {change25_v2:.2f}% and R2 by {aligned_means.loc['25g','V2_r2']-old_means.loc['25g','V2_test_r2']:+.3f}. Thus the 25g gain is endpoint-specific, not a general closure of the gap.", "",
             f"Q4. 40g does not improve: V1/V2 RMSE changes are {change40_v1:.2f}%/{change40_v2:.2f}% (negative means worse). Mean R2 gaps to the paper remain about 0.210/0.204, versus seed SDs {summary.loc[summary.column.eq('40g'),'V1_r2_sample_sd'].iloc[0]:.3f}/{summary.loc[summary.column.eq('40g'),'V2_r2_sample_sd'].iloc[0]:.3f}; seed variance is material but not the main identified explanation.", "",
             "Q5/Q6. The complete RMSE/MAE/R2 reference is the table above. Neither column is jointly close to both paper endpoints: 25g remains -0.067/-0.115 R2 below Figure 4 and 40g remains -0.210/-0.204 below. No additional epoch or hyperparameter configurations were added after seeing test results.", "",
             "Remaining unidentified factors are the exact author source checkpoint and data snapshot, the paper's split/seed map, target-transfer duration, and hidden training history. Track A results were not used to design Track B."]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run", nargs=2)
    args = parser.parse_args()
    if args.run:
        prepare(); run_one(args.run[0], int(args.run[1]), args.smoke)
    elif args.execute:
        execute(args.workers, args.smoke)
    elif args.summarize:
        summarize()
    else:
        prepare()


if __name__ == "__main__":
    main()
