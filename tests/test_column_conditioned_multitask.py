"""Executable architecture, label-boundary, sampling, and freeze contracts."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.studies import run_column_conditioned_multitask as r
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.transfer.column_conditioned_multitask import (
    BalancedTaskBatcher, build_multitask_model, film_is_zero_initialized,
    global_target_group_folds, gradient_geometry, head_copies_are_exact, TASKS,
)


@pytest.fixture(scope="module")
def real_batch():
    torch.set_num_threads(1)
    specification = r.protocol_payload()
    source = r.task_data("4g", "compound", r.SEEDS[0], specification)
    return r.make_batch({"4g": source}, {"4g": list(range(12))})


def test_qualified_source_and_split_identity_authority():
    assert r.sha(r.source_path()) == r.SOURCE_SHA
    model = load_predictor_checkpoint(r.source_path())
    assert model.model_variant == "qgeognn_v2"
    assert sum(p.numel() for p in model.parameters()) == 458952
    schedule = r.verify_schedule()
    assert set(schedule.column) == {"25g", "40g"}
    assert set(schedule.outer_seed) == set(r.SEEDS)


def test_exact_head_copy_source_equivalence_and_zero_film(real_batch):
    atom, angle, tasks = real_batch
    source = load_predictor_checkpoint(r.source_path()).eval()
    a1 = build_multitask_model(r.source_path(), r.ARMS["A1"]).eval()
    a2 = build_multitask_model(r.source_path(), r.ARMS["A2"]).eval()
    assert head_copies_are_exact(a1) and head_copies_are_exact(a2)
    assert film_is_zero_initialized(a2)
    for task in TASKS:
        for key, value in source.head.state_dict().items():
            torch.testing.assert_close(a1.heads[task].state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        expected = source(atom, angle)
        for index in range(3):
            ids = torch.full_like(tasks, index)
            first, second = a1(atom, angle, ids), a2(atom, angle, ids)
            torch.testing.assert_close(first, expected, rtol=0, atol=1e-6)
            torch.testing.assert_close(second, first, rtol=0, atol=1e-6)
            torch.testing.assert_close(second, a2(atom, angle, ids), rtol=0, atol=0)
            assert torch.isfinite(second).all()


def test_task_routing(real_batch):
    atom, angle, tasks = real_batch
    model = build_multitask_model(r.source_path(), r.ARMS["A1"]).eval()
    with torch.no_grad():
        for index, task in enumerate(TASKS):
            model.heads[task][0].weight.zero_()
            model.heads[task][0].bias.fill_(index + 1.)
        tasks = torch.arange(len(tasks)) % 3
        prediction = model(atom, angle, tasks)
    torch.testing.assert_close(prediction, (tasks + 1).float()[:, None].expand(-1, 6))
    with pytest.raises(ValueError):
        model(atom, angle, torch.full_like(tasks, 3))


def test_film_generators_and_embedding_receive_gradients(real_batch):
    atom, angle, tasks = real_batch
    torch.manual_seed(19)
    model = build_multitask_model(r.source_path(), r.ARMS["A2"]).eval()
    optimizer = model.optimizer()
    for step in range(2):
        optimizer.zero_grad()
        prediction = model(atom, angle, tasks)
        r.target_loss_recipe(atom.y[:, 0], atom.y[:, 1], prediction, {}, recipe="raw_quantile").backward()
        for generator in model.film_generators.values():
            assert generator.weight.grad is not None
            assert torch.isfinite(generator.weight.grad).all()
            assert torch.count_nonzero(generator.weight.grad) > 0
        if step:
            assert torch.count_nonzero(model.column_embedding.weight.grad) > 0
        optimizer.step()
    assert not any(p.requires_grad for p in model.backbone.convs[0].parameters())
    assert all(p.requires_grad for p in model.backbone.convs[3].parameters())


def test_cached_prefix_preserves_training_predictions_gradients_and_update(real_batch):
    atom, angle, tasks = real_batch
    uncached = atom.clone()
    del uncached.frozen_nodes
    del uncached.frozen_edges
    for arm in ("A1", "A2"):
        torch.manual_seed(42)
        cached_model = build_multitask_model(r.source_path(), r.ARMS[arm])
        torch.manual_seed(42)
        full_model = build_multitask_model(r.source_path(), r.ARMS[arm])
        first_optimizer, second_optimizer = cached_model.optimizer(), full_model.optimizer()
        for model in (cached_model, full_model):
            r.set_training_mode(model)
        cached_prediction = cached_model(atom, angle, tasks)
        full_prediction = full_model(uncached, angle, tasks)
        torch.testing.assert_close(cached_prediction, full_prediction, rtol=0, atol=1e-6)
        cached_prediction.square().mean().backward()
        full_prediction.square().mean().backward()
        for first, second in zip(cached_model.parameters(), full_model.parameters()):
            if first.requires_grad:
                torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=1e-5)
        first_optimizer.step()
        second_optimizer.step()
        for key, value in cached_model.state_dict().items():
            torch.testing.assert_close(value, full_model.state_dict()[key], rtol=0, atol=1e-6)


def test_balanced_participation_reproducible_with_wrapping():
    indices = {"4g": np.arange(3330), "25g": np.arange(13), "40g": np.arange(317)}
    first = BalancedTaskBatcher(indices, per_task_batch_size=128, seed=42)
    second = BalancedTaskBatcher(indices, per_task_batch_size=128, seed=42)
    assert first.participation(1) == {c: 384 for c in TASKS}
    for a, b in zip(first.epoch_batches(1), second.epoch_batches(1)):
        assert len(set(map(len, a.values()))) == 1
        for c in TASKS:
            np.testing.assert_array_equal(a[c], b[c])
            assert set(a[c]) <= set(indices[c])
    assert not np.array_equal(first._draws("4g", 1), first._draws("4g", 2))
    assert set(first._draws("40g", 1)) == set(indices["40g"])


def test_global_smiles_folds_prevent_cross_column_leakage():
    groups = [f"m{i}" for i in range(20)] * 2
    columns = ["25g"] * 20 + ["40g"] * 20
    folds = global_target_group_folds(groups, columns)
    seen = []
    for train, valid in folds:
        assert not set(np.array(groups)[train]) & set(np.array(groups)[valid])
        assert set(np.array(columns)[valid]) == {"25g", "40g"}
        seen.extend(valid)
    assert sorted(seen) == list(range(40))


def test_outer_truth_is_not_requested_and_graph_labels_scrubbed(monkeypatch):
    specification = r.protocol_payload()
    requests = []
    original = r._read_authorized_truth
    def spy(path, ids):
        requests.extend(ids)
        return original(path, ids)
    monkeypatch.setattr(r, "_read_authorized_truth", spy)
    data = r.task_data("25g", "compound", r.SEEDS[0], specification)
    assert requests == data["roles"]["gradient_train"]
    protected = data["positions"]["validation"] + data["positions"]["test"]
    assert np.count_nonzero(data["truth"][protected]) == 0
    assert all(torch.count_nonzero(data["atoms"][i].y) == 0 for i in protected)
    synthetic = {c: data for c in TASKS}
    training = {c: data["positions"]["gradient_train"] for c in TASKS}
    training["25g"] = protected[:1]
    with pytest.raises(RuntimeError, match="unauthorized fitting"):
        r.authorize_fit(synthetic, training)


def test_gradient_diagnostics_do_not_mutate_training_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    losses = {"4g": parameter.sum(), "25g": 2 * parameter.sum(), "40g": -parameter.sum()}
    result = gradient_geometry(losses, [parameter])
    assert parameter.grad is None
    assert result["cos_4g_25g"] == pytest.approx(1.)
    assert result["cos_4g_40g"] == pytest.approx(-1.)
    torch.stack(list(losses.values())).mean().backward()
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2 / 3))


def synthetic_evidence(gain=.05):
    rows, tails = [], []
    for seed in r.SEEDS:
        for fold in range(1, 6):
            for c in TASKS[1:]:
                for arm in r.ARMS:
                    value = 1. if arm == "A1" else 1 - gain
                    meta = dict(outer_seed=seed, inner_fold=fold, column=c, arm=arm)
                    rows.append({**meta, "combined_normalized_rmse": value, "V1_rmse": value, "V2_rmse": value})
                    tails.extend([{**meta, "endpoint": e, "tail_count": 2, "tail_sse": 2 * value ** 2} for e in ("V1", "V2")])
    return pd.DataFrame(rows), pd.DataFrame(tails)


def test_gate_requires_both_columns_complete_and_tail_guard():
    rows, tails = synthetic_evidence()
    assert r.gate_decision(rows, tails)["passed"]
    bad = deepcopy(tails)
    bad.loc[bad.column.eq("25g") & bad.arm.eq("A2"), "tail_sse"] = 3.
    assert not r.gate_decision(rows, bad)["passed"]
    assert not r.gate_decision(*synthetic_evidence(.02))["passed"]
    with pytest.raises(RuntimeError, match="exactly"):
        r.gate_decision(rows.iloc[:-1], tails)


def test_scoring_checks_gate_and_freeze_before_truth(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "STUDY", tmp_path)
    def forbidden(*args):
        raise AssertionError("truth accessed before freeze")
    monkeypatch.setattr(r, "_read_authorized_truth", forbidden)
    with pytest.raises(RuntimeError, match="no COMPOUND"):
        r.score("compound")
    monkeypatch.setattr(r, "require_gate", lambda: {"passed": True})
    with pytest.raises(RuntimeError, match="until all predictions"):
        r.score("compound")


def test_tail_threshold_is_supplied_training_threshold():
    truth = np.array([[1., 2.], [5., 6.], [10., 12.]])
    prediction = np.repeat(truth, 3, axis=1) + 1
    _, tail, _ = r.metric_rows(truth, prediction, {"V1": 1., "V2": 2.}, [4., 11.], [3., 9.])
    assert [v["tail_count"] for v in tail] == [2, 1]
    assert [v["threshold"] for v in tail] == [4., 11.]


def test_checkpoint_selection_uses_inner_scales_and_one_joint_epoch():
    values = np.array([[1., 10.], [3., 14.], [5., 18.]])
    assert r.scales(values) == pytest.approx(dict(zip(("V1", "V2"), values.std(axis=0))))
    first = r.point_metrics(values, np.repeat(values + 1, 3, axis=1), r.scales(values))
    expected = np.sqrt(.5 * sum((1 / scale) ** 2 for scale in values.std(axis=0)))
    assert first["combined_normalized_rmse"] == pytest.approx(expected)
    policy = r.protocol_payload()
    assert policy["inner_cv"]["selector"].startswith("equal-column")
    assert policy["outer_validation_or_test_used_for_fit_or_selection"] is False


def test_project_gate_rejects_single_column_gain_and_row_regression():
    from scripts.studies.summarize_column_conditioned_multitask import promotion_decision
    rows, tails = synthetic_evidence()
    rows = rows.loc[rows.inner_fold.eq(1)].rename(columns={"arm": "method"})
    tails = tails.loc[tails.inner_fold.eq(1)].rename(columns={"arm": "method"})
    tails["tail_rmse"] = np.sqrt(tails.tail_sse / tails.tail_count)
    result = promotion_decision(rows, rows.copy(), tails, tails)
    assert result["PROJECT_TRANSFER_GAIN"]
    regression = rows.copy()
    regression.loc[regression.method.eq("A2") & regression.column.eq("25g"), "combined_normalized_rmse"] = 1.03
    assert not promotion_decision(rows, regression, tails, tails)["PROJECT_TRANSFER_GAIN"]
    assert not promotion_decision(regression, rows, tails, tails)["PROJECT_TRANSFER_GAIN"]
    regression = rows.copy()
    regression.loc[regression.method.eq("A2") & regression.column.eq("25g"), "V1_rmse"] = 1.03
    assert not promotion_decision(rows, regression, tails, tails)["PROJECT_TRANSFER_GAIN"]
    tail_regression = tails.copy()
    tail_regression.loc[tail_regression.method.eq("A2"), "tail_rmse"] = 1.03
    assert not promotion_decision(rows, rows, tails, tail_regression)["PROJECT_TRANSFER_GAIN"]


def test_gradient_summary_flags_conflict_and_magnitude_imbalance():
    from scripts.studies.summarize_column_conditioned_multitask import summarize_gradients
    frame = pd.DataFrame({"arm": ["A1"] * 3, "cos_4g_25g": [-.1, -.2, .1], "cos_4g_40g": [.1, .2, .3],
                          "cos_25g_40g": [.5, .3, .1], "grad_norm_4g": [1., 1., 1.],
                          "grad_norm_25g": [20., 20., 20.], "grad_norm_40g": [5., 5., 5.]})
    output = summarize_gradients(frame).set_index("pair")
    assert output.loc["4g_25g", "negative_fraction"] == pytest.approx(2/3)
    assert output.loc["4g_25g", "median_magnitude_ratio"] == 20.


def test_negative_report_runs_end_to_end_without_outer_truth(monkeypatch, tmp_path):
    import json
    from scripts.studies import summarize_column_conditioned_multitask as summary
    monkeypatch.setattr(r, "STUDY", tmp_path)
    monkeypatch.setattr(r, "execution_provenance", lambda: "test")
    monkeypatch.setattr(r, "verify_completed", lambda run: True)
    monkeypatch.setattr(r, "verify_schedule", lambda: pd.DataFrame({"column": ["25g"], "protocol": ["compound"], "outer_seed": [r.SEEDS[0]], "role": ["test"], "sample_id": ["held-out"]}))
    def forbidden(*args):
        raise AssertionError("report must never read endpoint data")
    monkeypatch.setattr(r, "_read_authorized_truth", forbidden)
    rows, tails = synthetic_evidence(.02)
    for key in ("V1_mae", "V2_mae", "Center_rmse", "Width_rmse"):
        rows[key] = rows.V1_rmse
    rows["V1_r2"], rows["V2_r2"] = .5, .6
    rows["best_epoch"], rows["epochs_run"] = 100, 180
    rows["protocol"] = "compound"
    tails["total_sse"] = 2 * tails.tail_sse
    tails["threshold"] = 1.
    rows.to_csv(tmp_path / "INNER_RESULTS.csv", index=False)
    tails.to_csv(tmp_path / "TAIL_METRICS.csv", index=False)
    (tmp_path / "protocol.json").write_text('{}\n')
    decision = r.gate_decision(rows, tails)
    decision.update(inner_results_sha256=r.sha(tmp_path / "INNER_RESULTS.csv"), tail_metrics_sha256=r.sha(tmp_path / "TAIL_METRICS.csv"), completion_hashes={})
    for seed in r.SEEDS:
        for fold in range(1, 6):
            for arm in r.ARMS:
                relative = f"runtime/inner/compound/seed_{seed}/fold_{fold}/{arm}"
                run = tmp_path / relative
                run.mkdir(parents=True)
                (run / "complete.json").write_text('{}\n')
                pd.DataFrame({"epoch": [1], "train_loss_4g": [1.]}).to_csv(run / "history.csv", index=False)
                pd.DataFrame({"sample_id": ["inner-validation"], "V1_q50": [1.], "V2_q50": [2.]}).to_csv(run / "predictions.csv", index=False)
                decision["completion_hashes"][relative] = r.sha(run / "complete.json")
    (tmp_path / "INNER_DECISION.json").write_text(json.dumps(decision))
    pd.DataFrame({"arm": ["A1", "A2"], "cos_4g_25g": [-.1, -.2], "cos_4g_40g": [.1, .2],
                  "cos_25g_40g": [.5, .3], "grad_norm_4g": [1., 1.], "grad_norm_25g": [20., 20.],
                  "grad_norm_40g": [5., 5.]}).to_csv(tmp_path / "GRADIENT_DIAGNOSTICS.csv", index=False)
    pd.DataFrame([{"column": c, "arm": arm, "regime": "high", "V1_rmse": 1., "V2_rmse": 1., "n": 3}
                  for c in r.TASKS[1:] for arm in r.ARMS]).to_csv(tmp_path / "RETENTION_REGIMES.csv", index=False)
    summary.summarize()
    report = (tmp_path / "FINAL_REPORT.md").read_text()
    assert "NO_REPLICATED_COLUMN_FILM_GAIN" in report
    assert "Outer prediction and scoring were intentionally blocked" in report
    for number in range(1, 14):
        assert f"\n{number}. " in report
    assert not (tmp_path / "OUTER_RESULTS.csv").exists()
    assert not (tmp_path / "PREDICTION_FREEZE_MANIFEST.json").exists()
