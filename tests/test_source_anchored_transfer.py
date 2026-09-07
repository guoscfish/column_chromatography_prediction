import inspect
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.studies import run_source_anchored_transfer as run
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.transfer.source_anchored import SourceAnchoredTransfer, METHODS


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(1)
    usage = run.ledger("8g", "row", run.SEEDS[0], 30)
    preprocessing = torch.load(run.SOURCE, weights_only=False)["preprocessing"]
    target, source, source_ids, _, _ = run.load_fitting_inputs(usage, preprocessing)
    return usage, preprocessing, target, source, source_ids


@pytest.mark.parametrize("scope,expected", [("shallow", 36387), ("full", 458952)])
def test_zero_step_and_trainable_masks(inputs, scope, expected):
    original = load_predictor_checkpoint(run.SOURCE)
    model = SourceAnchoredTransfer(original, scope)
    a, b = run.batch(inputs[2]["gradient_train"])
    with torch.no_grad():
        assert torch.equal(original(a, b), model(a, b, "target"))
        assert torch.equal(original(a, b), model(a, b, "source"))
    inventory = model.parameter_inventory()
    assert inventory["trainable_parameters"] == expected
    assert inventory["total_parameters"] == 459726
    for name, p in model.named_parameters():
        assert not (name.startswith("source_head.") and p.requires_grad)
        if scope == "shallow":
            assert p.requires_grad == name.startswith(("backbone.convs.4.", "condition_branch.", "target_head."))
    assert model.source_head[0].weight.data_ptr() != model.target_head[0].weight.data_ptr()
    assert model.backbone.convs[4].eps.data_ptr() != original.backbone.convs[4].eps.data_ptr()


def test_heads_do_not_cross_and_replay_has_backbone_gradients(inputs):
    model = SourceAnchoredTransfer(load_predictor_checkpoint(run.SOURCE), "shallow")
    a, b = run.batch(inputs[2]["gradient_train"])
    before = model(a, b, "source").detach().clone()
    with torch.no_grad():
        model.target_head[0].bias.add_(10)
    assert torch.equal(before, model(a, b, "source"))
    assert not torch.equal(before, model(a, b, "target"))
    model.training_target()
    model(a, b)
    buffers = {n: p.clone() for n, p in model.named_buffers()}
    sa, sb = run.batch(inputs[3], range(22))
    with model.replay_mode():
        run.loss_pair(sa.y, model(sa, sb, "source")).backward()
    assert all(torch.equal(p, buffers[n]) for n, p in model.named_buffers())
    assert all(p.grad is None for p in model.source_head.parameters())
    assert all(p.grad is None for p in model.target_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.convs[4].parameters())
    assert model.backbone.convs[4].training


def test_training_step_does_not_change_frozen_parameters(inputs):
    model = SourceAnchoredTransfer(load_predictor_checkpoint(run.SOURCE), "shallow")
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    model.training_target()
    a, b = run.batch(inputs[2]["gradient_train"])
    run.loss_pair(a.y, model(a, b)).backward()
    optimizer.step()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(initial[name], p)
    assert model.parameter_drift(initial)["source_head_l2_drift"] == 0


def test_all_120_ledgers_and_nested_budgets():
    for c in run.COLUMNS:
        for p in ("row", "compound"):
            for s in run.SEEDS:
                previous = set()
                valid, test = None, None
                for b in run.BUDGETS:
                    usage = run.ledger(c, p, s, b)
                    acquired = set(usage["gradient_train"])
                    assert previous <= acquired
                    previous = acquired
                    assert usage["actual_budget"] == len(acquired)+len(usage["validation"])
                    if valid is not None:
                        assert valid == usage["validation"] and test == usage["test"]
                    valid, test = usage["validation"], usage["test"]


def test_selected_truth_never_parses_unpurchased_cells(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("sample_id,V1_ml,V2_ml\ntrain,1,2\ntest,DO_NOT_READ,DO_NOT_READ\ndonor,DO_NOT_READ,DO_NOT_READ\n")
    np.testing.assert_array_equal(run.selected_truth(path, ["train"]), [[1., 2.]])


def test_fitting_inputs_read_only_purchased_and_source_train(inputs, monkeypatch):
    calls = []
    original = run.selected_truth
    def guarded(path, ids):
        allowed = set(inputs[4]) if path == run.SOURCE_DATA else set(inputs[0]["gradient_train"]+inputs[0]["validation"])
        assert path in (run.SOURCE_DATA, run.target_path("8g"))
        assert set(ids) <= allowed
        calls.append((path, set(ids)))
        return original(path, ids)
    monkeypatch.setattr(run, "selected_truth", guarded)
    run.load_fitting_inputs(inputs[0], inputs[1])
    assert len(calls) == 3
    assert not set(inspect.signature(run.fit).parameters) & {"test", "test_truth", "probe", "other_target"}


def test_freeze_rejects_modified_or_missing_predictions(tmp_path):
    (tmp_path / "pred.csv").write_text("1,2\n")
    run.atomic_json(tmp_path / "frozen.json", {"files": {"pred.csv": run.sha256_file(tmp_path / "pred.csv")}})
    run.verify_manifest(tmp_path / "frozen.json")
    (tmp_path / "pred.csv").write_text("3,4\n")
    with pytest.raises(RuntimeError, match="changed"):
        run.verify_manifest(tmp_path / "frozen.json")


@pytest.mark.parametrize("method", ["source_anchored_shallow", "source_anchored_full"])
def test_deterministic_tiny_fit_and_resume(inputs, monkeypatch, tmp_path, method):
    monkeypatch.setitem(run.CONFIG, "maximum_epochs", 2)
    contract = {"test_fixture": True}
    results = [run.fit(method, 17, inputs[2], inputs[3], inputs[4], inputs[1], tmp_path / str(i), contract) for i in range(2)]
    states = [torch.load(tmp_path / str(i) / "best.pt", weights_only=False)["model"] for i in range(2)]
    assert all(torch.equal(v, states[1][k]) for k, v in states[0].items())
    assert results[0]["source_replay_draws"] == 44
    assert set(results[0]["source_replay_ids"]) <= set(inputs[4])
    assert results[0]["source_head_l2_drift"] == 0
    assert run.fit(method, 17, inputs[2], inputs[3], inputs[4], inputs[1], tmp_path / "0", contract) == results[0]


def test_evaluation_refuses_truth_before_global_freeze(monkeypatch):
    from scripts.studies import evaluate_source_anchored_transfer as ev
    def no_freeze():
        raise RuntimeError("incomplete freeze")
    monkeypatch.setattr(ev, "lock_evaluation", lambda: None)
    monkeypatch.setattr(ev, "validate_global_freeze", no_freeze)
    monkeypatch.setattr(run, "selected_truth", lambda *args: pytest.fail("truth read before freeze"))
    with pytest.raises(RuntimeError, match="incomplete freeze"):
        ev.evaluate()


def test_strata_macro_and_relative_error_contract():
    from scripts.studies import evaluate_source_anchored_transfer as ev
    feature = pd.DataFrame({"canonical_smiles": ["A", "A", "B"], "source_V1": [1., 2., 3.],
                            "source_V2": [1., 2., 3.], "EA_fraction": [.05, .25, .8]})
    truth = np.array([[0., 1.], [2., 2.], [3., 3.]])
    pred = truth+np.array([[1., 1.], [3., 3.], [10., 10.]])
    result, strata = ev.sensitivity(truth, pred, feature, feature, {})
    assert result["V1_macro_compound_mae"] == 6
    assert result["V1_macro_compound_rmse"] == pytest.approx((np.sqrt(5)+10)/2)
    assert result["V1_relative_excluded_rows"] == 1
    assert result["V1_median_relative_absolute_error"] == pytest.approx((1.5+10/3)/2)
    assert len(strata) == 22
    empty = ev.error_summary(np.array([]))
    assert empty["n"] == 0 and np.isnan(empty["rmse"])


def test_material_gate_requires_all_references_and_replication():
    from scripts.studies import evaluate_source_anchored_transfer as ev
    rows = []
    for column in ("25g", "40g"):
        for protocol in ("row", "compound"):
            for reference in ev.CALIBRATION:
                rows.append({"column": column, "protocol": protocol, "method": "standard_full_finetune",
                             "reference": reference, "endpoint": "aulc", "relative_gain": .06,
                             "material": protocol == "compound"})
    frame = pd.DataFrame(rows)
    assert ev.material_gate(frame, "standard_full_finetune", ev.CALIBRATION, "aulc")["replicated"]
    frame.loc[frame.column.eq("25g") & frame.reference.eq("scale_only"), "material"] = False
    assert not ev.material_gate(frame, "standard_full_finetune", ev.CALIBRATION, "aulc")["replicated"]


def test_interrupted_fit_resumes_to_identical_checkpoint(inputs, monkeypatch, tmp_path):
    method, seed, contract = "source_anchored_full", 29, {"fixture": "resume"}
    monkeypatch.setitem(run.CONFIG, "maximum_epochs", 2)
    complete = run.fit(method, seed, inputs[2], inputs[3], inputs[4], inputs[1], tmp_path / "complete", contract)
    monkeypatch.setitem(run.CONFIG, "maximum_epochs", 1)
    run.fit(method, seed, inputs[2], inputs[3], inputs[4], inputs[1], tmp_path / "interrupted", contract)
    (tmp_path / "interrupted" / "fit_summary.json").unlink()
    monkeypatch.setitem(run.CONFIG, "maximum_epochs", 2)
    resumed = run.fit(method, seed, inputs[2], inputs[3], inputs[4], inputs[1], tmp_path / "interrupted", contract)
    a = torch.load(tmp_path / "complete" / "best.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "interrupted" / "best.pt", weights_only=False)["model"]
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    assert complete["source_replay_draws"] == resumed["source_replay_draws"] == 44
    assert complete["source_replay_ids"] == resumed["source_replay_ids"]


def test_completed_experiment_artifacts_and_all_fit_ledgers():
    if not (run.STUDY / "all_predictions_frozen.json").exists():
        pytest.skip("formal fitting has not completed")
    from scripts.studies import evaluate_source_anchored_transfer as ev
    freeze = ev.validate_global_freeze()
    split = pd.read_csv(run.SOURCE_SPLIT)
    source_train = set(split.loc[split.split.eq("train"), "sample_id"])
    for name in freeze["files"]:
        output = (run.STUDY / name).parent
        usage = json.loads((output / "label_usage.json").read_text())
        fits = json.loads((output / "fit_audit.json").read_text())
        assert set(fits) == set(METHODS)
        expected = run.ledger(usage["column"], usage["protocol"], usage["seed"], usage["budget"])
        for key, value in expected.items():
            assert usage[key] == value
        for method, fit in fits.items():
            scope, anchored = METHODS[method]
            assert fit["fit_success"] and fit["source_head_l2_drift"] == 0
            assert fit["trainable_parameters"] == (36387 if scope == "shallow" else 458952)
            assert fit["total_parameters"] == 459726
            assert fit["target_train_count"] == len(usage["gradient_train"])
            assert fit["target_validation_count"] == len(usage["validation"])
            assert fit["source_replay_unique_rows"] == len(fit["source_replay_ids"])
            assert set(fit["source_replay_ids"]) <= source_train
            assert fit["source_replay_draws"] == (fit["epochs_run"]*fit["target_train_count"] if anchored else 0)
            assert fit["contract"]["target_train_ids"] == usage["gradient_train"]
            assert fit["contract"]["target_validation_ids"] == usage["validation"]
    if not (run.STUDY / "execution_audit.json").exists():
        return  # The same ledger audit can run before the first test-truth read.
    metrics = pd.read_csv(run.STUDY / "all_metrics.csv")
    assert len(metrics) == 1080
    assert np.isfinite(metrics[ev.POINT].to_numpy()).all()
    assert len(pd.read_csv(run.STUDY / "source_drift_metrics.csv")) == 480
    assert len(pd.read_csv(run.STUDY / "training_audit.csv")) == 480
