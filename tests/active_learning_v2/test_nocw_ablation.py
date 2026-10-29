"""No paid calls, fits or hidden-label reads. Exercise the actual serialized boundary."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.active_learning_v2 import nocw_study as study
from src.qgeognn_al.active_learning_v2 import nocw_inputs as views
from src.qgeognn_al.active_learning_v2.dialog_bridge import once, read
from src.qgeognn_al.active_learning_v2.protocol import ids_hash, stable_hash, RestrictedLabelStore
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import encode_catalog
from src.qgeognn_al.active_learning_v2.scientist_selector import ReadOnlyCatalog


def raw_catalog():
    conditions = {"PE/EA": "10/1", "Density g/ml": 1., "V/ul": 50.,
                  "loading solvent": "EA", "Volume of loading solvent/ul": 200}
    def row(i):
        return {"id": f"opaque_{i:04d}", "smiles": "CCO" if i % 2 else "CCN", "conditions": conditions.copy()}
    candidates = []
    for i in range(80):
        r = row(i)
        r.update({f"pred_{t}_{q}_ml": float(i+j+.123456789) for t in ("V1", "V2")
                  for j, q in enumerate(("q10", "q50", "q90"))})
        r.update(pred_V1_ml=r["pred_V1_q50_ml"], pred_V2_ml=r["pred_V2_q50_ml"], cw_distance_percentile=i/80)
        candidates.append(r)
    observed = [{**row(i), "V1_ml": 1., "V2_ml": 2., "record_kind": "initial_L333"} for i in range(100, 105)]
    return {"candidates": candidates, "observed": observed, "pending": [], "salt": "legacy:method:0"}


@pytest.fixture
def config():
    c = read(study.STUDY / "config.json")["transport"]
    c["retry_backoff_seconds"] = 0
    return c


def packet(arm=study.METHODS[1]):
    return views.input_view(raw_catalog(), 157, arm, 0, [])


def test_exact_pair_diff_and_shared_order():
    raw = raw_catalog()
    a, ar, ap = views.input_view(raw, 157, study.METHODS[0], 0, [])
    b, br, bp = views.input_view(raw, 157, study.METHODS[1], 0, [])
    diff = views.pair_diff(ap, bp)
    assert len(diff["removed"]) == 80
    assert ar["salt"] == br["salt"]
    assert a.key(next(iter(a.candidates.values()))) == b.key(next(iter(b.candidates.values())))
    assert ap["candidates"]["rows"][0][0] == bp["candidates"]["rows"][0][0]
    assert views.CW_FIELD not in json.dumps(br)
    assert raw == raw_catalog()  # no shared object mutation
    bp["molecules"]["m0"]["descriptors"]["LogP"] += .00001
    with pytest.raises(ValueError, match="unregistered"):
        views.pair_diff(ap, bp)


@pytest.mark.parametrize("injection", [
    {"nested": [{"cw_distance_percentile": .99}]},
    {"memory": {"reason": "CW rank 99"}},
    {"note": "center/width gradient distance 0.5"},
    {"hidden": {"gradient_coverage_score": 3.}},
    {"reason": "LCMD suggests row 7"},
    {"reason": "梯度距离分位数为0.8"},
    {"prompt": "coverage in gradient space is 0.5"},
])
def test_recursive_serialized_leak_audit(injection):
    with pytest.raises(ValueError):
        views.audit_no_cw(injection)


@pytest.mark.parametrize("field", ["nearest_gradient_distance", "coverage_rank", "V1_ml", "numeric_summary", "validation_score"])
def test_candidate_unknown_channels_rejected(field):
    raw = raw_catalog()
    raw["candidates"][0][field] = .5
    with pytest.raises(ValueError, match="unauthorized"):
        views.input_view(raw, 157, study.METHODS[1], 0, [])


def test_foreign_or_contaminated_memory_rejected():
    h = {"seed": 157, "method": study.METHODS[0], "round": 0, "records": [{}]*32,
         "hypotheses": [], "feedback_interpretation": "CW percentile 0.9"}
    with pytest.raises(ValueError, match="foreign"):
        views.validate_history([h], 157, study.METHODS[1], 1)
    h["method"] = study.METHODS[1]
    h["records"] = [{"source": "LLM"}]*32
    with pytest.raises(ValueError, match="NoCW"):
        views.validate_history([h], 157, study.METHODS[1], 1)


@pytest.mark.parametrize("failure", ["count", "duplicate", "observed", "unknown", "split", "evidence", "prose_evidence", "extra", "cw"])
def test_invalid_selection_never_repaired(failure, tmp_path, config):
    c, raw, p = packet()
    response = study.mock_plan(p)
    if failure == "count":
        response["choices"].pop()
    elif failure == "duplicate":
        response["choices"][1] = response["choices"][0]
    elif failure in ("observed", "unknown", "split"):
        response["choices"][0]["id"] = next(iter(c.observed)) if failure == "observed" else failure + "_row"
    elif failure == "evidence":
        response["hypotheses"] = [{"id": "h", "content": "Prediction hypothesis.", "basis": "model_behavior",
            "supporting_observed_ids": [response["choices"][0]["id"]], "contradicting_observed_ids": [],
            "confidence": "low", "status": "proposed", "why_it_matters_for_model_learning": "Test behavior.",
            "next_discriminating_question": "What changes?"}]
    elif failure == "prose_evidence":
        response["choices"][0]["reason"] = "Measured outcome of " + response["choices"][0]["id"]
    elif failure == "extra":
        response["choices"][0]["evidence"] = "untyped data"
    else:
        response["batch_rationale"] = "CW distance 0.9"
    calls = []
    def mock(*args):
        calls.append(True)
        return json.dumps(response), {"mock": True}
    with pytest.raises((ValueError, RuntimeError)):
        study.run_selection(p, c, tmp_path, "no_cw", config, mock)
    assert len(calls) <= 3
    assert not (tmp_path / "selection.json").exists()
    assert not (tmp_path / "batch_freeze.json").exists()


def test_bounded_correction_and_exact_replay(tmp_path, config):
    c, _, p = packet()
    calls = []
    def mock(messages, cfg):
        calls.append(messages)
        if len(calls) == 1:
            return "not json", {"mock": True}
        return json.dumps(study.mock_plan(p)), {"mock": True}
    result = study.run_selection(p, c, tmp_path, "no_cw", config, mock)
    assert len(calls) == 2
    assert len(json.loads(calls[0][1]["content"])["candidates"]["rows"]) == 80
    assert study.run_selection(p, c, tmp_path, "no_cw", config,
        lambda *a: pytest.fail("network during replay")) == result
    for path in tmp_path.glob("request_*.json"):
        views.audit_no_cw(read(path)["messages"])


def test_total_transport_cap_survives_restart(tmp_path, config):
    calls = []
    def bad(*args):
        calls.append(True)
        raise OSError("mock provider failure")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="budget exhausted"):
            study.request_with_retries([], config, tmp_path, 0, bad)
    assert len(calls) == 3


def test_ambiguous_request_fails_closed(tmp_path, config):
    once(tmp_path / "attempt_00_00.started.json", {"request_sha256": stable_hash({"messages": [], "config": config})})
    with pytest.raises(RuntimeError, match="ambiguous"):
        study.request_with_retries([], config, tmp_path, 0, lambda *a: pytest.fail("paid replay"))


def test_complete_catalog_tamper_rejected_before_call(tmp_path, config):
    c, _, p = packet()
    p["candidates"]["rows"].pop()
    with pytest.raises(ValueError, match="complete catalog"):
        study.run_selection(p, c, tmp_path, "no_cw", config, lambda *a: pytest.fail("network"))


def test_legacy_encoding_and_method_salt_unchanged():
    raw = raw_catalog()
    c = ReadOnlyCatalog(**raw)
    before = encode_catalog(c)
    views.input_view(raw, 157, study.METHODS[1], 0, [])
    assert encode_catalog(ReadOnlyCatalog(**raw)) == before
    assert "cw_distance_percentile" in before["candidates"]["columns"]
    assert c.salt == "legacy:method:0"


def test_expensive_actions_need_explicit_authorization(monkeypatch):
    monkeypatch.setattr(study, "Context", lambda *a: pytest.fail("opened labels/models"))
    with pytest.raises(PermissionError):
        study.select(157, study.METHODS[0])
    with pytest.raises(PermissionError):
        study.advance(157, study.METHODS[0])
    with pytest.raises(PermissionError):
        study.preregister()


def test_exact_live_api_configuration_is_frozen_without_network(monkeypatch, config):
    import openai
    from src.qgeognn_al.active_learning_v2 import scientist_transport
    client = MagicMock()
    client.__enter__.return_value = client
    response = SimpleNamespace(status="completed", model="gpt-6.1-sol", output=[],
                               output_text="{}", id="mock", usage=None)
    client.responses.stream.return_value.__enter__.return_value.get_final_response.return_value = response
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(study, "require_authorization", lambda: None)
    monkeypatch.setattr(study, "once", lambda *a: None)
    monkeypatch.setattr(scientist_transport, "provider_key", lambda c: "mock")
    study.live_call([], config)
    kw = client.responses.stream.call_args.kwargs
    assert kw == {"model": "gpt-6.1-sol", "input": [], "reasoning": {"effort": "xhigh"},
                  "tools": [], "store": False, "max_output_tokens": 32768}


def test_real_l333_dry_run_never_opens_truth_network_or_fit(tmp_path, monkeypatch):
    config = read(study.STUDY / "config.json")
    once(tmp_path / "config.json", config)
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "validate", lambda: {"protocol_sha256": "offline_mock"})
    monkeypatch.setattr(RestrictedLabelStore, "reveal", lambda *a: pytest.fail("truth read in dry-run"))
    monkeypatch.setattr(study, "Context", lambda *a: pytest.fail("model context in dry-run"))
    monkeypatch.setattr(study, "live_call", lambda *a: pytest.fail("paid call in dry-run"))
    result = study.dry_run()
    assert result["status"] == "PASS"
    assert result["new_real_labels_revealed"] == result["model_fits"] == result["real_llm_calls"] == 0
    assert not (tmp_path / "runtime").exists()
    for seed in study.SEEDS:
        raw, roles, _ = study.initial_state(seed)
        all_ids = roles["l0"] + roles["u0"]
        context = SimpleNamespace(seed=seed, roles={"l0": np.arange(333), "u0": np.arange(333, 3330)},
            l0_truth=np.array([[x["V1_ml"], x["V2_ml"]] for x in raw["observed"]]),
            ids=lambda indices: [all_ids[int(i)] for i in indices])
        monkeypatch.setattr(study, "selection_directory", lambda s, m, r: tmp_path / f"dry_run/seed_{s}/{m}/round_{r:02d}")
        arm_states = []
        for arm in study.METHODS:
            for r in (0, 1):
                d = tmp_path / f"dry_run/seed_{seed}/{arm}/round_{r:02d}"
                frozen = study.audit_batch(d)
                assert frozen["dry_run"] and len(frozen["batch_ids"]) == 32
            nxt = read(tmp_path / f"dry_run/seed_{seed}/{arm}/round_01/input.json")
            assert nxt["active_label_count"] == 365 and nxt["candidate_count"] == 2965
            d = tmp_path / f"dry_run/seed_{seed}/{arm}/round_00"
            synthetic = read(d / "synthetic_next_state.json")["feedback"][0]
            once(d / "feedback.json", synthetic)
            once(d / "label_access_receipt.json", {"feedback_sha256": stable_hash(synthetic),
                "batch_freeze_sha256": sha256_file(d / "batch_freeze.json"), "audit": {
                    "purpose": "after_acquisition_fit", "acquisitions_frozen": True,
                    "requested_ids_hash": ids_hash([x["candidate_id"] for x in synthetic["records"]])}})
            labeled, unlabeled, values, history = study.state(context, arm, 1)
            assert len(labeled) == 365 and len(unlabeled) == 2965
            assert np.all(values[-32:] == [-123., -122.])
            arm_states.append(set(context.ids(labeled)))
        assert arm_states[0] != arm_states[1]


def test_training_parameters_and_initialization_identical(monkeypatch, tmp_path):
    import pandas as pd
    from src.qgeognn_al.active_learning_v2 import short_sequential_runner as runner
    calls = []
    monkeypatch.setattr(runner, "fit_from_same_initialization", lambda **kw: calls.append(kw) or {})
    context = SimpleNamespace(seed=157, study=tmp_path, protocol_hash="protocol", atom=[], angle=[],
        normalization={"frozen": True}, preprocessing={"target_scales": {"V1": 1., "V2": 2.}},
        validation_truth=np.zeros((2, 2)), roles={"validation": np.array([10, 11]), "test": np.array([12])},
        partition=pd.DataFrame({"role": ["l0", "u0"]}), ids=lambda indices: [f"row_{i}" for i in indices])
    context.training_config = lambda: runner.ShortSequentialContext.training_config(context)
    for arm in study.METHODS:
        runner.ShortSequentialContext.fit(context, arm, 1, np.array([0, 1]), np.zeros((2, 2)), tmp_path / arm)
    assert calls[0]["initialization_seed"] == calls[1]["initialization_seed"] == 2000160
    assert calls[0]["training_config"] == calls[1]["training_config"]
    assert calls[0]["normalization"] == calls[1]["normalization"]
    assert calls[0]["preprocessing"] == calls[1]["preprocessing"]
    assert calls[0]["contract"]["method"] != calls[1]["contract"]["method"]


def test_label_store_rejects_unfrozen_candidate_and_test(tmp_path):
    # Guards check permissions before opening the deliberately nonexistent target file.
    import pandas as pd
    partition = pd.DataFrame({"sample_id": ["observed", "candidate", "test"], "role": ["l0", "u0", "test"]})
    store = RestrictedLabelStore(tmp_path / "never_open.csv", partition)
    with pytest.raises(PermissionError):
        store.reveal(["candidate"], "after_acquisition_fit")
    with pytest.raises(PermissionError):
        store.reveal(["test"], "final_test_evaluation")


def test_report_matrix_barrier_before_truth_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "validate", lambda: None)
    monkeypatch.setattr(study, "Context", lambda *a: pytest.fail("model/truth before complete matrix"))
    with pytest.raises(FileNotFoundError):
        study.report(525)
