"""Synthetic label and provider fixtures exercise the real A7 loop and receipts."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.qgeognn_al.active_learning_v2.runner import write_predictions
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import encode_catalog
from src.qgeognn_al.active_learning_v2.scientist_selector import ReadOnlyCatalog
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.context_builder import decode_context
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.frozen_state import FrozenState
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import load_4g_metadata
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportFailure, TransportStop,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    FormalContract, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.seed_runner import (
    FIT_ARTIFACTS, PROMPT_PATH, _approved_assets, _initial_state, qualification_status, reconcile_attempt,
    round_directory, run_seed, seed_directory,
)

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MOCK_PREDICTIONS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


class SyntheticBackend:
    is_mock = True

    def __init__(self, seed, metadata):
        self.seed = seed
        self.metadata = metadata
        ids = sorted(metadata)
        self.l0_ids, self.u0_ids = ids[:2], ids[2:74]
        self.l0_truth = [[2.0, 5.0], [3.0, 6.0]]
        self.split_sha256 = digest({"seed": seed, "roles": [self.l0_ids, self.u0_ids]})
        self.cache_receipt_sha256 = digest({"purpose": "synthetic"})
        self.reveal_calls = 0
        self.fit_calls = 0
        self.optimizer_runs = 0
        self.initial_calls = 0
        self.fail_before_fit = False
        self.fail_after_fit = False
        self.fail_after_optimizer = False

    def _write_fit(self, fit_dir, labeled, unlabeled):
        checkpoint, predictions = fit_dir / "best.pt", fit_dir / "predictions.csv.gz"
        checkpoint.write_bytes(f"synthetic-checkpoint-{self.seed}-{len(labeled)}".encode())
        write_predictions(predictions, unlabeled, np.tile(MOCK_PREDICTIONS, (len(unlabeled), 1)))
        metrics = {"best_epoch": 1, "checkpoint_sha256": file_digest(checkpoint), "combined_normalized_rmse": 0.4}
        once(fit_dir / "validation_metrics.json", metrics)
        return {"checkpoint_sha256": metrics["checkpoint_sha256"],
                "prediction_sha256": file_digest(predictions), "train_ids_hash": digest(sorted(labeled)),
                "train_rows": len(labeled), "prediction_rows": len(unlabeled),
                "test_labels_used_for_fit_or_checkpoint_selection": 0,
                "best_epoch": metrics["best_epoch"], "best_validation_combined_normalized_rmse": 0.4}

    def initial_state(self, anchor_run, fit_dir, *, allow_fresh_fit=False):
        assert anchor_run is None and allow_fresh_fit
        self.initial_calls += 1
        fit = self._write_fit(fit_dir, self.l0_ids, self.u0_ids)
        once(fit_dir / "fit_audit.json", fit)
        receipt = {**fit, "validation_metrics_sha256": file_digest(fit_dir / "validation_metrics.json")}
        once(fit_dir / "initial_state_receipt.json", receipt)

        def card(key):
            item = self.metadata[key]
            raw = item.public_conditions()
            return {"id": key, "smiles": item.canonical_smiles, "conditions": {
                "PE/EA": raw["PE/EA"], "Density g/ml": float(raw["density"]),
                "V/ul": float(raw["sample_volume"]), "loading solvent": raw["loading_solvent"],
                "Volume of loading solvent/ul": float(raw["loading_solvent_volume"]),
            }}

        targets = dict(zip(("V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"),
                           MOCK_PREDICTIONS, strict=True))
        candidates = [{**card(key), **{f"pred_{name}_ml": val for name, val in targets.items()},
                       "pred_V1_ml": 2.0, "pred_V2_ml": 5.0} for key in self.u0_ids]
        observed = [{**card(key), "V1_ml": value[0], "V2_ml": value[1], "record_kind": "initial_L333"}
                    for key, value in zip(self.l0_ids, self.l0_truth, strict=True)]
        packet = {"seed": self.seed, "round": 0, "study_version": "pure_llm_v3",
                  "active_label_count": len(observed), "selection_count": 32,
                  "objective": "synthetic test only", "validation_test_records": 0,
                  "delivery": "all_candidates_one_request", "candidate_count": len(candidates),
                  "observed_count": len(observed),
                  **encode_catalog(ReadOnlyCatalog(candidates, observed, [], salt="mock")),
                  "memory": {"previous_hypotheses": [], "previous_feedback_interpretation": "",
                             "previous_batch_rationale": "", "previous_selected_ids": [],
                             "recent_outcomes": [], "high_error_measured_examples": [], "memory_policy": "test"}}
        packet["packet_sha256"] = digest(packet)
        provenance = {"split_sha256": self.split_sha256, "expected_observed": self.l0_ids,
                      "expected_candidates": self.u0_ids, "resource_receipt_sha256": self.cache_receipt_sha256,
                      "initial_state_receipt_sha256": digest(receipt)}
        return FrozenState(self.seed, 0, fit["checkpoint_sha256"], packet, [], [], provenance)

    def reveal_batch(self, prior_ids, batch_ids):
        assert len(batch_ids) == 32 and set(batch_ids) <= set(self.u0_ids) - set(prior_ids)
        self.reveal_calls += 1
        audit = {"purpose": "after_acquisition_fit", "requested_rows": len(batch_ids),
                 "requested_ids_hash": digest(sorted(batch_ids)), "acquisitions_frozen": True, "roles": "u0"}
        return np.tile([3.0, 6.0], (len(batch_ids), 1)), audit

    def fit_next(self, *, labeled_ids, labeled_truth, unlabeled_ids, round_index, freeze_sha256, fit_dir):
        assert len(labeled_truth) == len(labeled_ids) == 2 + 32 * round_index
        assert set(labeled_ids) & set(unlabeled_ids) == set() and len(freeze_sha256) == 64
        self.fit_calls += 1
        if self.fail_before_fit:
            self.fail_before_fit = False
            (fit_dir / "history.csv").write_text("synthetic interrupted attempt\n")
            raise RuntimeError("interrupted mock fit")
        if not (fit_dir / "training_complete.json").exists():
            self.optimizer_runs += 1
            once(fit_dir / "training_complete.json", {"round_index": round_index, "freeze_sha256": freeze_sha256})
        if self.fail_after_optimizer:
            self.fail_after_optimizer = False
            raise RuntimeError("mock optimizer completed before interruption")
        once(fit_dir / "fit_audit.json", self._write_fit(fit_dir, labeled_ids, unlabeled_ids))
        once(fit_dir / "fit_device.json", {"device": "synthetic"})
        (fit_dir / "history.csv").write_text("synthetic successful fit\n")
        once(fit_dir / "fit_artifact_receipt.json", {name: file_digest(fit_dir / name) for name in FIT_ARTIFACTS})
        if self.fail_after_fit:
            self.fail_after_fit = False
            raise RuntimeError("mock fit completed before interruption")


def choice(card):
    return {"candidate_id": card["id"], "molecule_id": card["molecule_id"],
            "condition_anchor": {"PE/EA": card["conditions"]["PE/EA"]}, "reason": "Synthetic plan.",
            "hypothesis_id": None, "references": []}


class MockProvider:
    def __init__(self, *, invalid_first=False, pause_after=None):
        self.requests = []
        self.invalid_first = invalid_first
        self.pause_after = pause_after

    def __call__(self, request, config):
        self.requests.append(request["request_id"])
        inputs = request["payload"]["input"]
        packet = json.loads(inputs[1]["content"])
        cards = list(decode_context(packet)["candidates"].values())
        if request["logical_index"] == 0:
            choices = [choice(card) for card in cards[:32]]
            if self.invalid_first:
                choices[7]["candidate_id"] = "C-9999"
            answer = {"choices": choices, "hypothesis_updates": [], "batch_strategy": "Synthetic fixture.",
                      "feedback_interpretation": "", "batch_review": "Checked synthetic plan."}
        else:
            repair = json.loads("{" + inputs[-1]["content"].split("\n{", 1)[1])
            answer = {"mode": "patch", "reason": "Repair synthetic selection", "repair_id": repair["repair_id"],
                      "replace_choices": [{"slot": 7, "choice": choice(cards[32])}], "replace_hypothesis_updates": [],
                      "append_choices": [], "replace_fields": {}}
        if self.pause_after is not None:
            self.pause_after.touch()
        return ProviderReply(json.dumps(answer), f"mock-{request['request_id']}", config.served_model, {"input_tokens": 1})


@pytest.fixture(scope="module")
def metadata():
    return load_4g_metadata(ROOT)


@pytest.fixture
def formal():
    profile = {"confirmation_seeds": [157, 887], "initial_labels": 2, "target_labels": 66,
               "batch_size": 32, "qualification_seed": 157, "api_parallelism": 3, "device": "cuda"}
    transport = TransportConfig("mock", "mock-model", "mock-model", "ultra", "https://example.com",
                                timeout_seconds=1, max_output_tokens=2048)
    contract = FormalContract(profile, transport, {}, {})
    authorization = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": contract.digest,
                     "authorized_by": "synthetic test", "authorization_reference": "fixture only",
                     "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True}
    return contract, authorization


def execute(run, formal, backend, provider, **kwargs):
    contract, authorization = formal
    return run_seed(root=ROOT, run_directory=run, contract=contract, authorization=authorization, seed=backend.seed,
                    backend=backend, call=provider, mock=True, **kwargs)


def test_qualification_and_freeze_resume_with_local_repair(tmp_path, metadata, formal):
    run = tmp_path / "run"
    gated, first = SyntheticBackend(887, metadata), SyntheticBackend(157, metadata)
    provider = MockProvider(invalid_first=True)
    assert execute(run, formal, gated, provider).status == "awaiting_qualification"
    assert gated.initial_calls == len(provider.requests) == 0
    first.fail_before_fit = True
    with pytest.raises(RuntimeError, match="interrupted mock fit"):
        execute(run, formal, first, provider, max_rounds=1)
    assert len(provider.requests) == 2 and first.reveal_calls == first.fit_calls == 1
    assert qualification_status(run, *formal, mock=True) is None
    recovered = execute(run, formal, first, provider, max_rounds=1)
    assert (recovered.status, recovered.completed_rounds, recovered.active_labels) == ("yielded", 1, 34)
    assert len(provider.requests) == 2 and first.reveal_calls == first.initial_calls == 1 and first.fit_calls == 2
    seed = seed_directory(run, 157)
    assert (round_directory(seed, 1) / "fit_000" / "history.csv").exists()
    assert read(round_directory(seed, 1) / "transition.json")["fit_path"] == "fit_001"
    assert read(round_directory(seed, 0) / "batch_freeze.json")["accepted_logical_index"] == 1
    assert qualification_status(run, *formal, mock=True)["status"] == "qualified"
    assert execute(run, formal, gated, MockProvider(), max_rounds=1).completed_rounds == 1
    assert gated.reveal_calls == 1 and first.reveal_calls == 1


def test_safe_pause_and_completed_prefix_resume(tmp_path, metadata, formal, monkeypatch):
    run = tmp_path / "run"
    seed = SyntheticBackend(157, metadata)
    is_fitting = False

    def forbid_historical_loader(*_args, **_kwargs):
        pytest.fail("formal seed loop attempted to load a historical LLM runtime")

    legacy = "studies.active_learning.pure_llm_v3_20261124.pure_llm_v3"
    monkeypatch.setattr(f"{legacy}.runner.load_historical", forbid_historical_loader)
    monkeypatch.setattr(f"{legacy}.trajectory.load_historical", forbid_historical_loader)

    @contextmanager
    def fit_slot():
        nonlocal is_fitting
        assert not is_fitting
        is_fitting = True
        try:
            yield
        finally:
            is_fitting = False

    provider = MockProvider(pause_after=run / "pause.request")

    def wrapped_call(request, config):
        assert not is_fitting
        return provider(request, config)

    paused = execute(run, formal, seed, wrapped_call, fit_slot=fit_slot)
    assert paused.status == "paused" and paused.completed_rounds == 0
    assert seed.reveal_calls == seed.fit_calls == 0 and len(provider.requests) == 1
    assert (round_directory(seed_directory(run, 157), 0) / "batch_freeze.json").exists()
    (run / "pause.request").unlink()
    provider.pause_after = None
    complete = execute(run, formal, seed, wrapped_call, fit_slot=fit_slot)
    assert (complete.status, complete.completed_rounds, complete.active_labels) == ("complete", 2, 66)
    assert seed.reveal_calls == seed.fit_calls == len(provider.requests) == 2
    assert execute(run, formal, seed, wrapped_call).status == "complete"
    assert seed.reveal_calls == len(provider.requests) == 2
    broken_metrics = round_directory(seed_directory(run, 157), 2) / "fit_000" / "validation_metrics.json"
    broken_metrics.write_text("{}")
    with pytest.raises((KeyError, ValueError)):
        execute(run, formal, seed, wrapped_call)
    assert seed.fit_calls == len(provider.requests) == 2


def test_ambiguous_transport_requires_bound_reconciliation(tmp_path, metadata, formal):
    backend = SyntheticBackend(157, metadata)

    def timeout(*_):
        raise TransportFailure("timeout", ambiguous=True)

    with pytest.raises(TransportStop, match="timeout"):
        execute(tmp_path, formal, backend, timeout, max_rounds=1)
    with pytest.raises(TransportStop, match="timeout"):
        execute(tmp_path, formal, backend, MockProvider(), max_rounds=1)
    assert backend.reveal_calls == 0
    round_dir = round_directory(seed_directory(tmp_path, 157), 0)
    turn = round_dir / "turn_00"
    started = read(turn / "attempt_00.started.json")
    recovery = {"request_id": started["request_id"], "attempt_id": started["attempt_id"],
                "decision": "confirmed_not_processed", "authorized_by": "synthetic auditor",
                "evidence": "synthetic dispatch was not processed"}
    contract, authorization = formal
    reconcile_attempt(run_directory=tmp_path, contract=contract, authorization=authorization, seed=157,
                      round_index=0, logical_index=0, attempt=0, reconciliation=recovery, mock=True)
    provider = MockProvider()
    assert execute(tmp_path, formal, backend, provider, max_rounds=1).completed_rounds == 1
    assert len(provider.requests) == backend.reveal_calls == 1


def test_authorization_refuses_mismatch_before_any_model_work(tmp_path, metadata, formal):
    contract, authorization = formal
    backend = SyntheticBackend(157, metadata)
    provider = MockProvider()
    with pytest.raises(PermissionError, match="exact approved contract"):
        execute(tmp_path, (contract, {**authorization, "authorize_label_reveal": False}), backend, provider)
    assert backend.initial_calls == backend.reveal_calls == backend.fit_calls == 0
    assert not provider.requests


def test_completed_fit_and_physical_credential_replay(tmp_path, metadata, formal):
    backend = SyntheticBackend(157, metadata)
    backend.fail_after_fit = True
    provider = MockProvider()
    leases = []

    @contextmanager
    def attempt_slot(request, _config):
        leases.append(request["request_id"])
        yield "mock-slot-a"

    options = {"attempt_slot": attempt_slot}
    with pytest.raises(RuntimeError, match="fit completed before interruption"):
        execute(tmp_path, formal, backend, provider, max_rounds=1, transport_options=options)
    assert len(leases) == len(provider.requests) == backend.fit_calls == 1
    attempt = round_directory(seed_directory(tmp_path, 157), 0) / "turn_00" / "attempt_00.started.json"
    assert read(attempt)["credential_id"] == "mock-slot-a"
    assert execute(tmp_path, formal, backend, provider, max_rounds=1, transport_options=options).completed_rounds == 1
    assert len(leases) == len(provider.requests) == backend.reveal_calls == backend.fit_calls == 1


def test_optimizer_seal_resumes_prediction_in_same_fit_directory(tmp_path, metadata, formal):
    backend = SyntheticBackend(157, metadata)
    backend.fail_after_optimizer = True
    provider = MockProvider()
    with pytest.raises(RuntimeError, match="optimizer completed before interruption"):
        execute(tmp_path, formal, backend, provider, max_rounds=1)
    assert backend.optimizer_runs == 1
    assert execute(tmp_path, formal, backend, provider, max_rounds=1).completed_rounds == 1
    assert backend.fit_calls == 2 and backend.optimizer_runs == backend.reveal_calls == len(provider.requests) == 1
    transition = round_directory(seed_directory(tmp_path, 157), 1) / "transition.json"
    assert read(transition)["fit_path"] == "fit_000"


def test_active_run_rejects_prompt_and_metadata_edits_against_frozen_digests(tmp_path, formal):
    prompt = tmp_path / PROMPT_PATH
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Frozen A7 test prompt.")
    asset = tmp_path / "metadata_fixture.csv"
    asset.write_text("fixture,synthetic\n")
    contract = replace(formal[0], source_sha256={PROMPT_PATH.as_posix(): file_digest(prompt)},
                       asset_sha256={asset.name: file_digest(asset)})
    assert _approved_assets(tmp_path, contract, {"mock": False}) == "Frozen A7 test prompt."
    prompt.write_text("Different scientific goal.")
    with pytest.raises(ValueError, match="authorized A7 system prompt changed"):
        _approved_assets(tmp_path, contract, {"mock": False})
    asset.write_text("fixture,altered\n")
    with pytest.raises(ValueError, match="formal data, split or preprocessing asset changed"):
        _approved_assets(tmp_path, contract, {"mock": False})


@pytest.mark.parametrize(("seed", "predecessor"), [
    (157, {"seed_157_initial_checkpoint_sha256": "f" * 64}),
    (887, {"initial_checkpoint_sha256_by_seed": {"887": "f" * 64}}),
])
def test_seed_rejects_initial_checkpoint_that_differs_from_predecessor(tmp_path, metadata, formal, seed, predecessor):
    backend = SyntheticBackend(seed, metadata)
    contract = replace(formal[0], predecessor=predecessor)
    with pytest.raises(ValueError, match="checkpoint differs from the authenticated predecessor"):
        _initial_state(tmp_path, backend, contract, {"seed": seed, "mock": False}, None, nullcontext)
    assert not (tmp_path / "initial" / "source.json").exists()
    assert not (tmp_path / "initial" / "initial.json").exists()
    assert backend.reveal_calls == backend.fit_calls == 0
