"""Tests of real numerical code; toy inputs establish no scientific efficacy."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

STUDY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR / "phase2"))
from lora_adaptation import (ExactMLLHead, LoRAConfig, Projection, RSLoRALinear,
                             attach_native_lora, target_layer_names)
from preflight import ToyEncoder, toy_preflight
from text_embeddings import (EncoderConfig, LocalTextEncoder, embedding_identity,
                              last_token_pool, masked_average_pool, render_experiment_text)


def test_pooling_matches_gollum_and_padding():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [900., 900.]],
                           [[800., 800.], [5., 6.], [7., 8.]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    expected = torch.tensor([[2., 3.], [6., 7.]])
    assert torch.equal(masked_average_pool(hidden, mask), expected)
    assert torch.equal(last_token_pool(hidden, mask), torch.tensor([[3., 4.], [7., 8.]]))
    masked_average_pool(hidden, mask).sum().backward()
    assert torch.equal(hidden.grad[0, 2], torch.zeros(2))
    with pytest.raises(ValueError):
        masked_average_pool(hidden, torch.zeros_like(mask))


def test_lora_zero_initialization_and_official_top_fraction():
    torch.manual_seed(7)
    encoder = ToyEncoder()
    names = target_layer_names(encoder)
    assert names == ["encoder.block.3.linear"]
    inputs = {"input_ids": torch.tensor([[1, 2], [3, 4]]), "attention_mask": torch.ones(2, 2, dtype=torch.long)}
    before = encoder(inputs).detach()
    metadata = attach_native_lora(encoder, LoRAConfig(dropout=0))
    assert metadata["trainable_parameters"] == 64
    assert torch.equal(encoder(inputs), before)
    assert all("lora_" in name for name, p in encoder.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError):
        attach_native_lora(encoder)


def test_exact_mll_feature_gradcheck_with_repeated_rows():
    torch.manual_seed(5)
    features = torch.randn(5, 3, dtype=torch.float64)
    features[1] = features[0]
    features.requires_grad_(True)
    target = torch.tensor([1., 0.8, -0.2, 0.3, -0.5], dtype=torch.float64)
    gp = ExactMLLHead()
    assert torch.autograd.gradcheck(lambda x: gp(x, target), (features,), eps=1e-6, atol=1e-5, rtol=1e-4)
    loss = gp(features, target)
    covariance = gp.covariance(features)
    oracle = -torch.distributions.MultivariateNormal(
        torch.zeros(len(target), dtype=torch.float64), covariance_matrix=covariance).log_prob(target) / len(target)
    assert torch.allclose(loss, oracle, atol=1e-12)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    with pytest.raises(ValueError):
        gp(features[:3], target)


def test_parameter_finite_differences():
    torch.manual_seed(11)
    layer = RSLoRALinear(torch.nn.Linear(3, 4, dtype=torch.float64), LoRAConfig(dropout=0))
    projection = Projection(4, 3)
    gp = ExactMLLHead()
    x = torch.randn(6, 3, dtype=torch.float64)
    y = torch.tensor([-1., 0.2, 0.5, -0.3, 1.1, 0.1], dtype=torch.float64)
    objective = lambda: gp(projection(layer(x)), y)
    objective().backward()
    params = [(layer.lora_B, "lora_B"), (projection.linear.weight, "projection"),
              (gp.log_lengthscale, "lengthscale"), (gp.log_noise_excess, "noise")]
    for parameter, name in params:
        index = int(parameter.grad.abs().argmax())
        analytic = float(parameter.grad.reshape(-1)[index])
        original = float(parameter.detach().reshape(-1)[index])
        values = []
        for direction in (1, -1):
            with torch.no_grad():
                parameter.reshape(-1)[index] = original + direction * 1e-6
            values.append(float(objective().detach()))
        with torch.no_grad():
            parameter.reshape(-1)[index] = original
        numerical = (values[0] - values[1]) / 2e-6
        assert abs(analytic) > 1e-8, name
        assert numerical == pytest.approx(analytic, rel=2e-4, abs=1e-6), name


def test_full_batch_lora_and_projection_gradient_flow():
    result = toy_preflight(steps=3)
    assert len(result["variants"]) == 8
    assert all(all(row["checks"].values()) for row in result["variants"])
    assert not result["performance_claims_allowed"]
    # Zero-B initialization gives zero A gradient on step 1; both update later.
    for row in result["variants"]:
        if "lora" in row["variant"]:
            gradients = row["parameter_groups"]["lora"]["gradients"]
            assert all(value["norm"] > 0 for value in gradients.values())


def test_text_boundary_identity_and_distinct_volumes():
    record = {"canonical_smiles": "CCO", "conditions": {
        "PE/EA": "10/1", "Density g/ml": "0.83", "V/ul": "2.50",
        "loading solvent": "EA", "Volume of loading solvent/ul": "120"}}
    texts = [render_experiment_text(record, template=template) for template in ("A", "B")]
    assert all("2.50" in text and "120" in text and "0.83" in text and "10/1" in text for text in texts)
    assert texts[0] != texts[1]
    with pytest.raises(ValueError):
        render_experiment_text({**record, "V1": 1.0})
    with pytest.raises(ValueError):
        render_experiment_text({**record, "response": 1.0})
    identity = embedding_identity([1, 2], texts, {"revision": "x", "pooling": "mean"})
    assert identity != embedding_identity([2, 1], texts, {"revision": "x", "pooling": "mean"})
    assert identity != embedding_identity([1, 2], texts[::-1], {"revision": "x", "pooling": "mean"})
    with pytest.raises(ValueError):
        embedding_identity([1, 1], texts, {})


def test_offline_flag_is_unconditional(monkeypatch):
    transformers = pytest.importorskip("transformers")
    observed = {}
    def stop(*args, **kwargs):
        observed.update(kwargs)
        raise OSError("Deliberate local-cache miss fixture")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", stop)
    with pytest.raises(OSError, match="local-cache miss"):
        LocalTextEncoder(EncoderConfig())
    assert observed["local_files_only"] is True
    assert observed["trust_remote_code"] is False
    assert observed["revision"] == "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
