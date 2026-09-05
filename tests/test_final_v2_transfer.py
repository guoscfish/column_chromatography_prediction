import inspect
from dataclasses import asdict
import pytest
from scripts.studies import run_final_v2_transfer as transfer
from scripts.studies import run_r2_pruned_requalification as old
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.schemas.conditions import ConditionNormalization
import numpy as np
import pandas as pd

@pytest.fixture(scope="module")
def inputs():
    return old.inputs()

def test_transfer_scopes_and_test_isolation(inputs):
    *_, cn, norm, scales, batches = inputs
    model = build_predictor(ConditionNormalization(**asdict(norm)))
    counts = {}
    for mode in ("head_only", "last2", "full"):
        counts[mode] = transfer.configure_trainable(model, mode)[0]
    assert counts == {"head_only": 774, "last2": 126735, "full": 458952}
    source = inspect.getsource(transfer.train_adaptation)
    assert "test" not in inspect.signature(transfer.train_adaptation).parameters
    assert "test_idx" not in source and "test_truth" not in source
    assert transfer.METHODS == (
        "zero_shot", "affine", "target_head_only", "last2", "full_finetune"
    )
    assert transfer.CONFIG["target_normalization"] == "source_4g_train_only"
    assert transfer.CONFIG["active_acquisition"] is False


def test_transfer_masks_unrevealed_truth_and_evaluates_after_fit():
    source = inspect.getsource(transfer.run_context)
    assert 'fitting_data[["V1_ml", "V2_ml"]] = 0.' in source
    assert source.index("train_adaptation(") < source.index("test_truth =")
    assert source.index("train_adaptation(") < source.index("test_six =")


def test_affine_fits_target_training_rows_only():
    x = np.array([[1., 2.], [2., 4.], [3., 6.]])
    y = x * [2., 3.] + [1., -2.]
    target_x = np.array([[4., 8.]])
    np.testing.assert_allclose(transfer.affine_fit(y, x, target_x), target_x * [2., 3.] + [1., -2.])


def test_quantile_audit_keeps_crossing_and_signed_width():
    from src.qgeognn_al.uncertainty.quantiles import quantile_metrics
    frame = pd.DataFrame({"V1_true": [0., 1., 2., 3., 4.], "V1_q10": [-1., 0., 3., 2., 3.],
                          "V1_q50": [0., 1., 2., 3., 5.], "V1_q90": [1., 2., 1., 4., 7.]})
    result = quantile_metrics(frame, "V1")
    assert result["crossing_rate"] == .2
    assert result["negative_width_rate"] == .2
    assert result["empirical_coverage"] == .8
    assert result["top_uncertainty_rows"] == 1
    assert result["q50_mae"] == .2


def test_completed_transfer_evidence_and_label_contract():
    import json
    from src.qgeognn_al.artifacts import sha256_file
    root = transfer.STUDY
    completions = list((root / 'results').glob('seed_*/budget_*/completion.json'))
    assert len(completions) == 20
    protocol = json.loads((root / 'protocol.json').read_text())
    for completion_path in completions:
        record = json.loads(completion_path.read_text())
        for filename, digest in record['files'].items():
            assert sha256_file(completion_path.parent / filename) == digest
        usage = json.loads((completion_path.parent / 'label_usage.json').read_text())
        assert usage['train_rows'] == record['contract']['budget'] - 8
        assert usage['validation_rows'] == 8 and usage['test_rows'] == 58
        assert usage['test_rows_used_for_fit'] == usage['test_rows_used_for_checkpoint_selection'] == usage['target_rows_used_for_preprocessing_fit'] == 0
        fits = json.loads((completion_path.parent / 'fit_audit.json').read_text())
        assert set(fits) == set(transfer.NEURAL)
    table = pd.read_csv(root / 'results/per_seed_budget_metrics.csv')
    assert len(table) == 100
    assert np.isfinite(table.select_dtypes(include='number')).all().all()
    decision = json.loads((root / 'decision.json').read_text())
    assert decision['source_checkpoint_sha256'] == protocol['source_checkpoint_sha256']
    assert decision['active_transfer_executed'] is False
