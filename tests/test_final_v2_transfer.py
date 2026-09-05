import inspect
from dataclasses import asdict
import pytest
from scripts.studies import run_final_v2_transfer as transfer
from scripts.studies import run_r2_pruned_requalification as old
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.schemas.conditions import ConditionNormalization

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
