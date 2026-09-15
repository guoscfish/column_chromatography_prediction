import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.studies.run_qgeognn_v2_row_hybrid_extension import (
    BATCH_SIZE,
    CONFIRMATION_SEEDS,
    SHORTLIST_FRACTION,
    _safe_write_csv,
    v2_hybrid_select,
)


def _inputs(rows=2997):
    rng = np.random.default_rng(62)
    representation = rng.normal(size=(333 + rows, 128))
    predictions = rng.normal(size=(3, rows, 6))
    return representation, predictions


def test_hybrid_shortlist_is_ceil_exact_top_quarter_with_canonical_ties():
    representation, prediction = _inputs(rows=13)
    prediction[:] = 0
    prediction[:, :, 1] = np.array([[0] * 13, [0] * 13, list(range(13))])
    result = v2_hybrid_select(representation, prediction, 333, (1.0, 1.0), 3)
    assert result["shortlist_size"] == 4
    # Descending uncertainty, then stable source/canonical U0 position for tied scores.
    assert result["shortlist_positions"].tolist() == [12, 11, 10, 9]
    assert SHORTLIST_FRACTION == 0.25


def test_hybrid_selects_exactly_32_only_from_shortlist_and_is_deterministic():
    representation, prediction = _inputs()
    first = v2_hybrid_select(representation, prediction, 333, (7.0, 11.0))
    second = v2_hybrid_select(representation, prediction, 333, (7.0, 11.0))
    assert first["shortlist_size"] == 750
    assert len(first["selected_positions"]) == BATCH_SIZE == 32
    assert len(set(first["selected_positions"])) == BATCH_SIZE
    assert set(first["selected_positions"]) <= set(first["shortlist_positions"])
    for key in ("selected_positions", "shortlist_positions", "uncertainty_ranking", "uncertainty_scores"):
        assert np.array_equal(first[key], second[key])


def test_hybrid_acquisition_surface_has_no_target_or_label_argument():
    parameters = inspect.signature(v2_hybrid_select).parameters
    assert not {"labels", "targets", "truth", "test"} & set(parameters)


def test_extension_uses_current_v2_extract_representation_and_keeps_old_cohort():
    source = Path("scripts/studies/run_qgeognn_v2_row_hybrid_extension.py").read_text()
    assert "extract_representations(model, atom, angle, outer)" in source
    assert "model.extract_representation" not in source  # only the current V2 adapter owns that call
    assert CONFIRMATION_SEEDS == (157, 887, 2357, 6101, 12203)


def test_reuse_audit_is_strict_and_compares_initialization_and_cache_contracts():
    source = Path("scripts/studies/run_qgeognn_v2_row_hybrid_extension.py").read_text()
    assert "verify_cache(runtime / \"scrubbed_graphs.pt\", graph_contract)" in source
    assert "verify_cache(gradient_path, gradient_contract)" in source
    assert "old evaluation initialization differs from the frozen V2 rule" in source
    assert "test_labels_used_for_fit_or_checkpoint_selection" in source


def test_safe_csv_resume_accepts_float_text_roundtrip_but_not_identity_change(tmp_path):
    path = tmp_path / "selection.csv"
    first = pd.DataFrame({"sample_id": ["a", "b"], "uncertainty_score": [1.0 / 3.0, 2.0 / 3.0]})
    _safe_write_csv(first, path)
    _safe_write_csv(first.copy(), path)
    changed = first.copy(); changed.loc[0, "sample_id"] = "other"
    try:
        _safe_write_csv(changed, path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("identity change must not overwrite a frozen CSV")


def test_primary_metric_contract_uses_named_endpoint_scale_mapping():
    from src.qgeognn_al.active_learning_v2.benchmark_reporting import metric_row

    truth = np.array([[1.0, 2.0], [3.0, 6.0]])
    prediction = np.array([[1.0, 2.0, 1.0, 2.0, 2.0, 3.0], [2.0, 3.0, 5.0, 6.0, 4.0, 6.0]])
    result = metric_row(truth, prediction, {"V1": 1.0, "V2": 2.0})
    assert result["combined_normalized_RMSE"] > 0
