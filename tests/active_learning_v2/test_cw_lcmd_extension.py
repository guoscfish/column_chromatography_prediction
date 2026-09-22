from __future__ import annotations
import numpy as np
import pytest
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import select_current_cw_lcmd
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_study import ACTIVE_LABEL_BUDGETS, FINAL_ACTIVE_LABELS, METHOD, NEW_ACQUISITION_ROUNDS
def test_extension_scope_and_hard_stop():
    assert METHOD == "center_width_lcmd" and NEW_ACQUISITION_ROUNDS == (4, 5, 6)
    assert ACTIVE_LABEL_BUDGETS[-1] == FINAL_ACTIVE_LABELS == 525 and 557 not in ACTIVE_LABEL_BUDGETS
def test_current_lcmd_uses_all_current_centers(monkeypatch):
    seen = {}
    class R: selected_pool_positions = np.arange(32); trace = []
    def fake(pool, centers, batch): seen.update(pool=len(pool), centers=len(centers), batch=batch); return R()
    monkeypatch.setattr("src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner.lcmd_tp_select", fake)
    result = select_current_cw_lcmd(np.zeros((429 + 100, 512)), 429)
    assert seen == {"pool": 100, "centers": 429, "batch": 32}; assert len(result["selected_pool_positions"]) == 32
def test_selection_rejects_invalid_shape():
    with pytest.raises(ValueError): select_current_cw_lcmd(np.zeros((429, 511)), 429)
