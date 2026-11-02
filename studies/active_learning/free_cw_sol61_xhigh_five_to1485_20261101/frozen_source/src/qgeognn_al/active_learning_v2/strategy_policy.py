"""Acquisition policy inputs contain observed validation and label-free summaries."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StrategyState:
    active_label_count: int
    candidate_pool_size: int
    recent_validation_nrmse: tuple[float, ...] = ()
    recent_validation_slope: float | None = None
    gradient_effective_rank: float | None = None
    gradient_norm_summary: tuple[float, ...] = ()
    lcmd_coverage_statistic: float | None = None
    ivr_predicted_integrated_variance_reduction: float | None = None


class StrategyPolicy(Protocol):
    def select(self, state: StrategyState) -> str: ...


@dataclass(frozen=True)
class FixedSwitchPolicy:
    before: str = "gradient_lcmd"
    after: str = "kernel_ivr"
    switch_active_labels: int = 653

    def select(self, state: StrategyState) -> str:
        if state.active_label_count < 0 or state.candidate_pool_size < 0:
            raise ValueError("negative label or pool count")
        return self.before if state.active_label_count < self.switch_active_labels else self.after
