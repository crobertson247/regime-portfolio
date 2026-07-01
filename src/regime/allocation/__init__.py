"""
Regime-conditioned allocation module (Phase 4).

Turns the regime labels from Phase 3 and the asset returns from Phase 2 into a
long-only, fully-invested weight series, under the same causal walk-forward
discipline as the rest of the pipeline.

Allocators share the Allocator interface:
- EqualWeightAllocator: 1/N baseline.
- MeanVarianceAllocator: Markowitz QP (with a minimum-variance mode).
- RiskParityAllocator: equal risk contribution.
- MinCVaRAllocator: Rockafellar-Uryasev tail-risk LP (the crisis objective).
- HRPAllocator: hierarchical risk parity (regime-blind robustness benchmark).

RegimeSwitchingAllocator dispatches to one of these per detected regime;
allocate_walk_forward produces the daily held weights.
"""

from regime.allocation.base import Allocator, ConstraintSpec
from regime.allocation.config import AllocationConfig, load_allocation_config
from regime.allocation.cvar import MinCVaRAllocator
from regime.allocation.equal_weight import EqualWeightAllocator
from regime.allocation.estimators import estimate_covariance, estimate_mean, to_simple
from regime.allocation.hrp import HRPAllocator
from regime.allocation.mean_variance import MeanVarianceAllocator
from regime.allocation.regime_switching import RegimeSwitchingAllocator, allocate_walk_forward
from regime.allocation.risk_parity import RiskParityAllocator

ALLOCATORS = {
    "equal_weight": EqualWeightAllocator,
    "mean_variance": MeanVarianceAllocator,
    "risk_parity": RiskParityAllocator,
    "min_cvar": MinCVaRAllocator,
    "hrp": HRPAllocator,
}

__all__ = [
    "Allocator",
    "ConstraintSpec",
    "EqualWeightAllocator",
    "MeanVarianceAllocator",
    "RiskParityAllocator",
    "MinCVaRAllocator",
    "HRPAllocator",
    "RegimeSwitchingAllocator",
    "allocate_walk_forward",
    "AllocationConfig",
    "load_allocation_config",
    "ALLOCATORS",
    "estimate_covariance",
    "estimate_mean",
    "to_simple",
]
