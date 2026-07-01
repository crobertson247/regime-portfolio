"""
Configuration for the allocation module.

Defaults follow the methodological direction: long-only, fully invested weights;
a one-year estimation window; monthly rebalancing with an extra rebalance on a
regime change; and the regime-to-objective map calm -> mean-variance,
volatile -> risk parity, crisis -> minimum-CVaR. Static baselines (equal weight,
mean-variance, risk parity, HRP, minimum-CVaR) reuse the same allocators with no
regime input. Values can be overridden from config/allocation.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from regime.allocation.base import Allocator, ConstraintSpec
from regime.allocation.cvar import MinCVaRAllocator
from regime.allocation.equal_weight import EqualWeightAllocator
from regime.allocation.hrp import HRPAllocator
from regime.allocation.mean_variance import MeanVarianceAllocator
from regime.allocation.risk_parity import RiskParityAllocator

# Severity rank for each regime name in the three-state setup.
REGIME_RANK = {"calm": 0, "volatile": 1, "crisis": 2}


class AllocationConfig(BaseModel):
    """Parameters for the allocators and the walk-forward weighting."""

    assets: list[str] = Field(
        default_factory=lambda: ["SPY", "TLT", "LQD", "GLD", "DBC", "BIL"]
    )

    lookback: int = Field(default=252, ge=30)
    rebalance_every: int = Field(default=21, ge=1)
    on_regime_change: bool = True

    # Constraints
    w_max: float = Field(default=1.0, gt=0.0, le=1.0)
    turnover_penalty: float = Field(default=0.0, ge=0.0)

    # Mean-variance
    risk_aversion: float = 10.0
    mean_shrinkage: float = Field(default=0.5, ge=0.0, le=1.0)

    # CVaR
    cvar_beta: float = Field(default=0.95, gt=0.5, lt=1.0)

    # HRP
    hrp_linkage: str = "single"

    # Regime objective map (name -> allocator name)
    regime_map: dict[str, str] = Field(
        default_factory=lambda: {
            "calm": "mean_variance",
            "volatile": "risk_parity",
            "crisis": "min_cvar",
        }
    )

    # Static baselines to also produce
    baselines: list[str] = Field(
        default_factory=lambda: [
            "equal_weight",
            "mean_variance",
            "risk_parity",
            "hrp",
            "min_cvar",
        ]
    )

    def constraints(self) -> ConstraintSpec:
        return ConstraintSpec(w_max=self.w_max, turnover_penalty=self.turnover_penalty)

    def build_allocator(self, name: str) -> Allocator:
        """Construct a single allocator by name."""
        c = self.constraints()
        if name == "equal_weight":
            return EqualWeightAllocator(c)
        if name == "mean_variance":
            return MeanVarianceAllocator(
                risk_aversion=self.risk_aversion, mean_shrinkage=self.mean_shrinkage, constraints=c
            )
        if name == "min_variance":
            return MeanVarianceAllocator(min_variance=True, constraints=c)
        if name == "risk_parity":
            return RiskParityAllocator(constraints=c)
        if name == "min_cvar":
            return MinCVaRAllocator(beta=self.cvar_beta, constraints=c)
        if name == "hrp":
            return HRPAllocator(linkage_method=self.hrp_linkage, constraints=c)
        raise ValueError(f"unknown allocator: {name}")

    def regime_allocators(self) -> dict[int, Allocator]:
        """Map severity rank -> allocator from regime_map."""
        return {REGIME_RANK[name]: self.build_allocator(meth) for name, meth in self.regime_map.items()}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AllocationConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Allocation config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw.get("allocation", raw))


def load_allocation_config(path: Optional[str | Path] = None) -> AllocationConfig:
    """Load the allocation config from YAML, or return defaults if path is None."""
    if path is None:
        return AllocationConfig()
    return AllocationConfig.from_yaml(path)
