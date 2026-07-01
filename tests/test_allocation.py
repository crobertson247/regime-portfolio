"""
Tests for the allocation module (Phase 4).

The properties checked are the ones the allocators are supposed to guarantee:
long-only and fully invested weights, equal risk contributions for risk parity,
lower variance for minimum-variance, a lower loss tail for minimum-CVaR, correct
regime dispatch, and a walk-forward harness that is causal and warms up. Where an
allocator estimates moments internally, the synthetic data uses a long window so
the estimate is close to the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.allocation import (
    EqualWeightAllocator,
    HRPAllocator,
    MeanVarianceAllocator,
    MinCVaRAllocator,
    RegimeSwitchingAllocator,
    RiskParityAllocator,
    allocate_walk_forward,
)
from regime.allocation.base import Allocator, ConstraintSpec
from regime.allocation.estimators import estimate_covariance


def random_returns(n_assets=5, t=800, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, scale, (t, n_assets))


def correlated_returns(n_assets=4, t=3000, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_assets, n_assets))
    cov = (a @ a.T) / n_assets * 1e-4
    chol = np.linalg.cholesky(cov)
    return rng.standard_normal((t, n_assets)) @ chol.T


ALLOCATORS = [
    EqualWeightAllocator(),
    MeanVarianceAllocator(),
    MeanVarianceAllocator(min_variance=True),
    RiskParityAllocator(),
    MinCVaRAllocator(),
    HRPAllocator(),
]


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=lambda a: type(a).__name__ + ("_minvar" if getattr(a, "min_variance", False) else ""))
def test_long_only_and_fully_invested(alloc):
    """Every allocator returns non-negative weights that sum to one."""
    w = alloc.weights(random_returns())
    assert w.shape == (5,)
    assert np.all(w >= -1e-9), f"{type(alloc).__name__} produced a short position"
    assert abs(w.sum() - 1.0) < 1e-6


def test_equal_weight_is_uniform():
    w = EqualWeightAllocator().weights(random_returns(n_assets=4))
    assert np.allclose(w, 0.25)


def test_per_asset_cap_respected():
    cap = 0.3
    w = MeanVarianceAllocator(constraints=ConstraintSpec(w_max=cap)).weights(random_returns())
    assert w.max() <= cap + 1e-6


def test_risk_parity_equal_risk_contribution():
    """Risk contributions are equal under the covariance the allocator uses."""
    x = correlated_returns(n_assets=4)
    w = RiskParityAllocator().weights(x)
    sigma = estimate_covariance(x)
    rc = w * (sigma @ w)
    rc = rc / rc.sum()
    assert np.max(np.abs(rc - 0.25)) < 1e-5


def test_min_variance_prefers_low_vol_asset():
    rng = np.random.default_rng(3)
    low = rng.normal(0, 0.004, 4000)
    high = rng.normal(0, 0.02, 4000)
    x = np.column_stack([low, high])
    w = MeanVarianceAllocator(min_variance=True).weights(x)
    assert w[0] > w[1]
    assert w[0] > 0.7


def test_min_cvar_avoids_the_crash_asset():
    """Minimum-CVaR underweights an asset with fat left-tail losses."""
    rng = np.random.default_rng(4)
    t = 1500
    risky = rng.normal(0.0006, 0.01, t)
    risky[rng.choice(t, 40, replace=False)] -= 0.08  # crashes
    safe = rng.normal(0.0002, 0.0006, t)
    x = np.column_stack([risky, safe])
    w = MinCVaRAllocator(beta=0.95).weights(x)
    assert w[1] > 0.8  # piles into the safe asset

    def cvar(weights):
        loss = -(x @ weights)
        var = np.quantile(loss, 0.95)
        return loss[loss >= var].mean()

    assert cvar(w) <= cvar(np.array([0.5, 0.5]))


def test_turnover_penalty_reduces_churn():
    x = random_returns(seed=7)
    prev = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # far from optimum
    free = MeanVarianceAllocator(min_variance=True).weights(x, prev)
    pen = MeanVarianceAllocator(
        min_variance=True, constraints=ConstraintSpec(turnover_penalty=50.0)
    ).weights(x, prev)
    assert np.sum(np.abs(pen - prev)) < np.sum(np.abs(free - prev))


# --- regime switching and the walk-forward harness ---------------------------


class _Const(Allocator):
    """Stub allocator returning fixed weights, for dispatch tests."""

    def __init__(self, vec):
        super().__init__()
        self.vec = np.array(vec, dtype=float)

    def weights(self, returns, prev_weights=None):
        return self.vec


def test_regime_dispatch_picks_mapped_allocator():
    sw = RegimeSwitchingAllocator({0: _Const([1, 0]), 1: _Const([0.5, 0.5]), 2: _Const([0, 1])})
    assert np.allclose(sw.for_regime(0).weights(None), [1, 0])
    assert np.allclose(sw.weights(2, None), [0, 1])


def test_walk_forward_switches_on_regime():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2010-01-01", periods=300, freq="B")
    x = pd.DataFrame(rng.normal(0, 0.01, (300, 2)), index=idx, columns=["A", "B"])
    regimes = pd.Series(0, index=idx, dtype=float)
    regimes.iloc[150:] = 2
    sw = RegimeSwitchingAllocator({0: _Const([1, 0]), 1: _Const([0.5, 0.5]), 2: _Const([0, 1])})
    w = allocate_walk_forward(x, sw.for_regime, regimes, lookback=50, rebalance_every=10)
    assert np.allclose(w.iloc[120].to_numpy(), [1, 0])   # calm regime
    assert np.allclose(w.iloc[-1].to_numpy(), [0, 1])    # crisis regime


def test_walk_forward_warmup_and_validity():
    x = pd.DataFrame(
        random_returns(n_assets=4, t=500),
        index=pd.date_range("2010-01-01", periods=500, freq="B"),
        columns=list("ABCD"),
    )
    alloc = MeanVarianceAllocator(min_variance=True)
    w = allocate_walk_forward(x, lambda r: alloc, None, lookback=100, rebalance_every=20)
    assert w.iloc[:99].isna().all().all()
    assert w.iloc[100:].notna().all().all()
    assert np.allclose(w.dropna().sum(axis=1), 1.0)


def test_walk_forward_is_causal():
    """Held weights up to t are identical whether computed on the full panel or
    on the panel truncated at t."""
    x = pd.DataFrame(
        random_returns(n_assets=4, t=600, seed=9),
        index=pd.date_range("2010-01-01", periods=600, freq="B"),
        columns=list("ABCD"),
    )
    alloc = MeanVarianceAllocator(min_variance=True)
    full = allocate_walk_forward(x, lambda r: alloc, None, lookback=100, rebalance_every=20)
    for t in (250, 400, 575):
        trunc = allocate_walk_forward(x.iloc[: t + 1], lambda r: alloc, None, lookback=100, rebalance_every=20)
        common = full.iloc[: t + 1].dropna().index
        assert np.allclose(full.loc[common].to_numpy(), trunc.loc[common].to_numpy()), f"not causal at {t}"
