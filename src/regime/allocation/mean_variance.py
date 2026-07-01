"""
Mean-variance allocator.

Solves the long-only Markowitz problem

    max_w  mu' w - (gamma / 2) w' Sigma w
    s.t.   sum w = 1,  0 <= w_i <= w_max,

with Ledoit-Wolf covariance and a shrunk mean. gamma is the risk-aversion: large
gamma (or mean_shrinkage = 1) drops the return term and gives the minimum-variance
portfolio, which sidesteps the unstable expected-return estimate entirely. An
optional quadratic turnover term penalises moving away from the previous weights.

The problem is a small convex QP, solved with SLSQP on the simplex.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from regime.allocation.base import Allocator, ConstraintSpec
from regime.allocation.estimators import estimate_covariance, estimate_mean


class MeanVarianceAllocator(Allocator):
    """Long-only mean-variance optimiser.

    Args:
        risk_aversion: gamma in the objective; higher means more weight on risk.
        mean_shrinkage: shrink the sample mean toward zero (1.0 = minimum variance).
        min_variance: if True, ignore the mean term outright (minimum-variance).
        constraints: per-asset cap and turnover penalty.
    """

    def __init__(
        self,
        risk_aversion: float = 10.0,
        mean_shrinkage: float = 0.5,
        min_variance: bool = False,
        constraints: Optional[ConstraintSpec] = None,
    ):
        super().__init__(constraints)
        self.risk_aversion = risk_aversion
        self.mean_shrinkage = mean_shrinkage
        self.min_variance = min_variance

    def weights(self, returns: np.ndarray, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        n = returns.shape[1]
        sigma = estimate_covariance(returns)
        mu = np.zeros(n) if self.min_variance else estimate_mean(returns, self.mean_shrinkage)

        gamma = self.risk_aversion
        tau = self.constraints.turnover_penalty
        w_prev = np.zeros(n) if prev_weights is None else np.asarray(prev_weights, dtype=float)

        def objective(w):
            val = -mu @ w + 0.5 * gamma * (w @ sigma @ w)
            if tau > 0:
                val += tau * np.sum((w - w_prev) ** 2)
            return val

        def gradient(w):
            g = -mu + gamma * (sigma @ w)
            if tau > 0:
                g = g + 2.0 * tau * (w - w_prev)
            return g

        cap = self.constraints.w_max
        w0 = w_prev if prev_weights is not None and np.isclose(w_prev.sum(), 1.0) else self._equal(n)
        res = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, cap)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
            options={"maxiter": 300, "ftol": 1e-12},
        )
        w = res.x if res.success else w0
        return self._finalize(w)
