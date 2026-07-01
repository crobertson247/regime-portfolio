"""
Risk-parity (equal risk contribution) allocator.

Each asset is sized so it contributes the same share of total portfolio
variance, which spreads risk rather than capital and avoids the expected-return
estimate altogether. The weights solve the equal-risk-contribution condition

    w_i * (Sigma w)_i = w_j * (Sigma w)_j   for all i, j,

found here by cyclic coordinate descent: holding the other weights fixed, the
update for asset i is the positive root of a_ii x^2 + c_i x - b_i = 0, where c_i
is its covariance with the rest of the portfolio. The iteration converges for a
positive-definite covariance and the result is normalised to sum to one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from regime.allocation.base import Allocator, ConstraintSpec
from regime.allocation.estimators import estimate_covariance


class RiskParityAllocator(Allocator):
    """Long-only equal-risk-contribution portfolio.

    Args:
        max_iter: coordinate-descent sweeps.
        tol: stop when the weights move by less than this in a sweep.
        constraints: per-asset cap (applied as a final projection).
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-8,
        constraints: Optional[ConstraintSpec] = None,
    ):
        super().__init__(constraints)
        self.max_iter = max_iter
        self.tol = tol

    def weights(self, returns: np.ndarray, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        sigma = estimate_covariance(np.asarray(returns, dtype=float))
        n = sigma.shape[0]
        budget = np.full(n, 1.0 / n)  # equal risk budget

        x = np.full(n, 1.0 / n)
        for _ in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                a = sigma[i, i]
                c = sigma[i] @ x - a * x[i]  # covariance with the rest
                if a <= 0:
                    continue
                x[i] = (-c + np.sqrt(c * c + 4.0 * a * budget[i])) / (2.0 * a)
            if np.max(np.abs(x - x_old)) < self.tol:
                break

        w = x / x.sum()
        cap = self.constraints.w_max
        if cap < 1.0 and np.any(w > cap):
            w = np.minimum(w, cap)
        return self._finalize(w)
