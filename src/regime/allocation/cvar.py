"""
Minimum-CVaR allocator (the crisis-regime objective).

Conditional value-at-risk at level beta is the expected loss in the worst
(1 - beta) tail. Rockafellar and Uryasev show it minimises through an auxiliary
variable alpha (which recovers the VaR at the optimum), turning the problem into
a linear program over the return scenarios:

    min_{w, alpha, u}  alpha + 1 / ((1 - beta) T) * sum_t u_t
    s.t.  u_t >= -r_t' w - alpha,  u_t >= 0,
          sum w = 1,  0 <= w_i <= w_max.

The scenarios r_t are the simple returns in the lookback window (empirical CVaR).
Because it targets the loss tail directly rather than variance, it is the natural
objective once a crisis regime is detected, where returns are fat-tailed and
variance treats upside and downside alike. Solved with the HiGHS LP backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog

from regime.allocation.base import Allocator, ConstraintSpec


class MinCVaRAllocator(Allocator):
    """Long-only minimum-CVaR portfolio over the window's empirical scenarios.

    Args:
        beta: tail confidence level (0.95 means the worst 5% of days).
        constraints: per-asset cap.
    """

    def __init__(self, beta: float = 0.95, constraints: Optional[ConstraintSpec] = None):
        super().__init__(constraints)
        self.beta = beta

    def weights(self, returns: np.ndarray, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        r = np.asarray(returns, dtype=float)
        t_n, n = r.shape
        beta = self.beta
        coef = 1.0 / ((1.0 - beta) * t_n)

        # Decision vector: [ w (n) , alpha (1) , u (T) ].
        c = np.concatenate([np.zeros(n), [1.0], np.full(t_n, coef)])

        # u_t + r_t' w + alpha >= 0  ->  -r_t' w - alpha - u_t <= 0
        a_ub = np.hstack([-r, -np.ones((t_n, 1)), -np.eye(t_n)])
        b_ub = np.zeros(t_n)

        # sum w = 1
        a_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1 + t_n))])
        b_eq = np.array([1.0])

        cap = self.constraints.w_max
        bounds = [(0.0, cap)] * n + [(None, None)] + [(0.0, None)] * t_n

        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        w = res.x[:n] if res.success else self._equal(n)
        return self._finalize(w)
