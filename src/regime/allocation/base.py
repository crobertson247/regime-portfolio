"""
Common interface for portfolio allocators (Phase 4).

An allocator turns a window of recent asset returns into a vector of portfolio
weights. Every allocator here is long-only and fully invested: weights are
non-negative and sum to one, so cash (the BIL holding) is just another asset
that absorbs any de-risking. Shared constraints, a per-asset cap and an optional
turnover penalty against the previous weights, live in ConstraintSpec so each
allocator applies them the same way.

The weight at a rebalance date uses only returns up to that date, mirroring the
causal discipline of the detection module: an allocation never sees the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConstraintSpec:
    """Weight constraints shared by the allocators.

    Args:
        w_max: per-asset cap (1.0 means no cap).
        turnover_penalty: weight on a quadratic ||w - w_prev||^2 term, used by
            the optimiser-based allocators to discourage churn between rebalances.
    """

    w_max: float = 1.0
    turnover_penalty: float = 0.0


class Allocator(ABC):
    """Base class for portfolio allocators.

    Subclasses implement weights(): given a (T, N) window of simple asset
    returns, return an N-vector of long-only weights that sum to one. prev_weights
    is the portfolio held going into the rebalance, used only by allocators that
    penalise turnover.
    """

    def __init__(self, constraints: Optional[ConstraintSpec] = None):
        self.constraints = constraints or ConstraintSpec()

    @abstractmethod
    def weights(self, returns: np.ndarray, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Long-only, sum-to-one weights for a (T, N) window of simple returns."""

    def _finalize(self, w: np.ndarray) -> np.ndarray:
        """Clean small numerical violations and renormalise to sum to one."""
        w = np.asarray(w, dtype=float).ravel()
        w[~np.isfinite(w)] = 0.0
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0:
            return np.full(len(w), 1.0 / len(w))
        return w / total

    @staticmethod
    def _equal(n: int) -> np.ndarray:
        return np.full(n, 1.0 / n)
