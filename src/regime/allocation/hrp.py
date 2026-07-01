"""
Hierarchical risk parity (HRP) allocator.

HRP (Lopez de Prado) allocates without inverting the covariance matrix, so it
stays usable when that matrix is ill-conditioned, the case where mean-variance
breaks down. Three steps: cluster the assets by a correlation distance, reorder
the covariance so similar assets sit together (quasi-diagonalisation), then split
the budget top-down, at each split sharing risk between the two halves in inverse
proportion to their cluster variance. It serves here as a regime-blind robustness
benchmark.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from regime.allocation.base import Allocator, ConstraintSpec
from regime.allocation.estimators import cov_to_corr, estimate_covariance


class HRPAllocator(Allocator):
    """Long-only hierarchical risk parity.

    Args:
        linkage_method: agglomeration rule for the clustering ("single").
        constraints: kept for interface symmetry; HRP is naturally diversified.
    """

    def __init__(self, linkage_method: str = "single", constraints: Optional[ConstraintSpec] = None):
        super().__init__(constraints)
        self.linkage_method = linkage_method

    @staticmethod
    def _quasi_diag(link: np.ndarray) -> list[int]:
        """Leaf order that places similar assets next to each other."""
        link = link.astype(int)
        n = link[-1, 3]  # total number of original items
        order = [link[-1, 0], link[-1, 1]]
        while max(order) >= n:
            new = []
            for item in order:
                if item < n:
                    new.append(item)
                else:
                    row = item - n
                    new.append(link[row, 0])
                    new.append(link[row, 1])
            order = new
        return order

    @staticmethod
    def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
        sub = cov[np.ix_(items, items)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    def _recursive_bisection(self, cov: np.ndarray, order: list[int]) -> np.ndarray:
        n = cov.shape[0]
        w = np.ones(n)
        clusters = [order]
        while clusters:
            clusters = [
                c[half]
                for c in clusters
                for half in (slice(0, len(c) // 2), slice(len(c) // 2, len(c)))
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                left, right = clusters[i], clusters[i + 1]
                var_l = self._cluster_var(cov, left)
                var_r = self._cluster_var(cov, right)
                alpha = 1.0 - var_l / (var_l + var_r)
                w[left] *= alpha
                w[right] *= 1.0 - alpha
        return w

    def weights(self, returns: np.ndarray, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
        cov = estimate_covariance(np.asarray(returns, dtype=float), annualize=False)
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])
        corr = cov_to_corr(cov)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        link = linkage(squareform(dist, checks=False), method=self.linkage_method)
        order = self._quasi_diag(link)
        w = self._recursive_bisection(cov, order)
        return self._finalize(w)
