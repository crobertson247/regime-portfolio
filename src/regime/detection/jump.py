"""
Statistical jump model regime detector.

A jump model is k-means with a penalty for changing state, so it favours fewer,
longer regimes. It minimises

    sum_t ||z_t - mu_{s_t}||^2  +  lambda * sum_t 1[s_t != s_{t-1}]

over the state means mu and the sequence s. The penalty lambda sets persistence:
at lambda = 0 it is plain k-means, and larger values give longer regimes.
Fitting alternates a dynamic-programming assignment of s with a centroid update,
from a k-means start.

Filtered (causal) labels use the forward DP on data up to day t; smoothed
(full-sample) labels use the same DP with back-tracking. States are ordered
calm < volatile < crisis by their mean over the stress features.

References: Nystrup, Lindstrom and Madsen; Shu, Yu and Mulvey.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from regime.detection.base import RegimeDetector
from regime.utils.logging import get_logger

logger = get_logger(__name__)


class JumpModelDetector(RegimeDetector):
    """Jump model detector with causal forward-DP inference.

    Args:
        n_states: number of regimes.
        jump_penalty: cost lambda charged for each state switch.
        severity_indices: feature columns used to order states by severity.
        n_init: full coordinate-descent restarts; the lowest-loss fit is kept.
        max_iter: coordinate-descent iterations per restart.
        tol: stop when the loss improves by less than this.
        random_state: base seed for the k-means initialisations.
    """

    def __init__(
        self,
        n_states: int = 3,
        jump_penalty: float = 50.0,
        severity_indices: Optional[list[int]] = None,
        n_init: int = 10,
        max_iter: int = 50,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        super().__init__(n_states)
        self.jump_penalty = jump_penalty
        self.severity_indices = severity_indices
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # --- dynamic programming -------------------------------------------------

    def _local_cost(self, X: np.ndarray, means: np.ndarray) -> np.ndarray:
        """Squared distance from each observation to each centroid, (T, K)."""
        return ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)

    def _forward(self, cost: np.ndarray, backtrack: bool):
        """Forward DP. Returns accumulated cost V (T, K) and, if requested,
        back-pointers for Viterbi decoding."""
        t_n, k = cost.shape
        lam = self.jump_penalty
        switch = lam * (1.0 - np.eye(k))  # switch[k, j] = lam if j != k else 0
        v = np.empty((t_n, k))
        v[0] = cost[0]
        bp = np.empty((t_n, k), dtype=int) if backtrack else None
        for t in range(1, t_n):
            m = v[t - 1][None, :] + switch  # rows = target k, cols = previous j
            v[t] = cost[t] + m.min(axis=1)
            if backtrack:
                bp[t] = m.argmin(axis=1)
        return (v, bp) if backtrack else v

    def _viterbi(self, cost: np.ndarray) -> np.ndarray:
        v, bp = self._forward(cost, backtrack=True)
        t_n = cost.shape[0]
        s = np.empty(t_n, dtype=int)
        s[-1] = int(v[-1].argmin())
        for t in range(t_n - 1, 0, -1):
            s[t - 1] = bp[t][s[t]]
        return s

    # --- fitting -------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "JumpModelDetector":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")

        best_means = None
        best_loss = np.inf
        for r in range(self.n_init):
            km = KMeans(n_clusters=self.n_states, n_init=1, random_state=self.random_state + r)
            means = km.fit(X).cluster_centers_
            prev_loss = np.inf
            for _ in range(self.max_iter):
                cost = self._local_cost(X, means)
                s = self._viterbi(cost)
                # centroid update; keep the old centroid for an empty state
                new_means = means.copy()
                for j in range(self.n_states):
                    if np.any(s == j):
                        new_means[j] = X[s == j].mean(axis=0)
                means = new_means
                loss = cost[np.arange(len(X)), s].sum() + self.jump_penalty * int(
                    np.count_nonzero(np.diff(s))
                )
                if prev_loss - loss < self.tol:
                    break
                prev_loss = loss
            if loss < best_loss:
                best_loss, best_means = loss, means

        self.means_ = best_means
        self._order_by_severity(X.shape[1])
        self.fitted_ = True
        logger.info(
            "Fitted %d-state jump model (penalty %.0f, loss %.1f, severity order %s)",
            self.n_states,
            self.jump_penalty,
            best_loss,
            self.state_order_.tolist(),
        )
        return self

    def _order_by_severity(self, n_features: int) -> None:
        idx = self.severity_indices or list(range(n_features))
        score = self.means_[:, idx].sum(axis=1)
        self.state_order_ = np.argsort(score)
        self.rank_of_state_ = np.argsort(self.state_order_)

    # --- inference -----------------------------------------------------------

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        v = self._forward(self._local_cost(X, self.means_), backtrack=False)
        internal = v.argmin(axis=1)
        ranks = self.rank_of_state_[internal]
        proba = np.zeros((len(X), self.n_states))
        proba[np.arange(len(X)), ranks] = 1.0
        return proba

    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        s = self._viterbi(self._local_cost(np.asarray(X, dtype=float), self.means_))
        return self.rank_of_state_[s].astype(int)
