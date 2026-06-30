"""
Change-point regime detector.

This detector makes few assumptions: it finds where the feature distribution
shifts, then labels each segment with a regime. Change points come from the
ruptures library (Truong, Oudre and Vayatis), and each segment takes the nearest
of a few severity-ordered prototypes (k-means centroids of the features).

Change-point search is offline, so the two paths differ in what they segment.
Smoothed labels segment the whole series once. Filtered (causal) labels segment
only the trailing window ending at day t, take the current run, and label it by
its mean, so no label depends on the future. Prototypes are fixed at fit time
and ordered calm < volatile < crisis by their stress means.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import ruptures as rpt
from sklearn.cluster import KMeans

from regime.detection.base import RegimeDetector
from regime.utils.logging import get_logger

logger = get_logger(__name__)


class ChangePointDetector(RegimeDetector):
    """Change-point detector with ruptures segmentation and nearest-prototype
    regime labels.

    Args:
        n_states: number of regimes.
        severity_indices: feature columns used to order the prototypes.
        model: ruptures cost model ("l2" detects mean shifts).
        min_size: minimum segment length.
        pen_scale: penalty multiplier; the ruptures penalty is
            pen_scale * n_features * log(segment length).
        window: trailing window length for the causal filtered path.
        random_state: seed for the k-means prototypes.
    """

    def __init__(
        self,
        n_states: int = 3,
        severity_indices: Optional[list[int]] = None,
        model: str = "l2",
        min_size: int = 10,
        pen_scale: float = 2.0,
        window: int = 126,
        random_state: int = 42,
    ):
        super().__init__(n_states)
        self.severity_indices = severity_indices
        self.model = model
        self.min_size = min_size
        self.pen_scale = pen_scale
        self.window = window
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "ChangePointDetector":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        km = KMeans(n_clusters=self.n_states, n_init=10, random_state=self.random_state)
        km.fit(X)
        self.centroids_ = km.cluster_centers_
        self._order_by_severity(X.shape[1])
        self.fitted_ = True
        logger.info(
            "Fitted change-point detector (%s model, severity order %s)",
            self.model,
            self.state_order_.tolist(),
        )
        return self

    def _order_by_severity(self, n_features: int) -> None:
        idx = self.severity_indices or list(range(n_features))
        score = self.centroids_[:, idx].sum(axis=1)
        self.state_order_ = np.argsort(score)
        self.rank_of_state_ = np.argsort(self.state_order_)

    def _classify(self, mean_vec: np.ndarray) -> int:
        """Severity rank of the prototype nearest a segment mean."""
        internal = int(((self.centroids_ - mean_vec) ** 2).sum(axis=1).argmin())
        return int(self.rank_of_state_[internal])

    def _breakpoints(self, segment: np.ndarray) -> list[int]:
        n = len(segment)
        pen = self.pen_scale * segment.shape[1] * np.log(max(n, 2))
        algo = rpt.Pelt(model=self.model, min_size=self.min_size).fit(segment)
        return algo.predict(pen=pen)  # list of segment-end indices, last == n

    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        labels = np.empty(len(X), dtype=int)
        prev = 0
        for end in self._breakpoints(X):
            labels[prev:end] = self._classify(X[prev:end].mean(axis=0))
            prev = end
        return labels

    def filtered_states(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        w = self.window
        labels = np.empty(len(X), dtype=int)
        for t in range(len(X)):
            window = X[max(0, t - w + 1) : t + 1]
            if len(window) < 2 * self.min_size:
                start = 0
            else:
                bkps = self._breakpoints(window)
                start = bkps[-2] if len(bkps) >= 2 else 0
            labels[t] = self._classify(window[start:].mean(axis=0))
        return labels

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        labels = self.filtered_states(X)
        proba = np.zeros((len(labels), self.n_states))
        proba[np.arange(len(labels)), labels] = 1.0
        return proba
