"""
Common interface for regime detectors (Phase 3).

A detector assigns each trading day to one of n_states regimes. States are
returned in a fixed severity order: 0 is the calmest regime and n_states-1 the
most severe (crisis). Mapping a model's internal, arbitrarily-numbered states
onto this order is done once at fit time, so labels stay comparable across
refits and across detectors.

Two inference paths are provided:
- filtered_proba / filtered_states: causal. The estimate at day t uses only
  data up to and including t, which is what a walk-forward backtest may use.
- smoothed_states: full-sample (uses the whole series). Useful for plotting and
  for comparing against the filtered path, but not admissible inside a trading
  rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Severity-ordered regime labels for the three-state case.
CALM, VOLATILE, CRISIS = 0, 1, 2
REGIME_NAMES = {CALM: "calm", VOLATILE: "volatile", CRISIS: "crisis"}


def regime_name(label: int, n_states: int = 3) -> str:
    """Map a severity rank to a name, defaulting to calm/volatile/crisis."""
    if n_states == 3:
        return REGIME_NAMES.get(int(label), str(label))
    if label == 0:
        return "calm"
    if label == n_states - 1:
        return "crisis"
    return f"state{int(label)}"


class RegimeDetector(ABC):
    """Base class for regime detectors.

    Subclasses fit a model on a feature matrix and expose causal (filtered) and
    full-sample (smoothed) state estimates, both already mapped to the severity
    order described in the module docstring.
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "RegimeDetector":
        """Fit the detector on X, shape (n_samples, n_features)."""

    @abstractmethod
    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """Causal filtered posteriors, shape (n_samples, n_states).

        Row t is the posterior over regimes given X[:t+1] only. Columns are in
        severity order (column 0 = calm).
        """

    @abstractmethod
    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        """Full-sample most-likely state per row, in severity order."""

    def filtered_states(self, X: np.ndarray) -> np.ndarray:
        """Causal hard labels: the severity rank with the highest filtered
        posterior at each day."""
        return np.argmax(self.filtered_proba(X), axis=1).astype(int)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("detector is not fitted; call fit() first")
