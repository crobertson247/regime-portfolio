"""
Gaussian hidden Markov model regime detector.

A GaussianHMM is fitted with hmmlearn. The fitted parameters (initial
distribution, transition matrix, per-state means and covariances) are then used
directly, so the causal filtered inference does not depend on hmmlearn
internals. Filtering runs the forward recursion by hand: the filtered posterior
at day t,

    alpha_t(j) proportional to b_j(x_t) * sum_i alpha_{t-1}(i) a_{ij},

uses observations up to and including t only. Smoothed (full-sample) states come
from hmmlearn's Viterbi decoding and are provided for comparison, not for
trading.

Internal HMM states are unordered, so after fitting they are sorted by a
severity score (the summed standardised mean over the stress features) and
remapped to the calm < volatile < crisis order used throughout the module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from hmmlearn import hmm as _hmm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from regime.detection.base import RegimeDetector
from regime.utils.logging import get_logger

logger = get_logger(__name__)

_LOG_FLOOR = 1e-300


class HMMDetector(RegimeDetector):
    """Gaussian HMM detector with causal filtered inference.

    Args:
        n_states: number of regimes.
        severity_indices: column positions (within the fitted feature matrix)
            of the features that indicate stress. States are ordered by the sum
            of their means over these columns. Defaults to all columns.
        covariance_type: hmmlearn covariance type ("diag" or "full").
        n_iter: maximum EM iterations per fit.
        tol: EM convergence tolerance.
        n_restarts: random restarts; the fit with the highest log-likelihood is
            kept (EM is only locally optimal).
        random_state: base seed; restart r uses random_state + r.
        min_covar: floor added to the diagonal of the covariances for numerical
            stability.
    """

    def __init__(
        self,
        n_states: int = 3,
        severity_indices: Optional[list[int]] = None,
        covariance_type: str = "diag",
        n_iter: int = 100,
        tol: float = 1e-4,
        n_restarts: int = 8,
        random_state: int = 42,
        min_covar: float = 1e-3,
    ):
        super().__init__(n_states)
        self.severity_indices = severity_indices
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.min_covar = min_covar

    def fit(self, X: np.ndarray) -> "HMMDetector":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")

        best_model = None
        best_ll = -np.inf
        for r in range(self.n_restarts):
            model = _hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                min_covar=self.min_covar,
                random_state=self.random_state + r,
            )
            try:
                model.fit(X)
                ll = model.score(X)
            except Exception as exc:  # a restart may diverge; try the next
                logger.debug("HMM restart %d failed: %s", r, exc)
                continue
            if np.isfinite(ll) and ll > best_ll:
                best_ll, best_model = ll, model

        if best_model is None:
            raise RuntimeError("HMM failed to fit on every restart")

        self.model_ = best_model
        self.startprob_ = best_model.startprob_
        self.transmat_ = best_model.transmat_
        self.means_ = best_model.means_
        # covars_ returns full (n_states, d, d) matrices for any covariance_type.
        self.covars_ = best_model.covars_

        self._order_by_severity(X.shape[1])
        self.fitted_ = True
        logger.info(
            "Fitted %d-state HMM (log-likelihood %.1f, severity order %s)",
            self.n_states,
            best_ll,
            self.state_order_.tolist(),
        )
        return self

    def _order_by_severity(self, n_features: int) -> None:
        idx = self.severity_indices
        if not idx:
            idx = list(range(n_features))
        score = self.means_[:, idx].sum(axis=1)  # per internal state
        # ascending severity: lowest score -> calm, highest -> crisis
        self.state_order_ = np.argsort(score)
        # rank_of_state_[internal_state] = severity rank
        self.rank_of_state_ = np.argsort(self.state_order_)

    def _emission_logprob(self, X: np.ndarray) -> np.ndarray:
        """Log emission probability per state, shape (n_samples, n_states)."""
        n, k = X.shape[0], self.n_states
        log_b = np.empty((n, k))
        for j in range(k):
            log_b[:, j] = multivariate_normal.logpdf(
                X, mean=self.means_[j], cov=self.covars_[j], allow_singular=True
            )
        return log_b

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        log_b = self._emission_logprob(X)
        log_a = np.log(self.transmat_ + _LOG_FLOOR)
        log_pi = np.log(self.startprob_ + _LOG_FLOOR)

        n = X.shape[0]
        log_alpha = np.empty((n, self.n_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, n):
            # logsumexp over the previous state, done inline for speed
            m = log_alpha[t - 1][:, None] + log_a
            mx = m.max(axis=0)
            log_alpha[t] = log_b[t] + mx + np.log(np.exp(m - mx).sum(axis=0))

        # Normalise each row to a posterior, then reorder columns to severity rank.
        filt = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        return filt[:, self.state_order_]

    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        internal = self.model_.predict(np.asarray(X, dtype=float))
        return self.rank_of_state_[internal].astype(int)
