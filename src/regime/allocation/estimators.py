"""
Moment estimation for the allocators.

Covariance uses Ledoit-Wolf shrinkage, which pulls the sample covariance toward
a structured target and stays well conditioned on the short windows used here;
the raw sample matrix is the input the mean-variance literature calls an "error
maximiser". Expected returns use the window sample mean, the noisiest input, so
it is shrunk toward zero by a configurable factor. Features store log returns,
so they are converted to simple returns first, since a portfolio return is a
weighted sum of simple returns, not log returns.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def to_simple(log_returns: np.ndarray) -> np.ndarray:
    """Convert log returns to simple returns (exp(r) - 1)."""
    return np.expm1(np.asarray(log_returns, dtype=float))


def estimate_covariance(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of a (T, N) simple-return window."""
    returns = np.asarray(returns, dtype=float)
    cov = LedoitWolf().fit(returns).covariance_
    return cov * TRADING_DAYS if annualize else cov


def estimate_mean(returns: np.ndarray, shrinkage: float = 0.0, annualize: bool = True) -> np.ndarray:
    """Window sample mean, optionally shrunk toward zero.

    shrinkage = 0 keeps the raw mean; shrinkage = 1 returns a zero vector, which
    turns mean-variance into pure minimum-variance.
    """
    mu = np.asarray(returns, dtype=float).mean(axis=0)
    mu = (1.0 - shrinkage) * mu
    return mu * TRADING_DAYS if annualize else mu


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance matrix."""
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1.0
    corr = cov / np.outer(d, d)
    return np.clip(corr, -1.0, 1.0)
