"""
Performance and significance metrics.

Standard return and risk statistics (annualised return and volatility, Sharpe
ratio, maximum drawdown), plus the probabilistic and deflated Sharpe ratios of
Bailey and Lopez de Prado. The deflated Sharpe ratio corrects the observed
Sharpe for the number of configurations tried and for non-normal returns, so an
apparent edge has to clear the bar set by the best of many trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def annualised_return(net: pd.Series) -> float:
    net = net.dropna()
    if len(net) == 0:
        return float("nan")
    return float((1.0 + net).prod() ** (TRADING_DAYS / len(net)) - 1.0)


def annualised_vol(net: pd.Series) -> float:
    return float(net.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(net: pd.Series, rf: pd.Series | float = 0.0, annualise: bool = True) -> float:
    """Sharpe ratio of the excess return; annualised by default."""
    r = net.dropna()
    excess = r - (rf if isinstance(rf, (int, float)) else rf.reindex(r.index).fillna(0.0))
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    sr = excess.mean() / sd
    return float(sr * np.sqrt(TRADING_DAYS)) if annualise else float(sr)


def max_drawdown(net: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative value (a positive number)."""
    eq = (1.0 + net.dropna()).cumprod()
    if len(eq) == 0:
        return float("nan")
    peak = eq.cummax()
    return float(((peak - eq) / peak).max())


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n: int, g3: float, g4: float) -> float:
    """PSR: probability that the true (per-period) Sharpe exceeds sr_benchmark.

    sr and sr_benchmark are per-period (not annualised); g3 and g4 are the
    skewness and kurtosis of the returns; n is the number of observations.
    """
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe across n_trials independent trials."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(net: pd.Series, trial_sharpes: list, rf: pd.Series | float = 0.0) -> float:
    """Deflated Sharpe ratio of a strategy given the Sharpes of all trials.

    trial_sharpes are the per-period (non-annualised) excess-return Sharpe ratios
    of every configuration examined, used to set the deflation benchmark. rf is
    the risk-free series, subtracted so a near-cash strategy is not flattered.
    Returns the probability that the strategy's Sharpe beats the expected best of
    the trials.
    """
    r = net.dropna()
    n = len(r)
    if n < 3:
        return float("nan")
    excess = r - (rf if isinstance(rf, (int, float)) else rf.reindex(r.index).fillna(0.0))
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    sr = excess.mean() / sd
    sr0 = expected_max_sharpe(np.var(trial_sharpes, ddof=1), len(trial_sharpes))
    g3 = float(skew(excess))
    g4 = float(kurtosis(excess, fisher=False))  # non-excess kurtosis (normal = 3)
    return probabilistic_sharpe_ratio(sr, sr0, n, g3, g4)


def summarise(bt: pd.DataFrame, rf: pd.Series | float = 0.0) -> dict:
    """Headline metrics for one backtest result (from run_backtest)."""
    net = bt["net"]
    years = len(net) / TRADING_DAYS
    return {
        "ann_return": annualised_return(net),
        "ann_vol": annualised_vol(net),
        "sharpe": sharpe_ratio(net, rf),
        "max_drawdown": max_drawdown(net),
        "ann_turnover": float(bt["turnover"].sum() / (2 * years)) if years > 0 else float("nan"),
    }
